import os
from glob import glob
import numpy as np
from tqdm import tqdm, trange
import time

from photutils import psf
from photutils.centroids import centroid_1dg, centroid_2dg
from photutils import CircularAperture, aperture_photometry, CircularAnnulus

from astroquery.gaia import Gaia
from astroquery.mast import Observations
from astroquery.sdss import SDSS
from astroquery.vizier import Vizier

from astropy.table import Table
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.coordinates import SkyCoord
from astropy.io import fits, ascii
import astropy.units as u
from astropy.modeling.fitting import NonFiniteValueError

import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib.patches import Circle, Ellipse, Rectangle, Polygon
from astropy.visualization import (imshow_norm,
                                   ZScaleInterval,
                                   SquaredStretch,
                                   LinearStretch,
                                  LogStretch,
                                  MinMaxInterval,
                                  ImageNormalize)

from IPython import embed

def gaia_to_AB(gaia_mag, filtname):
    if filtname == 'G':
        zpt_gaia = 2861.  # Jy
    elif filtname == 'BP':
        zpt_gaia = 3478.
    elif filtname == 'RP':
        zpt_gaia = 2461.

    flux_jy = 10 ** (-0.4 * gaia_mag) * zpt_gaia
    m_AB = -2.5 * np.log10(flux_jy / 3631)
    return m_AB

def _2mass_to_AB(_2mass_mag, filtname):
    if filtname == 'H':
        zpt_2mass = 1011.
    elif filtname == 'J':
        zpt_2mass =  1574.
    elif filtname == 'K':
        zpt_2mass = 654.
    else:
        print("pick a valid 2MASS filter")
    flux_jy = 10**(-0.4 * _2mass_mag) * zpt_2mass
    m_AB = -2.5 * np.log10(flux_jy/3631)
    return m_AB

def profile_2D(psf):
    x_marg = [np.sum(psf[:, i]) for i in range(psf.shape[0])]
    y_marg = [np.sum(psf[j, :]) for j in range(psf.shape[1])]
    return np.array(x_marg), np.array(y_marg)


def bg_subtract(image, rms_clean_level):
    im_shape = image.shape
    m1, m2 = int((image.shape[0] / 2) - 5), int((image.shape[0] / 2) + 5)

    mask_array = image.copy()
    mask_array[m1:m2, m1:m2] = np.nan
    mask_flat = mask_array.flatten()
    mask_flat_fin = mask_flat[np.isnan(mask_flat) == False]
    med = np.median(mask_flat_fin)
    rms = np.sqrt(np.average(mask_flat_fin ** 2))

    center = image.shape[0] / 2

    # measure the background in an annulus around star
    aper_annulus = CircularAnnulus((center, center), r_in=27, r_out=30)
    phot = aperture_photometry(image, aper_annulus)
    bkg_mean = phot['aperture_sum'][0] / aper_annulus.area

    # subtract background
    im_new = image - bkg_mean
    im_flat = im_new.flatten()

    # if there are remaining positive peaks, set them to local rms
    if rms_clean_level != 0:
        mask_new = im_new.copy()
        mask_new[m1:m2, m1:m2] = np.nan
        mask_new_flat = mask_new.flatten()
        mask_new_flat_fin = mask_new_flat[np.isnan(mask_new_flat) == False]
        rms_new = np.sqrt(np.average(mask_new_flat_fin ** 2))
        mask_new_flat[mask_new_flat > (rms_clean_level * rms_new)] = rms_new

        new_cutout_bkgsub = np.reshape(mask_new_flat, im_shape)
        new_cutout_bkgsub[m1:m2, m1:m2] = im_new[m1:m2, m1:m2]

    else:
        new_cutout_bkgsub = im_new

    return new_cutout_bkgsub

class create_single_psf():
    '''
    Builds psf for each filter by creating cutouts around Gaia-selected stars, performing background
    subtraction, resampling/recentering stars, and stacking them to a final psf.
    Used inside build_psfs class.
    '''

    def __init__(self, image_dir, filtername, gaia_coords, gaia_cat, ext=0):

        self.image_dir = image_dir

        fitsfile = fits.open(glob(self.image_dir + '*' + filtername + '*sci.fits')[0])[ext]

        self.image = fitsfile.data
        self.filtername = filtername
        self.header = fitsfile.header
        self.wcs_info = WCS(self.header)
        self.gaia_coords = gaia_coords
        self.gaia_cat = gaia_cat
        self.xpix, self.ypix = self.wcs_info.world_to_pixel(self.gaia_coords)
        self.image[np.where(np.isnan(self.image)==True)] = -99
        self.image[np.where(np.isinf(self.image) == True)] = -99

        checkdir = glob(self.image_dir + 'psfs*')
        if not checkdir:
            os.makedirs(self.image_dir + 'psfs')

    def cutouts(self, cutout_size, plot=True, messages=True,offset_lim=5):
        cutouts = []
        Z = int(cutout_size / 2)
        im_cent = Z + 1
        cutout_indices = []
        cutout_count = 0
        self.source_ids = []
        self.cutout_size = cutout_size

        ymax, xmax = self.image.shape
        print("central offset limit: ", offset_lim, " pixels")

        for j, (x, y) in enumerate(zip(self.xpix, self.ypix)):
            # initially make the cutouts larger than what we ask for
            x1, x2 = int(round(x) - Z), int(round(x) + Z + 1)
            y1, y2 = int(round(y) - Z), int(round(y) + Z + 1)

            if (x1 > 0 and x2 < xmax and y1 > 0 and y2 < ymax):
                cutout_count += 1
                cutout_indices.append(j)
                self.source_ids.append(self.gaia_cat['ID'][j])

                print("Index for this star: ", self.gaia_cat['ID'][j])

                # Give a bit of a buffer so we can get a ZxZ cutout after adjusting centroid
                new_cutout_badastro = self.image[y1 - 10:y2 + 10, x1 - 10:x2 + 10]
                if new_cutout_badastro.shape[0] == 0 or new_cutout_badastro.shape[1] == 0:
                    print("Cutout is empty")
                    continue
                
                
                try:
                    x_centroid, y_centroid = centroid_1dg(new_cutout_badastro)
                except NonFiniteValueError or ValueError:
                    # eliminate if image contains non-finite values
                    print("Non-finite values in image")
                    continue

                new_cutout_badastro[np.where(np.isnan(new_cutout_badastro)==True)[0]] = 0
                new_cutout_badastro[np.where(np.isfinite(new_cutout_badastro) == False)[0]] = 0


                # if the centroid is more than 5 pixels off from the star position you
                # either have an astrometry issue or there's a bright neighbor/binary
                # so just take it out of the sample

                x_centroid_lim_1  =  new_cutout_badastro.shape[0]//2 - offset_lim
                x_centroid_lim_2 = new_cutout_badastro.shape[0]//2 + offset_lim
                y_centroid_lim_1 = new_cutout_badastro.shape[1]//2 - offset_lim
                y_centroid_lim_2 = new_cutout_badastro.shape[1]//2 + offset_lim

                if x_centroid < x_centroid_lim_1 or x_centroid > x_centroid_lim_2:
                    print("x_centroid_lim_1: ", x_centroid_lim_1, "x_centroid_lim_2: ", x_centroid_lim_2)
                    print("x_centroid: ", x_centroid)
                    print("bad centroid for star %s, skipping"%self.gaia_cat['ID'][j])
                    continue
                if y_centroid < y_centroid_lim_1 or y_centroid > y_centroid_lim_2:
                    print("y_centroid_lim_1: ", y_centroid_lim_1, "y_centroid_lim_2: ", y_centroid_lim_2)
                    print("y_centroid: ", y_centroid)
                    print("bad centroid for star %s, skipping" % self.gaia_cat['ID'][j])
                    continue

                # Then oversample it by 10x to find any sub-pixel astrometric offset
                psf1_pixscale = np.sqrt(self.wcs_info.proj_plane_pixel_area()).to('arcsec').value
                new_cutout_badastro_fine = psf.resize_psf(new_cutout_badastro, psf1_pixscale, psf1_pixscale / 10)
                new_cutout_badastro_fine[np.where(np.isfinite(new_cutout_badastro_fine) == False)] = 0

                # Find new centroid
                x_offset, y_offset = centroid_1dg(new_cutout_badastro_fine)

                x1_corr, x2_corr = int(round(x_offset - (10 * Z + 5))), int(round(x_offset + (10 * Z + 6)))
                y1_corr, y2_corr = int(round(y_offset - (10 * Z + 5))), int(round(y_offset + (10 * Z + 6)))

                new_cutout_fine = new_cutout_badastro_fine[y1_corr:y2_corr, x1_corr:x2_corr]
                try:
                    cutout_centroid_x, cutout_centroid_y = centroid_1dg(new_cutout_fine)
                except ValueError:
                    # eliminate if image contains non-finite values
                    print("Non-finite values in image")
                    continue

                new_cutout = psf.resize_psf(new_cutout_fine, psf1_pixscale / 10, psf1_pixscale)
                final_centroid_x, final_centroid_y = centroid_1dg(new_cutout)

                from astropy.visualization import ImageNormalize
                from astropy.visualization import ZScaleInterval, LogStretch
                norm = ImageNormalize(new_cutout, stretch=LogStretch(), interval=ZScaleInterval(contrast=0.1))

                plt.imshow(new_cutout, origin='lower', norm=norm, cmap='gray')
                plt.savefig(f'/home/dyang/data/JWST_DataCollection/CEERS_DR05_cat_daming/tmp_fig/{j}_cutout.png')

                nc_shape = new_cutout.shape
                nc_flat = new_cutout.flatten()

                if messages:
                    print("original xy centroid: %.3f, %.3f" % (x_centroid, y_centroid))
                    print("10x over-sampled centroid: %.3f, %.3f" % (x_offset, y_offset))
                    print("10x over-sampled centroid, corrected: %.3f, %.3f" % (cutout_centroid_x, cutout_centroid_y))
                    print("final centroid on native pixel scale: %.3f, %.3f" % (final_centroid_x, final_centroid_y))

                # Subtract background
                aper_annulus = CircularAnnulus((Z, Z), r_in=27, r_out=30)
                phot = aperture_photometry(new_cutout, aper_annulus)
                bkg_mean = phot['aperture_sum'][0] / aper_annulus.area
                if messages:
                    print("bkg mean: %.3f"%bkg_mean)

                new_cutout_bkgsub = new_cutout - bkg_mean

                # normalize background-subtracted cutout
                new_cutout_norm = new_cutout_bkgsub / np.max(new_cutout_bkgsub)
                cutouts.append(new_cutout_norm)

                if plot:
                    fig, ax1 = plt.subplots(figsize=(3, 3))
                    n = new_cutout_norm

                    try:
                        im, norma = imshow_norm(n, ax1, origin='lower',  # interval=ZScaleInterval(),
                                                stretch=LogStretch())
                    except ValueError:
                        im, norma = imshow_norm(new_cutout, ax1, origin='lower',  # interval=ZScaleInterval(),
                                                stretch=LogStretch())

                    fig.colorbar(im)

                    radii = np.array([27, 30])
                    c_patches = [Circle(xy=(Z, Z), radius=r, color='k', fill=False) for r in radii]
                    pp = [ax1.add_patch(p) for p in c_patches]

                    plt.axvline(final_centroid_x)
                    plt.axhline(final_centroid_y)
                    plt.title("star index: " + str(self.gaia_cat['ID'][j]))

                    plt.savefig(f'~/data/JWST_DataCollection/CEERS_DR05_cat_daming/tmp_fig/test_{self.filtername}_cutout.pdf')

        self.star_cutouts = cutouts
        self.cutout_indices = cutout_indices

        return cutouts

    def cutout_cog(self, cutout_size, min_radius, max_radius):
        cogs = []
        for ind, n in enumerate(self.star_cutouts):
            cog = curve_of_growth(n, cutout_size, min_radius, max_radius)
            cogs.append(cog)

        self.cogs = cogs

        return cogs

    def sum_med_psf(self, exclusion_list=None):

        # Give the list of indices for stars you don't want included
        if exclusion_list is None:
            exclusion_list = []

        sum_psf = np.sum([yy for xx, yy in enumerate(self.star_cutouts) if xx not in exclusion_list], axis=0)
        med_psf = np.median([yy for xx, yy in enumerate(self.star_cutouts) if xx not in exclusion_list], axis=0)

        # re-centroid the final product one last time - don't want an offset for psf-matching
        x_offset_sum, y_offset_sum = centroid_1dg(sum_psf)
        xdiff_sum, ydiff_sum = np.abs(self.cutout_size / 2 - x_offset_sum), np.abs(self.cutout_size / 2 - y_offset_sum)

        new_sum_psf = np.roll(sum_psf, int(xdiff_sum), axis=0)
        new_sum_psf = np.roll(new_sum_psf, int(ydiff_sum), axis=1)

        x_offset_med, y_offset_med = centroid_1dg(med_psf)
        xdiff_med, ydiff_med = np.abs(self.cutout_size / 2 - x_offset_med), np.abs(self.cutout_size / 2 - y_offset_med)

        new_med_psf = np.roll(med_psf, int(round(xdiff_med)), axis=0)
        new_med_psf = np.roll(new_med_psf, int(round(ydiff_med)), axis=1)

        sum_psf_bg_norm = bg_subtract(new_sum_psf, rms_clean_level=0)
        sum_psf_bg_norm /= np.sum(sum_psf_bg_norm)

        med_psf_bg_norm = bg_subtract(new_med_psf, rms_clean_level=0)
        med_psf_bg_norm /= np.sum(med_psf_bg_norm)

        self.sum_psf = sum_psf_bg_norm
        self.med_psf = med_psf_bg_norm

        self.sum_psf_profile2d = profile_2D(self.sum_psf)
        self.med_psf_profile2d = profile_2D(self.med_psf)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6, 3))
        imshow_norm(self.sum_psf, ax1, origin='lower',  # interval=ZScaleInterval(),
                    stretch=LogStretch(), )

        centroid = self.sum_psf.shape[0]/2
        ax1.axhline(centroid, color='white', linewidth=0.5)
        ax1.axvline(centroid, color='white', linewidth=0.5)
        ax1.set_title("sum-stacked")

        imshow_norm(self.med_psf, ax2, origin='lower',  # interval=ZScaleInterval(),
                    stretch=LogStretch())
        ax2.axhline(centroid, color='white', linewidth=0.5)
        ax2.axvline(centroid, color='white', linewidth=0.5)
        ax2.set_title("median-stacked")

        x_sum, y_sum = self.sum_psf_profile2d
        ax3.plot(np.arange(self.sum_psf.shape[0]), x_sum / x_sum.max(), linewidth=1)
        ax3.plot(np.arange(self.sum_psf.shape[1]), y_sum / y_sum.max(), linewidth=1)
        ax3.set_ylabel("normalized 1D profile")
        ax3.axhline(0, color='black', linewidth=0.5)
        ax3.axvline(centroid, color='black', linewidth=0.5)

        x_med, y_med = self.med_psf_profile2d
        ax4.plot(np.arange(self.med_psf.shape[0]), x_med / x_med.max(), linewidth=1)
        ax4.plot(np.arange(self.med_psf.shape[1]), y_med / y_med.max(), linewidth=1)
        ax4.axhline(0, color='black', linewidth=0.5)
        ax4.axvline(centroid, color='black', linewidth=0.5)


        return sum_psf_bg_norm, med_psf_bg_norm

    def save_psf(self, empirical_psf, filename):

        hdr = fits.Header()
        hdr['NAXIS'] = 2
        hdr['NAXIS1'] = empirical_psf.shape[0]
        hdr['NAXIS2'] = empirical_psf.shape[1]

        hdr['PIXSCALE'] = np.sqrt(self.wcs_info.proj_plane_pixel_area()).to('arcsec').value

        new_hdu = fits.PrimaryHDU(empirical_psf, header=hdr)
        new_hdulist = fits.HDUList([new_hdu])

        new_psf = new_hdulist.writeto(self.image_dir + 'psfs/' + filename + '_psf.fits', overwrite=True)


class build_psfs():

    def __init__(self, fieldname, images_dir, cutout_size=101,suffix='NIRCAM',offset_lim=5,fov_arcmin=4, ext=0,filt=None,image_list=None,center=None,maglim=25):
        '''
        Args:
            fieldname: Field  name
            images_dir: directory where your science images live
            cutout_size: in pixels, default is 101
        This class queries Gaia at the center position of the pointing and then builds the psf
        using the psf class above.
        '''
        self.fieldname = fieldname
        self.image_dir = images_dir + '/' if images_dir[-1] != '/' else images_dir
        if image_list is None:
            self.image_list = glob(self.image_dir + '*sci.fits')
        else:
            self.image_list = image_list
        
        if filt:
            self.image_list = [x for x in self.image_list if filt in x]

        if not self.image_list:
            print("Check the name of your directory.")
        else:
            print("load images:", self.image_list)

        qso_fits = fits.open(self.image_list[0])
        self.qso_wcs = WCS(qso_fits[ext].header)
        if center is None:
            central_x, central_y = qso_fits[ext].data.shape[0] / 2, qso_fits[ext].data.shape[1] / 2
            self.central_coord = self.qso_wcs.pixel_to_world(central_x, central_y)
        else:
            self.central_coord = SkyCoord(center[0],center[1],unit='deg')

        print("central coordinate in this field:", self.central_coord)
        qso_fits.close()

        self.ext = ext
        self.cutout_size = cutout_size
        self.suffix = suffix
        self.psf = {}

        if os.path.exists(self.image_dir + self.fieldname + '_gaiaquery.txt'):
            print('Gaia table exists. Skip query.')
            gaia_table = Table.read(self.image_dir + self.fieldname + '_gaiaquery.txt', format='ascii')
            self.gaia_coords = SkyCoord(ra=gaia_table['ra'], dec=gaia_table['dec'], unit=(u.hourangle, u.degree))
            self.gaia_cat = gaia_table
        else:
            self.query_gaia(fov_arcmin=fov_arcmin, maglim=maglim)
        
        for im in self.image_list:
            self.empirical_psf(im,offset_lim)

    def query_gaia(self, fov_arcmin=4, maglim=25):
        print("Querying Gaia for stars in the field of view ({} arcmin)".format(fov_arcmin))
        gaia_query_qso = Gaia.query_object_async(coordinate=self.central_coord, radius=fov_arcmin * u.arcmin)
        reduced_query = gaia_query_qso['ra', 'dec', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']

        gaia_table = Table(names=['ID', 'ra', 'dec', 'g_mag', 'bp_mag', 'rp_mag', 'bp-rp', 'Jmag', 'Hmag'],
                           dtype=('int', 'S9', 'S9', 'float', 'float', 'float', 'float', 'float', 'float'))

        im_limits = self.qso_wcs.array_shape
        for s, star in enumerate(reduced_query):
            if star['phot_g_mean_mag'] > maglim:
                pass
            else:
                coord = SkyCoord(star['ra'], star['dec'], unit='deg')

                is_in_pointing = self.qso_wcs.world_to_pixel(coord)
                #print("number %s has pixel coord %.2f, %.2f"%(s, is_in_pointing[0], is_in_pointing[1]))
                # make sure it's in the field and it's not on the edge
                size_lim = self.cutout_size/2
                if is_in_pointing[1] < size_lim + 11\
                        or is_in_pointing[1] > im_limits[0] - size_lim - 11\
                        or is_in_pointing[0] < size_lim + 11\
                        or is_in_pointing[0] > im_limits[1] - size_lim - 11:
                    continue
                else:
                    ra = coord.ra.to_string(unit=u.hourangle, sep=':')
                    dec = coord.dec.to_string(unit=u.degree, sep=':')
                    gmag_AB = gaia_to_AB(star['phot_g_mean_mag'], 'G')
                    bpmag_AB = gaia_to_AB(star['phot_bp_mean_mag'], 'BP')
                    rpmag_AB = gaia_to_AB(star['phot_rp_mean_mag'], 'RP')
                    bp_rp = bpmag_AB - rpmag_AB

                    ROW = [s, ra, dec, gmag_AB, bpmag_AB, rpmag_AB, bp_rp, 0, 0]
                    gaia_table.add_row(ROW)

            # Some fields do not have suitable Gaia stars (they might be saturated or the field
            # has low stellar density) so we query 2MASS point sources as well.

            viz = Vizier(columns=['RAJ2000', 'DEJ2000', 'Jmag', 'Hmag', 'Qflg'])
            viz_query = viz.query_region(self.central_coord, radius='4min', catalog='2MASS-PSC')[0]
            working_gaia_cat = SkyCoord(ra=gaia_table['ra'], dec=gaia_table['dec'], unit=(u.hourangle, u.degree))
            for s, star in enumerate(viz_query):
                coord = SkyCoord(star['RAJ2000'], star['DEJ2000'], unit='deg')
                is_in_pointing = self.qso_wcs.world_to_pixel(coord)
                # make sure it's in the field and it's not on the edge
                size_lim = self.cutout_size / 2
                if is_in_pointing[1] < size_lim + 11 \
                        or is_in_pointing[1] > im_limits[0] - size_lim - 11 \
                        or is_in_pointing[0] < size_lim + 11 \
                        or is_in_pointing[0] > im_limits[1] - size_lim - 11:
                    pass
                else:
                    ra = coord.ra.to_string(unit=u.hourangle, sep=':')
                    dec = coord.dec.to_string(unit=u.degree, sep=':')
                    # check to see if it's already in our Gaia catalog
                    gaia_row, gaia_sep, _ = coord.match_to_catalog_sky(working_gaia_cat)
                    if gaia_sep < 0.2 * u.arcsec:
                        gaia_table[gaia_row]['Hmag'] = star['Hmag']
                        gaia_table[gaia_row]['Jmag'] = star['Jmag']
                    else:
                        ROW = [100+s, ra, dec, 0, 0, 0, -99, star['Jmag'], star['Hmag']]
                        gaia_table.add_row(ROW)


        self.gaia_coords = SkyCoord(ra=gaia_table['ra'], dec=gaia_table['dec'], unit=(u.hourangle, u.degree))
        self.gaia_cat = gaia_table
        ascii.write(gaia_table, self.image_dir + self.fieldname + '_gaiaquery.txt', overwrite=True)


    def empirical_psf(self, im, offset_lim):
        """
        im: image name, e.g. 'CEERS_HDR-f606w_sci.fits'
        offset_lim: 
        """

        filtername = 'F' + im.split('-f')[1].split('_')[0]
        filtername = filtername.upper()

        self.psf[filtername] = create_single_psf(self.image_dir, filtername.lower(), self.gaia_coords, self.gaia_cat, ext = self.ext)
        working_psf = self.psf[filtername]
        print(f"######## working on filter {filtername}")

        working_psf_cutout = working_psf.cutouts(self.cutout_size, offset_lim=offset_lim)

        time.sleep(10)
        exclusion_list = [int(item) for item in input("give exclusion indices separated by commas, 99 if none: ").split(',')]
        working_psf_sum, working_psf_median = working_psf.sum_med_psf(exclusion_list)
        working_psf.save_psf(working_psf_sum, self.fieldname + f'_{self.suffix.upper()}_' + filtername + '_sum')
        working_psf.save_psf(working_psf_median, self.fieldname + f'_{self.suffix.upper()}_' + filtername + '_median')



