from astropy.io import fits
import matplotlib.pyplot as plt
import os
from glob import glob
from astropy.visualization import ZScaleInterval
from astropy.visualization import ImageNormalize
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import matplotlib.gridspec as gridspec
from astropy.table import Table
from scipy import interpolate
import scipy.ndimage as ndimage
from photutils.aperture import SkyEllipticalAperture, SkyCircularAperture
from photutils.aperture import aperture_photometry, ApertureStats
from astropy.visualization import simple_norm
import pandas as pd
from astropy.stats import SigmaClip
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve, convolve_fft

#source position: 150.15949, 2.1914772
#nebula position:
#0923 to do list: background source detection and rms random aperture and arrange parameters
class MultiBandPhotometry:
    def __init__(self, source_position, z, drop_telescopes=[], drop_bands=[], psf_homo=None):
        self.all_bands = {'HST':['F606W', 'F814W', 'F125W', 'F160W'],'JWST':['F115W', 'F150W', 'F277W', 'F444W'], 'SUBARU':['HSC-G','HSC-I','HSC-R','HSC-Y','HSC-Z','IB0945','NB0387','NB0527','NB0718','NB0816','NB0921','NB0973'],
             'CFHT':['H','i','Ks','u'], 'SPITZER':['ch1','ch2','ch3','ch4'], 'GEMINI':['Ks'], 'UKIRT':['J'], 'VISTA':['H','J','Ks','Y','NB118']}
        self.valid_bands = self.__drop_bands(drop_telescopes, drop_bands)
        self.source_position = source_position
        self.image_paths, self.image_files, self.header_files, self.wcss = self.__load_images()
        self.z = z
        self.psf_homo = psf_homo
        self.__load_zeropoints()
        self.filter_paths = dict(zip([telescope+'_'+band for telescope in self.valid_bands.keys() for band in self.valid_bands[telescope]],
                                [glob(f'/home/lupengjun/OII_emitter/data/filter/{telescope}/*{band}*')[0] for telescope in self.valid_bands.keys() for band in self.valid_bands[telescope]]))

    def __drop_bands(self, drop_telescopes, drop_bands):
        valid_bands = {}
        for key in self.all_bands:
            if key not in drop_telescopes:
                valid_bands[key] = [band for band in self.all_bands[key] if band not in drop_bands]
        return valid_bands
    
    def load_psf_kernel(self, telescope_band, psf_type):
        return fits.getdata(f'/home/lupengjun/OII_emitter/photometry/output/psf_kernel_pypher/kernel_{telescope_band}_to_{psf_type}.fits')

    def __load_images(self,):
        '''
        I: Chosen telescopes and band
        O: (A dictionary of image data, a dictionary of headers, a dictionary of rms map) 
        '''
        paths = {}
        datas = {}
        headers = {}
        wcss = {}
        for telescope in self.valid_bands.keys():
            for band in self.valid_bands[telescope]:
                if telescope == 'HST':
                    path = '/home/lupengjun/OII_emitter/data/image/COSMOS-archive/candels_cutouts/' #also acs_cutout
                    with fits.open(glob(path+f'*{band.lower()}*.fits')[0]) as hdu:
                        paths[telescope+'_'+band] = glob(path+f'*{band.lower()}*.fits')[0]
                        datas[telescope+'_'+band] = hdu[0].data
                        header = hdu[0].header.copy()
                        #remove SIP header with keywords start with A_ and B_ to avoid wcs misunderstood
                        for keyword in hdu[0].header:
                            if (keyword.startswith('A_')) or (keyword.startswith('B_')):
                                del header[keyword]
                        headers[telescope+'_'+band] = header  
                        wcss[telescope+'_'+band] = WCS(headers[telescope+'_'+band])
                elif telescope == 'JWST':
                    path = '/home/lupengjun/OII_emitter/data/image/COSMOS-Web_sep/'
                    with fits.open(glob(path+f'*{band.upper()}*.fits')[0]) as hdu:
                        paths[telescope+'_'+band] = glob(path+f'*{band.upper()}*.fits')[0]
                        datas[telescope+'_'+band] = hdu[f'{band.upper()}-CLEAR', 'SCI'].data
                        headers[telescope+'_'+band] = hdu[f'{band.upper()}-CLEAR', 'SCI'].header
                        wcss[telescope+'_'+band] = WCS(headers[telescope+'_'+band])
                elif telescope == 'SUBARU': # also subaru_tiles
                    path = '/home/lupengjun/OII_emitter/data/image/SUBARU/'
                    with fits.open(glob(path+f'{band}*/image-{band}*.fits')[0]) as hdu:
                        paths[telescope+'_'+band] = glob(path+f'{band}*/image-{band}*.fits')[0]
                        datas[telescope+'_'+band] = hdu[0].data
                        headers[telescope+'_'+band] = hdu[0].header
                        wcss[telescope+'_'+band] = WCS(headers[telescope+'_'+band])
                elif telescope == 'CFHT':
                    path = '/home/lupengjun/OII_emitter/data/image/COSMOS-archive/cfht_tiles/'
                    with fits.open(glob(path+f'*{band}*sci*fits')[0]) as hdu:
                        paths[telescope+'_'+band] = glob(path+f'*{band}*sci*fits')[0]
                        datas[telescope+'_'+band] = hdu[0].data
                        headers[telescope+'_'+band] = hdu[0].header
                        wcss[telescope+'_'+band] = WCS(headers[telescope+'_'+band])
                elif telescope == 'SPITZER':
                    path = '/home/lupengjun/OII_emitter/data/image/COSMOS-archive/irac_sci/'
                    with fits.open(glob(path+f'*{band}*fits')[0]) as hdu:
                        paths[telescope+'_'+band] = glob(path+f'*{band}*fits')[0]
                        datas[telescope+'_'+band] = hdu[0].data
                        headers[telescope+'_'+band] = hdu[0].header
                        wcss[telescope+'_'+band] = WCS(headers[telescope+'_'+band])
                elif telescope == 'GEMINI':
                    path = '/home/lupengjun/OII_emitter/data/image/COSMOS-archive/kpno_sci/'
                    with fits.open(glob(path+f'*{band}*sci*fits')[0]) as hdu:
                        paths[telescope+'_'+band] = glob(path+f'*{band}*sci*fits')[0]
                        datas[telescope+'_'+band] = hdu[0].data
                        headers[telescope+'_'+band] = hdu[0].header
                        wcss[telescope+'_'+band] = WCS(headers[telescope+'_'+band])
                elif telescope == 'UKIRT':
                    path = '/home/lupengjun/OII_emitter/data/image/COSMOS-archive/ukirt_tiles/'
                    with fits.open(glob(path+f'*{band}*sci*fits')[0]) as hdu:
                        paths[telescope+'_'+band] = glob(path+f'*{band}*sci*fits')[0]
                        datas[telescope+'_'+band] = hdu[0].data
                        headers[telescope+'_'+band] = hdu[0].header
                        wcss[telescope+'_'+band] = WCS(headers[telescope+'_'+band])
                elif telescope == 'VISTA':
                    path = '/home/lupengjun/OII_emitter/data/image/COSMOS-archive/ultravista_sci/'
                    with fits.open(glob(path+f'*{band}*fits')[0]) as hdu:
                        paths[telescope+'_'+band] = glob(path+f'*{band}*fits')[0]
                        datas[telescope+'_'+band] = hdu[0].data
                        headers[telescope+'_'+band] = hdu[0].header
                        wcss[telescope+'_'+band] = WCS(headers[telescope+'_'+band])

        return paths, datas, headers, wcss,

    def unit_convert(self, telescope_band):
        '''
        Convert image data to microJanskys
        '''
        telescope, band = telescope_band.split('_')
        if telescope == 'HST': # * {header[PHOTPLAM]/A} **2 *{header[PHOTFLAM]/ergcm^(-2)A^(-1)eletrons^(-1)} * {value}/ electrons*s^(-1)
            rate = 3.33564*10**10*self.header_files[telescope_band]['PHOTPLAM']**2*self.header_files[telescope_band]['PHOTFLAM']
        elif telescope == 'JWST':#weight map
            rate = 10**(-2) # 10*nanoJansky
        elif telescope == 'SUBARU':
            rate = 10**(6-(27-8.9)/2.5) #zeropoint = 27
        elif telescope == 'CFHT': #rms map
            rate = 10**(-3) #nanoJansky
        elif telescope == 'SPITZER':
            rate = (u.MJy/u.sr).to(u.uJy/u.degree**2)*np.abs(self.header_files[telescope_band]['CDELT1']*self.header_files[telescope_band]['CDELT2']) #MJy/Sr*degree**2
        elif telescope == 'GEMINI':
            rate = 10**(-3) #nanoJansky
        elif telescope == 'UKIRT':#rms map
            rate = 10**(-3) #nJy
        elif telescope == 'VISTA':#only skysub
            rate = 10**(6-(30-8.9)/2.5) #zeropoint = 30
        return rate

    def __measure_single_flux(self, aperture, telescope_band, bkg_regenerate=False, error_method='global', ax=None, cutout_size=10*u.arcsec): 
        data, header, wcs = self.image_files[telescope_band], self.header_files[telescope_band], self.wcss[telescope_band]
        bkg_map, rms_map, source_mask = self.load_background(telescope_band, regenerate=bkg_regenerate)
        data_bkgsub = data-bkg_map
        if self.psf_homo and telescope_band.split('_')[-1]!=self.psf_homo:
            kernel = self.__load_psf_kernel(telescope_band, self.psf_homo)
            data_bkgsub = convolve_fft(data_bkgsub, kernel, normalize_kernel=True, allow_huge=True)
            print(f'{telescope_band} convolved to target psf!')
        if error_method == 'global':
            phot_stats = ApertureStats(data_bkgsub, error=rms_map, aperture=aperture, wcs=wcs,)
            single_error = phot_stats.sum_err*self.unit_convert(telescope_band) #convert data to mJy
        elif error_method == 'local':
            phot_stats = ApertureStats(data_bkgsub, error=rms_map, aperture=aperture, wcs=wcs,)
            single_error = self.estimate_local_rms(data, wcs, source_mask, aperture)*self.unit_convert(telescope_band) #convert data to mJy
        single_flux = phot_stats.sum*self.unit_convert(telescope_band) #convert data to mJy
        single_area = phot_stats.sum_aper_area
        if ax:
            cutout = Cutout2D(data_bkgsub, position=aperture.positions, size=cutout_size, wcs=wcs)
            norm = ImageNormalize(cutout.data, interval=ZScaleInterval())
            ax.imshow(cutout.data, norm=norm, origin='lower')
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()    
            
            ax.set_xlabel('$\Delta$ RA[arcsec]')
            ax.set_ylabel('$\Delta$ DEC[arcsec]')
            ax.text(0.95*xmax, 0.95*ymin, ' '.join(telescope_band.split('_')), color='white', ha='right', va='bottom', fontstyle='italic', fontweight='bold')
            aperture.to_pixel(wcs=cutout.wcs).plot(ax, color='white', lw=2)
            #change pixel coordinate to arcsec
            new_xticks = (ax.get_xticks()-cutout.position_cutout[0])*wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value
            new_yticks = (ax.get_yticks()-cutout.position_cutout[1])*wcs.proj_plane_pixel_scales()[1].to(u.arcsec).value
            ax.set_xticklabels(np.around(new_xticks, 1).astype(str))
            ax.set_yticklabels(np.around(new_yticks, 1).astype(str))
        return single_flux, single_error, single_area

    def __load_zeropoints(self):
        self.zeropoints = {}
        for telescope_band in self.image_paths.keys():
            rate = self.unit_convert(telescope_band)
            self.zeropoints[telescope_band] = -2.5*np.log10(rate)+23.9
            
    def measure_flux(self, method='small', bkg_regenerate=False, error_method='global'):
        '''
        simple: Measure a small aperture that the inside is pure source at all bands(what about psf?)
        aperture: Use a empirical ellptical aperture that can conclude all the source to measure all bands
        galfit: Use galfit to fit morphology at JWST F150W. Change amplitude to measure the flux at each band.
        '''
        if method == 'small':
            aperture = SkyCircularAperture(self.source_position, r=0.4*u.arcsec)
        elif method == 'ellipse':
            aperture = SkyEllipticalAperture(self.source_position, a=1.2609045*u.arcsec, b=0.95159205*u.arcsec, theta=232*u.degree)
        elif method == 'upright':
            aperture = SkyCircularAperture(SkyCoord(150.1594161, 2.1915121, unit='deg'), r=0.3*u.arcsec)
        elif method == 'downleft':
            aperture = SkyCircularAperture(SkyCoord(150.1595557, 2.1914293, unit='deg'), r=0.3*u.arcsec)
        else:
            raise ValueError('method must be small or ellipse or upright or downleft!')
        fig = plt.figure(figsize=(12, 48))
        gs = fig.add_gridspec(15, 3,)
        i = -1
        flux_dic = {}
        error_dic = {}
        pixel_area_dic = {}
        for telescope in self.valid_bands.keys():
            flux_dic[telescope]  = {}
            error_dic[telescope] = {}
            pixel_area_dic[telescope]  = {}
            for band in self.valid_bands[telescope]:
                i += 1
                ax = fig.add_subplot(gs[i//3, i%3])
                flux, error, area = self.__measure_single_flux(aperture, '_'.join([telescope,band]), bkg_regenerate, error_method, ax)
                flux_dic[telescope][band] = flux
                error_dic[telescope][band] = error
                pixel_area_dic[telescope][band] = area
        return pd.DataFrame(flux_dic), pd.DataFrame(error_dic), pd.DataFrame(pixel_area_dic)

    def load_background(self, telescope_band, regenerate=False, use_originrms=True):
        telescope, band = telescope_band.split('_')
        if os.path.exists(f'/home/lupengjun/OII_emitter/photometry/output/bkg_map/{telescope_band}_bkg_map.fits') and not regenerate:
            with fits.open(f'/home/lupengjun/OII_emitter/photometry/output/bkg_map/{telescope_band}_bkg_map.fits') as hdu:
                bkg_map = hdu['BKG'].data
                rms_map = hdu['RMS'].data
                source_mask = hdu['MASK'].data
        else:
            bkg_map, rms_map, source_mask = self.estimate_background(telescope_band,)
            hdul = fits.HDUList([fits.PrimaryHDU(header=self.wcss[telescope_band].to_header()), fits.ImageHDU(data=bkg_map, name='BKG'), fits.ImageHDU(data=rms_map, name='RMS'), fits.ImageHDU(data=source_mask.astype(int), name='MASK')])
            hdul.writeto(f'/home/lupengjun/OII_emitter/photometry/output/bkg_map/{telescope_band}_bkg_map.fits', overwrite=True)
        if use_originrms and telescope in ['JWST', 'CFHT', 'UKIRT']:
            if telescope == 'JWST':
                with fits.open(glob('/home/lupengjun/OII_emitter/data/image/COSMOS-Web_sep/'+f'*{band.upper()}*.fits')[0]) as hdu:
                    rms_map = hdu[f'{band.upper()}-CLEAR', 'WHT'].data
                rms_map = np.sqrt(1/rms_map) #weight map = 1/sigma***2
            else:
                rms_map = fits.getdata(glob(f'/home/lupengjun/OII_emitter/data/image/COSMOS-archive/{telescope.lower()}_tiles/*{band}*rms*fits')[0])                
        return bkg_map, rms_map, source_mask
    
    def estimate_background(self, telescope_band, method='photoutils', plot=True):
        '''
        photoutils.Background2D to generate background map and plot bkgmap and bkg_sub map
        When measureing one source, do other sources considered as the background noise? 
        '''
        from astropy.convolution import convolve_fft, Gaussian2DKernel
        from photutils.background import Background2D, SExtractorBackground, StdBackgroundRMS, BkgZoomInterpolator
        from photutils.segmentation import detect_threshold, detect_sources
        
        data = self.image_files[telescope_band]
        wcs = self.wcss[telescope_band]
        if method == 'photoutils':
            #smooth data using gaussian kernel, sigma=5pixels
            kernel = Gaussian2DKernel(x_stddev=3, y_stddev=3) 
            image_convolved = convolve_fft(data, kernel)
            #source detection
            sigclip = SigmaClip(sigma=3, maxiters=10) #rough background estimate for source segmetation
            threshold = detect_threshold(data, nsigma=2.5, sigma_clip=sigclip)
            segment_img = detect_sources(image_convolved, threshold, npixels=10)
            source_mask = segment_img.make_source_mask()
            #avoid non-converage sky
            coverage_mask = data==0
            #do background estimation
            if 'SPITZER' in telescope_band: 
                box_size = data.shape[0] #evaluate SPITZER background as a whole to avoid over substraction
                BKG = Background2D(data=data, box_size=box_size, mask=source_mask, coverage_mask=coverage_mask, fill_value=0,
                                filter_size=(9, 9), exclude_percentile=50)
            else:
                box_size = int(1.26*u.arcsec/wcs.proj_plane_pixel_scales()[0].to(u.arcsec))*3 #box_size should be larger than the source
                BKG = Background2D(data=data, box_size=box_size, mask=source_mask, coverage_mask=coverage_mask, fill_value=0,
                                filter_size=(9, 9),)#default: bkg_estimator=SExtractorBackground(), bkgrms_estimator=StdBackgroundRMS(),interpolator=BkgZoomInterpolator()
            bkg_image = BKG.background
            rms_image = BKG.background_rms
            source_cutout = Cutout2D(data, position=self.source_position, size=10*u.arcsec, wcs=wcs)
            bkg_cutout = Cutout2D(bkg_image, position=self.source_position, size=10*u.arcsec, wcs=wcs)
            rms_cutout = Cutout2D(rms_image, position=self.source_position, size=10*u.arcsec, wcs=wcs)
            mask_cutout = Cutout2D(source_mask, position=self.source_position, size=10*u.arcsec, wcs=wcs)
            if plot:
                fig, axes = plt.subplots(6,2, figsize=(5,12))
                for ax, image, title in zip(axes.flatten(), [data, image_convolved, source_mask, bkg_image, data-bkg_image, rms_image, source_cutout.data, mask_cutout.data, bkg_cutout.data, source_cutout.data-bkg_cutout.data, rms_cutout.data],
                    ['original', 'smoothed', 'source_detection', 'background', 'background-sub','RMS', 'source', 'source-mask', 'source-bkg', 'source-bkgsub', 'source-rms']):
                    norm = ImageNormalize(image, interval=ZScaleInterval())
                    bar = ax.imshow(image, norm=norm, origin='lower')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.colorbar(bar, ax=ax, location='left')
                    ax.set_title(title)
                #BKG.plot_meshes(ax=axes[0,0], outlines=True, marker='.', color='white', alpha=0.3)
                fig.suptitle(telescope_band.replace('_', ' '))
                plt.savefig(f'/home/lupengjun/OII_emitter/photometry/output/bkg_map/photoutils_bkg_estimate_{telescope_band}.png')
                plt.close()

        return bkg_image, rms_image, source_mask

    def estimate_local_rms(self, img, wcs, source_mask, aperture, numbers=200,):
        combined_mask = source_mask + (img==0).astype(int)
        combined_mask[combined_mask>1] = 1 #avoid non-coverage sky
        img_cutout = Cutout2D(img, position=aperture.positions, wcs=wcs, size=30*u.arcsec)
        mask_cutout = Cutout2D(combined_mask, position=aperture.positions, wcs=wcs, size=30*u.arcsec)
        local_aperture = aperture.to_pixel(mask_cutout.wcs)
        n_positions = 0
        valid_positions = []
        while n_positions < numbers:
            local_aperture.positions = np.random.randint(low=0, high=img_cutout.data.shape[0], size=(numbers*5, 2))
            phot_table = aperture_photometry(mask_cutout.data, local_aperture)
            if (n_positions + np.sum(phot_table['aperture_sum']==0))<numbers: #dont have enough source aperture
                n_positions += np.sum(phot_table['aperture_sum']==0)
                positions = np.vstack([phot_table[phot_table['aperture_sum']==0]['xcenter'].value, 
                                    phot_table[phot_table['aperture_sum']==0]['ycenter'].value]).T
                valid_positions.append(positions) #choose numbers from top
            else: #valid position satisfy the numbers
                positions = np.vstack([phot_table[phot_table['aperture_sum']==0][:(numbers-n_positions)]['xcenter'].value, 
                                    phot_table[phot_table['aperture_sum']==0][:(numbers-n_positions)]['ycenter'].value]).T
                valid_positions.append(positions) #choose numbers from top
                break
        valid_positions = np.vstack(valid_positions)
        local_aperture.positions = valid_positions
        bkg_table = aperture_photometry(img_cutout.data, local_aperture)
        return np.std(bkg_table['aperture_sum'])

    def load_psf(self, telescope_band):
        '''
        Load psfs generated by PSFEx with 25 pixel length, stored in /home/lupengjun/OII_emitter/photometry/output/psfmodels
        JWST's psf are generated by WebbPSF with the same pixel scale of COSMOS-web and 51 pixel length, stored in /home/lupengjun/find_quiescent/WebbPSF/psf
        IRAC psf need to be specificly dealt with for the future.
        '''
        telescope, band = telescope_band.split('_')
        if telescope == 'JWST':
            psf = fits.getdata(f'/home/lupengjun/find_quiescent/WebbPSF/psf/NIRCam_{band.upper()}_51pixels.fits', 1).astype(np.float64)
        else:
            psf = fits.getdata(f'/home/lupengjun/OII_emitter/photometry/output/psfmodels/{telescope_band}_SEcat.psf', 1)['PSF_MASK'][0][0].astype(np.float64) #field-constant psf only have one image
        
        return psf

    
