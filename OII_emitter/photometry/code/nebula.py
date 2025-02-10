import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import multibandphotometry as mbp
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval
from astropy.visualization import ImageNormalize
from scipy.stats import norm
from scipy import ndimage
import pyregion
from astropy.stats import SigmaClip, sigma_clip
from photutils.segmentation import detect_threshold, detect_sources
from scipy.optimize import curve_fit

def int_filter_curve(file):
    '''
    Read the filter curve file and return the integral of the filter curve.
    '''
    data = np.loadtxt(file, unpack=True) #wavelength in angstrom
    frequencies = 3e18/data[0] #Hz
    return np.trapz(data[1][::-1], frequencies[::-1])

def calc_surface_brightness(nebula, mask, xscale, yscale, mode='flux'):
    # flux_sum = np.sum(nebula[mask])*10**(6-(27-8.9)/2.5)*int_filter #zeropoint 27 to microJy
    if mode == 'flux':
        flux_sum = np.sum(nebula[mask])
    elif mode == 'rms':
        flux_sum = np.sqrt(np.sum(nebula[mask]**2))
    else:
        TypeError('The mode is unknown.')
    area = np.sum(mask)*xscale*yscale #arcsec^2
    return flux_sum/area

class nebula:
    def __init__(self, name, size=30*u.arcsec, sigma=3, apply_cutout=True, choose_std=1):
        self.name = name
        self.size = size
        self.sigma = sigma
        self.choose_std = choose_std
        self.mbp = self.__init_mbp()
        self.data, self.cutout = self.get_nebula()
        self.wcs = self.cutout.wcs
        self.rmsmap = self.calc_total_rms_map()
        self.contour_mask, self.contour = self.get_contour_mask()
        if size != 10*u.arcsec and apply_cutout: 
            self.data, self.rmsmap, self.contour_mask, self.smoothed = self.apply_cutout()
        self.region_masks, self.combined_region_mask = self.load_region()
        self.intermasks = self.get_intermasks()

    def __init_mbp(self):
        self.source_position = SkyCoord(150.15949, 2.1914772, frame='icrs', unit=u.deg)
        z = 0.924
        return mbp.MultiBandPhotometry(self.source_position, z, drop_telescopes=['HST', 'JWST', 'CFHT', 'USKIRT', 'VISTA', 'SPIZER'])

    def __get_contour_threshold(self, ):
        '''
        Only support 1-sigma above the background value for now
        '''
        def gaussian(x, mu, sigma, A):
            return A * np.exp(-(x-mu)**2/(2*sigma**2))
        bkg_map, rms_map, source_mask = self.mbp.load_background(self.telescope_band)
        source_mask_cutout = Cutout2D(source_mask, self.source_position, size=self.size, wcs=self.mbp.wcss[self.telescope_band])
        masked_nebula = np.ma.array(self.data, mask=source_mask_cutout.data)
        # #source detection
        # sigclip = SigmaClip(sigma=3, maxiters=None) #rough background estimate for source segmetation
        # threshold = detect_threshold(self.smoothed, nsigma=2.5, sigma_clip=sigclip)
        # segment_img = detect_sources(self.smoothed, threshold, npixels=10)
        # mask_smoothed = segment_img.make_source_mask()
        # #
        # masked_nebula = np.ma.array(self.smoothed, mask=mask_smoothed) #use smoothed image to get the background value
        # masked_nebula_1D = masked_nebula.data.flatten()[~masked_nebula.mask.flatten()]
        # masked_nebula_1D = masked_nebula_1D[~np.isnan(masked_nebula_1D)]
        # masked_nebula_1D = sigma_clip(self.smoothed.flatten(), sigma=3, maxiters=None, masked=False)
        masked_nebula_1D = masked_nebula.data.flatten()[~masked_nebula.mask.flatten()]
        mu, std = norm.fit(masked_nebula_1D)
        # mu, std = norm.fit(masked_nebula_1D, loc=np.median(masked_nebula_1D), scale=np.std(masked_nebula_1D)/2)
        # nm = ImageNormalize(self.smoothed, interval=ZScaleInterval())
        nm = ImageNormalize(masked_nebula, interval=ZScaleInterval())
        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.imshow(self.smoothed, origin='lower', norm=nm)
        plt.subplot(1,2,2)
        n, bins, _ = plt.hist(masked_nebula_1D, bins=1000, edgecolor='blue', histtype='step', density=True)
        xlim0 = np.abs([np.percentile(masked_nebula_1D, 0.1), np.percentile(masked_nebula_1D, 99.9)]).max()
        plt.xlim(-xlim0, xlim0)
        ymin, ymax = plt.ylim()        
        x = (bins[:-1] + bins[1:]) / 2
        y = n
        popt, _ = curve_fit(gaussian, x, y, p0=[mu, std, np.max(y)])
        mu, std, A = popt
        # x = np.linspace(-xlim0, xlim0, 10000) 
        # p = norm.pdf(x, mu, std) 
        # plt.plot(x, p, 'k')
        plt.plot(x, gaussian(x, *popt), 'r-')
        plt.vlines([np.median(masked_nebula_1D)], [ymin], [ymax], color='red', linestyle='dashed')
        plt.vlines([np.percentile(masked_nebula_1D, 16), np.percentile(masked_nebula_1D, 84)], ymin, ymax, color='red', linestyle='dotted') # 1 sigma
        plt.fill_between(x, gaussian(x, *popt), where=((x>(mu-std))&(x<(mu+std))), facecolor='k', alpha=0.2)
        plt.title(self.telescope_band)
        plt.show()

        return mu+self.choose_std*std
        # return np.mean(masked_nebula)+self.choose_std*std
    
    def __get_contour(self,):
        nebula_smoothed = ndimage.gaussian_filter(self.data, sigma=self.sigma)
        self.smoothed = nebula_smoothed
        self.threshold = self.__get_contour_threshold()        
        nm = ImageNormalize(nebula_smoothed, interval=ZScaleInterval())
        plt.imshow(self.data, origin='lower', norm=nm)
        contour = plt.contour(nebula_smoothed, levels=[self.threshold], colors='red')
        # self.contout_path = contour.collections[0].get_paths()[-1]
        return contour
    
    def cont_sub_func(self, fnu_NB, fnu_BB, mode='flux'):
        '''
        Calculated by Mingyu
        fnu_NB, fnu_BB should in microJy, i.e. 10^-29 erg s^-1 cm^-2 Hz^-1
        '''
        if self.name == 'OII':
            epsilon_NB = 109.67 #Angstrom
            epsilon_BB = [80.70, 33.60] #Angstrom
            lambda_line = 7175.54 #Angstrom
        elif self.name == 'OIII':
            epsilon_NB = 264.31 #Angstrom
            epsilon_BB = [319.64] #Angstrom
            lambda_line = 9637.35 #Angstrom
        if mode == 'flux':
            img_NB = epsilon_NB*fnu_NB
            img_BB = np.sum([epsilon*fnu for epsilon, fnu in zip(epsilon_BB, fnu_BB)], axis=0)
            return img_NB*3e18/(lambda_line**2)*1e-29, img_BB*3e18/(lambda_line**2)*1e-29, (img_NB - img_BB)*3e18/(lambda_line**2)*1e-29 #erg s^-1 cm^-2
        elif mode == 'rms':
            rms_NB = [epsilon_NB*fnu_NB]
            rms_BB = [epsilon*fnu for epsilon, fnu in zip(epsilon_BB, fnu_BB)]
            square_sum = np.sum(np.array(rms_NB+rms_BB)**2, axis=0)
            return np.sqrt(square_sum)*3e18/(lambda_line**2)*1e-29 #erg s^-1 cm^-2
        else:
            TypeError('The mode is unknown.')

    def get_nebula(self):
        if self.name == 'OII':
            self.telescope_band = 'SUBARU_NB0718'
            self.bkg_telescope_band = ['SUBARU_HSC-R', 'SUBARU_HSC-Z']
        elif self.name == 'OIII':
            self.telescope_band = 'SUBARU_NB0973'
            self.bkg_telescope_band = ['SUBARU_HSC-Z']
        else:
            TypeError('The name of nebula is unknown.')
        return mbp.plot_intensity_map(self.mbp, self.telescope_band, self.bkg_telescope_band, self.cont_sub_func, size=self.size, to_pixelscl=None, mode='photometry')

    def get_contour_mask(self,):
        def create_mask_from_path(path_list, shape):
            mask = np.zeros(shape, dtype=bool)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for path in path_list:
                        mask[i, j] = path.contains_point([j, i])
                        if mask[i, j]: break
            return mask 
        contour = self.__get_contour()
        path_list = contour.collections[0].get_paths()
        mask = create_mask_from_path(path_list, self.data.shape)
        masked_nebula = np.ma.array(self.data, mask=mask)
        plt.imshow(masked_nebula, origin='lower')
        plt.show()
        return mask, contour        
    
    def load_region(self):
        region_masks = []
        for i in range(1, 6):
            region = pyregion.open(f'/home/lupengjun/OII_emitter/photometry/code/radial_profile/pandas_suit_series/pandas_suit{i}.reg')
            mask = region.get_mask(shape=self.data.shape)
            region_masks.append(mask)
        combined_mask = np.zeros(self.data.shape, dtype=int)
        for i, mask in enumerate(region_masks, start=1):
            combined_mask[mask] = i
        return region_masks, combined_mask
    
    def get_intermasks(self, ):
        intermasks = []
        for mask in self.region_masks:
            intermasks.append(mask&self.contour_mask)
        return intermasks
    
    def calc_total_rms_map(self,):
        telescope_band_list = [self.telescope_band]+self.bkg_telescope_band
        big_rms_map_list = [self.mbp.load_background(telescope_band)[1]*self.mbp.unit_convert(telescope_band) for telescope_band in telescope_band_list]
        rms_map_list = [Cutout2D(big_rms_map, self.source_position, size=self.size, wcs=self.mbp.wcss[telescope_band]).data for big_rms_map, telescope_band in zip(big_rms_map_list, telescope_band_list)]
        return self.cont_sub_func(rms_map_list[0], rms_map_list[1:], mode='rms')

    def apply_cutout(self, ):
        size = 10*u.arcsec
        new_list = []
        for img in [self.data, self.rmsmap, self.contour_mask, self.smoothed]:
            img_ = Cutout2D(img, self.source_position, size=size, wcs=self.wcs)
            img = img_.data
            new_list.append(img)
        self.wcs = img_.wcs
        print('Cutout applied.')
        return new_list

    def radial_profile(self, mode='mag'):
        xscale, yscale = self.wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value, self.wcs.proj_plane_pixel_scales()[1].to(u.arcsec).value
        # radius = np.linspace(0, 22.667181*np.sqrt(xscale**2+yscale**2), 5)
        radius = np.linspace(0, 3.808, 11)[1::2] #arcsec read from ds9, devided into 5 bins, radius in median of each bin

        surface_brightness = []
        surface_rms = []
        for mask in self.intermasks:
            surface_brightness.append(calc_surface_brightness(self.data, mask, xscale, yscale, mode='flux'))
            surface_rms.append(calc_surface_brightness(self.rmsmap, mask, xscale, yscale, mode='rms'))
        surface_brightness = np.array(surface_brightness)
        surface_rms = np.array(surface_rms)

        # if mode == 'mag':
        #     return radius, -2.5*np.log10(surface_brightness)+8.9, np.abs(np.vstack((2.5*np.log10((surface_brightness+surface_rms)/surface_brightness), -2.5*np.log10((surface_brightness-surface_rms)/surface_brightness)))) #arcsec, ABmag/arcsec^2
        if mode == 'flux':
            return radius, surface_brightness/1e-17, surface_rms/1e-17  #10^-17 erg s^-1 cm^-2/arcsec^2
        else:
            TypeError('The mode is unknown.')

if __name__ == '__main__':
    nebula = nebula('OII')
    print(nebula.radial_profile(mode='mag'))
    print(nebula.radial_profile(mode='flux'))
    nebula = nebula('OIII')
    print(nebula.radial_profile(mode='mag'))
    print(nebula.radial_profile(mode='flux'))