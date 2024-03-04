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

def int_filter_curve(file):
    '''
    Read the filter curve file and return the integral of the filter curve.
    '''
    data = np.loadtxt(file, unpack=True) #wavelength in angstrom
    frequencies = 3e18/data[0] #Hz
    return np.trapz(data[1][::-1], frequencies[::-1])

def calc_surface_brightness(nebula, mask, xscale, yscale):
    # flux_sum = np.sum(nebula[mask])*10**(6-(27-8.9)/2.5)*int_filter #zeropoint 27 to microJy
    flux_sum = np.sum(nebula[mask])
    area = np.sum(mask)*xscale*yscale #arcsec^2
    return flux_sum/area

class nebula:
    def __init__(self, name, size=30*u.arcsec):
        self.name = name
        self.size = size
        self.mbp = self.__init_mbp()
        self.data, self.cutout = self.get_nebula()
        self.wcs = self.cutout.wcs
        self.contour_mask = self.get_contour_mask()
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
        bkg_map, rms_map, source_mask = self.mbp.load_background(self.telescope_band)
        source_mask_cutout = Cutout2D(source_mask, self.source_position, size=self.size, wcs=self.mbp.wcss[self.telescope_band])
        masked_nebula = np.ma.array(self.data, mask=source_mask_cutout.data)
        masked_nebula_1D = masked_nebula.data.flatten()[~masked_nebula.mask.flatten()]
        mu, std = norm.fit(masked_nebula_1D)
        nm = ImageNormalize(masked_nebula, interval=ZScaleInterval())

        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.imshow(masked_nebula, origin='lower', norm=nm)
        plt.subplot(1,2,2)
        plt.hist(masked_nebula_1D, bins=30, edgecolor='blue', histtype='step', density=True)
        xlim0 = np.abs([np.percentile(masked_nebula_1D, 0.1), np.percentile(masked_nebula_1D, 99.9)]).max()
        plt.xlim(-xlim0, xlim0)
        ymin, ymax = plt.ylim()
        x = np.linspace(-xlim0, xlim0, 10000) 
        p = norm.pdf(x, mu, std) 
        plt.plot(x, p, 'k') 
        plt.vlines([np.median(masked_nebula_1D)], [ymin], [ymax], color='red', linestyle='dashed')
        plt.vlines([np.percentile(masked_nebula_1D, 16), np.percentile(masked_nebula_1D, 84)], ymin, ymax, color='red', linestyle='dotted') # 1 sigma
        plt.fill_between(x, p, where=((x>(mu-std))&(x<(mu+std))), facecolor='k', alpha=0.2)
        plt.title(self.telescope_band)
        plt.show()
        return mu+std   
    
    def __get_contour(self, sigma=3):
        nebula_smoothed = ndimage.gaussian_filter(self.data, sigma=sigma)
        self.smoothed = nebula_smoothed
        self.threshold = self.__get_contour_threshold()        
        nm = ImageNormalize(nebula_smoothed, interval=ZScaleInterval())
        plt.imshow(self.data, origin='lower', norm=nm)
        contour = plt.contour(nebula_smoothed, levels=[self.threshold], colors='red')
        self.contout_path = contour.collections[0].get_paths()[-1]
        return contour
    
    def get_nebula(self):
        if self.name == 'OII':
            self.telescope_band = 'SUBARU_NB0718'
            self.bkg_telescope_band = ['SUBARU_HSC-R', 'SUBARU_HSC-I']
        elif self.name == 'OIII':
            self.telescope_band = 'SUBARU_NB0973'
            self.bkg_telescope_band = ['SUBARU_HSC-Z', 'SUBARU_NB0921']
        else:
            TypeError('The name of nebula is unknown.')
        return mbp.plot_intensity_map(galaxy=self.mbp, telescope_band=self.telescope_band, size=self.size, to_pixelscl=None, mode='photometry')

    def get_contour_mask(self, sigma=3):
        def create_mask_from_path(path, shape):
            mask = np.zeros(shape, dtype=bool)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    mask[i, j] = path.contains_point([j, i])
            return mask 
        contour = self.__get_contour(sigma=sigma)
        path = contour.collections[0].get_paths()[-1]
        mask = create_mask_from_path(path, self.data.shape)
        masked_nebula = np.ma.array(self.data, mask=mask)
        plt.imshow(masked_nebula, origin='lower')
        plt.show()
        return mask        
    
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
        telescope_band_list = self.bkg_telescope_band+[self.telescope_band]
        big_rms_map_list = [self.mbp.load_background(telescope_band)[1] for telescope_band in telescope_band_list]
        rms_map_list = [Cutout2D(big_rms_map, self.source_position, size=self.size, wcs=self.mbp.wcss[telescope_band]).data for big_rms_map, telescope_band in zip(big_rms_map_list, telescope_band_list)]
        return np.sqrt(np.sum(np.array(rms_map_list)**2, axis=0))

    def radial_profile(self, mode='mag'):
        xscale, yscale = self.wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value, self.wcs.proj_plane_pixel_scales()[1].to(u.arcsec).value
        radius = np.linspace(0, 22.667181*(xscale+yscale)/2, 5)
        
        surface_brightness = []
        surface_rms = []
        for mask in self.intermasks:
            surface_brightness.append(calc_surface_brightness(self.data, mask, xscale, yscale))
            surface_rms.append(calc_surface_brightness(self.calc_total_rms_map(), mask, xscale, yscale))
        surface_brightness = np.array(surface_brightness)
        surface_rms = np.array(surface_rms)

        if mode == 'mag':
            return radius, -2.5*np.log10(surface_brightness)+27, np.abs(np.vstack((2.5*np.log10((surface_brightness+surface_rms)/surface_brightness), -2.5*np.log10((surface_brightness-surface_rms)/surface_brightness)))) #arcsec, ABmag/arcsec^2
        elif mode == 'flux':
            return radius, surface_brightness*10**(-(27-8.9)/2.5)*int_filter_curve(self.mbp.filter_paths[self.telescope_band])*1e-6, surface_rms*10**(-(27-8.9)/2.5)*int_filter_curve(self.mbp.filter_paths[self.telescope_band])*1e-6#arcsec, 10^-17 erg s^-1 cm^-2/arcsec^2
        else:
            TypeError('The mode is unknown.')

if __name__ == '__main__':
    nebula = nebula('OII')
    print(nebula.radial_profile(mode='mag'))
    print(nebula.radial_profile(mode='flux'))
    nebula = nebula('OIII')
    print(nebula.radial_profile(mode='mag'))
    print(nebula.radial_profile(mode='flux'))