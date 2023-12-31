import build_psfs
import numpy as np
import webbpsf
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.convolution import Moffat2DKernel
from astropy.convolution import convolve, convolve_fft
from astropy.visualization import ImageNormalize
from astropy.visualization import ZScaleInterval, LogStretch, SqrtStretch
import os
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry

def remove_tick(ax):
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])

def radial_profile(image, bins=50):
    center = np.array(image.shape)/2
    radii = np.linspace(1, int(image.shape[0]/2), num=bins)
    apertures = [CircularAperture(center, r=r) for r in radii]
    phot_table = aperture_photometry(image, apertures)
    # convert measurement into array
    return radii, phot_table.to_pandas().values[0][3:]

def save_psf(psf, pixel_scale, path):
    #write psf to path as pypher required type
    new_hdu = fits.PrimaryHDU(data = psf)
    new_hdu.header['PIXSCALE'] = pixel_scale
    new_hdulist = fits.HDUList([new_hdu])
    new_hdulist.writeto(path, overwrite=True)

class PSF_homo():

    def __init__(self, img_path, img_ext, img_band, psf_type, pixel_size=33):
        with fits.open(img_path) as hdu:
            self.img_shape = hdu[img_ext].data.shape
            self.pixel_scale = WCS(hdu[img_ext].header).proj_plane_pixel_scales()[0].to(u.arcsec).value #unit: arcsec
        self.img_path = img_path
        self.img_ext = img_ext
        self.length_arcmin = self.img_shape[0]*self.pixel_scale/60
        self.img_band = img_band
        self.psf_type = psf_type
        self.pixel_size = pixel_size
    
    def build_self_psf(self, mode='Gaia', write_path='/home/lupengjun/OII_emitter/photometry/output/psf_for_pypher/'):
        '''
        'Gaia' mode:
        Build point source psf selected by Gaia from Xiaojing, Lin's code and choose which stacking type you want(sum or median)
        'PSFEx' mode:
        Read the existed PSF model generated by SExtractor+PSFEx, plan to rewrite for the futrue
        'WebbPSF' mode:
        Especially for JWST NIRCam images to generate modeled psf using WebbPSF
        '''
        psf_path = write_path+f'{self.img_band}_{mode}_psf.fits'
        if mode == 'Gaia':
            if os.path.exists(psf_path) and input('PSF file already exists, do you want to regenerate? Y/N:') !='Y':
                self.psf = fits.getdata(psf_path, 0)
            else:
                psf_ = build_psfs.build_psfs(fieldname='COSMOS', images_dir='/home/lupengjun/OII_emitter/photometry/output/psf_gaia/', images_band=[self.img_band], cutout_size=self.pixel_size, fov_arcmin=self.length_arcmin, ext=self.img_ext, image_list=[self.img_path])
                choose_type = input('Choose sum or median')
                if choose_type == 'sum':
                    self.psf = psf_.working_psf_sum
                elif choose_type == 'median':
                    self.psf = psf_.working_psf_median
                else:
                    raise TypeError('Inappropriate input type')
        elif mode == 'PSFEx': #for future to complete
            self.psf = fits.getdata(f'/home/lupengjun/OII_emitter/photometry/output/psfmodels/{self.img_band}_SEcat.psf', 1)['PSF_MASK'].astype(np.float64)[0][0]
        elif mode == 'WebbPSF':
            inst = webbpsf.instrument('NIRCam')
            inst.pixelscale = self.pixel_scale
            inst.filter = self.img_band.split('_')[-1].upper()
            psf_ = inst.calc_psf(fov_pixels=self.pixel_size, oversample=1)
            self.psf = psf_[0].data
        #psf normalization
        self.psf = self.psf/np.sum(self.psf)
        self.psf_path = psf_path
        save_psf(self.psf, self.pixel_scale, self.psf_path)

    def build_target_psf(self, write_path='/home/lupengjun/OII_emitter/photometry/output/psf_for_pypher/target_psfs/'):
        '''
        Moffat psf:
        Generate Moffat psf FWHM=0.8 arcsec, beta=2.5 with the same pixel scale as the image psf
        F444W psf:
        Use WebbPSF to generate NIRCam F444W PSF with the same pixel scale as the image psf
        '''
        #check if already exists target psf file:
        filename = f'{self.psf_type}_{self.pixel_scale:.2f}scl_{self.pixel_size}pix.fits'
        
        if os.path.exists(write_path+filename):
            print('Target PSF already exists! Read the existed file.')
            self.target_psf = fits.getdata(write_path+filename, 0)
        else:
            if self.psf_type == 'Moffat':
                FWHM_arcsec = 0.8
                beta = 2.5
                R_arcsec = FWHM_arcsec/np.sqrt(2**(1/beta)-1)/2
                R_pix = R_arcsec/self.pixel_scale
                self.target_psf = Moffat2DKernel(R_pix, beta, x_size=self.pixel_size, y_size=self.pixel_size).array
            elif self.psf_type == 'F444W':
                inst = webbpsf.instrument('NIRCam')
                inst.pixelscale = self.pixel_scale
                inst.filter = 'F444W'
                psf_ = inst.calc_psf(fov_pixels=self.pixel_size, oversample=1)
                self.target_psf = psf_[0].data
            else:
                raise TypeError('No type match your requirement')
            #psf normalization
            self.target_psf = self.target_psf/np.sum(self.target_psf)
            save_psf(self.target_psf, self.pixel_scale, write_path+filename)
        self.target_psf_path = write_path+filename

    def kernel_gen(self, mode='pypher', write_path='/home/lupengjun/OII_emitter/photometry/output/psf_kernel_pypher/'):
        '''
        Using pypher to read the kernel, lower the regularization factor for HST matching to JWST.
        '''
        if mode == 'pypher':
            if self.psf_type == 'F444W':
                r_factor = 3.e-03
            else:
                r_factor = 1.e-04 #pypher default value
            
            self.kernel_path = f'{write_path}kernel_{self.img_band}_to_{self.psf_type}.fits'
            os.system(f'pypher {self.psf_path} {self.target_psf_path} {self.kernel_path} -r {r_factor}')

        else:
            raise TypeError('No mode match your requirement')
        
    def test_psfmatching(self, write_path='/home/lupengjun/OII_emitter/photometry/output/test_conv_psf/'):
        '''
        Plot check image and radial profile of original, convolved and target PSF. Adapted from Daming, Yang
        '''
        kernel = fits.getdata(self.kernel_path, 0)
        self.psf_conv = convolve_fft(self.psf, kernel, normalize_kernel=True, allow_huge=True)
        self.psf_conv = self.psf_conv / np.sum(self.psf_conv)
        
        psf_src = self.psf
        psf_src_conv = self.psf_conv
        psf_tar = self.target_psf

        #plot check image
        fig, ax = plt.subplots(3, 2, figsize=(8,8))
        ax_imshow = []
        norm = ImageNormalize(psf_src, stretch=SqrtStretch(), interval=ZScaleInterval())
        ax_imshow.append(ax[0][0].imshow(psf_src, norm=norm, cmap='gray'))
        ax[0][0].set_title('before')
        remove_tick(ax[0][0])

        ax_imshow.append(ax[0][1].imshow(psf_src_conv, norm=norm, cmap='gray'))
        ax[0][1].set_title('after')
        remove_tick(ax[0][1])

        norm_k = ImageNormalize(kernel, stretch=SqrtStretch(), interval=ZScaleInterval())
        ax_imshow.append(ax[1][0].imshow(kernel, norm=norm_k, cmap='gray'))
        ax[1][0].set_title('kernel')
        remove_tick(ax[1][0])

        ax_imshow.append(ax[1][1].imshow(psf_tar, norm=norm, cmap='gray'))
        ax[1][1].set_title(f'{self.psf_type}')
        remove_tick(ax[1][1])

        res = psf_src_conv-psf_tar
        ax_imshow.append(ax[2][0].imshow(res, norm=norm, cmap='gray'))
        ax[2][0].set_title('residual')
        remove_tick(ax[2][0])

        res = np.abs(psf_src_conv-psf_tar)/psf_tar
        norm_k = ImageNormalize(res, stretch=SqrtStretch(), interval=ZScaleInterval())
        ax_imshow.append(ax[2][1].imshow(res, norm=norm_k, cmap='gray'))
        ax[2][1].set_title(f'residual/{self.psf_type}')
        remove_tick(ax[2][1])

        ax_flat = ax.flatten()
        for _ax in ax_flat:
            remove_tick(_ax)
        for _ax in ax_imshow:
            fig.colorbar(_ax)
        title = f'{self.img_band}->{self.psf_type}'
        fig.suptitle(title, fontsize=15)
        fig.tight_layout()
        plt.savefig(write_path+f'img_{title}.png')
        plt.show()
        plt.clf()
        plt.close()

        #plot radial profile
        r, profile = radial_profile(psf_src, bins=20)
        plt.plot(r, profile, label=self.img_band)
        r, profile = radial_profile(psf_tar, bins=20)
        plt.plot(r, profile, label=self.psf_type)
        r, profile = radial_profile(psf_src_conv, bins=20)
        plt.plot(r, profile, label=f'{self.img_band} (convolved)', ls='--')
        plt.yscale('log')
        plt.legend()
        plt.savefig(write_path+f'radial_{title}.png')
        plt.show()
        plt.clf()
        plt.close()