import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from datetime import datetime
import os
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval, ImageNormalize, make_lupton_rgb
from scipy import interpolate
from scipy import ndimage
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import zoom
from astropy.convolution import convolve, convolve_fft

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
mpl.rcParams['axes.linewidth'] = 1

def source_cutout(galaxy, telescope_band, size=10*u.arcsec, ax=None):
    wcs = galaxy.wcss[telescope_band]
    cutout = Cutout2D(galaxy.image_files[telescope_band], position=galaxy.source_position, size=size, wcs=wcs)
    # aperture.to_pixel(wcs=cutout.wcs).plot(ax, color='red', lw=1)
    if ax:
        extent = np.concatenate([((np.array(cutout.bbox_cutout[0])-cutout.position_cutout[0])*wcs.proj_plane_pixel_scales()[0]).to(u.arcsec).value, 
                                ((np.array(cutout.bbox_cutout[1])-cutout.position_cutout[1])*wcs.proj_plane_pixel_scales()[1]).to(u.arcsec).value])
        norm = ImageNormalize(cutout.data, interval=ZScaleInterval())
        ax.imshow(cutout.data, norm=norm, extent=extent, origin='lower')
        ax.set_xlabel('$\Delta$ RA[arcsec]')
        ax.set_ylabel('$\Delta$ DEC[arcsec]')
    return cutout

def plot_cutout_aligned(axes, galaxy, telescope_band_list, color_list, aperture_type='ellipse', size=10*u.arcsec):
    axes_1d = axes.flatten()
    assert len(axes_1d) == len(telescope_band_list), 'Number of axes should be equal to the number of telescope bands!'
    if type(aperture_type) == str:
        apertures = [galaxy.load_aperture(aperture_type)]
    elif type(aperture_type) == tuple:
        apertures = [galaxy.load_aperture(aperture_type_) for aperture_type_ in aperture_type]
    else:
        raise TypeError('Wrong aperture type!')
    for ax, telescope_band in zip(axes_1d, telescope_band_list):
        wcs = galaxy.wcss[telescope_band]
        cutout = Cutout2D(galaxy.image_files[telescope_band], position=galaxy.source_position, size=size, wcs=wcs)
        # extent = np.concatenate([((np.array(cutout.bbox_cutout[0])-cutout.position_cutout[0])*wcs.proj_plane_pixel_scales()[0]).to(u.arcsec).value, 
        #                 ((np.array(cutout.bbox_cutout[1])-cutout.position_cutout[1])*wcs.proj_plane_pixel_scales()[1]).to(u.arcsec).value])
        norm = ImageNormalize(cutout.data, interval=ZScaleInterval())
        ax.imshow(cutout.data, norm=norm, origin='lower')
        for i, aperture in enumerate(apertures):
            aperture.to_pixel(wcs=cutout.wcs).plot(ax, color=color_list[i], lw=1.5)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(0.95*xmax, 0.95*ymin, telescope_band.split('_')[-1], color='white', ha='right', va='bottom', fontstyle='italic', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])


def plot_filter_curve(galaxy, telescope_band, ax, range_mode='related', length=8000, xmin=None, xmax=None, facecolor='lightblue'):
    emission_lines = {'$OVI$':1033.82, '$Ly\\alpha$':1215.24, '$NV$':1240.81, '$OI$':1305.53, '$CII$':1335.31, '$SiIV$': 1397.61, '$OIV$': 1399.8, '$CIV$':1549.48, '$HeII$':1640.4, '$OIII$':1665.85, 
                    '$AlIII$':1857.4, '$CIII$':1908.734, '$CII$':2326.0, '$NeIV$':2439.5, '$MgII$':2799.117, '$NeV$':3346.79, '$NeVI$': 3426.85, '$OII$':3727.09, '$OII$':3729.88, '$HeI$':3889.0, 
                    '$SII$':4072.3, '$H\Delta$':4102.89, '$H\gamma$':4341.68, '$OIII$':4364.44, '$OIII$':4960.30, '$OIII$':5008.24, '$OI$':6302.05, '$OI$':6365.54, '$NI$':6529.03, '$NII$':6549.86, 
                    '$H\\alpha$':6564.61, '$NII$':6585.27, '$SII$':6718.29,'$SII$':6732.67}
    trans_curve = np.loadtxt(galaxy.filter_paths[telescope_band])
    wave_min, wave_max = trans_curve[:,0].min(), trans_curve[:,0].max()
    wave_c = (trans_curve[:,0]*trans_curve[:,1]).sum()/trans_curve[:,1].sum()
    if range_mode == 'related':
        x_min = wave_c - 1.5*(wave_c - wave_min)
        x_max = wave_c + 1.5*(wave_max - wave_c)
    elif range_mode == 'fixed':
        x_min = wave_c - 0.5*length
        x_max = wave_c + 0.5*length
    elif range_mode == 'manual':
        x_min, x_max = (xmin, xmax)
        if wave_c > x_max or wave_c < x_min: return None        
    emission_lines_array = np.array(list(emission_lines.values()))
    emission_names_array = np.array(list(emission_lines.keys()))
    emission_lines_array_shift = emission_lines_array*(1+galaxy.z)
    visible_ind = np.where((emission_lines_array_shift>x_min)&(emission_lines_array_shift<x_max))[0]
    visible_lines = emission_lines_array_shift[visible_ind]
    visible_names = emission_names_array[visible_ind]
    #plot emission lines
    ax.vlines(visible_lines, 0, 1, colors='r', linestyle='dashed',)
    for i, (x, text) in enumerate(zip(visible_lines, visible_names)):
        if i%2 == 1:
            ax.text(x, 0.8, text, fontsize=10)
        else:
            ax.text(x, 1, text, fontsize=10)
    #plot filter curves
    p = ax.plot(trans_curve[:,0], trans_curve[:,1])
    color = p[0].get_color()
    ax.fill_between(trans_curve[:,0], trans_curve[:,1], facecolor=color, alpha=0.3)        
    ax.text(wave_c, 0.5, telescope_band.split('_')[-1], color=color, fontsize=10)
    ax.set_ylim(0,1.05)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('Wavelength(A)')

def plot_intensity_map(galaxy, telescope_band, bkg_telescope_band, cont_sub_func, size=10*u.arcsec, sigma=0.7, to_pixelscl=0.05, mode='plot'):
    '''
    Plot intensity map of a galaxy in a specific telescope band.
    Only available for NB0718 now.
    Subtracted HSC-R and HSC-I band and normalized.
    Return contours of the intensity map.
    12.3: Try to add [OIII] intensity map, by stacking NB973, HSC-Y and IB0945, and subtracting HSC-z and NB0921.
    cont_sub_func should be defined as cont_sub_func(fnu_NB, fnu_BB),
    return is the flux map.
    '''
    def cutout_data(telescope_band): #return cutout data
        data = galaxy.image_files[telescope_band]
        wcs = galaxy.wcss[telescope_band]
        cutout_ = Cutout2D(data, position=galaxy.source_position, size=size, wcs=wcs)
        cutout = cutout_.data
        if to_pixelscl:
            pixelscl = (wcs.proj_plane_pixel_scales()[0]).to(u.arcsec).value
            ratio = pixelscl/to_pixelscl
            cutout = zoom(cutout, ratio)/ratio**2
        return cutout
    def subtract_bkg(data, sigma=3): # sigma clipping
        return data-sigma_clipped_stats(data, sigma=sigma)[1]
    # def stack_cutout(cutout_list): #stack and normalize
    #     # if mode == 'plot': #get the pure nebula
    #     #     cutout = np.zeros_like(cutout_list[0])
    #     #     for cutout_ in cutout_list:
    #     #         cutout += cutout_
    #     #     return cutout/np.max(cutout)
    #     # elif mode == 'photometry': #remain the flux unit
    #     cutout_list = np.array(cutout_list)
    #     return np.mean(cutout_list, axis=0)
    def set_extent():
        data = galaxy.image_files[telescope_band]
        wcs = galaxy.wcss[telescope_band]
        cutout_ = Cutout2D(data, position=galaxy.source_position, size=size, wcs=wcs)
        extent = np.concatenate([((-np.array(cutout_.bbox_cutout[0])+cutout_.position_cutout[0])*wcs.proj_plane_pixel_scales()[0]).to(u.arcsec).value, 
                                    ((np.array(cutout_.bbox_cutout[1])-cutout_.position_cutout[1])*wcs.proj_plane_pixel_scales()[1]).to(u.arcsec).value])
        return extent
    if 'NB0718' in telescope_band:
        nebular_name = '[OII]'
    elif 'NB0973' in telescope_band:
        nebular_name = '[OIII]'
    else:
        raise TypeError('Wrong telescope band!')
    rate = galaxy.unit_convert(telescope_band) #img data to microJy
    bkg_telescope_band = list(bkg_telescope_band)
    fnu_img_NB = subtract_bkg(cutout_data(telescope_band))*rate
    fnu_img_BB = [subtract_bkg(cutout_data(bkg_telescope_band_))*rate for bkg_telescope_band_ in bkg_telescope_band]
    img_NB, img_BB, F_img = cont_sub_func(fnu_img_NB, fnu_img_BB)
    # nebular_cutout_list = [subtract_bkg(cutout_data(telescope_band_)) for telescope_band_ in nebular_list]
    # continuum_cutout_list = [subtract_bkg(cutout_data(telescope_band_)) for telescope_band_ in continuum_list]
    # stack_nebular = stack_cutout(nebular_cutout_list)
    # stack_continuum = stack_cutout(continuum_cutout_list)
    # cutout_nebular = stack_nebular - stack_continuum
    # cutout_nebular = subtract_bkg(cutout_nebular)
    # cutout_nebular = cutout_nebular/np.max(cutout_nebular)
    extent = set_extent()
    smooth_nebular = ndimage.gaussian_filter(F_img, sigma=sigma)
    smooth_nebular = smooth_nebular/np.max(smooth_nebular)

    nx, ny = smooth_nebular.shape
    if nx%2 == 0: array_x = smooth_nebular[(nx//2-1):(nx//2+1)].mean(axis=0)
    else: array_x = smooth_nebular[nx//2]
    if ny%2 == 0: array_y = smooth_nebular[:, (ny//2-1):(ny//2+1)].mean(axis=1)
    else: array_y = smooth_nebular[:, ny//2]
    #debug plots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(img_NB, extent=extent, origin='lower')
    #axs[0].set_title('+'.join([telescope_band_.split('_')[1] for telescope_band_ in nebular_list]))
    axs[0].set_title("Narrow band")
    axs[1].imshow(img_BB, extent=extent, origin='lower')
    #axs[1].set_title('+'.join([telescope_band_.split('_')[1] for telescope_band_ in continuum_list]))
    axs[1].set_title("Broad band continuum")
    axs[2].imshow(F_img, extent=extent, origin='lower')
    axs[2].set_title('Subtracted')
    plt.show()
    #normal plots
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                    left=0.1, right=0.9, bottom=0.1, top=0.9,
                    wspace=0.05, hspace=0.05)
    ax_img = fig.add_subplot(gs[1, 0])
    ax_x = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[1, 1])
    norm = ImageNormalize(smooth_nebular, interval=ZScaleInterval())
    ax_img.imshow(smooth_nebular, norm=norm, extent=extent, origin='lower')
    CS = ax_img.contour(smooth_nebular, extent=extent, colors='white', levels=np.arange(0.1, 1, 0.2))
    xmin, xmax = ax_img.get_xlim()
    ymin, ymax = ax_img.get_ylim() 
    ax_img.clabel(CS, CS.levels)
    ax_img.vlines([0], ymin, ymax, colors='white', linestyle='dashed')
    ax_img.hlines([0], xmin, xmax, colors='white', linestyle='dashed')
    ax_img.set_xlabel('$\Delta$ RA[arcsec]')
    ax_img.set_ylabel('$\Delta$ DEC[arcsec]')
    ax_img.text(0.95*xmax, 0.95*ymin, nebular_name, color='white', ha='right', va='bottom', fontstyle='italic', fontweight='bold')
    ax_x.plot(array_x, 'k-')
    ax_x.set_xticks([])
    ax_y.plot(array_y, np.arange(array_y.shape[0]), 'k-')
    ax_y.set_yticks([])
    
    if mode=='plot': return (smooth_nebular, Cutout2D(galaxy.image_files[telescope_band], position=galaxy.source_position, size=size, wcs=galaxy.wcss[telescope_band]))
    elif mode=='photometry': return (F_img, Cutout2D(galaxy.image_files[telescope_band], position=galaxy.source_position, size=size, wcs=galaxy.wcss[telescope_band]))

def plot_rgb_image(galaxy, telescope_bands, contours_data=None, levels=None, apertures=None, size=10*u.arcsec, stretch=5, Q=8, ax=None):
    '''
    Plot RGB image of the galaxy.
    Input: 
        telescope_bands, a list of telescope band, in the order of [blue, green, red].
        contours_data, if in 2D shape, it is the contours of the [OII] nebular intensity map.
                        if in 3D shape, it is the contours of the [OII] and [OIII] nebular intensity map.
    Output: RGB image with contours of the nebular intensity map.
    '''
    datas = []
    for telescope_band in telescope_bands:
        data, header, wcs = galaxy.image_files[telescope_band], galaxy.header_files[telescope_band], galaxy.wcss[telescope_band]
        bkg_map, rms_map, source_mask = galaxy.load_background(telescope_band, regenerate=False)
        data_bkgsub = data-bkg_map
        pixelscl = wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value
        #convolve to the same psf
        if galaxy.psf_homo and telescope_band.split('_')[-1]!=galaxy.psf_homo:
            kernel = galaxy.load_psf_kernel(telescope_band, galaxy.psf_homo)
            data_bkgsub = convolve_fft(data_bkgsub, kernel, normalize_kernel=True, allow_huge=True)
            print(f'{telescope_band} convolved to target psf!')
        cutout_ = Cutout2D(data_bkgsub, position=galaxy.source_position, size=size, wcs=galaxy.wcss[telescope_band])
        cutout = cutout_.data
        #resample to the same pixel scale
        if pixelscl != 0.05:
            ratio = pixelscl/0.05
            cutout = zoom(cutout, ratio)/ratio**2
        datas.append(cutout*galaxy.unit_convert(telescope_band))
    datas = np.array(datas)
    extent = np.concatenate([((-np.array(cutout_.bbox_cutout[0])+cutout_.position_cutout[0])*wcs.proj_plane_pixel_scales()[0]).to(u.arcsec).value, 
                                ((np.array(cutout_.bbox_cutout[1])-cutout_.position_cutout[1])*wcs.proj_plane_pixel_scales()[1]).to(u.arcsec).value])
    #plot
    if ax == None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,1,1)
    if len(telescope_bands) == 3: #single band for single color
        rgb_img = make_lupton_rgb(datas[2], datas[1], datas[0], stretch=stretch, Q=Q)
    elif len(telescope_bands) == 6: #two bands for single color
        rgb_img = make_lupton_rgb((datas[4]+datas[5])/2, (datas[2]+datas[3])/2, (datas[0]+datas[1])/2, stretch=stretch, Q=Q)
    else:
        print('Wrong number of bands!')
        return None
    ax.imshow(rgb_img, extent=extent, origin='lower',)
    #levels=np.arange(0.1, 1, 0.2)
    if np.any(contours_data): #draw nebular contours
        if contours_data.ndim == 2:
            assert levels.ndim == 1
            CS = ax.contour(contours_data, extent=extent, colors='white', alpha=0.6, levels=levels)
            # ax.clabel(CS, levels=CS.levels)
            ax.text(0.95*extent[0], 0.95*extent[-2], '$\\rm{[O\\uppercase\\expandafter{\\romannumeral2}]}$ nebula', color='white', ha='left', va='top', fontstyle='italic', fontweight='bold')
        elif contours_data.ndim == 3:
            assert levels.ndim == 2
            CS = ax.contour(contours_data[0], extent=extent, colors='white', alpha=0.4, levels=levels[0]) 
            # ax.clabel(CS, levels=CS.levels)
            ax.text(0.95*extent[0], 0.90*extent[-2], '$\\rm{[O\\uppercase\\expandafter{\\romannumeral2}]}$ nebula', color='white', ha='left', va='top', fontstyle='italic', fontweight='bold')
            CS = ax.contour(contours_data[1], extent=extent, colors='lightgreen', linestyles='dotted', alpha=0.8, levels=levels[1])
            # ax.clabel(CS, levels=CS.levels)
            ax.text(0.50*extent[0], 0.90*extent[-2], '$\\rm{[O\\uppercase\\expandafter{\\romannumeral3}]}$ nebula', color='lightgreen', ha='left', va='top', fontstyle='italic', fontweight='bold')
    if len(telescope_bands) == 3:
        ax.text(0.95*extent[0], 0.95*extent[-1], telescope_bands[0].split('_')[-1], color='lightblue', ha='left', va='top', fontstyle='italic', fontweight='bold')
        ax.text(0.70*extent[0], 0.95*extent[-1], telescope_bands[1].split('_')[-1], color='g', ha='left', va='top', fontstyle='italic', fontweight='bold')
        ax.text(0.45*extent[0], 0.95*extent[-1], telescope_bands[2].split('_')[-1], color='r', ha='left', va='top', fontstyle='italic', fontweight='bold')
    elif len(telescope_bands) == 6:
        ax.text(0.95*extent[0], 0.95*extent[-1], '+'.join([telescope_bands[0].split('_')[-1], telescope_bands[1].split('_')[-1]]), color='lightblue', ha='left', va='top', fontstyle='italic', fontweight='bold')
        ax.text(0.95*extent[0], 0.75*extent[-1], '+'.join([telescope_bands[2].split('_')[-1], telescope_bands[3].split('_')[-1]]), color='g', ha='left', va='top', fontstyle='italic', fontweight='bold')
        ax.text(0.95*extent[0], 0.55*extent[-1], '+'.join([telescope_bands[4].split('_')[-1], telescope_bands[5].split('_')[-1]]), color='r', ha='left', va='top', fontstyle='italic', fontweight='bold')
    ax.set_xlabel('$\Delta$ RA[arcsec]')
    ax.set_ylabel('$\Delta$ DEC[arcsec]')
    ax.tick_params(axis='both', which='both', color='white', grid_color='white')
    if apertures:
        for aperture in apertures:
            aperture.plot(ax=ax, color='lightgreen', lw=2)
    if ax == None:
        plt.savefig('_'.join(telescope_bands)+'.png')
        plt.show()
    return ax, extent