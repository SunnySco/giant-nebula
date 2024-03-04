import numpy as np 
import bagpipes as pipes
from astropy.io import fits
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u
import sys
from datetime import datetime
import os
mpl.rcParams['text.usetex'] = True
drop_telescope, drop_band = [], []

def load_photometry_data(version,):
    fluxes = pd.read_csv(f'/home/lupengjun/OII_emitter/photometry/output/flux_sum_{version}.csv', index_col=0).drop(drop_telescope, axis=1).drop(drop_band, axis=0).stack().values
    errors = pd.read_csv(f'/home/lupengjun/OII_emitter/photometry/output/error_sum_{version}.csv', index_col=0).drop(drop_telescope, axis=1).drop(drop_band, axis=0).stack().values
    photometry = np.c_[fluxes, errors]
    return photometry
def load_filter_path(version,):
    stack_df = pd.read_csv(f'/home/lupengjun/OII_emitter/photometry/output/flux_sum_{version}.csv', index_col=0).drop(drop_telescope,axis=1).drop(drop_band, axis=0).stack().reset_index()
    filters = stack_df['level_0']
    telescopes = stack_df['level_1']
    filter_list = [glob(f'/home/lupengjun/OII_emitter/data/filter/{telescope}/*{band}*')[0] for telescope, band in zip(telescopes, filters)]
    return filter_list
def load_SFH_model(mode):
    #Tau-model
    exponential = {}                                  # Tau-model star-formation history component
    exponential["age"] = (0.1, 15.)                   # Vary age between 100 Myr and 15 Gyr. In practice 
                                            # the code automatically limits this to the age of
                                            # the Universe at the observed redshift.
    exponential["tau"] = (0.3, 10.)                   # Vary tau between 300 Myr and 10 Gyr
    exponential["massformed"] = (1., 15.)             # vary log_10(M*/M_solar) between 1 and 15
    exponential["metallicity"] = (0., 2.5)            # vary Z between 0 and 2.5 Z_oldsolar

    #delayed Tau-model
    delayed = {}                         # Delayed Tau model t*e^-(t/tau)
    delayed["age"] = (0.1, 15.)           # Time since SF began: Gyr
    delayed["tau"] = (0.3, 10.)           # Timescale of decrease: Gyr
    delayed["massformed"] = (1., 15.)
    delayed["metallicity"] = (0., 2.5)

    #double-power law
    dblplaw = {}                        
    dblplaw["tau"] = (0., 15.)                # Vary the time of peak star-formation between
                                            # the Big Bang at 0 Gyr and 15 Gyr later. In 
                                            # practice the code automatically stops this
                                            # exceeding the age of the universe at the 
                                            # observed redshift.            
    dblplaw["alpha"] = (0.01, 1000.)          # Vary the falling power law slope from 0.01 to 1000.
    dblplaw["beta"] = (0.01, 1000.)           # Vary the rising power law slope from 0.01 to 1000.
    dblplaw["alpha_prior"] = "log_10"         # Impose a prior which is uniform in log_10 of the 
    dblplaw["beta_prior"] = "log_10"          # parameter between the limits which have been set 
                                            # above as in Carnall et al. (2017).
    dblplaw["massformed"] = (1., 15.)
    dblplaw["metallicity"] = (0., 2.5)

    #lognormal
    lognormal = {}                       # lognormal SFH
    lognormal["tmax"] = (0.1, 15)        # Age of Universe at peak SF: Gyr
    lognormal["fwhm"] = (0.1, 15)        # Full width at half maximum SF: Gyr
    lognormal["massformed"] = (1., 15.)
    lognormal["metallicity"] = (0., 2.5)

    return eval(mode)    

def load_fit_instructions(SFH_model, add_agn=False):
    #dust
    dust = {}                                 # Dust component
    dust["type"] = "Calzetti"                 # Define the shape of the attenuation curve
    dust["Av"] = (0., 5.)                     # Vary Av between 0 and 2 magnitudes
    #nebula
    nebular = {}
    nebular["logU"] = (-4, -2)
    #AGN
    agn = {}
    agn['alphalam'] = (-2.5, -0.5) #Power law slope of the AGN continuum at lambda < 5000A
    agn['betalam'] = (-0.5, 1.5) #Power law slope of the AGN continuum at lambda > 5000A
    agn['hanorm'] = (0., 2.5*10**(-17)) #Halpha luminosity erg/s/cm^2
    agn['sigma'] = (1000., 5000.) #Velocity dispersion km/s
    agn['f5100A'] = (0., 10**(-19)) #5100A luminosity erg/s/cm^2

    fit_instructions = {}                     # The fit instructions dictionary
    fit_instructions["redshift"] = 0.924  # Vary observed redshift from 0.9 to 1 #spetrum redshift=0.924
    fit_instructions[SFH_model] = load_SFH_model(SFH_model)   
    fit_instructions["dust"] = dust
    fit_instructions["nebular"] = nebular #add nebular emission
    if add_agn: fit_instructions["agn"] = agn
    return fit_instructions

def SED_fitting(version, SFH_model, add_agn=False):
    galaxy = pipes.galaxy(version, load_photometry_data, spectrum_exists=False, filt_list=load_filter_path(version=version,))
    if add_agn: agn = 'agn'
    else: agn = 'noagn'
    fit = pipes.fit(galaxy, load_fit_instructions(SFH_model, add_agn), run=f'{version}_{SFH_model}_{agn}')
    fit.fit(verbose=False)
    #plot results
    fit.plot_spectrum_posterior(save=True, show=False)
    fit.plot_sfh_posterior(save=True, show=False)
    fit.plot_corner(save=True, show=False)          

def diagnostic_plot(version, SFH_model, add_agn=False):
    if add_agn: agn = 'agn'
    else: agn = 'fix_z'

    galaxy = pipes.galaxy(version, load_photometry_data, spectrum_exists=False, filt_list=load_filter_path(version=version,))
    fit = pipes.fit(galaxy, load_fit_instructions(SFH_model, add_agn), run=f'{version}_{SFH_model}_{agn}')
    fit.posterior.get_advanced_quantities()
    plt.figure(figsize=(12, 14))
    gs = mpl.gridspec.GridSpec(11, 5, hspace=4., wspace=0.1)
    ax1 = plt.subplot(gs[:4, :])
    ax2 = plt.subplot(gs[4:8, :])
    pipes.plotting.add_observed_photometry(fit.galaxy, ax1, zorder=10)
    pipes.plotting.add_photometry_posterior(fit, ax1)
    pipes.plotting.add_sfh_posterior(fit, ax2)
    labels = ["sfr", "ssfr", "dust:Av", "mass_weighted_age", "stellar_mass"]
    post_quantities = dict(zip(labels, [fit.posterior.samples[l] for l in labels]))
    axes = []
    for i in range(5):
        axes.append(plt.subplot(gs[8:, i]))
        pipes.plotting.hist1d(post_quantities[labels[i]], axes[-1], smooth=True, label=labels[i])
    plt.suptitle(f'{version} {SFH_model}', fontsize=20)
    savepath = f'/home/lupengjun/OII_emitter/SED_fitting/output/diag_plots/{version}'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    plt.savefig(f'{savepath}/{SFH_model}_{version}.png')
    plt.clf()
    plt.close()
    return fit
   
def results_plot(versions=['homo_ellipse_v1', 'space_homo_ellipse_v1', 'space_homo_downleft', 'space_homo_upright'], labels=["sfr", "dust:Av"], SFH_models=['exponential', 'delayed', 'lognormal', 'dblplaw'], add_agn=False):
    def get_diclabel(label):
        if label == 'sfr':
            return 'SFR'
        elif label == 'ssfr':
            return 'sSFR'
        else:
            return label
    if add_agn: agn = 'agn'
    else: agn = 'fix_z'

    plt.rcParams.update({'font.size': 12})
    nicknames = {'homo_ellipse_v1': 'Galaxy',
                 'homo_ellipse_v1_dered': 'Galaxy',
                'space_homo_ellipse_v1': 'HST+JWST Galaxy',
                'space_homo_downleft': 'Arm',
                'space_homo_downleft_dered': 'Arm',
                'space_homo_upright': 'Bulge',
                'space_homo_upright_dered': 'Bulge',
                }
    colors = {'homo_ellipse_v1': 'tab:blue',
                'space_homo_ellipse_v1': 'tab:purple',
                'space_homo_downleft': 'tab:green',
                'space_homo_upright': 'tab:orange'
                }
    ylims = {'sfr': (-5, 205),
             'ssfr': (-12, -7),
             'dust:Av': (2.0, 3.3),
                }

    # Creating subplots
    n_rows = len(labels)
    fig, axs = plt.subplots(n_rows, 1, figsize=(8, 4*n_rows))
    versions_results_dic = {}
    #versions_samples_dic = {}    
    for version in versions:
        # load results
        results_dic = {}
        #samples_dic = {}
        for label in labels:
            new_label = get_diclabel(label)
            results_dic[new_label] = {}
        for SFH_model in SFH_models:
            galaxy = pipes.galaxy(version, load_photometry_data, spectrum_exists=False, filt_list=load_filter_path(version=version,))
            fit = pipes.fit(galaxy, load_fit_instructions(SFH_model, add_agn), run=f'{version}_{SFH_model}_{agn}')
            fit.posterior.get_advanced_quantities()
            for label in labels:
                new_label = get_diclabel(label)
                results_dic[new_label][SFH_model] = np.percentile(fit.posterior.samples[label], [50, 16, 84])     
            #samples_dic[SFH_model] = fit.posterior.samples
        for label, ax in zip(labels, axs.flatten()):
            new_label = get_diclabel(label)
            # Plotting
            x = [key for key in results_dic[new_label].keys()]
            y50 = np.array([value[0] for value in results_dic[new_label].values()])
            y1684 = np.array([value[1:] for value in results_dic[new_label].values()]).T
            ax.errorbar(x, y50, np.abs(y1684-y50), fmt='o', capsize=5, label=nicknames[version])
            ax.set_ylabel(new_label)
            ax.set_ylim(*ylims[label])
        versions_results_dic[version] = results_dic
        #versions_samples_dic[version] = samples_dic
    # General plot adjustments
    axs.flatten()[0].legend()
    plt.xlabel('SFH Models')
    plt.tight_layout()
    plt.show()
    return versions_results_dic #, versions_samples_dic
    
if __name__ == '__main__':
    version = sys.argv[1:][0]
    ###create log file for a specific version
    log_file = open(f'/home/lupengjun/OII_emitter/SED_fitting/output/pipes_{version}.log', 'a+')
    sys.stdout = log_file
    ###
    print(datetime.now())
    print('--->Running SED fitting to the version', version)
    pre_drop_tel = input("Input drop telescope, seperated by comma:")
    if pre_drop_tel:
        drop_telescope = pre_drop_tel.split(',')
    pre_drop_band = input("Input drop band, seperated by comma:")
    if pre_drop_band:
        drop_band = pre_drop_band.split(',')
    print("Dropping", drop_telescope, drop_band)

    pre_add_agn = input("Add AGN? (y/n)")
    if pre_add_agn == 'y':
        add_agn = True
    else:
        add_agn = False
    print("Add AGN?", add_agn)

    SFH_models = ['exponential', 'delayed', 'lognormal', 'dblplaw']
    for SFH_model in SFH_models:
        SED_fitting(version, SFH_model, add_agn=add_agn)
    print('Done!')
    log_file.close()
    