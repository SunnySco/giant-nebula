import time, sys

import numpy as np
from sedpy.observate import load_filters
from astropy.cosmology import WMAP9 as cosmo

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

# --------------
# RUN_PARAMS
# When running as a script with argparsing, these are ignored.  Kept here for backwards compatibility.
# --------------

run_params = {'verbose': True,
              'debug': False,
              'outfile': 'demo_galphot',
              'output_pickles': False,
              # Optimization parameters
              'do_powell': False,
              'ftol': 0.5e-5, 'maxfev': 5000,
              'do_levenberg': True,
              'nmin': 10,
              # emcee fitting parameters
              'nwalkers': 128,
              'nburn': [16, 32, 64],
              'niter': 512,
              'interval': 0.25,
              'initial_disp': 0.1,
              # dynesty Fitter parameters
              'nested_bound': 'multi',  # bounding method
              'nested_sample': 'unif',  # sampling method
              'nested_nlive_init': 100,
              'nested_nlive_batch': 100,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.05,
              'nested_weight_kwargs': {"pfrac": 1.0},
              'nested_target_n_effective': 10000,
              # Obs data parameters
              'objid': 0,
              'phottable': 'demo_photometry.dat',
              'luminosity_distance': 1e-5,  # in Mpc
              # Model parameters
              'add_neb': False,
              'add_duste': False,
              # SPS parameters
              'zcontinuous': 1,
              }

# Helper functions
def if_parametric_SFH(SFH_model):
    if SFH_model in ['delayed', 'exponential', 'delayed_bursty', 'exponential_bursty']:
        return True
    else:
        return False

def get_agebins(nbin):
    assert (nbin>6)&(nbin<10), "The number of agebins should be larger than 6 and less than 9."
    #following the 7 agebins in the paper: https://ui.adsabs.harvard.edu/abs/2019ApJ...876....3L
    agelims = [0, 30*1e6, 100*1e6, 330*1e6, 1.1*1e9, 3.6*1e9, 11.7*1e9, 13.7*1e9]
    #add agebins in the front by myself
    attachlims = [3*1e6, 10*1e6]
    for i in range(nbin-7):
        agelims.insert(1, attachlims[-1+i])
    agebins = np.array([agelims[:-1], agelims[1:]]).T
    agebins[0][0] = 1
    return agelims, np.log10(agebins) #agelims in yr, agebins in log10(yr)
def adjust_agebins_with_zred(agelims, zred):
    tuniv = cosmo.lookback_time(zred).value*1e9 #yr
    agelims = np.asarray(agelims)
    new_agelims = agelims*(agelims.max()-tuniv)/agelims.max()
    new_agebins = np.array([new_agelims[:-1], new_agelims[1:]]).T
    new_agebins[0][0] = 1
    new_agelims[0] = 1
    return np.log10(new_agelims), np.log10(new_agebins) #agelims in log10(yr), agebins in log10(yr)
 
# --------------
# Model Definition
# --------------

def build_model(z, sfh_type='delayed', nbin=8, add_agn=False, **extras):
    """
    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models.templates import adjust_dirichlet_agebins, adjust_continuity_agebins
    from prospect.models import priors
    from prospect.models import SpecModel
    from prospect.models import transforms
    
    if z != 0:
        z_ = z
    else:
        z_ = 0.924
    
    #parametric SFH
    if 'delayed' in sfh_type:
        #delayed tau SFH
        model_params = TemplateLibrary["parametric_sfh"]
        model_params["sfh"]["init"] = 4
        model_params['tage']['prior'] = priors.TopHat(mini=0.1, maxi=13.8) #    delayed["age"] = (0.1, 15.)           # Time since SF began: Gyr
        model_params['tau']['prior'] = priors.TopHat(mini=0.3, maxi=10.) #    delayed["tau"] = (0.3, 10.)           # Timescale of decrease: Gyr
        model_params['mass']['prior'] = priors.LogUniform(mini=10., maxi=1e15)#    delayed["massformed"] = (1., 15.)
        if 'bursty' in sfh_type:
            model_params.update(TemplateLibrary["burst_sfh"]) #recent burst
            model_params['fage_burst']['isfree'] = True
            model_params['fage_burst']['init'] = 0.95
            model_params['fage_burst']['prior'] = priors.TopHat(mini=0.9, maxi=1)    
            model_params['fburst']['isfree'] = True
            model_params['fburst']['init'] = 0.3
            model_params['fburst']['prior'] = priors.TopHat(mini=0.1, maxi=0.5) 
    elif 'exponential' in sfh_type:
        model_params = TemplateLibrary["parametric_sfh"]
        model_params["sfh"]["init"] = 1
        model_params['tage']['prior'] = priors.TopHat(mini=0.1, maxi=13.8) #    delayed["age"] = (0.1, 15.)           # Time since SF began: Gyr
        model_params['tau']['prior'] = priors.TopHat(mini=0.3, maxi=10.) #    delayed["tau"] = (0.3, 10.)           # Timescale of decrease: Gyr
        model_params['mass']['prior'] = priors.LogUniform(mini=10., maxi=1e15)#    delayed["massformed"] = (1., 15.)
        if 'bursty' in sfh_type:
            model_params.update(TemplateLibrary["burst_sfh"]) #recent burst
            model_params['fage_burst']['isfree'] = True
            model_params['fage_burst']['init'] = 0.95
            model_params['fage_burst']['prior'] = priors.TopHat(mini=0.9, maxi=1)
            model_params['fburst']['isfree'] = True
            model_params['fburst']['init'] = 0.3
            model_params['fburst']['prior'] = priors.TopHat(mini=0.1, maxi=0.5)        
    #nonparametric SFH
    elif sfh_type == 'logM':
        model_params = TemplateLibrary['logm_sfh'] # Using a (perhaps dangerously) simple nonparametric model of mass in fixed time bins with a logarithmic prior.
        mass_min = np.zeros(nbin)+1e5
        mass_max = np.zeros(nbin)+1e11
        model_params['agebins'] = {'N': nbin, 'isfree': False,
                        'init': adjust_agebins_with_zred(get_agebins(nbin)[0], z_)[1],
                        'units': 'log(yr)'} 
        model_params["mass"]    = {'N': nbin, 'isfree': True, 'units': r'M$_\odot$',
                                    'init': np.zeros(nbin) + 1e7,
                                    'prior': priors.LogUniform(mini=mass_min, maxi=mass_max)}
    elif sfh_type == 'continuity':
        model_params = TemplateLibrary['continuity_sfh']
        adjust_continuity_agebins(model_params, nbins=nbin, agebins=adjust_agebins_with_zred(get_agebins(nbin)[0], z_)[1])
    elif sfh_type == 'dirichlet':
        model_params = TemplateLibrary['dirichlet_sfh']
        adjust_dirichlet_agebins(model_params, agelims=adjust_agebins_with_zred(get_agebins(nbin)[0], z_)[0])
    else:
        raise ValueError('SFH type not recognized')
    #nebular emission with logU varies
    model_params.update(TemplateLibrary["nebular"])
    model_params['gas_logu']['isfree'] = True
    #model_params['gas_logu']['prior'] = priors.TopHat(mini=-4., maxi=-2.)
    model_params['gas_logu']['prior'] = priors.TopHat(mini=-5., maxi=-1.)
    #Kroupa IMF
    model_params['imf_type']['init'] = 2 #Kroupa
    #dust extinction
    model_params['dust_type']['init'] = 2 #Calzetti
    #model_params['dust2']['prior'] = priors.TopHat(mini=0., maxi=5.) #uniform prior    dust["Av"] = (0., 5.)                     # Vary Av between 0 and 5 magnitudes
    model_params['dust2']['prior'] = priors.TopHat(mini=0., maxi=6.)
    #fixed spectroscopy redshift
    if z != 0:
        model_params["zred"]["init"] = z
        model_params["zred"]["isfree"] = False
    else:
        model_params["zred"]["init"] = 0.9
        model_params["zred"]["prior"] = priors.TopHat(mini=0.8, maxi=1.0)
        model_params["zred"]["isfree"] = True
    ###tau, mass, metalicity prior need to be tuned###
    #model_params['logzsol']['prior'] = priors.TopHat(mini=-1, maxi=0.5)#    delayed["metallicity"] = (0., 2.5)
    model_params['logzsol']['prior'] = priors.TopHat(mini=-2, maxi=0.6)
    if add_agn:
        model_params.update(TemplateLibrary["agn"])
        model_params['fagn']['isfree'] = True
        model_params['agn_tau']['isfree'] = True
    
    model = SpecModel(model_params)
    print(model)

    return model

# --------------
# Observational Data
# --------------

def build_obs(version, z, **kwargs):
    """
    :returns obs:
        Dictionary of observational data.
    """
    # Writes your code here to read data.  Can use FITS, h5py, astropy.table,
    # sqlite, whatever.
    # e.g.:
    # import astropy.io.fits as pyfits
    # catalog = pyfits.getdata(phottable)
    from sedpy.observate import load_filters
    from prospect.utils.obsutils import fix_obs
    from glob import glob
    import pandas as pd
    #photometry
    rate = 10**(-0.4*23.9) #uJy to maggies(10**(-0.4*m_AB)), equals to 1/3631/10**6
    maggies = pd.read_csv(f'/home/lupengjun/OII_emitter/photometry/output/flux_sum_{version}.csv', index_col=0).drop(drop_telescope, axis=1).drop(drop_band, axis=0).stack().values*rate
    maggerr = pd.read_csv(f'/home/lupengjun/OII_emitter/photometry/output/error_sum_{version}.csv', index_col=0).drop(drop_telescope, axis=1).drop(drop_band, axis=0).stack().values*rate
    #filters
    stack_df = pd.read_csv(f'/home/lupengjun/OII_emitter/photometry/output/flux_sum_{version}.csv', index_col=0).drop(drop_telescope,axis=1).drop(drop_band, axis=0).stack().reset_index()
    filters = stack_df['level_0']
    telescopes = stack_df['level_1']
    filter_list = [glob(f'/home/lupengjun/OII_emitter/data/filter/{telescope}/*{band}*')[0] for telescope, band in zip(telescopes, filters)]
    print(filter_list)
    filters = load_filters(filter_list) #use sedpy to load filters, but change the source code to direct the filter path to mine
    #construct obs dictionary
    if z != 0:
        obs = dict(wavelength=None, spectrum=None, unc=None, redshift=z, # redshift from z_spec
            maggies=maggies, maggies_unc=maggerr, filters=filters) #only photometry
    else:
        obs = dict(wavelength=None, spectrum=None, unc=None,# redshift from z_spec
            maggies=maggies, maggies_unc=maggerr, filters=filters) #only photometry
    obs = fix_obs(obs)

    return obs


# --------------
# SPS Object
# --------------

def build_sps(sfh_type, zcontinuous=1, **extras):
    from prospect.sources import CSPSpecBasis
    from prospect.sources import FastStepBasis
    if if_parametric_SFH(sfh_type):
        sps = CSPSpecBasis(zcontinuous=zcontinuous)
    else:
        sps = FastStepBasis(zcontinuous=zcontinuous)

    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------

def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))

if __name__ == '__main__':
    drop_band, drop_telescope = [], []
    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--version', type=str, default='homo_ellipse_v1',
                        help="Photometry data version.")
    parser.add_argument('--sfh_type', type=str, default='delayed',
                        help="SFH type.")
    parser.add_argument('--nbin', type=int, default=8,
                        help="number of bins for non-parametric sfh")
    parser.add_argument('--add_agn', type=bool, default=False,
                        help="whether add agn continuum")
    parser.add_argument('--z', type=float, default=0.924,
                    help="redshift")
    args = parser.parse_args()
    run_params = vars(args)

    ###create log file for a specific version
    ts = time.strftime("%y%b%d-%H.%M", time.localtime())
    sfh_type = run_params['sfh_type']
    log_file = open(f'/home/lupengjun/OII_emitter/SED_fitting/output/prospect_{args.version}_{sfh_type}.log', 'a+')
    sys.stdout = log_file
    ###
    print(ts)
    print('--->Running SED fitting to the version', args.version)
    pre_drop_tel = input("Input drop telescope, seperated by comma:\n")
    if pre_drop_tel:
        drop_telescope = pre_drop_tel.split(',')
    pre_drop_band = input("Input drop band, seperated by comma:\n")
    if pre_drop_band:
        drop_band = pre_drop_band.split(',')
    print("Dropping", drop_telescope, drop_band)

    obs, model, sps, noise = build_all(**run_params)

    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__

    print(model)

    if args.debug:
        sys.exit()

    #hfile = setup_h5(model=model, obs=obs, **run_params)
    hfile = "../output/prospect_results/{0}_{1}_result.h5".format(args.version, ts)

    output = fit_model(obs, model, sps, noise, **run_params)

    print("writing to {}".format(hfile))
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1],
                      sps=sps)

    try:
        hfile.close()
    except(AttributeError):
        pass
