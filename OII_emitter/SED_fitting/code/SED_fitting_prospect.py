import time, sys

import numpy as np
from sedpy.observate import load_filters

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

# --------------
# Model Definition
# --------------

def build_model(z=0.924, sfh_type='delayed_tau', **extras):
    """
    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors
    from prospect.models import SpecModel
    from prospect.models import transforms
    if sfh_type == 'delayed_tau':
        #delayed tau SFH
        model_params = TemplateLibrary["parametric_sfh"]
        model_params['tage']['prior'] = priors.TopHat(mini=0.1, maxi=13.8) #    delayed["age"] = (0.1, 15.)           # Time since SF began: Gyr
        model_params['tau']['prior'] = priors.TopHat(mini=0.3, maxi=10.) #    delayed["tau"] = (0.3, 10.)           # Timescale of decrease: Gyr
        model_params['mass']['prior'] = priors.LogUniform(mini=10., maxi=1e15)#    delayed["massformed"] = (1., 15.)
    # elif sfh_type == 'exponential':
    # elif sfh_type == 'dblplaw':
    # elif sfh_type == 'lognormal':
    elif sfh_type == 'binned':
        #nonparametric SFH
        model_params = TemplateLibrary['logm_sfh'] # Using a (perhaps dangerously) simple nonparametric model of mass in fixed time bins with a logarithmic prior.
        model_params["zred"]["init"] = z
        model_params['agebins']['init'] = np.array(model_params['agebins']['init'])
        model_params['agebins']['depends_on'] = transforms.zred_to_agebins
        #agebins, mass need to be tuned
    elif sfh_type == 'continuity':
        #nonparametric SFH
        model_params = TemplateLibrary['continuity_sfh']
        model_params["zred"]["init"] = z
    elif sfh_type == 'flexible':
        #nonparametric SFH
        model_params = TemplateLibrary['continuity_flex_sfh']
        model_params["zred"]["init"] = z
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
    model_params["zred"]["init"] = z
    model_params["zred"]["isfree"] = False
    ###tau, mass, metalicity prior need to be tuned###
    #model_params['logzsol']['prior'] = priors.TopHat(mini=-1, maxi=0.5)#    delayed["metallicity"] = (0., 2.5)
    model_params['logzsol']['prior'] = priors.TopHat(mini=-2, maxi=0.6)
    model = SpecModel(model_params)
    print(model)

    return model

# --------------
# Observational Data
# --------------

def build_obs(version, z=0.924, **kwargs):
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
    obs = dict(wavelength=None, spectrum=None, unc=None, redshift=z, # redshift from z_spec
           maggies=maggies, maggies_unc=maggerr, filters=filters) #only photometry
    obs = fix_obs(obs)

    return obs


# --------------
# SPS Object
# --------------

def build_sps(sfh_type, zcontinuous=1, **extras):
    from prospect.sources import CSPSpecBasis
    from prospect.sources import FastStepBasis
    if sfh_type == 'delayed_tau':
        sps = CSPSpecBasis(zcontinuous=zcontinuous)
    elif sfh_type == 'binned' or sfh_type == 'continuity' or sfh_type == 'flexible':
        sps = FastStepBasis(zcontinuous=zcontinuous)
    else:
        raise ValueError('SFH type not recognized')
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

# def results_plot(versions=['homo_ellipse_v1', 'space_homo_downleft', 'space_homo_upright'], labels=["mass", "SFR", "sSFR", "dust2"], SFH_models=['delayed', 'binned'], pipes_results=None):
#     from prospect.io import read_results as reader
#     from prospect.plotting.sfh import parametric_sfr

#     path = '/home/lupengjun/OII_emitter/SED_fitting/output/prospect_results/'

#     def get_percentile(out, label):
#         q = np.atleast_1d([0.16, 0.5, 0.84])
#         label_mask = np.atleast_1d(out['theta_labels'])==label
#         if not np.any(label_mask): raise ValueError("Wrong input parameter label!")
#         samples_1d = (out['chain'].T)[label_mask][0]
#         weights_1d = (out['weights'].T)[label_mask][0]
#         idx = np.argsort(samples_1d)  # sort samples
#         sw = weights_1d[idx]  # sort weights
#         cdf = np.cumsum(sw)[:-1]  # compute CDF
#         cdf /= cdf[-1]  # normalize CDF
#         cdf = np.append(0, cdf)  # ensure proper span
#         quantiles = np.interp(q, cdf, samples_1d[idx]).tolist()
#         return quantiles
    

#     plt.rcParams.update({'font.size': 12})
#     nicknames = {'homo_ellipse_v1': 'Galaxy',
#                 'space_homo_ellipse_v1': 'HST+JWST Galaxy',
#                 'space_homo_downleft': 'Arm',
#                 'space_homo_upright': 'Bulge'
#                 }
#     # Creating subplots
#     n_rows = len(labels)
#     fig, axs = plt.subplots(n_rows, 1, figsize=(8, 4*n_rows))
#     versions_results_dic = {}
#     #versions_samples_dic = {}    
#     for version in versions:
#         # load results
#         results_dic = {}
#         #samples_dic = {}
#         for label in labels:
#             results_dic[label] = {}
#         for SFH_model in SFH_models:
#             out, out_obs, out_model = reader.results_from(path+'{version}_{SFH_model}.h5')
#             for label in labels:
#                 if label in out['theta_labels']:
#                     results_dic[label][SFH_model] = get_percentile(out, label)
#                 elif label == 'SFR' and SFH_model == 'delayed':
#                     # sfr = parametric_sfr(times=np.array([0]), mass=mass, tage=tage, tau=tau)
#                     results_dic[label][SFH_model] = [parametric_sfr(times=np.array([0]), mass=mass, tage=tage, tau=tau)[0] 
#                                                     for mass, tage, tau in zip(get_percentile(out, 'mass'), get_percentile(out, 'tage'), get_percentile(out, 'tau'))]
#                 elif label == 'SFR' and SFH_model in ['binned', 'very_binned']:
#                     #use the recent mass in the recent time bin to calculate SFR
#                     agebin = 10**out['model'].params['agebins'][-1, 1] - 10**out['model'].params['agebins'][-1, 0]
#                     for key in out['theta_labels']:
#                         if 'mass' in key:
#                             mass_key = key # get last mass key
#                     results_dic[label][SFH_model] = [mass/agebin for mass in get_percentile(out, mass_key)]
#                 elif label == 'sSFR':
#                     if 'SFR' in results_dic.keys() and 'mass' in results_dic.keys() and SFH_model in results_dic['SFR'].keys():
#                         results_dic[label][SFH_model] = [sfr/mass for sfr, mass in zip(results_dic['SFR'][SFH_model], results_dic['mass'][SFH_model])]
#                     else:
#                         results_dic[label][SFH_model] = [parametric_sfr(times=np.array([0]), mass=mass, tage=tage, tau=tau)[0]/mass 
#                                     for mass, tage, tau in zip(get_percentile(out, 'mass'), get_percentile(out, 'tage'), get_percentile(out, 'tau'))]
#                 elif label == 'dust:Av':
#                     results_dic[label][SFH_model] = get_percentile(out, 'dust2')
#         for label, ax in zip(labels, axs.flatten()):
#             # Plotting
#             x = [key for key in results_dic[label].keys()]
#             y50 = np.array([value[1] for value in results_dic[label].values()])
#             y1684 = np.array([value[0::2] for value in results_dic[label].values()]).T
#             ax.errorbar(x, y50, np.abs(y1684-y50), fmt='o', capsize=5, label=nicknames[version])
#             if pipes_results and version in pipes_results.keys() and label in pipes_results[version].keys():
#                 x = [key for key in pipes_results[version][label].keys()]
#                 y50 = np.array([value[0] for value in pipes_results[version][label].values()])
#                 y1684 = np.array([value[1:] for value in pipes_results[version][label].values()]).T
#                 ax.errorbar(x, y50, np.abs(y1684-y50), fmt='s', capsize=5, label=nicknames[version]) #pipes results marked by square
#             ax.set_ylabel(label)
#         versions_results_dic[version] = results_dic
#         #versions_samples_dic[version] = samples_dic
#     # General plot adjustments
#     axs.flatten()[0].legend()
#     plt.xlabel('SFH Models')
#     plt.tight_layout()
#     plt.show()
#     return versions_results_dic #, versions_samples_dic

if __name__ == '__main__':
    drop_band, drop_telescope = [], []
    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--version', type=str, default='homo_ellipse_v1',
                        help="Photometry data version.")
    parser.add_argument('--sfh_type', type=str, default='delayed_tau',
                        help=("SFH type."))
    args = parser.parse_args()
    run_params = vars(args)

    ###create log file for a specific version
    ts = time.strftime("%y%b%d-%H.%M", time.localtime())
    log_file = open(f'/home/lupengjun/OII_emitter/SED_fitting/output/prospect_{args.version}.log', 'a+')
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
