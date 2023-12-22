import time, sys

import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

def results_plot(versions=['homo_ellipse_v1', 'space_homo_downleft', 'space_homo_upright'], labels=["mass", "SFR", "sSFR", "dust:Av"], SFH_models=['delayed', 'binned'], pipes_results=None):
    from prospect.io import read_results as reader
    from prospect.plotting.sfh import parametric_sfr

    path = '/home/lupengjun/OII_emitter/SED_fitting/output/prospect_results/'

    def get_percentile(out, label):
        q = np.atleast_1d([0.16, 0.5, 0.84])
        label_mask = np.atleast_1d(out['theta_labels'])==label
        if not np.any(label_mask): raise ValueError("Wrong input parameter label!")
        samples_1d = (out['chain'])[:, label_mask][:, 0]
        weights_1d = np.atleast_1d(out['weights'])
        idx = np.argsort(samples_1d)  # sort samples
        sw = weights_1d[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, samples_1d[idx]).tolist()
        return quantiles
    

    plt.rcParams.update({'font.size': 12})
    nicknames = {'homo_ellipse_v1': 'Galaxy',
                'space_homo_ellipse_v1': 'HST+JWST Galaxy',
                'space_homo_downleft': 'Arm',
                'space_homo_upright': 'Bulge'
                }
    colors = {'homo_ellipse_v1': 'tab:blue',
                'space_homo_ellipse_v1': 'tab:purple',
                'space_homo_downleft': 'tab:green',
                'space_homo_upright': 'tab:orange'
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
            results_dic[label] = {}
        for SFH_model in SFH_models:
            out, out_obs, out_model = reader.results_from(path+f'{version}_{SFH_model}.h5')
            for label in labels:
                if label in out['theta_labels']:
                    results_dic[label][SFH_model] = get_percentile(out, label)
                elif label == 'mass':
                    results_dic[label][SFH_model] = np.sum([get_percentile(out, key) for key in out['theta_labels'] if 'mass' in key], axis=0).tolist()
                elif label == 'SFR' and SFH_model == 'delayed':
                    # sfr = parametric_sfr(times=np.array([0]), mass=mass, tage=tage, tau=tau)
                    results_dic[label][SFH_model] = [parametric_sfr(times=np.array([0]), mass=mass, tage=tage, tau=tau)[0] 
                                                    for mass, tage, tau in zip(get_percentile(out, 'mass'), get_percentile(out, 'tage'), get_percentile(out, 'tau'))]
                elif label == 'SFR' and SFH_model in ['binned', 'very_binned']:
                    #use the recent mass in the recent time bin to calculate SFR
                    agebin = 10**out['model'].params['agebins'][-1, 1] - 10**out['model'].params['agebins'][-1, 0]
                    for key in out['theta_labels']:
                        if 'mass' in key:
                            mass_key = key # get last mass key
                    results_dic[label][SFH_model] = [mass/agebin for mass in get_percentile(out, mass_key)]
                elif label == 'sSFR'and SFH_model == 'delayed':
                    # if 'SFR' in results_dic.keys() and 'mass' in results_dic.keys() and SFH_model in results_dic['SFR'].keys():
                    #     results_dic[label][SFH_model] = [sfr/mass for sfr, mass in zip(results_dic['SFR'][SFH_model], results_dic['mass'][SFH_model])]
                    # else:
                    mass_sum = np.sum([get_percentile(out, key) for key in out['theta_labels'] if 'mass' in key], axis=0)
                    results_dic[label][SFH_model] = [parametric_sfr(times=np.array([0]), mass=mass, tage=tage, tau=tau)[0]/mass 
                                for mass, tage, tau in zip(get_percentile(out, 'mass'), get_percentile(out, 'tage'), get_percentile(out, 'tau'))]
                elif label == 'sSFR' and SFH_model in ['binned', 'very_binned']:
                    #use the recent mass in the recent time bin to calculate SFR
                    agebin = 10**out['model'].params['agebins'][-1, 1] - 10**out['model'].params['agebins'][-1, 0]
                    mass_sum = np.sum([get_percentile(out, key) for key in out['theta_labels'] if 'mass' in key], axis=0)
                    for key in out['theta_labels']:
                        if 'mass' in key:
                            mass_key = key # get last mass key
                    results_dic[label][SFH_model] = (np.array([mass/agebin for mass in get_percentile(out, mass_key)])/mass_sum).tolist()
                elif label == 'dust:Av':
                    results_dic[label][SFH_model] = get_percentile(out, 'dust2')
        for label, ax in zip(labels, axs.flatten()):
            # Plotting
            x = [key for key in results_dic[label].keys()]
            y50 = np.array([value[1] for value in results_dic[label].values()])
            y1684 = np.array([value[0::2] for value in results_dic[label].values()]).T
            ax.errorbar(x, y50, np.abs(y1684-y50), color=colors[version], fmt='o', capsize=5, label=nicknames[version])
            if pipes_results and version in pipes_results.keys() and label in pipes_results[version].keys():
                x = [key for key in pipes_results[version][label].keys()]
                y50 = np.array([value[0] for value in pipes_results[version][label].values()])
                y1684 = np.array([value[1:] for value in pipes_results[version][label].values()]).T
                ax.errorbar(x, y50, np.abs(y1684-y50), color=colors[version], fmt='s', capsize=5, label=nicknames[version]) #pipes results marked by square
            ax.set_ylabel(label)
        versions_results_dic[version] = results_dic
        #versions_samples_dic[version] = samples_dic
    # General plot adjustments
    axs.flatten()[0].legend()
    plt.xlabel('SFH Models')
    plt.tight_layout()
    plt.show()
    return versions_results_dic #, versions_samples_dic