import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
mpl.rcParams['text.usetex'] = True
from prospect.io import read_results as reader
from prospect.plotting.sfh import parametric_sfr, sfh_quantiles, params_to_sfh
from prospect.models.transforms import zfrac_to_sfrac, zfrac_to_masses, logsfr_ratios_to_sfrs


def get_agediff(out):
    agebins = 10**[dic['init'] for dic in out['model_params'] if dic['name']=='agebins'][0]
    return agebins, np.diff(agebins).flatten()

def get_SFR_perbin(out, SFH): #return (nsamples, nbin)
    if 'logM' in SFH:
        label_mask = np.array(['mass_' in label_ for label_ in out['theta_labels']])
        samples_masses = out['chain'][:, label_mask]
        return samples_masses/get_agediff(out)[1]
    elif 'continuity' in SFH:
        agebins = get_agediff(out)[0]
        label_mask = np.array(['logsfr_ratios_' in label_ for label_ in out['theta_labels']])
        label_mask_ = np.array(['logmass' in label_ for label_ in out['theta_labels']])
        samples_sratios = out['chain'][:, label_mask]
        samples_logmass = out['chain'][:, label_mask_]
        samples_sfrs = np.asarray([logsfr_ratios_to_sfrs(samples_logmass[i], samples_sratios[i], np.log10(agebins)) for i in range(len(samples_sratios))])
        return samples_sfrs    
    elif 'dirichlet' in SFH:
        agebins = get_agediff(out)[0]
        label_mask = np.array(['z_fraction_' in label_ for label_ in out['theta_labels']])
        label_mask_ = np.array(['total_mass' in label_ for label_ in out['theta_labels']])
        samples_zfrac = out['chain'][:, label_mask]
        samples_total_mass = out['chain'][:, label_mask_]
        #samples_sfrac = np.asarray([zfrac_to_sfrac(samples_zfrac[i]) for i in range(len(samples_zfrac))])
        samples_masses = np.asarray([zfrac_to_masses(samples_total_mass[i], samples_zfrac[i], np.log10(agebins)) for i in range(len(samples_zfrac))])
        return samples_masses/get_agediff(out)[1]
    else:
        raise ValueError('SFH not recognized')

def get_samples_1d(out, label):
    label_mask = np.atleast_1d(out['theta_labels'])==label
    if not np.any(label_mask) and label=='mass': 
        label_mask = np.array(['mass_' in label_ for label_ in out['theta_labels']]) #for logM SFH
        samples_1d = np.sum((out['chain'])[:, label_mask], axis=1)#the sum of mass 
        if not np.any(label_mask):
            label_mask = np.array(['total_mass' in label_ for label_ in out['theta_labels']]) #for dirichlet SFH
            samples_1d = out['chain'][:, label_mask].flatten()
            if not np.any(label_mask):
                label_mask = np.array(['logmass' in label_ for label_ in out['theta_labels']]) #for continuity SFH
                samples_1d = 10**out['chain'][:, label_mask].flatten()
    else:
        samples_1d = (out['chain'])[:, label_mask][:, 0]
    return samples_1d

def get_percentile(out, label, samples_1d=None): #see prospect.plotting.corner.quantile
    q = np.atleast_1d([0.16, 0.5, 0.84])
    if label:
        label_mask = np.atleast_1d(out['theta_labels'])==label
        if not np.any(label_mask) and label=='mass': #don't have mass in the params but want total mass (non-param SFH)
            label_mask = np.array(['mass_' in label_ for label_ in out['theta_labels']]) #for logM and continuity SFH
            samples_1d = np.sum((out['chain'])[:, label_mask], axis=1)#the sum of mass                
            if not np.any(label_mask): #don't have mass_ but want total mass 
                label_mask = np.array(['total_mass' in label_ for label_ in out['theta_labels']]) # for dirichlet SFH
                samples_1d = out['chain'][:, label_mask].flatten()
        elif not np.any(label_mask) and label=='zfrac_to_mass':
            agebins = get_agediff(out)[0]
            label_mask = np.array(['z_fraction_' in label_ for label_ in out['theta_labels']])
            label_mask_ = np.array(['total_mass' in label_ for label_ in out['theta_labels']])
            samples_zfrac = out['chain'][:, label_mask]
            samples_total_mass = out['chain'][:, label_mask_]
            samples_masses = np.asarray([zfrac_to_masses(samples_total_mass[i], samples_zfrac[i], np.log10(agebins)) for i in range(len(samples_zfrac))])
            samples_1d = samples_masses[:, 0]
        else:
            samples_1d = (out['chain'])[:, label_mask][:, 0]
        weights_1d = np.atleast_1d(out['weights'])
        idx = np.argsort(samples_1d)  # sort samples
        sw = weights_1d[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, samples_1d[idx]).tolist()
    elif np.any(samples_1d):
        weights_1d = np.atleast_1d(out['weights'])
        idx = np.argsort(samples_1d)  # sort samples
        sw = weights_1d[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, samples_1d[idx]).tolist()    
    else:
        quantiles = [[],[],[]]
        weights_1d = np.atleast_1d(out['weights'])
        for parames_vector in out['chain'].T:
            idx = np.argsort(parames_vector)
            sw = weights_1d[idx]
            cdf = np.cumsum(sw)[:-1]
            cdf /= cdf[-1]
            cdf = np.append(0, cdf)
            quantiles_ = np.interp(q, cdf, parames_vector[idx]).tolist()
            for i in range(3):
                quantiles[i].append(quantiles_[i])
    return quantiles

def if_parametric_SFH(SFH_model):
    if SFH_model in ['delayed', 'exponential', 'delayed_agn', 'exponential_agn']:
        return True
    else:
        return False

def calc_stellar_mass(res, obs, model):
    '''
    Question remained:
    Can the median stellar mass and the mfrac calculated by all median parameters multiply together?
    Or should I only use the median stellar mass sample to get the mfrac?
    '''
    sps = reader.get_sps(res)
    total_stellar_mass_1d = get_samples_1d(res, 'mass')
    samples_2d = []
    for label in res['theta_labels']:
        samples_2d.append(get_samples_1d(res, label))
    samples_2d = np.array(samples_2d).T
    mfrac_1d = [model.predict(vector, obs=obs, sps=sps)[2] for vector in samples_2d]
    mfrac_1d = np.array(mfrac_1d)
    surviving_stellar_mass_1d = total_stellar_mass_1d * mfrac_1d
    return total_stellar_mass_1d, surviving_stellar_mass_1d

def calc_recent_SFR(out, obs, model, SFH_model, surviving_stellar_mass_1d):
    if if_parametric_SFH(SFH_model):
        samples_tau, samples_tage, samples_mass = [get_samples_1d(out, label) for label in ['tau', 'tage', 'mass']]
        params = dict(tage=samples_tage, tau=samples_tau, mass=samples_mass)
        if 'exponential' in SFH_model:
            params['sfh'] = 1
        elif 'delayed' in SFH_model:
            params['sfh'] = 4
        else:
            raise ValueError('SFH not recognized')
        _, sfrs, _ = params_to_sfh(params, time=np.array([0])) #(nsamples, 1)
        SFR_1d = sfrs.flatten()
    else:
        SFR_1d = get_SFR_perbin(out, SFH_model)[:, 0] #only the recent bin
    sSFR_1d = SFR_1d / surviving_stellar_mass_1d
    SFR_quantiles = get_percentile(out, label=None, samples_1d=SFR_1d)
    sSFR_quantiles = get_percentile(out, label=None, samples_1d=sSFR_1d)
    return SFR_quantiles, sSFR_quantiles

def results_plot(versions=['homo_ellipse_v1', 'space_homo_downleft', 'space_homo_upright'], labels=["formed_mass", "SFR", "sSFR", "dust:Av"], SFH_models=['delayed', 'binned'], pipes_results=None):
    path = '/home/lupengjun/OII_emitter/SED_fitting/output/prospect_results/'
    plt.rcParams.update({'font.size': 12})
    nicknames = {'homo_ellipse_v1': 'Galaxy',
                'space_homo_ellipse_v1': 'HST+JWST Galaxy',
                'space_homo_downleft': 'Arm',
                'space_homo_upright': 'Bulge',
                'BCG': 'BCG'
                }
    colors = {'homo_ellipse_v1': 'tab:blue',
                'space_homo_ellipse_v1': 'tab:purple',
                'space_homo_downleft': 'tab:green',
                'space_homo_upright': 'tab:orange',
                'BCG': 'tab:blue',
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
            print(f'----{version}----{SFH_model}----')
            out, out_obs, out_model = reader.results_from(path+f'{version}_{SFH_model}.h5')
            if 'formed_mass' in labels or 'surviving_mass' in labels or 'SFR' in labels or 'sSFR' in labels:
                if not os.path.exists(f'./prospect_results/{version}_{SFH_model}_stellar_mass_1d.txt'):
                    total_stellar_mass_1d, surviving_stellar_mass_1d = calc_stellar_mass(out, out_obs, out_model)
                    np.savetxt(f'./prospect_results/{version}_{SFH_model}_stellar_mass_1d.txt', [total_stellar_mass_1d, surviving_stellar_mass_1d])
                else:
                    total_stellar_mass_1d, surviving_stellar_mass_1d = np.loadtxt(f'./prospect_results/{version}_{SFH_model}_stellar_mass_1d.txt')
                SFR, sSFR = calc_recent_SFR(out, out_obs, out_model, SFH_model, surviving_stellar_mass_1d)
                total_stellar_mass = get_percentile(out, label=None, samples_1d=total_stellar_mass_1d)
                surviving_stellar_mass = get_percentile(out, label=None, samples_1d=surviving_stellar_mass_1d)
            for label in labels: 
                if label == 'SFR': #what does bagpipes do?
                    results_dic[label][SFH_model] = SFR
                elif label == 'sSFR':
                    results_dic[label][SFH_model] = sSFR
                elif label == 'formed_mass':
                    results_dic[label][SFH_model] = total_stellar_mass
                elif label == 'surviving_mass':
                    results_dic[label][SFH_model] = surviving_stellar_mass
                elif label == 'dust:Av':
                    results_dic[label][SFH_model] = get_percentile(out, 'dust2')
                else:
                    results_dic[label][SFH_model] = get_percentile(out, label)
        for label, ax in zip(labels, np.array([axs]).flatten()):
            version_key = [key for key in nicknames.keys() if key in version ]
            # Plotting
            x = [key for key in results_dic[label].keys()]
            y50 = np.array([value[1] for value in results_dic[label].values()])
            y1684 = np.array([value[0::2] for value in results_dic[label].values()]).T
            ax.errorbar(x, y50, np.abs(y1684-y50), color=colors[version_key[0]], fmt='o', capsize=5, label=nicknames[version_key[0]])
            if pipes_results and version in pipes_results.keys() and label in pipes_results[version].keys():
                x = [key for key in pipes_results[version][label].keys()]
                y50 = np.array([value[0] for value in pipes_results[version][label].values()])
                y1684 = np.array([value[1:] for value in pipes_results[version][label].values()]).T
                ax.errorbar(x, y50, np.abs(y1684-y50), color=colors[version_key[0]], fmt='s', capsize=5, label=nicknames[version_key[0]]) #pipes results marked by square
            ax.set_ylabel(label)
        versions_results_dic[version] = results_dic
        #versions_samples_dic[version] = samples_dic
    # General plot adjustments
    np.array([axs]).flatten()[0].legend()
    plt.xlabel('SFH Models')
    plt.tight_layout()
    plt.show()
    return versions_results_dic #, versions_samples_dic

if __name__ == '__main__':
    # results_plot(versions=['homo_ellipse_v1_dered', 'space_homo_downleft_dered', 'space_homo_upright_dered'], SFH_models=['exponential_agn', 'delayed_agn', 'logM_agn', 'continuity_agn'], labels=["surviving_mass", "formed_mass", "SFR", "sSFR", "dust:Av"])
    # results_plot(versions=['homo_ellipse_v1_dered', 'space_homo_upright_dered'], SFH_models=['exponential_agn', 'delayed_agn', 'logM_agn', 'continuity_agn','dirichlet_agn'], labels=["surviving_mass", "formed_mass", "SFR", "sSFR", "dust:Av"])
    results_plot(versions=['space_homo_upright_dered'], SFH_models=['delayed', 'logM',], labels=["formed_mass", "surviving_mass", "SFR", "sSFR", "dust:Av"])