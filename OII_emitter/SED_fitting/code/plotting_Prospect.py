import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
from prospect.io import read_results as reader
from prospect.plotting.sfh import parametric_sfr

def results_plot(versions=['homo_ellipse_v1', 'space_homo_downleft', 'space_homo_upright'], labels=["formed_mass", "SFR", "sSFR", "dust:Av"], SFH_models=['delayed', 'binned'], pipes_results=None):
    path = '/home/lupengjun/OII_emitter/SED_fitting/output/prospect_results/'
    def get_percentile(out, label): #see prospect.plotting.corner.quantile
        q = np.atleast_1d([0.16, 0.5, 0.84])
        if label:
            label_mask = np.atleast_1d(out['theta_labels'])==label
            if not np.any(label_mask) and label=='mass':
                label_mask = np.array(['mass' in label_ for label_ in out['theta_labels']])
                samples_1d = np.sum((out['chain'])[:, label_mask], axis=1)#the sum of mass
            else:
                samples_1d = (out['chain'])[:, label_mask][:, 0]
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
        if SFH_model in ['delayed', 'exponential']:
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
        total_stellar_mass = get_percentile(res, 'mass')
        mfrac = [model.predict(vector, obs=obs, sps=sps)[2] for vector in get_percentile(res, None)]
        surviving_stellar_mass = [frac*mass for frac, mass in zip(mfrac, total_stellar_mass)]
        return total_stellar_mass, surviving_stellar_mass
    
    def calc_SFR(out, SFH_model, surviving_stellar_mass):
        if if_parametric_SFH(SFH_model):
            SFR = [parametric_sfr(times=np.array([0]), mass=mass, tage=tage, tau=tau)[0] 
            for mass, tage, tau in zip(get_percentile(out, 'mass'), get_percentile(out, 'tage'), get_percentile(out, 'tau'))]
        else:
            agebin = 10**out['model'].params['agebins'][-1, 1] - 10**out['model'].params['agebins'][-1, 0]
            for key in out['theta_labels']:
                if 'mass' in key:
                    mass_key = key # get last mass key
            SFR = [mass/agebin for mass in get_percentile(out, mass_key)]
        sSFR = [sfr/mass for sfr, mass in zip(SFR, surviving_stellar_mass)]
        return SFR, sSFR

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
            print(f'----{version}----{SFH_model}----')
            out, out_obs, out_model = reader.results_from(path+f'{version}_{SFH_model}.h5')
            total_stellar_mass, surviving_stellar_mass = calc_stellar_mass(out, out_obs, out_model)
            SFR, sSFR = calc_SFR(out, SFH_model, surviving_stellar_mass)
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
        for label, ax in zip(labels, axs.flatten()):
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
    axs.flatten()[0].legend()
    plt.xlabel('SFH Models')
    plt.tight_layout()
    plt.show()
    return versions_results_dic #, versions_samples_dic