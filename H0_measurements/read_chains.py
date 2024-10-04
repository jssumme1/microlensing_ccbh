import numpy as np
import h5py
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = (0, (1, 5))
mpl.rcParams['grid.color'] = 'grey'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['legend.handlelength'] = 3
mpl.rcParams['legend.fontsize'] = 20

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

BURNIN = 6000
H0_IDX = -1

chain_paths = ['posterior_yCL_HF_baseline.h5',
               'posterior_yCL_HF_a15.h5',
               'posterior_yCL_HF_a20.h5',
               'posterior_yCL_HF_a28.h5',
               'posterior_yCL_Panth_baseline.h5',
               'posterior_yCL_Panth_a15.h5',
               'posterior_yCL_Panth_a20.h5',
               'posterior_yCL_Panth_a28.h5']
HF = [True, True, True, True, False, False, False, False]
alpha = [0.0, 1.5, 2.0, 2.83, 0.0, 1.5, 2.0, 2.83]

samples, medians = [], []
for chain in chain_paths:
    posterior = h5py.File(chain, 'r')['mcmc']['chain'][:,:,:]
    samples_burnin = posterior[BURNIN:,:,:]
    fivelogH0 = samples_burnin[:,:,H0_IDX].flatten()
    H0_samples = 10**(fivelogH0/5)
    median = np.median(H0_samples)
    samples.append(H0_samples)
    medians.append(median)

fig, ax = plt.subplots()
fig.set_size_inches(6, 4)

cbar_loc = [0.9, 0.11, 0.04, 0.77]
cmap = cm.get_cmap('viridis')
norm = mcolors.Normalize(vmin=0, vmax=3)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes(cbar_loc)  # [left, bottom, width, height]
cbar = plt.colorbar(sm, cax=cbar_ax, shrink=1)
cbar_ax.set_yticks([0, 1.5, 2.0, 2.83])
cbar_ax.set_ylabel(r'$\alpha$')

bins = np.linspace(68, 80, 24)
ax.hist([0], bins=1, color='black', alpha=1, histtype='step', lw=2, ls='-', label='Riess')
ax.hist([0], bins=1, color='black', alpha=1, histtype='step', lw=2, ls=':', label='Pantheon+')

for i in range(len(chain_paths)):
    if HF[i] == True:
        linestyle = '-'
    else:
        linestyle = ':'
    ax.hist(samples[i], bins=bins, density=True, histtype='step', 
           ls=linestyle, edgecolor=cmap(norm(alpha[i])), lw=2)
    ax.plot([medians[i], medians[i]], [0,1e4], color=cmap(norm(alpha[i])), lw=1, ls=linestyle)

ax.set_xlim(68, 80)
ax.set_ylim(0, 0.6)
ax.set_xlabel(r'$H_0 ~ [{\rm km}~{\rm s}^{-1}~{\rm Mpc}]$')
ax.set_ylabel(r'$p(H_0)$')
ax.legend(framealpha=0, loc='upper right', fontsize=15, handlelength=1.5)

ml1 = MultipleLocator(0.5)
ml2 = MultipleLocator(0.05)
ax.yaxis.set_minor_locator(ml2)
ax.xaxis.set_minor_locator(ml1)
ax.tick_params(axis='both', which='major', direction='in', 
                    bottom=True, top=True, left=True, right=True)
ax.tick_params(axis='both', which='minor', direction='in', 
                    bottom=True, top=True, left=True, right=True)

fig.savefig('plots/SNe_H0_posterior_large_alpha.pdf', bbox_inches='tight')
