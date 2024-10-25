import numpy as np
#from colossus.cosmology import cosmology
#cosmo = cosmology.setCosmology('planck18')
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import astropy.constants as c
from numba import njit
import time

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

#########################################################################

# precompute function to get distances
redshift_grid = np.logspace(-4, 0.5, 10000)
comoving_distance_grid = cosmo.comoving_distance(redshift_grid).value
#@njit
def comoving_distance_from_redshift(redshift):
    comoving_distance = np.interp(redshift, redshift_grid, comoving_distance_grid)
    return comoving_distance
#@njit
def redshift_from_comoving_distance(comoving_distance):
    redshift = np.interp(comoving_distance, comoving_distance_grid, redshift_grid)
    return redshift

# parameters of the simulation
M = 5e5 * u.Msun # Msun
G1Mc2 = (c.G * (1 * u.Msun).to(u.kg) / c.c**2).to(u.Mpc).value
M = M.value
R_tube = 1e-3 # Mpc
z_source = 2

# cosmological parameters
Omega_m = cosmo.Om0
rho_c = cosmo.critical_density0 
rho_c = rho_c.to(u.Msun / u.Mpc**3).value # Msun / Mpc^3
# size of tube
L_tube = cosmo.comoving_distance(z_source).value # Mpc
V_tube = np.pi * R_tube**2 * L_tube  # Mpc^3

# number of disruptors
rho_disruptors = rho_c * Omega_m
mean_disruptors = rho_disruptors * V_tube / M
N_disruptors = np.random.poisson(lam=mean_disruptors)
print(f'running with {N_disruptors} disruptors')

# generate random positions for disruptors
distances = np.random.uniform(low=0, high=L_tube, size=N_disruptors)
distances.sort()
radii = R_tube * np.sqrt(np.random.uniform(size=N_disruptors))
theta = 2 * np.pi * np.random.uniform(size=N_disruptors)

# Convert to Cartesian coordinates
x = radii * np.cos(theta)
y = radii * np.sin(theta)
z = distances

# full vector of disruptors
xdis = np.array([x, y, z])

# Rauch Eq 4
#@njit
def xj(x, xdis, dj, M):
    GMc2 = M * G1Mc2

    zj = redshift_from_comoving_distance(dj)
    Dj = dj
    D1 = x[2, 0]

    # thing at the end of the sun
    #diff = (x[0:2, :] - xdis[0:2, :]) / np.linalg.norm(x[0:2, :] - xdis[0:2, :], axis=0)**2
    diff = (x[0:2, :] - xdis[0:2, :]) / ((x[0, :] - xdis[0, :])**2 + (x[1, :] - xdis[1, :])**2)

    # full equation
    ans = Dj * x[:2, 0] / D1 - 4 * GMc2 * np.sum((1 + zj) * (Dj - x[2, :]) * diff, axis=1)
    return ans

# recursively finding source positions
#@njit
def compute_source_position(image_position, xdis, N_screens=None):
    # choose between running through all screens or just some
    if N_screens == None:
        N = xdis.shape[1]
    else:
        N = N_screens

    # solve for all image positions, starting from the nearest plane to observer
    image_position.append(xdis[2, 0])
    ximg = np.array([image_position])
    for ii in range(1, N):
        x = ximg.T
        x_disruptors = xdis[:, 0:ii]
        dj = xdis[2, ii]
        xii = xj(x, x_disruptors, dj, M) 
        #xii = xii.tolist()
        #xii.append(float(dj))
        xii = np.append(xii, dj)
        ximg = np.append(ximg, xii).reshape(ii+1,3)

    return ximg.T

img_pos = compute_source_position([0.0,0.0], xdis, N_screens=None)

x = np.random.uniform(5e-4, 5e-4, size=100)
y = np.random.uniform(5e-4, 5e-4, size=100)
t0 = time.time()
for i in range(len(x)):
    img_pos = compute_source_position([x[i],y[i]], xdis, N_screens=None)

t1 = time.time()
tottime = (t1-t0)/len(x)
print(f'avg time {tottime}')

r = np.hypot(img_pos[0, :], img_pos[1, :])
d = img_pos[2, :]

fig, ax = plt.subplots()
fig.set_size_inches(8, 5)

ax.scatter(d, r, s=0.5, color='black')

ax.set_xlabel(r'comoving LOS distance [Mpc]')
ax.set_ylabel(r'comoving transverse dist [Mpc]')

ax.tick_params(axis='both', which='major', direction='in', 
                    bottom=True, top=True, left=True, right=True)
ax.tick_params(axis='both', which='minor', direction='in', 
                    bottom=True, top=True, left=True, right=True)

fig.savefig('images.pdf', bbox_inches='tight')