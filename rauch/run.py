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
@njit
def comoving_distance_from_redshift(redshift):
    comoving_distance = np.interp(redshift, redshift_grid, comoving_distance_grid)
    return comoving_distance
@njit
def redshift_from_comoving_distance(comoving_distance):
    redshift = np.interp(comoving_distance, comoving_distance_grid, redshift_grid)
    return redshift

# parameters of the simulation
M = 1e7 * u.Msun # Msun
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

# Rauch Eq 4 ish
@njit
def xj(x, xdis, Dj, M):
    GMc2 = M * G1Mc2

    zj = redshift_from_comoving_distance(Dj)
    D1 = x[2, 0]

    # thing at the end of the sum
    diff = (x[0:2, :] - xdis[0:2, :]) / ((x[0, :] - xdis[0, :])**2 + (x[1, :] - xdis[1, :])**2)

    # full equation
    ans = Dj * x[:2, 0] / D1 - 4 * GMc2 * np.sum((1 + zj) * (Dj - x[2, :]) * diff, axis=1)
    return ans

# SEF Eq 9.10
@njit
def Ui(x, xdis, M):
    # technically this is just \partial \alpha_i / \partial \xi_i
    # i include the extra fractor of D_i in the summation in Aj()
    mat = np.zeros((2, 2, x.shape[1]))

    # values
    GMc2 = M * G1Mc2
    zi = redshift_from_comoving_distance(x[2, :])
    norm2 = (x[0, :] - xdis[0, :])**2 + (x[1, :] - xdis[1, :])**2

    # final equations from differentiating alpha (and cancelling the Dis/Ds factor)
    # (1+z)^2 is from converting from angular diamter distances to comoving distances
    mat[0, 0, :] = 4 * GMc2 * (1 + zi)**2 / norm2**2 * (norm2 - 2 * (x[0, :] - xdis[0, :])**2)
    mat[1, 0, :] = 4 * GMc2 * (1 + zi)**2 / norm2**2 * (-2 * (x[0, :] - xdis[0, :]) * (x[1, :] - xdis[1, :]))
    mat[0, 1, :] = 4 * GMc2 * (1 + zi)**2 / norm2**2 * (-2 * (x[0, :] - xdis[0, :]) * (x[1, :] - xdis[1, :]))
    mat[1, 1, :] = 4 * GMc2 * (1 + zi)**2 / norm2**2 * (norm2 - 2 * (x[1, :] - xdis[1, :])**2)
    return mat

@njit
def matmul(A, B):
    out = np.zeros(A.shape)
    out[0, 0, :] = A[0, 0, :] * B[0, 0, :] + A[0, 1, :] * B[1, 0, :]
    out[0, 1, :] = A[0, 0, :] * B[0, 1, :] + A[0, 1, :] * B[1, 1, :]
    out[1, 0, :] = A[1, 0, :] * B[0, 0, :] + A[1, 1, :] * B[1, 0, :]
    out[1, 1, :] = A[1, 0, :] * B[0, 1, :] + A[1, 1, :] * B[1, 1, :]
    return out

# SEF Eq 9.12 ish
@njit
def Aj(x, Amat, xdis, Dj, M):
    I = np.identity(2)
    zj = redshift_from_comoving_distance(Dj)
    Dfactor = (Dj - x[2, :]) / (1 + zj) * x[2, :] / Dj

    # \partial \alpha_i / \partial \xi_i
    Umat = Ui(x, xdis, M)
    # calculate Aj from previous Ai
    A = I - np.sum(Dfactor * matmul(Umat, Amat), axis=2)
    return A

# recursively finding source positions
@njit
def compute_img_position_and_mag(image_position, xdis, N_screens=None):
    # choose between running through all screens or just some
    if N_screens == None:
        N = xdis.shape[1]
    else:
        N = N_screens

    # instantiate empty arrays
    ximg = np.zeros((3, N))
    Amat = np.zeros((2, 2, N))
    ximg[:, 0] = [image_position[0], image_position[1], xdis[2, 0]]

    # solve for all image positions, starting from the nearest plane to observer
    for ii in range(1, N):
        # shorten the arrays / get values
        x = ximg[:, 0:ii]
        x_disruptors = xdis[:, 0:ii]
        A = Amat[:, :, 0:ii]
        Dj = xdis[2, ii]

        # calculate stuff
        xi = xj(x, x_disruptors, Dj, M) 
        Ai = Aj(x, A, x_disruptors, Dj, M)

        # save results
        ximg[:, ii] = [xi[0], xi[1], Dj]
        Amat[:, :, ii] = Ai

    return ximg, Amat

@njit
def matrix_inverse(A):
    B = np.zeros(A.shape)
    B[0, 0] = A[1, 1]
    B[1, 1] = A[0, 0]
    B[0, 1] = -A[0, 1]
    B[1, 0] = -A[1, 0]
    return B / np.linalg.det(A)

@njit
def matmul_2d1d(A, B):
    out = np.zeros(2)
    out[0] = A[0, 0] * B[0] + A[0, 1] * B[1]
    out[1] = A[1, 0] * B[0] + A[1, 1] * B[1]
    return out

#@njit
def newton_raphson(initial_guess, xdis):
    guess = np.array(initial_guess)
    for i in range(10):
        img_pos, mag = compute_img_position_and_mag(guess, xdis)
        Jinv = matrix_inverse(mag[:, :, -1])
        guess = guess - matmul_2d1d(Jinv, img_pos[:2, -1])

    return guess

img_pos, mag = compute_img_position_and_mag([0.0000,0.00000], xdis)
print('mu = ', 1 / np.linalg.det(mag[:,:,-1]))

x = np.linspace(-1e-3, 1e-3, 100)
best = []
best_r = -1
for i in range(len(x)):
    for j in range(len(x)):
        g = newton_raphson([x[i], x[j]], xdis)
        img_pos, mag = compute_img_position_and_mag(g, xdis)
        r = np.hypot(img_pos[0, -1], img_pos[1, -1])
        if r < best_r or best_r == -1:
            best = img_pos
            best_r = r

img_pos = best

'''
x = np.random.uniform(5e-6, 5e-6, size=100)
y = np.random.uniform(5e-6, 5e-6, size=100)
t0 = time.time()
mus = []
for i in range(len(x)):
    img_pos, mag = compute_img_position_and_mag([x[i],y[i]], xdis, N_screens=None)
    mus.append(1 / np.linalg.det(mag[:,:,-1]))

t1 = time.time()
tottime = (t1-t0)/len(x)
print(f'avg time {tottime}')
print(f'median mu = {np.std(mus)}')
'''

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