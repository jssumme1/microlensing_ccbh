import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import RegularGridInterpolator

# not useful rn but saved in case i need it
SNe_names = ('2011fe 2006D 2007A 2005W 1999dq 2009ig 2002fk 2012fr 2001el 2021pit 2005df 2005df_ANU 2015F 2018gv 2001bg 1995al 1997bq '
            +'2008fv 2008fv_comb 2021hpr 2019np 1994ae 2012ht 2015so ASASSN-15so 2011by 1998aq 2007sr Anchor 2012cg 1981B 1990N 1997bp 1999cp 2002cr '
            +'2007af 2013aa 2017cbv 2009Y 2017erp 2005cf 2013dy 2006bh 1998dh 2002dp 2003du').split()

def get_yCL(alpha = 0, SNe_data = 'data/Pantheon+SH0ES.dat', SNe_cov = 'data/Pantheon+SH0ES_STAT+SYS.cov',
                  Riess_Y = 'data/ally_shoes_ceph_topantheonwt6.0_112221.fits', Riess_L = 'data/alll_shoes_ceph_topantheonwt6.0_112221.fits',
                  Riess_C = 'data/allc_shoes_ceph_topantheonwt6.0_112221.fits'):
    '''
    Inputs:
        alpha - effective density of disruptor matter between SNe and observer, relative to DM density
        SNe_data - string filepath to Pantheon+SH0ES table of SNe
        SNe_cov - string filepath to Pantheon+SH0ES covariance matrix
        Riess_Y - string filepath to Riess 2022 Y vector
        Riess_L - string filepath to Riess 2022 L matrix
        Riess_C - string filepath to Riess 2022 C matrix

    Outputs:
        1) list of hubble_flow outputs (only including SNe used in Riess 2022 primary analysis)
            1a) hubble_flow[0] --> Y vector (with all SNe mags input from Pantheon+SH0ES)
            1b) hubble_flow[1] --> C matrix (with all SNe variance/covariance input from Pantheon+SH0ES)
            1c) hubble_flow[2] --> L matrix (same as Riess 2022)
        2) list of pantheon outputs (including SNe used in Riess 2022 analysis variant 34: 0.0233<z<0.8 plus calibrator SNe)
            2a) pantheon[0] --> Y vector (with all SNe mags input from Pantheon+SH0ES)
            2b) pantheon[1] --> C matrix (with all SNe variance/covariance input from Pantheon+SH0ES)
            2c) pantheon[2] --> L matrix (similar to Riess 2022, but extended to work with more SNe)

    Notes:
    the Riess 2022 Y and C values do not seem to match up directly with those from the Pantheon+SH0ES data. to match up the two catalogs,
    i minimized differences between them. but the match is still not great. in particular, calibrator SNe variances seem to be overestimated
    in these outputs, and a handful of Y vector values are off by >0.5 mag.
    '''
    # load the SNe in Hubble flow from Pantheon+
    t = Table.read(SNe_data, format='ascii')
    # indices of Riess baseline analysis
    HF = np.where(t['USED_IN_SH0ES_HF'])
    # indices of Riess analysis variant 34 (most of Pantheon+)
    Panth = np.where((t['zHD'] > 0.0233) & (t['zHD'] < 0.8) & (t['IS_CALIBRATOR'] == False))
    # indices of Cepheid host SNe (calibrators)
    cal = np.where(t['IS_CALIBRATOR'])

    # q0 and j0 from Riess 2016 pg. 11
    q0 = -0.55
    j0 = 1

    # y vector entries
    HF_ydata = t[HF]['m_b_corr'] - 5 * np.log10(299792.458*t[HF]['zHD'] * (1 + (1/2) * (1 - q0) * t[HF]['zHD'] - (1/6) * (1 - q0 - 3 * q0**2 + j0) * t[HF]['zHD']**2)) - 25
    Panth_ydata = t[Panth]['m_b_corr'] - 5 * np.log10(299792.458*t[Panth]['zHD'] * (1 + (1/2) * (1 - q0) * t[Panth]['zHD'] - (1/6) * (1 - q0 - 3 * q0**2 + j0) * t[Panth]['zHD']**2)) - 25
    cal_ydata = t[cal]['m_b_corr']

    ### shift magnitudes due to microlensing ###
    HF_ydata += shift_mags(t[HF]['zHD'], alpha)
    Panth_ydata += shift_mags(t[Panth]['zHD'], alpha)
    cal_ydata += shift_mags(t[cal]['zHD'], alpha)
        
    # covariance matrix (just get entries for the SNe we are using)
    cov = np.loadtxt(SNe_cov)[1:]
    cov = cov.reshape(len(t), len(t))
    # start getting the covariance entries. also will need covariance between calibrators and Pantheon SNe
    HF_cov = cov[np.ix_(HF[0], HF[0])]
    Panth_cov = cov[np.ix_(Panth[0], Panth[0])]
    cal_cov = cov[np.ix_(cal[0], cal[0])]

    # Riess data vector
    Y = fits.open(Riess_Y)[0].data
    # Riess equation matrix 
    L = fits.open(Riess_L)[0].data
    # Riess covariance matrix
    C = fits.open(Riess_C)[0].data

    n1 = 3215 # index of first SNe entry
    n2 = 3130 # index of first calibrator SNe
    n3 = 3207 # index of last calibrator SNe (not inclusive -- last is 3206)

    # need to (try to) match the Pantheon and Riess data
    # variance and magnitude arrays for Riess and Pantheon catalogs
    c_riess = np.array([C[i, i] for i in range(n2, n3)])
    c_panth = np.array([cov[i, i] for i in cal])[0]
    y_riess = np.array(Y[n2:n3])
    y_panth = np.array(t[cal]['m_b_corr'])

    # calculate the difference between the catalog magnitudes and variances (all combinations)
    magnitude_distances = np.abs(y_panth[:, np.newaxis] - y_riess[np.newaxis, :])
    variance_distances = np.abs(c_panth[:, np.newaxis] - c_riess[np.newaxis, :])

    # normalize the distances
    normalized_magnitude_distances = magnitude_distances**2 / np.median(magnitude_distances)**2
    normalized_variance_distances = variance_distances**2 / np.median(variance_distances)**2

    # combine the two normalized distances into a total distance matrix
    total_distances = normalized_magnitude_distances + normalized_variance_distances

    # use Hungarian algorithm to find the best matching indices that minimize the total distance
    row_indices, col_indices = linear_sum_assignment(total_distances.T)
    # col_indices --> indices to match them

    ### Y VECTOR ###
    # prepare data vectors of the correct size
    HF_Y = np.zeros(n1 + len(HF_ydata))
    Panth_Y = np.zeros(n1 + len(Panth_ydata))

    # duplicate entreis for non-SNE entries in Y vector
    HF_Y[:n1] = Y[:n1]
    Panth_Y[:n1] = Y[:n1]

    # replace calibrator entries in Y vector
    HF_Y[n2:n3] = cal_ydata[col_indices]
    Panth_Y[n2:n3] = cal_ydata[col_indices]

    # add new data entries from Pantheon+ to Y vector
    HF_Y[n1:] = HF_ydata
    Panth_Y[n1:] = Panth_ydata

    ### C MATRIX ###
    # prepare covariance matrices of the correct size
    HF_C = np.zeros((n1 + len(HF_ydata), n1 + len(HF_ydata)))
    Panth_C = np.zeros((n1 + len(Panth_ydata), n1 + len(Panth_ydata)))

    # duplicate entries for non-SNe from Riess paper covariance
    HF_C[:n1, :n1] = C[:n1, :n1]
    Panth_C[:n1, :n1] = C[:n1, :n1]

    # replace calibrator entries in covariance matrix
    HF_C[n2:n3, n2:n3] = cal_cov[np.ix_(col_indices, col_indices)]
    Panth_C[n2:n3, n2:n3] = cal_cov[np.ix_(col_indices, col_indices)]

    # add new covariance entries from Pantheon+ non-calibrators
    HF_C[n1:, n1:] = HF_cov
    Panth_C[n1:, n1:] = Panth_cov

    # get covariance between Pantheon+ and calibrators
    HF_oddcov = cov[np.ix_(HF[0], cal[0][col_indices])]
    Panth_oddcov = cov[np.ix_(Panth[0], cal[0][col_indices])]

    # insert these covariance terms into the overall matrices
    HF_C[n2:n3, n1:] = HF_oddcov.T
    HF_C[n1:, n2:n3] = HF_oddcov
    Panth_C[n2:n3, n1:] = Panth_oddcov.T
    Panth_C[n1:, n2:n3] = Panth_oddcov

    ### L MATRIX ###
    # HF has same equation matrix as Riess
    HF_L = L

    # Need to add rows to the Pantheon L matrix since we have more SNe
    Panth_L = np.zeros((L.shape[0], n1+len(Panth_ydata)))
    Panth_L[:,:n1] = L[:,:n1]
    Panth_L[:,n1:] = L[:,-1][:,np.newaxis] * np.ones(len(Panth_ydata))[np.newaxis,:]

    # pack up the HF and Panth yCL
    hubble_flow = [HF_Y, HF_C, HF_L]
    pantheon = [Panth_Y, Panth_C, Panth_L]
    return hubble_flow, pantheon

def Delta_mu(alpha, z):
    '''
    Take alpha and redshift and return $\Delta\mu$ fit from Zumalacarregui 2018.
    alpha - float
    z - 1D array of floats
    '''
    # median values scraped from Zumalacarregui 2018
    scraped_redshifts = [0, 0.34, 0.69, 1.10]
    scraped_alphas = [0, 0.5, 1.0]
    scraped_mu = np.array([[0, 0, 0, 0], 
                   [0, -0.006593406593406681, -0.02637362637362639, -0.05934065934065946],
                   [0, -0.013186813186813251, -0.04945054945054961, -0.11868131868131870]])

    # 2D interpolation over alpha and z
    zamu = RegularGridInterpolator((scraped_alphas, scraped_redshifts), scraped_mu, method='slinear', bounds_error=False, fill_value=None)
    Dmu = zamu(np.array([np.ones(len(z))*alpha, z]).T) # transpose to make it work for arrays
    return Dmu


def shift_mags(z, alpha = 0):
    '''
    Convert the median $\Delta\mu$ value for a given redshift and alpha into an additive shift in apparent magnitude.
    The intrinsic brightness is higher than the observed brightness because the SNe is slightly demagnified.
    Note that $\Delta\mu < 0$ so this magnitude correction is indeed <0.
    ${\rm mag}_{\rm actual} = {\rm mag}_{\rm seen} + 2.5 \log(1 + \Delta\mu)$
    '''
    Dmu = Delta_mu(alpha, z)
    return 2.5 * np.log10(1 + Dmu)
