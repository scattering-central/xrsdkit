from collections import OrderedDict

import numpy as np 

from . import standardize_array, pearson
from ..fitting import fit_I0

profile_keys = list([
    'Imax_over_Imean',
    'Imax_sharpness',
    'I_fluctuation',
    'logI_fluctuation',
    'logI_max_over_std',
    'r_fftIcentroid',
    'r_fftImax',
    'q_Icentroid',
    'q_logIcentroid',
    'pearson_q',
    'pearson_q2',
    'pearson_expq',
    'pearson_invexpq'])

profile_keys_1 = []
profile_keys_1.extend(profile_keys) # a short set of features for diffuse and cristaline classifications

gp_profile_keys = list([
    'I0_over_Imean',
    'I0_curvature',
    'q_at_half_I0'])
spherical_profile_keys = list([
    'q_at_Iq4_min1',
    'pIq4_qwidth',
    'pI_qvertex',
    'pI_qwidth'])

profile_keys_2 = []
profile_keys_2.extend(profile_keys)
profile_keys_2.extend(gp_profile_keys)

profile_keys.extend(gp_profile_keys)
profile_keys.extend(spherical_profile_keys)
 
def profile_spectrum(q_I):
    """Numerical profiling of a SAXS spectrum.

    Profile a saxs spectrum (n-by-2 array `q_I`) 
    by computing several relatively fast numerical metrics.
    The metrics should be invariant with respect to intensity scaling,
    and should not be strongly affected by minor details of the data
    (e.g. the limits of the reported q-range or the q-resolution).

    This method should execute gracefully
    for any n-by-2 input array,
    such that it can be used to profile any type of spectrum. 
    TODO: document the returned metrics here.

    Parameters
    ----------
    q_I : array
        n-by-2 array of scattering vector q and scattered intensity I
    
    Returns
    -------
    features : dict
        Dictionary of metrics computed from input spectrum `q_I`.
        The features are:

        - 'Imax_over_Imean': maximum over mean intensity on the full q-range

        - 'Imax_sharpness': maximum over mean intensity for q-values 
            from 0.9*q(Imax) to 1.1*q(Imax)

        - 'I_fluctuation': sum of difference in I between adjacent points,
            multiplied by q-width of each point,
            divided by the intensity range (Imax minus Imin)

        - 'logI_fluctuation': same as I_fluctuation,
            but for log(I) and including only points where I>0

        - 'logI_max_over_std': max(log(I)) divided by std(log(I)),
            including only points with I>0

        - 'r_fftIcentroid': real-space centroid of the magnitude squared
            of the fourier transform of the scattering spectrum

        - 'r_fftImax': real-space maximum of the magnitude squared
            of the fourier transform of the scattering spectrum
 
        - 'q_Icentroid': q-space centroid of the scattering intensity

        - 'q_logIcentroid': q-space centroid of log(I)

        - 'pearson_q': Pearson correlation between q and I(q)

        - 'pearson_q2': Pearson correlation between q squared and I(q)

        - 'pearson_expq': Pearson correlation between exp(q) and I(q)
 
        - 'pearson_invexpq': Pearson correlation between exp(-q) and I(q) 
    """ 
    q = q_I[:,0]
    I = q_I[:,1]
    # I metrics
    idxmax = np.argmax(I)
    idxmin = np.argmin(I)
    I_min = I[idxmin]
    I_max = I[idxmax] 
    q_Imax = q[idxmax]
    I_range = I_max - I_min
    I_mean = np.mean(I)
    Imax_over_Imean = I_max/I_mean
    # log(I) metrics
    nz = I>0
    q_nz = q[nz]
    I_nz = I[nz]
    logI_nz = np.log(I_nz)
    logI_max = np.max(logI_nz)
    logI_min = np.min(logI_nz)
    logI_range = logI_max - logI_min
    logI_std = np.std(logI_nz)
    logI_max_over_std = logI_max / logI_std
    # I_max peak shape analysis
    idx_around_max = ((q > 0.9*q_Imax) & (q < 1.1*q_Imax))
    Imean_around_max = np.mean(I[idx_around_max])
    Imax_sharpness = I_max / Imean_around_max

    ### integration and intensity centroid
    dq = q[1:] - q[:-1]
    qcenter = 0.5 * (q[1:] + q[:-1])
    Itrap = 0.5 * (I[1:] + I[:-1])
    I_qint = np.sum(dq*Itrap)
    qI_qint = np.sum(qcenter*dq*Itrap)
    q_Icentroid = qI_qint / I_qint
    # same thing for log(I)
    dq_nz = q_nz[1:] - q_nz[:-1]
    qcenter_nz = 0.5 * (q_nz[1:] + q_nz[:-1])
    logItrap_nz = 0.5 * (logI_nz[1:] + logI_nz[:-1])
    logI_qint_nz = np.sum(dq_nz*logItrap_nz)
    qlogI_qint_nz = np.sum(qcenter_nz*dq_nz*logItrap_nz)
    q_logIcentroid = qlogI_qint_nz / logI_qint_nz

    ### fluctuation analysis
    nn_diff = I[1:]-I[:-1]
    I_fluctuation = np.sum(np.abs(nn_diff)*dq)/I_range
    nn_logdiff = logI_nz[1:]-logI_nz[:-1]
    logI_fluctuation = np.sum(np.abs(nn_logdiff)*dq_nz)/logI_range
    # keep indices where the sign of this difference changes.
    # also keep first index
    nn_diff_prod = nn_diff[1:]*nn_diff[:-1]
    idx_keep = np.hstack((np.array([True]),nn_diff_prod<0))
    fluc = np.sum(np.abs(nn_diff[idx_keep]))
    logI_fluctuation = fluc/logI_range

    ### correlation analysis
    pearson_q = pearson(q,I)
    pearson_q2 = pearson(q**2,I)
    pearson_expq = pearson(np.exp(q),I)
    pearson_invexpq = pearson(np.exp(-1*q),I)

    ### fourier analysis
    fftI = np.fft.fft(I)
    fftampI = np.abs(fftI)
    r = np.fft.fftfreq(q.shape[-1])
    idx_rpos = (r>0)
    r_pos = r[idx_rpos]
    fftampI_rpos = fftampI[idx_rpos]
    
    dr_pos = r_pos[1:] - r_pos[:-1]
    rcenter = 0.5 * (r_pos[1:] + r_pos[:-1])
    fftItrap = 0.5 * (fftampI_rpos[1:] + fftampI_rpos[:-1])
    fftI_rint = np.sum(dr_pos*fftItrap)
    rfftI_rint = np.sum(rcenter*dr_pos*fftItrap)
    r_fftIcentroid = rfftI_rint / fftI_rint 
    r_fftImax = r_pos[np.argmax(fftampI_rpos)]

    features = OrderedDict.fromkeys(profile_keys_1) # we need NOT EXTENDED profile_kesy. If we use extended version,
    features['Imax_over_Imean'] = Imax_over_Imean   # we have NONs for "extra" features
    features['Imax_sharpness'] = Imax_sharpness
    features['I_fluctuation'] = I_fluctuation
    features['logI_fluctuation'] = logI_fluctuation
    features['logI_max_over_std'] = logI_max_over_std
    features['r_fftIcentroid'] = r_fftIcentroid
    features['r_fftImax'] = r_fftImax
    features['q_Icentroid'] = q_Icentroid
    features['q_logIcentroid'] = q_logIcentroid
    features['pearson_q'] = pearson_q 
    features['pearson_q2'] = pearson_q2
    features['pearson_expq'] = pearson_expq
    features['pearson_invexpq'] = pearson_invexpq
    return features 

def guinier_porod_profile(q_I):
    """Numerical profiling of guinier_porod scattering intensities.

    Computes the intensity at q=0 and the q value 
    at which the intensity drops to half of I(q=0).

    Parameters
    ----------
    q_I : array
        n-by-2 array of scattering vector q and scattered intensity I

    Returns
    -------
    features : dict
        Dictionary of metrics computed from input spectrum `q_I`.
        The features are:

        - 'I0_over_Imean': intensity at q=0, obtained by polynomial fitting
            with the slope at q=0 constrained to be 0,
            divided by the average intensity. 

        - 'I0_curvature': curvature of the polynomial used in 'I0_over_Imean',
            evaluated at q=0, normalized by the mean intensity.

        - 'q_at_half_I0': q-value at which the intensity
            first drops to half of I(q=0)
    """
    q = q_I[:,0]
    I = q_I[:,1]
    q_s,q_mean,q_std = standardize_array(q)
    I_s,I_mean,I_std = standardize_array(q)
    I_at_0, p_I0 = fit_I0(q,I,4)
    dpdq = np.polyder(p_I0)
    d2pdq2 = np.polyder(dpdq)
    I0_curv = (np.polyval(d2pdq2,-1*q_mean/q_std)*I_std+I_mean)/I_mean
    idx_half_I0 = np.min(np.where(I<0.5*I_at_0))
    
    features = OrderedDict.fromkeys(gp_profile_keys)
    features['I0_over_Imean'] = I_at_0/I_mean
    features['q_at_half_I0'] = q[idx_half_I0]
    features['I0_curvature'] = I0_curv
    return features

def spherical_normal_profile(q_I):
    """Numerical profiling of spherical_normal scattering intensities.
    
    Computes several properties of the I(q) and I(q)*q**4 curves.

    Parameters
    ----------
    q_I : array
        n-by-2 array of scattering vector q and scattered intensity I

    Returns
    -------
    features : dict
        Dictionary of metrics computed from input spectrum `q_I`.
        The features are:

        - 'q_at_Iqqqq_min1': q value at first minimum of I*q^4

        - 'pIqqqq_qwidth': Focal q-width of polynomial fit to I*q^4 
            near first minimum of I*q^4 

        - 'pI_qvertex': q value of vertex of polynomial fit to I(q) 
            near first minimum of I*q^4  

        - 'pI_qwidth': Focal q-width of polynomial fit to I(q) 
            near first minimum of I*q^4
    """
    q = q_I[:,0]
    I = q_I[:,1]
    #######
    # 1: Find the first local max
    # and subsequent local minimum of I*q**4 
    Iqqqq = I*q**4
    # Window width for determining local extrema: 
    w = 10
    idxmax1, idxmin1 = 0,0
    stop_idx = len(q)-w-1
    test_range = np.arange(w,stop_idx)
    for idx in test_range:
        if np.argmax(Iqqqq[idx-w:idx+w+1]) == w and idxmax1 == 0:
            idxmax1 = idx
        if np.argmin(Iqqqq[idx-w:idx+w+1]) == w and idxmin1 == 0 and not idxmax1 == 0:
            idxmin1 = idx
    features = OrderedDict.fromkeys(spherical_profile_keys)
    if idxmin1 == 0 or idxmax1 == 0:
        return features 
    #######
    # 2: Characterize I*q**4 around idxmin1. 
    idx_around_min1 = (q>0.9*q[idxmin1]) & (q<1.1*q[idxmin1])
    q_min1_mean = np.mean(q[idx_around_min1])
    q_min1_std = np.std(q[idx_around_min1])
    q_min1_s = (q[idx_around_min1]-q_min1_mean)/q_min1_std
    Iqqqq_min1_mean = np.mean(Iqqqq[idx_around_min1])
    Iqqqq_min1_std = np.std(Iqqqq[idx_around_min1])
    Iqqqq_min1_s = (Iqqqq[idx_around_min1]-Iqqqq_min1_mean)/Iqqqq_min1_std
    p_min1 = np.polyfit(q_min1_s,Iqqqq_min1_s,2,None,False,np.ones(len(q_min1_s)),False)
    # quadratic vertex horizontal coord is -b/2a
    qs_at_min1 = -1*p_min1[1]/(2*p_min1[0])
    features['q_at_Iq4_min1'] = qs_at_min1*q_min1_std+q_min1_mean
    # quadratic focal width is 1/a 
    p_min1_fwidth = abs(1./p_min1[0])
    features['pIq4_qwidth'] = p_min1_fwidth*q_min1_std
    #######
    # 3: Characterize I(q) around idxmin1.
    I_min1_mean = np.mean(I[idx_around_min1])
    I_min1_std = np.std(I[idx_around_min1])
    I_min1_s = (I[idx_around_min1]-I_min1_mean)/I_min1_std
    pI_min1 = np.polyfit(q_min1_s,I_min1_s,2,None,False,np.ones(len(q_min1_s)),False)
    # quadratic vertex horizontal coord is -b/2a
    qs_vertex = -1*pI_min1[1]/(2*pI_min1[0])
    features['pI_qvertex'] = qs_vertex*q_min1_std+q_min1_mean
    # quadratic focal width is 1/a 
    pI_fwidth = abs(1./pI_min1[0])
    features['pI_qwidth'] = pI_fwidth*q_min1_std
    return features 

def full_profile(q_I):
    profs = OrderedDict.fromkeys(profile_keys)
    profs.update(profile_spectrum(q_I))    
    try:
        gp_prof = guinier_porod_profile(q_I)
        profs.update(gp_prof)
    except:
        pass
    try:
        sph_prof = spherical_normal_profile(q_I)
        profs.update(sph_prof)
    except:
        pass
    return profs

