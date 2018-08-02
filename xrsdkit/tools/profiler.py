from collections import OrderedDict

import numpy as np 

from . import standardize_array, pearson
from . import peak_math

profile_keys = list([
    'Imax_over_Imean',
    'Ilowq_over_Imean',
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
    'pearson_invexpq',
    'q_best_pk',
    'q_best_vly',
    'best_pk_qwidth',
    'best_vly_qwidth'])

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

        - 'Ilowq_over_Imean': mean intensity on the lower 10% of the q-range,
            divided by the mean intensity on the full q-range

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

        - 'q_best_pk' : Most peak-like q-value (see xrsdkit.tools.peak_math.peakness())

        - 'q_best_vly' : Most negative-peak-like q-value, like q_best_pk for -1*I(q) 

        - 'best_pk_qwidth' : Focal width of parabola fit to 
            standardized intensities within +/-10% of q_best_pk 

        - 'best_vly_qwidth' : Focal width of parabola fit to 
            standardized intensities within +/-10% of q_best_vly 
    """ 
    q = q_I[:,0]
    I = q_I[:,1]
    # q, I metrics
    idxmax = np.argmax(I)
    idxmin = np.argmin(I)
    I_min = I[idxmin]
    I_max = I[idxmax] 
    q_Imax = q[idxmax]
    I_range = I_max - I_min
    I_mean = np.mean(I)
    I_std = np.std(I)
    Is = (I-I_mean)/I_std
    q_mean = np.mean(q)
    q_std = np.std(q)
    qs = (q-q_mean)/q_std
    idx_lowq = (q < q[0]+0.1*(q[-1]-q[0]))
    I_lowq = np.mean(I[idx_lowq])
    # log(I) metrics
    nz = I>0
    q_nz = q[nz]
    I_nz = I[nz]
    logI_nz = np.log(I_nz)
    logI_max = np.max(logI_nz)
    logI_min = np.min(logI_nz)
    logI_range = logI_max - logI_min
    logI_std = np.std(logI_nz)
    # I_max peak shape analysis
    idx_around_Imax = ((q > 0.9*q_Imax) & (q < 1.1*q_Imax))
    Imean_around_Imax = np.mean(I[idx_around_Imax])

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

    ### heuristic fluctuation analysis 
    nn_diff = I[1:]-I[:-1]
    nn_difflog = logI_nz[1:]-logI_nz[:-1]
    #I_fluctuation = np.sum(np.abs(nn_diff)*dq)/I_range
    #logI_fluctuation = np.sum(np.abs(nn_logdiff)*dq_nz)/logI_range
    # count indices where the sign of the nearest-neighbor difference changes 
    nn_diff_prod = nn_diff[1:]*nn_diff[:-1]
    idx_keep = np.hstack((np.array([True]),nn_diff_prod<0))
    nn_difflog_prod = nn_difflog[1:]*nn_difflog[:-1]
    idx_keep_log = np.hstack((np.array([True]),nn_difflog_prod<0))
    I_fluctuation = np.sum(np.abs(nn_diff[idx_keep]))/I_range
    logI_fluctuation = np.sum(np.abs(nn_difflog[idx_keep_log]))/logI_range

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

    # peak and valley analysis
    pkness = peak_math.peakness(q,I)
    vlyness = peak_math.peakness(q,-1*I)
    idx_best_pk = np.argmax(pkness)
    idx_best_vly = np.argmax(vlyness)
    idx_near_pk = (q>0.9*q[idx_best_pk]) & (q<1.1*q[idx_best_pk])
    idx_near_vly = (q>0.9*q[idx_best_vly]) & (q<1.1*q[idx_best_vly])
    pIs_pk = np.polyfit(qs[idx_near_pk],Is[idx_near_pk],2)
    pIs_vly = np.polyfit(qs[idx_near_vly],Is[idx_near_vly],2)
    # quadratic vertex horizontal coord is -b/2a
    #qs_pk = -1*pIs_pk[1]/(2*pIs_pk[0])
    #features['pIs_qvertex'] = qs_pk*q_std+q_mean
    # quadratic focal width is 1/a 
    pIs_pk_fwidth = abs(1./pIs_pk[0])
    pIs_vly_fwidth = abs(1./pIs_vly[0])

    features = OrderedDict.fromkeys(profile_keys)
    features['Imax_over_Imean'] = I_max / I_mean   
    features['Ilowq_over_Imean'] = I_lowq / I_mean
    features['Imax_sharpness'] = I_max / Imean_around_Imax
    features['I_fluctuation'] = I_fluctuation
    features['logI_fluctuation'] = logI_fluctuation
    features['logI_max_over_std'] = logI_max / logI_std
    features['r_fftIcentroid'] = r_fftIcentroid
    features['r_fftImax'] = r_fftImax
    features['q_Icentroid'] = q_Icentroid
    features['q_logIcentroid'] = q_logIcentroid
    features['pearson_q'] = pearson_q 
    features['pearson_q2'] = pearson_q2
    features['pearson_expq'] = pearson_expq
    features['pearson_invexpq'] = pearson_invexpq
    features['q_best_pk'] = q[idx_best_pk]
    features['q_best_vly'] = q[idx_best_vly]
    features['best_pk_qwidth'] = pIs_pk_fwidth*q_std
    features['best_vly_qwidth'] = pIs_vly_fwidth*q_std
    # NOTE: considered these features, decidedly too arbitrary:
    #features['q_min'] = q[0]
    #features['q_max'] = q[-1]

    return features

