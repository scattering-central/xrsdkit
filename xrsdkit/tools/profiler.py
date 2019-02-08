from collections import OrderedDict

import numpy as np 

from . import pearson
from . import peak_math

#profile_keys = list(profile_defs.keys())
profile_keys = [\
'Imax_over_Imean',\
'Ilowq_over_Imean',\
'Imax_sharpness',\
'I_fluctuation',\
'logI_fluctuation',\
'logI_max_over_std',\
'r_fftIcentroid',\
'q_Icentroid',\
'q_logIcentroid',\
'pearson_q',\
'pearson_q2',\
'pearson_expq',\
'pearson_invexpq',\
'q_best_hump',\
'q_best_trough',\
'best_hump_qwidth',\
'best_trough_qwidth',\
'q_best_hump_log',\
'q_best_trough_log',\
'best_hump_qwidth_log',\
'best_trough_qwidth_log']

profile_defs = OrderedDict.fromkeys(profile_keys)
profile_defs.update(
    Imax_over_Imean = 'maximum over mean intensity on the full q-range',
    Ilowq_over_Imean = 'mean intensity on the lower 10% of the q-range, '\
                    'divided by the mean intensity on the full q-range',
    Imax_sharpness = 'maximum over mean intensity for q-values '\
                    'from 0.9*q(Imax) to 1.1*q(Imax)',
    I_fluctuation = 'sum of difference in I between adjacent points, '\
                    'multiplied by q-width of each point, '\
                    'divided by the intensity range (Imax minus Imin)',
    logI_fluctuation = 'same as I_fluctuation but for log(I)',
    logI_max_over_std = 'max(log(I)) divided by std(log(I))',
    r_fftIcentroid = 'centroid (in length units) of the magnitude squared '\
                    'of the fourier transform of the scattering spectrum',
    #r_fftImax = 'location (in length units) of the maximum of the '\
    #                'squared magnitude of the fourier transform of the pattern' 
    q_Icentroid = 'q-space centroid of the scattering intensity',
    q_logIcentroid = 'q-space centroid of log(I)',
    pearson_q = 'Pearson correlation between q and I(q)',
    pearson_q2 = 'Pearson correlation between q squared and I(q)',
    pearson_expq = 'Pearson correlation between exp(q) and I(q)',
    pearson_invexpq = 'Pearson correlation between exp(-q) and I(q)',
    q_best_hump = 'q-vertex of parabola fit to intensity values near the best hump',
    q_best_trough = 'q-vertex of parabola fit to intensity values near the best trough',
    best_hump_qwidth = 'Focal width of same parabola used for q_best_hump',
    best_trough_qwidth = 'Focal width of same parabola used for q_best_trough',
    q_best_hump_log = 'like q_best_hump, but fit to standardized log(I)',
    q_best_trough_log = 'like q_best_trough, but fit to standardized log(I)',
    best_hump_qwidth_log = 'like best_hump_qwidth, but fit to standardized log(I)',
    best_trough_qwidth_log = 'like best_trough_qwidth, but fit to standardized log(I)'
    )

def profile_pattern(q,I):
    """Numerical profiling of a scattering or diffraction pattern.

    Profile a 1d scattering or diffraction pattern 
    (consisting of scattering vectors `q` and intensities `I`) 
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
    q : array
        array of scattering vector magnitudes
    I : array
        array of integrated scattering intensities corresponding to `q`
    
    Returns
    -------
    features : dict
        Dictionary of numerical features extracted from input pattern.
    """ 
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
    qs_nz = qs[nz]
    I_nz = I[nz]
    Is_nz = Is[nz]
    logI_nz = np.log(I_nz)
    logI_max = np.max(logI_nz)
    logI_min = np.min(logI_nz)
    logI_range = logI_max - logI_min
    logI_std = np.std(logI_nz)
    logI_mean = np.mean(logI_nz)
    logIs = (logI_nz-logI_mean)/logI_std
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

    # heuristic hump and trough analysis
    humpness,troughness = peak_math.humpness(qs_nz,logIs)
    idx_best_hump = np.argmax(humpness)
    idx_best_trough = np.argmax(troughness)
    idx_near_hump = (q_nz>q_nz[idx_best_hump]-0.1*q_std) & (q_nz<q_nz[idx_best_hump]+0.1*q_std)
    idx_near_trough = (q_nz>q_nz[idx_best_trough]-0.1*q_std) & (q_nz<q_nz[idx_best_trough]+0.1*q_std)
    p_Is_hump = np.polyfit(qs_nz[idx_near_hump],Is_nz[idx_near_hump],2)
    p_Is_trough = np.polyfit(qs_nz[idx_near_trough],Is_nz[idx_near_trough],2)
    p_logIs_hump = np.polyfit(qs_nz[idx_near_hump],logIs[idx_near_hump],2)
    p_logIs_trough = np.polyfit(qs_nz[idx_near_trough],logIs[idx_near_trough],2)
    # quadratic vertex horizontal coord is -b/2a
    q_best_hump = -1*p_Is_hump[1]/(2*p_Is_hump[0])*q_std+q_mean
    q_best_trough = -1*p_Is_trough[1]/(2*p_Is_trough[0])*q_std+q_mean
    q_best_hump_log = -1*p_logIs_hump[1]/(2*p_logIs_hump[0])*q_std+q_mean
    q_best_trough_log = -1*p_logIs_trough[1]/(2*p_logIs_trough[0])*q_std+q_mean
    # quadratic focal width is 1/a 
    best_hump_qwidth = abs(1./p_Is_hump[0])*q_std
    best_trough_qwidth = abs(1./p_Is_trough[0])*q_std
    best_hump_qwidth_log = abs(1./p_logIs_hump[0])*q_std
    best_trough_qwidth_log = abs(1./p_logIs_trough[0])*q_std

    features = OrderedDict.fromkeys(profile_keys)
    features['Imax_over_Imean'] = I_max / I_mean   
    features['Ilowq_over_Imean'] = I_lowq / I_mean
    features['Imax_sharpness'] = I_max / Imean_around_Imax
    features['I_fluctuation'] = I_fluctuation
    features['logI_fluctuation'] = logI_fluctuation
    features['logI_max_over_std'] = logI_max / logI_std
    features['r_fftIcentroid'] = r_fftIcentroid
    #features['r_fftImax'] = r_fftImax
    features['q_Icentroid'] = q_Icentroid
    features['q_logIcentroid'] = q_logIcentroid
    features['pearson_q'] = pearson_q 
    features['pearson_q2'] = pearson_q2
    features['pearson_expq'] = pearson_expq
    features['pearson_invexpq'] = pearson_invexpq
    features['q_best_hump'] = q_best_hump
    features['q_best_trough'] = q_best_trough
    features['best_hump_qwidth'] = best_hump_qwidth
    features['best_trough_qwidth'] = best_trough_qwidth
    features['q_best_hump_log'] = q_best_hump_log
    features['q_best_trough_log'] = q_best_trough_log
    features['best_hump_qwidth_log'] = best_hump_qwidth_log
    features['best_trough_qwidth_log'] = best_trough_qwidth_log
    # NOTE: considered these features, decidedly too arbitrary:
    #features['q_min'] = q[0]
    #features['q_max'] = q[-1]

    return features

