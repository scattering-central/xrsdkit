"""
SAXS data fitting functions by SSRL.

Citations:
Lenson A. Pellouchoud, Amanda P. Fournier, Christopher J. Tassone (2017).
ssrlsaxsfit [software]. 
Retrieved from https://github.com/scatteringcentral/pysaxs.
"""
from __future__ import print_function
from collections import OrderedDict

import numpy as np
from scipy.optimize import minimize as scipimin

def compute_saxs(q,params):
    """
    Given q and a dict of parameters,
    compute the saxs spectrum.
    Supported parameters are the same as 
    SaxsParameterization Operation outputs,
    and should include at least the following keys:
    I_at_0, precursor_flag, form_flag, structure_flag.

    TODO: Document the equation.
    """
    pre_flag = params['precursor_flag']
    f_flag = params['form_flag']
    s_flag = params['structure_flag']
    I_at_0 = params['I_at_0']
    if not any([pre_flag,f_flag]):
        return np.ones(len(q))*I_at_0
    else:
        I = np.zeros(len(q))
        if pre_flag:
            I0_pre = params['I0_pre']
            r0_pre = params['r0_pre']
            I_pre = compute_spherical_normal_saxs(q,r0_pre,0)
            I += I0_pre*I_pre
        if f_flag: 
            I0_sph = params['I0_form']
            r0_sph = params['r0_form']
            sigma_sph = params['sigma_form']
            I_sph = compute_spherical_normal_saxs(q,r0_sph,sigma_sph)
            I += I0_sph*I_sph
        #if s_flag:
        #    I0_pk = params['I0_pk']
        #    I_pk = compute_peaks()
    return I

def profile_spectrum(q,I,dI=None):
    """
    Profile a saxs spectrum (q,I) 
    by taking several fast numerical metrics 
    on the measured data.

    :returns: dictionary of metrics and spectrum profiling flags.
    Dict keys and descriptions: 
    - 'ERROR_MESSAGE': Any errors or warnings are reported here. 
    - 'bad_data_flag': Boolean indicating that the spectrum is unfamiliar or mostly made of noise. 
    - 'precursor_flag': Boolean indicating presence of precursor terms. 
    - 'form_flag': Boolean indicating presence of form factor terms. 
    - 'form_id': Form factor identity, e.g. sphere, cube, rod, etc. 
    - 'structure_flag': Boolean indicating presence of structure factor terms. 
    - 'structure_id': Structure identity, e.g. fcc, hcp, bcc, etc. 
    - 'low_q_ratio': fraction of integrated intensity for q<0.4 
    - 'high_q_ratio': fraction of integrated intensity for q>0.4 
    - 'low_q_logratio': fraction of total log(I)-min(log(I)) over q<0.4 
    - 'high_q_logratio': fraction of total log(I)-min(log(I)) over q>0.4 
    - 'high_freq_ratio': Ratio of the upper half to the lower half 
        of the power spectrum of the discrete fourier transform of the intensity. 
        fluctuation_strength': Integrated fluctuation of intensity 
        (sum of difference in intensity between adjacent points 
        taken only where this difference changes sign), 
        divided by the range (maximum minus minimum) of intensity. 
    - 'Imax_over_Imean': maximum intensity divided by mean intensity. 
    - 'Imax_over_Imean_local': maximum intensity divided by 
        mean intensity over q values within 10% of the q value of the maximum. 
    - 'Imax_over_Ilowq': Maximum intensity divided by mean intensity over q<0.1. 
    - 'Imax_over_Ihighq': Maximum intensity divided by mean intensity over q>0.4. 
    - 'Ilowq_over_Ihighq': Mean intensity for q<0.1 divided by mean intensity for q>0.4. 
    - 'low_q_logcurv': Curvature of parabola fit to log(I) versus log(q) for q<0.1. 
    - 'q_Imax': q value of the maximum intensity. 
    - 'bin_strengths': log(I) integrated in q-bins of 0.1 1/Angstrom from 0 to 1 1/Angstrom. 
    """ 
    ### amplitude analysis:
    idxmax = np.argmax(I)
    Imax = I[idxmax] 
    q_Imax = q[idxmax]
    idxmin = np.argmin(I)
    Imin = I[idxmin]
    Irange = Imax - Imin
    Imean = np.mean(I)
    Imax_over_Imean = float(Imax)/float(Imean)
    idx_around_max = ((q > 0.9*q_Imax) & (q < 1.1*q_Imax))
    Imean_around_max = np.mean(I[idx_around_max])
    Imax_over_Imean_local = Imax / Imean_around_max

    ### fourier analysis
    n_q = len(q)
    fftI = np.fft.fft(I)
    ampI = np.abs(fftI)
    powI = np.abs(fftI)**2
    high_freq_ratio = np.sum(powI[n_q/4:n_q/2])/np.sum(powI[1:n_q/4])
    #high_freq_ratio = np.sum(ampI[n_q/4:n_q/2])/np.sum(ampI[1:n_q/4])

    ### fluctuation analysis
    # array of the difference between neighboring points:
    nn_diff = I[1:]-I[:-1]
    # keep indices where the sign of this difference changes.
    # also keep first index
    nn_diff_prod = nn_diff[1:]*nn_diff[:-1]
    idx_keep = np.hstack((np.array([True]),nn_diff_prod<0))
    fluc = np.sum(np.abs(nn_diff[idx_keep]))
    #fluctuation_strength = fluc/Irange
    fluctuation_strength = fluc/Imean

	# correlations on intensity
    #q_corr = compute_pearson(I,np.linspace(0,1,n_q))
    #qsquared_corr = compute_pearson(I,np.linspace(0,1,n_q)**2)
    #cosq_corr = compute_pearson(I,np.cos(q*np.pi/(2*q[-1])))
    #cos2q_corr = compute_pearson(I,np.cos(q*np.pi/(2*q[-1]))**2)
    #invq_corr = compute_pearson(I,q**-1)
    #invq4_corr = compute_pearson(I,q**-4)

    # correlations on log intensity
    #idx_nz = np.invert(I <= 0)
    #q_logcorr = compute_pearson( np.log(I[idx_nz]) , np.linspace(0,1,n_q)[idx_nz] )
    #qsquared_logcorr = compute_pearson( np.log(I[idx_nz]) , (np.linspace(0,1,n_q)[idx_nz])**2 )
    #cosq_logcorr = compute_pearson( np.log(I[idx_nz]) , np.cos(q*np.pi/(2*q[-1]))[idx_nz] )
    #cos2q_logcorr = compute_pearson( np.log(I[idx_nz]) , (np.cos(q*np.pi/(2*q[-1]))[idx_nz])**2 )
    #invq_logcorr = compute_pearson( np.log(I[idx_nz]) , q[idx_nz]**-1 )

    ### bin-integrated intensity analysis
    bin_strengths = np.zeros(10)
    for i in range(10):
        qmini, qmaxi = i*0.1, (i+1)*0.1 
        idxi = ((q>=qmini) & (q<qmaxi))
        if any(idxi):
            qi = q[ idxi ]
            Ii = I[ idxi ]
            dqi = qi[1:]-qi[:-1]
            Ii = (Ii[1:]+Ii[:-1])/2
            bin_strengths[i] = np.sum(np.log(Ii) * dqi) / (qi[-1]-qi[0]) 
    idx_nz = (I>0)
    q_nz = q[idx_nz] 
    I_nz_log = np.log(I[idx_nz])
    # make values positive:
    I_nz_log = I_nz_log-np.min(I_nz_log)
    I_logsum = np.sum(I_nz_log)
    low_q_logratio = np.sum(I_nz_log[(q_nz<0.4)])/I_logsum
    high_q_logratio = np.sum(I_nz_log[(q_nz>=0.4)])/I_logsum
    I_sum = np.sum(I)
    low_q_ratio = np.sum(I[(q<0.4)])/I_sum
    high_q_ratio = np.sum(I[(q>=0.4)])/I_sum

    ### curve shape analysis
    lowq_idx = q<0.1
    highq_idx = q>0.4
    lowq = q[lowq_idx]
    highq = q[highq_idx]
    I_lowq = I[lowq_idx]
    I_highq = I[highq_idx]
    I_lowq_mean = np.mean(I_lowq)
    I_highq_mean = np.mean(I_highq)
    lowq_mean = np.mean(lowq)
    lowq_std = np.std(lowq)
    I_lowq_std = np.std(I_lowq)
    I_lowq_s = I_lowq/I_lowq_std
    lowq_s = (lowq - lowq_mean)/lowq_std
    #p_lowq = fit_with_slope_constraint(lowq_s,np.log(I_lowq_s),-1*lowq_mean/lowq_std,0,3) 
    #p_lowq = fit_with_slope_constraint(lowq_s,np.log(I_lowq_s),lowq_s[-1],0,3) 
    nz = (I_lowq_s>0)
    p_lowq = np.polyfit(lowq_s[nz],np.log(I_lowq_s[nz]),2)
    low_q_logcurv = p_lowq[0]
    Imax_over_Ilowq = float(Imax)/I_lowq_mean
    Imax_over_Ihighq = float(Imax)/I_highq_mean
    Ilowq_over_Ihighq = I_lowq_mean/I_highq_mean

    # Flagging bad data: 
    # Data with high noise are bad.
    # Data that are totally flat or increasing in q are bad.
    bad_data_flag = ( (fluctuation_strength > 20 and Imax_over_Imean_local < 2)
                    or low_q_ratio/high_q_ratio < 1
                    or Ilowq_over_Ihighq < 10 )

    form_id = None 
    structure_id = None 
    if bad_data_flag:
        form_flag = False
        precursor_flag = False
        structure_flag = False
    else:
        # Flagging form factor:
        # Intensity should be quite decreasing in q.
        # Low-q region should be quite flat.
        # Low-q mean intensity should be much larger than low-q fluctuations.
        form_flag = low_q_ratio/high_q_ratio > 10 
        if form_flag:
            # TODO: determine form factor here
            #form_id = 'sphere'
            form_id = 'NOT_IMPLEMENTED'
 
        # Flagging precursors: 
        # Intensity should decrease in q, at least mildly.
        # Intensity should decrease more sharply at high q.
        # Low-q region of spectrum should be quite flat.
        # Noise levels may be high if only precursors are present.
        # More high-q intensity than form factor alone.
        nz_bins = np.invert(np.array((bin_strengths==0)))
        s_nz = bin_strengths[nz_bins]
        precursor_flag = ( low_q_ratio/high_q_ratio > 2 
                        and high_q_ratio > 1E-3 
                        #and np.argmin(s_nz) == np.sum(nz_bins)-1 
                        #and (s_nz[-2]-s_nz[-1]) > (s_nz[-3]-s_nz[-2]) 
                        and Imax_over_Ilowq < 4 )

        # Flagging structure:
        # Structure is likely to cause max intensity to be outside the low-q region. 
        # Maximum intensity should be large relative to its 'local' mean intensity. 
        structure_flag = Imax_over_Imean_local > 2 and q_Imax > 0.06 
        if structure_flag:
            # TODO: determine structure factor here
            structure_id = 'NOT_IMPLEMENTED'
    d_r = OrderedDict() 
    d_r['bad_data_flag'] = bad_data_flag
    d_r['precursor_flag'] = precursor_flag
    d_r['form_flag'] = form_flag
    d_r['structure_flag'] = structure_flag
    d_r['structure_id'] = structure_id 
    d_r['form_id'] = form_id 
    d_r['low_q_logcurv'] = low_q_logcurv
    d_r['Imax_over_Imean'] = Imax_over_Imean
    d_r['Imax_over_Imean_local'] = Imax_over_Imean_local
    d_r['Imax_over_Ilowq'] = Imax_over_Ilowq 
    d_r['Imax_over_Ihighq'] = Imax_over_Ihighq 
    d_r['Ilowq_over_Ihighq'] = Ilowq_over_Ihighq 
    d_r['low_q_logratio'] = low_q_logratio 
    d_r['high_q_logratio'] = high_q_logratio 
    d_r['low_q_ratio'] = low_q_ratio 
    d_r['high_q_ratio'] = high_q_ratio 
    d_r['high_freq_ratio'] = high_freq_ratio 
    d_r['fluctuation_strength'] = fluctuation_strength
    d_r['q_Imax'] = q_Imax
    d_r['bin_strengths'] = bin_strengths
    #d_r['q_logcorr'] = q_logcorr
    #d_r['qsquared_logcorr'] = qsquared_logcorr
    #d_r['cos2q_logcorr'] = cos2q_logcorr
    #d_r['cosq_logcorr'] = cosq_logcorr
    #d_r['invq_logcorr'] = invq_logcorr
    #d_r['q_corr'] = q_corr
    #d_r['qsquared_corr'] = qsquared_corr
    #d_r['cos2q_corr'] = cos2q_corr
    #d_r['cosq_corr'] = cosq_corr
    #d_r['invq_corr'] = invq_corr
    return d_r

def parameterize_spectrum(q,I,metrics,fixed_params={}):
    """
    Determine a parameterization for a scattering equation,
    beginning with the measured spectrum (q,I) 
    and a dict of features (metrics).
    Dict metrics is similar or equal to 
    the output dict of profile_spectrum().
    """
    if metrics['bad_data_flag']:
        # stop
        return metrics 
    fix_keys = fixed_params.keys()

    pre_flag = metrics['precursor_flag']
    f_flag = metrics['form_flag']
    s_flag = metrics['structure_flag']

    # Check for overconstrained system
    if 'I_at_0' in fix_keys and 'I0_form' in fix_keys and 'I0_pre' in fix_keys:
        val1 = fixed_params['I_at_0']
        val2 = fixed_params['I0_form'] + fixed_params['I0_pre'] 
        if not val1 == val2:
            msg = str('Spectrum intensity is overconstrained. '
            + 'I_at_0 is constrained to {}, '
            + 'but I0_form + I0_pre = {}. '.format(val1,val2))
            raise ValueError(msg)
    
    if 'I0_form' in fix_keys and 'I0_pre' in fix_keys:
        I_at_0 = fixed_params['I0_pre'] + fixed_params['I0_form']
    else:
        # Perform a low-q spectrum fit to get I(q=0).
        if s_flag:
            # If structure factor scattering,
            # expect low-q SNR to be high, use q<0.06,
            # fit to second order.
            idx_lowq = (q<0.06)
            I_at_0 = fit_I0(q[idx_lowq],I[idx_lowq],2)
            metrics['I_at_0'] = I_at_0
            #
            #
            # And now bail, until structure factor parameterization is in.
            #
            #
            metrics['ERROR_MESSAGE'] = '[{}] structure factor parameterization not yet supported'.format(__name__)
            return metrics
        elif f_flag:
            # If form factor scattering, fit 3rd order. 
            # Use q<0.1 and disregard lowest-q values if they are far from the mean, 
            # as these points are likely dominated by experimental error.
            idx_lowq = (q<0.1)
            Imean_lowq = np.mean(I[idx_lowq])
            Istd_lowq = np.std(I[idx_lowq])
            idx_good = ((I[idx_lowq] < Imean_lowq+Istd_lowq) & (I[idx_lowq] > Imean_lowq-Istd_lowq))
            #dI_lowq = (I[idx_lowq][1:] - I[idx_lowq][:-1])/Imed_lowq
            #idx_good = np.hstack( ( (abs(dI_lowq)<0.1) , np.array([True]) ) )
            I_at_0 = fit_I0(q[idx_lowq][idx_good],I[idx_lowq][idx_good],3)
        else:
            # If only precursor scattering, fit the entire spectrum.
            I_at_0 = fit_I0(q,I,4)
        if I_at_0 > 10*I[0] or I_at_0 < 0.1*I[0]:
            # stop 
            msg = 'polynomial fit for I(q=0) deviates too far from low-q values' 
            metrics['ERROR_MESSAGE'] = msg
            metrics['I_at_0'] = I_at_0
            metrics['bad_data_flag'] = True 
            return metrics

    # If we make it to the point, I_at_0 should be set.
    metrics['I_at_0'] = I_at_0

    r0_form = None
    sigma_form = None
    if f_flag:
        if ('r0_form' in fix_keys and 'sigma_form' in fix_keys):
            r0_form = fixed_params['r0_form']
            sigma_form = fixed_params['sigma_form']
        else:
            # get at least one of r0_form or sigma_form from spherical_normal_heuristics()
            r0_form, sigma_form = spherical_normal_heuristics(q,I,I_at_0=I_at_0)
            if 'r0_form' in fix_keys:
                r0_form = fixed_params['r0_form']
            if 'sigma_form' in fix_keys:
                sigma_form = fixed_params['sigma_form']
    metrics['r0_form'] = r0_form
    metrics['sigma_form'] = sigma_form
    
    r0_pre = None
    if pre_flag:
        if 'r0_pre' in fix_keys:
            r0_pre = fixed_params['r0_pre']
        else:
            r0_pre = precursor_heuristics(q,I,I_at_0=I_at_0)
    metrics['r0_pre'] = r0_pre 

    I0_pre = None
    I0_form = None
    if pre_flag and f_flag:
        if 'I0_pre' in fix_keys and 'I0_form' in fix_keys:
            I0_pre = fixed_params['I0_pre']
            I0_form = fixed_params['I0_form']
        elif 'I0_pre' in fix_keys:
            I0_pre = fixed_params['I0_pre']
            I0_form = I_at_0 - I0_pre
        elif 'I0_form' in fix_keys:
            I0_form = fixed_params['I0_form']
            I0_pre = I_at_0 - I0_form
        else:
            I_pre = compute_spherical_normal_saxs(q,r0_pre,0)
            I_form = compute_spherical_normal_saxs(q,r0_form,sigma_form)
            I_nz = np.invert((I<=0))
            I_error = lambda x: np.sum( (np.log(I_at_0*(x*I_pre+(1.-x)*I_form)[I_nz])-np.log(I[I_nz]))**2 )
            x_res = minimize(I_error,[0.1],bounds=[(0.0,1.0)]) 
            x_fit = x_res.x[0]
            I0_form = (1.-x_fit)*I_at_0
            I0_pre = x_fit*I_at_0
    elif pre_flag:
        if 'I0_pre' in fix_keys:
            I0_pre = fixed_params['I0_pre']
        else:
            I0_pre = I_at_0 
    elif f_flag:
        if 'I0_form' in fix_keys:
            I0_form = fixed_params['I0_form'] 
        else:
            I0_form = I_at_0 
    metrics['I0_pre'] = I0_pre 
    metrics['I0_form'] = I0_form 

    q0_structure = None
    I0_structure = None
    sigma_structure = None
    #if s_flag:
    #   .....   
    metrics['q0_structure'] = q0_structure 
    metrics['I0_structure'] = I0_structure 
    metrics['sigma_structure'] = sigma_structure

    I_guess = compute_saxs(q,p)
    q_I_guess = np.array([q,I_guess]).T
    nz = ((I>0)&(I_guess>0))
    logI_nz = np.log(I[nz])
    logIguess_nz = np.log(I_guess[nz])
    Imean = np.mean(logI_nz)
    Istd = np.std(logI_nz)
    logI_nz_s = (logI_nz - Imean) / Istd
    logIguess_nz_s = (logIguess_nz - Imean) / Istd
    metrics['R2log_guess'] = compute_Rsquared(logI_nz,logIguess_nz)
    metrics['chi2log_guess'] = compute_chi2(logI_nz_s,logIguess_nz_s)
    return metrics

def fit_spectrum(q,I,objfun,features,x_keys,constraints=[]):
    """
    Fit a saxs spectrum (I(q) vs q) to the theoretical spectrum 
    for one or several scattering populations.
    Input objfun (string) specifies objective function to use in optimization.
    Input features (dict) describes spectrum and scatterer populations.
    Input x_keys (list of strings) are the keys of variables that will be optimized.
    Every item in x_keys should be a key in the features dict.
    Input constraints (list of strings) to specify constraints.
    
    Supported objective functions: 
    (1) 'chi2': sum of difference squared across entire q range. 
    (2) 'chi2log': sum of difference of logarithm, squared, across entire q range. 
    (3) 'chi2norm': sum of difference divided by measured value, squared, aross entire q range. 
    (4) 'low_q_chi2': sum of difference squared in only the lowest half of measured q range. 
    (5) 'low_q_chi2log': sum of difference of logarithm, squared, in lowest half of measured q range. 
    (6) 'pearson': pearson correlation between measured and modeled spectra. 
    (7) 'pearson_log': pearson correlation between logarithms of measured and modeled spectra.
    (8) 'low_q_pearson': pearson correlation between measured and modeled spectra. 
    (9) 'low_q_pearson_log': pearson correlation between logarithms of measured and modeled spectra. 

    Supported constraints: 
    (1) 'fix_I0': keeps I(q=0) fixed while fitting x_keys.

    TODO: document the objective functions, etc.
    """
    pre_flag = features['precursor_flag']
    form_flag = features['form_flag']
    structure_flag = features['structure_flag']

    # trim non-flagged populations out of x_keys
    if not pre_flag:
        if 'I0_pre' in x_keys:
            x_keys.pop(x_keys.index('I0_pre')) 
        if 'r0_pre' in x_keys:
            x_keys.pop(x_keys.index('r0_pre')) 
    if not form_flag:
        if 'I0_form' in x_keys:
            x_keys.pop(x_keys.index('I0_form')) 
        if 'r0_form' in x_keys:
            x_keys.pop(x_keys.index('r0_form')) 
        if 'sigma_form' in x_keys:
            x_keys.pop(x_keys.index('sigma_form')) 
    if not structure_flag:
        if 'I0_structure' in x_keys:
            x_keys.pop(x_keys.index('I0_structure')) 
        if 'q0_structure' in x_keys:
            x_keys.pop(x_keys.index('q0_structure')) 
        if 'sigma_structure' in x_keys:
            x_keys.pop(x_keys.index('sigma_structure')) 

    c = []
    if 'fix_I0' in constraints: 
        I_keys = []
        if 'I0_pre' in x_keys:
            I_keys.append('I0_pre')
        if 'I0_form' in x_keys:
            I_keys.append('I0_form')
        if 'I0_structure' in x_keys:
            I_keys.append('I0_structure')
        if len(I_keys) == 0:
            # constraint inherently satisfied: do nothing.
            pass
        elif len(I_keys) == 1:
            # overconstrained: skip the constraint, reduce dimensionality. 
            x_keys.pop(x_keys.index(I_keys[0])) 
        else:
            # find the indices of the relevant x_keys and form a constraint function.
            iargs = []
            Icons = 0
            for ik,k in zip(range(len(x_keys)),x_keys):
                if k in I_keys:
                    iargs.append(ik)
                    Icons += features[k]
            cfun = lambda x: sum([x[i] for i in iargs]) - Icons 
            c_fixI0 = {'type':'eq','fun':cfun}
            c.append(c_fixI0)

    x_init = [] 
    x_bounds = [] 
    for k in x_keys:
        # features.keys() may not include k,
        # e.g. if the relevant population was not flagged.
        if k in features.keys():
            x_init.append(features[k])
            if k in ['r0_pre','r0_form']:
                x_bounds.append((1E-3,None))
            elif k in ['I0_pre','I0_form']:
                x_bounds.append((0,None))
            elif k in ['sigma_form']:
                x_bounds.append((0.0,0.3))

    # Only proceed if there is still work to do.
    d_opt = OrderedDict()
    x_opt = []
    if any(x_init):
        saxs_fun = lambda q,x,d: compute_saxs_with_substitutions(q,d,x_keys,x)
        I_nz = (I>0)
        n_q = len(q)
        idx_lowq = (q<0.4)
        if objfun == 'chi2':
            fit_obj = lambda x: compute_chi2( saxs_fun(q,x,features) , I )
        elif objfun == 'chi2log':
            fit_obj = lambda x: compute_chi2( np.log(saxs_fun(q[I_nz],x,features)) , np.log(I[I_nz]) )
        elif objfun == 'chi2norm':
            fit_obj = lambda x: compute_chi2( saxs_fun(q[I_nz],x,features) , I[I_nz] , weights=float(1)/I[I_nz] )
        elif objfun == 'low_q_chi2':
            fit_obj = lambda x: compute_chi2( saxs_fun(q[idx_lowq],x,features) , I[idx_lowq] )
        elif objfun == 'pearson':
            fit_obj = lambda x: -1*compute_pearson( saxs_fun(q,x,features) , I )
        elif objfun == 'low_q_pearson':
            fit_obj = lambda x: -1*compute_pearson( saxs_fun(q[idx_lowq],x,features) , I[idx_lowq] )
        elif objfun == 'low_q_chi2log':
            fit_obj = lambda x: compute_chi2( np.log(saxs_fun(q[I_nz][idx_lowq[I_nz]],x,features)) , np.log(I[I_nz][idx_lowq[I_nz]]) ) 
        elif objfun == 'pearson_log':
            fit_obj = lambda x: -1*compute_pearson( np.log(saxs_fun(q[I_nz],x,features)) , np.log(I[I_nz]) ) 
        elif objfun == 'low_q_pearson_log':
            fit_obj = lambda x: -1*compute_pearson( np.log(saxs_fun(q[I_nz][idx_lowq[I_nz]],x,features)) , np.log(I[I_nz][idx_lowq[I_nz]]) ) 
        else:
            msg = 'objective function {} not supported'.format(objfun)
            raise ValueError(msg)
        d_opt['objective_before'] = fit_obj(x_init)
        #try:
        res = scipimin(fit_obj,x_init,bounds=x_bounds,constraints=c)
        #except:
        #    import pdb; pdb.set_trace()
        x_opt = res.x
        d_opt['objective_after'] = fit_obj(x_opt)
        if d_opt['objective_after'] > d_opt['objective_before']:
            print('WARNING: optimization has increased the objective function. why?')
            print('x_init: {}'.format(x_init))
            print('obj_init: {}'.format(d_opt['objective_before']))
            print('x_opt: {}'.format(x_opt))
            print('obj_opt: {}'.format(d_opt['objective_after']))
            #import pdb; pdb.set_trace()
            x_opt = x_init
        for k,xk in zip(x_keys,x_opt):
            d_opt[k] = xk
    return d_opt    

def compute_spherical_normal_saxs(q,r0,sigma):
    """
    Given q, a mean radius r0, 
    and the fractional standard deviation of radius sigma,
    compute the saxs spectrum assuming spherical particles 
    with normal size distribution.
    The returned intensity is normalized 
    such that I(q=0) is equal to 1.
    """
    q_zero = (q == 0)
    q_nz = np.invert(q_zero) 
    I = np.zeros(q.shape)
    if sigma < 1E-9:
        x = q*r0
        V_r0 = float(4)/3*np.pi*r0**3
        I[q_nz] = V_r0**2 * (3.*(np.sin(x[q_nz])-x[q_nz]*np.cos(x[q_nz]))*x[q_nz]**-3)**2
        I_zero = V_r0**2 
    else:
        sigma_r = sigma*r0
        dr = sigma_r*0.02
        rmin = np.max([r0-5*sigma_r,dr])
        rmax = r0+5*sigma_r
        I_zero = 0
        for ri in np.arange(rmin,rmax,dr):
            xi = q*ri
            V_ri = float(4)/3*np.pi*ri**3
            # The normal-distributed density of particles with radius r_i:
            rhoi = 1./(np.sqrt(2*np.pi)*sigma_r)*np.exp(-1*(r0-ri)**2/(2*sigma_r**2))
            I_zero += V_ri**2 * rhoi*dr
            I[q_nz] += V_ri**2 * rhoi*dr*(3.*(np.sin(xi[q_nz])-xi[q_nz]*np.cos(xi[q_nz]))*xi[q_nz]**-3)**2
    if any(q_zero):
        I[q_zero] = I_zero
    I = I/I_zero 
    return I

def precursor_heuristics(q,I,I_at_0=None):
    """
    Makes an educated guess for the radius of a small scatterer
    that would produce the input q, I(q).
    Result is bounded between 0 and 10 Angstroms.
    """
    n_q = len(q)
    # optimize the pearson correlation in the upper half of the q domain
    fit_obj = lambda r: -1*compute_pearson(compute_spherical_normal_saxs(q[n_q/2:],r,0),I[n_q/2:])
    #res = scipimin(fit_ojb,[0.1],bounds=[(0,0.3)]) 
    res = scipimin(fit_obj,[5],bounds=[(0,10)])
    r_opt = res.x[0]
    return r_opt
    # Assume the first dip of the precursor form factor occurs around the max of our q range.
    # First dip = qmax*r_pre ~ 4.5
    #r0_pre = 4.5/q[-1] 

def spherical_normal_heuristics(q,I,I_at_0=None):
    """
    This algorithm was developed and 
    originally contributed by Amanda Fournier.    

    Performs some heuristic measurements on the input spectrum,
    in order to make educated guesses 
    for the parameters of a size distribution
    (mean and standard deviation of radius)
    for a population of spherical scatterers.

    TODO: Document algorithm here.
    """
    if I_at_0 is None:
        I_at_0 = fit_I0(q,I)
    m = saxs_Iq4_metrics(q,I)
    width_metric = m['pI_qwidth']/m['q_at_Iqqqq_min1']
    intensity_metric = m['I_at_Iqqqq_min1']/I_at_0
    #######
    #
    # POLYNOMIALS FITTED FOR q0+/-10%,
    # where q0 is the argmin of a parabola
    # that is fit around the first minimum of I*q**4.
    # The function spherical_normal_heuristics_setup()
    # (in this same module) can be used to generate these polynomials.
    # polynomial for qr0 focus (with x=sigma_r/r0):
    # -8.05459639763x^2 + -0.470989868709x + 4.50108683096
    p_qr0_focus = [-8.05459639763,-0.470989868709,4.50108683096]
    # polynomial for width metric (with x=sigma_r/r0):
    # 3.12889797288x^2 + -0.0645231661487x + 0.0576604958693
    p_w = [3.12889797288,-0.0645231661487,0.0576604958693]
    # polynomial for intensity metric (with x=sigma_r/r0):
    # -1.33327411025x^3 + 0.432533640102x^2 + 0.00263776123775x + -1.27646761062e-05
    p_I = [-1.33327411025,0.432533640102,0.00263776123775,-1.27646761062e-05]
    #
    #######
    # Now find the sigma_r/r0 value that gets the extracted metrics
    # as close as possible to p_I and p_w.
    width_error = lambda x: (np.polyval(p_w,x)-width_metric)**2
    intensity_error = lambda x: (np.polyval(p_I,x)-intensity_metric)**2
    # TODO: make the objective function weight all errors equally
    heuristics_error = lambda x: width_error(x) + intensity_error(x)
    res = scipimin(heuristics_error,[0.1],bounds=[(0,0.3)]) 
    if not res.success:
        self.outputs['return_code'] = 2 
        msg = str('[{}] function minimization failed during '
        + 'form factor parameter extraction.'.format(__name__))
        self.outputs['features']['message'] = msg
    sigma_over_r = res.x[0]
    qr0_focus = np.polyval(p_qr0_focus,sigma_over_r)
    # qr0_focus = x1  ==>  r0 = x1 / q1
    r0 = qr0_focus/m['q_at_Iqqqq_min1']
    #sigma_r = sigma_over_r * r0 
    return r0,sigma_over_r

def saxs_Iq4_metrics(q,I):
    """
    From an input spectrum q and I(q),
    compute several properties of the I(q)*q^4 curve.
    This was designed for spectra that are 
    dominated by a dilute spherical form factor term.
    The metrics extracted by this Operation
    were originally intended as an intermediate step
    for estimating size distribution parameters 
    for a population of dilute spherical scatterers.

    Returns a dict of metrics.
    Dict keys and meanings:
    q_at_Iqqqq_min1: q value at first minimum of I*q^4
    I_at_Iqqqq_min1: I value at first minimum of I*q^4
    Iqqqq_min1: I*q^4 value at first minimum of I*q^4
    pIqqqq_qwidth: Focal q-width of polynomial fit to I*q^4 near first minimum of I*q^4 
    pIqqqq_Iqqqqfocus: Focal point of polynomial fit to I*q^4 near first minimum of I*q^4
    pI_qvertex: q value of vertex of polynomial fit to I(q) near first minimum of I*q^4  
    pI_Ivertex: I(q) at vertex of polynomial fit to I(q) near first minimum of I*q^4
    pI_qwidth: Focal q-width of polynomial fit to I(q) near first minimum of I*q^4
    pI_Iforcus: Focal point of polynomial fit to I(q) near first minimum of I*q^4

    TODO: document the algorithm here.
    """
    d = {}
    #if not dI:
    #    # uniform weights
    #    wt = np.ones(q.shape)   
    #else:
    #    # inverse error weights, 1/dI, 
    #    # appropriate if dI represents
    #    # Gaussian uncertainty with sigma=dI
    #    wt = 1./dI
    #######
    # Heuristics step 1: Find the first local max
    # and subsequent local minimum of I*q**4 
    Iqqqq = I*q**4
    # w is the number of adjacent points to consider 
    # when examining the I*q^4 curve for local extrema.
    # A greater value of w filters out smaller extrema.
    w = 10
    idxmax1, idxmin1 = 0,0
    stop_idx = len(q)-w-1
    test_range = iter(range(w,stop_idx))
    idx = test_range.next() 
    while any([idxmax1==0,idxmin1==0]) and idx < stop_idx-1:
        if np.argmax(Iqqqq[idx-w:idx+w+1]) == w and idxmax1 == 0:
            idxmax1 = idx
        if np.argmin(Iqqqq[idx-w:idx+w+1]) == w and idxmin1 == 0 and not idxmax1 == 0:
            idxmin1 = idx
        idx = test_range.next()
    if idxmin1 == 0 or idxmax1 == 0:
        ex_msg = str('unable to find first maximum and minimum of I*q^4 '
        + 'by scanning for local extrema with a window width of {} points'.format(w))
        d['message'] = ex_msg 
        raise Exception(ex_msg)
    #######
    # Heuristics 2: Characterize I*q**4 around idxmin1, 
    # by locally fitting a standardized polynomial.
    idx_around_min1 = (q>0.9*q[idxmin1]) & (q<1.1*q[idxmin1])
    q_min1_mean = np.mean(q[idx_around_min1])
    q_min1_std = np.std(q[idx_around_min1])
    q_min1_s = (q[idx_around_min1]-q_min1_mean)/q_min1_std
    Iqqqq_min1_mean = np.mean(Iqqqq[idx_around_min1])
    Iqqqq_min1_std = np.std(Iqqqq[idx_around_min1])
    Iqqqq_min1_s = (Iqqqq[idx_around_min1]-Iqqqq_min1_mean)/Iqqqq_min1_std
    p_min1 = np.polyfit(q_min1_s,Iqqqq_min1_s,2,None,False,np.ones(len(q_min1_s)),False)
    # polynomial vertex horizontal coord is -b/2a
    qs_at_min1 = -1*p_min1[1]/(2*p_min1[0])
    d['q_at_Iqqqq_min1'] = qs_at_min1*q_min1_std+q_min1_mean
    # polynomial vertex vertical coord is poly(-b/2a)
    Iqqqqs_at_min1 = np.polyval(p_min1,qs_at_min1)
    d['Iqqqq_min1'] = Iqqqqs_at_min1*Iqqqq_min1_std+Iqqqq_min1_mean
    d['I_at_Iqqqq_min1'] = d['Iqqqq_min1']*float(1)/(d['q_at_Iqqqq_min1']**4)
    # The focal width of the parabola is 1/a 
    p_min1_fwidth = float(1)/p_min1[0] 
    d['pIqqqq_qwidth'] = p_min1_fwidth*q_min1_std
    # The focal point is at -b/2a,poly(-b/2a)+1/(4a)
    p_min1_fpoint = Iqqqqs_at_min1+float(1)/(4*p_min1[0])
    d['pIqqqq_Iqqqqfocus'] = p_min1_fpoint*Iqqqq_min1_std+Iqqqq_min1_mean
    #######
    # Heuristics 2b: Characterize I(q) near min1 of I*q^4.
    I_min1_mean = np.mean(I[idx_around_min1])
    I_min1_std = np.std(I[idx_around_min1])
    I_min1_s = (I[idx_around_min1]-I_min1_mean)/I_min1_std
    pI_min1 = np.polyfit(q_min1_s,I_min1_s,2,None,False,np.ones(len(q_min1_s)),False)
    # polynomial vertex horizontal coord is -b/2a
    qs_vertex = -1*pI_min1[1]/(2*pI_min1[0])
    d['pI_qvertex'] = qs_vertex*q_min1_std+q_min1_mean
    # polynomial vertex vertical coord is poly(-b/2a)
    Is_vertex = np.polyval(pI_min1,qs_vertex)
    d['pI_Ivertex'] = Is_vertex*I_min1_std+I_min1_mean
    # The focal width of the parabola is 1/a 
    pI_fwidth = float(1)/pI_min1[0]
    d['pI_qwidth'] = pI_fwidth*q_min1_std
    # The focal point is at -b/2a,poly(-b/2a)+1/(4a)
    pI_fpoint = Is_vertex+float(1)/(4*pI_min1[0])
    d['pI_Ifocus'] = pI_fpoint*I_min1_std+I_min1_mean
    #######
    return d

def compute_saxs_with_substitutions(q,d,x_keys,x_vals):
    d_sub = d.copy()
    for k,v in zip(x_keys,x_vals):
        d_sub[k] = v
    return compute_saxs(q,d_sub)


def compute_chi2(y1,y2,weights=None):
    """
    Compute the sum of the difference squared between input arrays y1 and y2.
    """
    if weights is None:
        return np.sum( (y1 - y2)**2 )
    else:
        weights = weights / np.sum(weights)
        return np.sum( (y1 - y2)**2*weights )

def compute_Rsquared(y1,y2):
    """
    Compute the coefficient of determination between input arrays y1 and y2.
    """
    sum_var = np.sum( (y1-np.mean(y1))**2 )
    sum_res = np.sum( (y1-y2)**2 ) 
    return float(1)-float(sum_res)/sum_var

def compute_pearson(y1,y2):
    """
    Compute the Pearson correlation coefficient between input arrays y1 and y2.
    """
    y1mean = np.mean(y1)
    y2mean = np.mean(y2)
    y1std = np.std(y1)
    y2std = np.std(y2)
    return np.sum((y1-y1mean)*(y2-y2mean))/(np.sqrt(np.sum((y1-y1mean)**2))*np.sqrt(np.sum((y2-y2mean)**2)))

def fit_I0(q,I,order=4):
    """
    Find an estimate for I(q=0) by polynomial fitting.
    All of the input q, I(q) values are used in the fitting.
    """
    I_mean = np.mean(I)
    I_std = np.std(I)
    q_mean = np.mean(q)
    q_std = np.std(q)
    I_s = (I-I_mean)/I_std
    q_s = (q-q_mean)/q_std
    p = fit_with_slope_constraint(q_s,I_s,-1*q_mean/q_std,0,order) 
    I_at_0 = np.polyval(p,-1*q_mean/q_std)*I_std+I_mean
    return I_at_0

def fit_with_slope_constraint(q,I,q_cons,dIdq_cons,order,weights=None):
    """
    Perform a polynomial fitting 
    of the low-q region of the spectrum
    with dI/dq(q=0) constrained to be zero.
    This is performed by forming a Lagrangian 
    from a quadratic cost function 
    and the Lagrange-multiplied constraint function.
    
    TODO: Explicitly document cost function, constraints, Lagrangian.

    Inputs q and I are not standardized in this function,
    so they should be standardized beforehand 
    if standardized fitting is desired.
    At the provided constraint point, q_cons, 
    the returned polynomial will have slope dIdq_cons.

    Because of the form of the Lagrangian,
    this constraint cannot be placed at exactly zero.
    This would result in indefinite matrix elements.
    """
    Ap = np.zeros( (order+1,order+1),dtype=float )
    b = np.zeros(order+1,dtype=float)
    # TODO: vectorize the construction of Ap
    for i in range(0,order):
        for j in range(0,order):
            Ap[i,j] = np.sum( q**j * q**i )
        Ap[i,order] = -1*i*q_cons**(i-1)
    for j in range(0,order):
        Ap[order,j] = j*q_cons**(j-1)
        b[j] = np.sum(I*q**j)
    b[order] = dIdq_cons
    p_fit = np.linalg.solve(Ap,b) 
    p_fit = p_fit[:-1]  # throw away Lagrange multiplier term 
    p_fit = p_fit[::-1] # reverse coefs to get np.polyfit format
    #from matplotlib import pyplot as plt
    #plt.figure(3)
    #plt.plot(q,I)
    #plt.plot(q,np.polyval(p_fit,q))
    #plt.plot(np.arange(q_cons,q[-1],q[-1]/100),np.polyval(p_fit,np.arange(q_cons,q[-1],q[-1]/100)))
    #plt.plot(q_cons,np.polyval(p_fit,q_cons),'ro')
    #plt.show()
    return p_fit
 
def spherical_normal_heuristics_setup():
    sigma_over_r = []
    width_metric = []
    intensity_metric = []
    qr0_focus = []
    # TODO: generate heuristics on a 1-d grid of sigma/r
    # instead of the 2-d grid being used here now.
    r0_vals = np.arange(10,41,10,dtype=float)              #Angstrom
    for ir,r0 in zip(range(len(r0_vals)),r0_vals):
        q = np.arange(0.001/r0,float(10)/r0,0.001/r0)       #1/Angstrom
        sigma_r_vals = np.arange(0*r0,0.21*r0,0.01*r0)      #Angstrom
        for isig,sigma_r in zip(range(len(sigma_r_vals)),sigma_r_vals):
            I = compute_spherical_normal_saxs(q,r0,sigma_r/r0) 
            I_at_0 = compute_spherical_normal_saxs(0,r0,sigma_r/r0) 
            d = saxs_spherical_normal_heuristics(q,I)
            sigma_over_r.append(float(sigma_r)/r0)
            qr0_focus.append(d['q_at_Iqqqq_min1']*r0)
            width_metric.append(d['pI_qwidth']/d['q_at_Iqqqq_min1'])
            intensity_metric.append(d['I_at_Iqqqq_min1']/I_at_0)
    # TODO: standardize before fitting, then revert after
    p_qr0_focus = np.polyfit(sigma_over_r,qr0_focus,2,None,False,None,False)
    p_w = np.polyfit(sigma_over_r,width_metric,2,None,False,None,False)
    p_I = np.polyfit(sigma_over_r,intensity_metric,3,None,False,None,False)
    print('polynomial for qr0 focus (with x=sigma_r/r0): {}x^2 + {}x + {}'.format(p_qr0_focus[0],p_qr0_focus[1],p_qr0_focus[2]))
    print('polynomial for width metric (with x=sigma_r/r0): {}x^2 + {}x + {}'.format(p_w[0],p_w[1],p_w[2]))
    print('polynomial for intensity metric (with x=sigma_r/r0): {}x^3 + {}x^2 + {}x + {}'.format(p_I[0],p_I[1],p_I[2],p_I[3]))
    plot = True
    if plot: 
        from matplotlib import pyplot as plt
        plt.figure(1)
        plt.scatter(sigma_over_r,width_metric)
        plt.plot(sigma_over_r,np.polyval(p_w,sigma_over_r))
        plt.figure(2)
        plt.scatter(sigma_over_r,intensity_metric)
        plt.plot(sigma_over_r,np.polyval(p_I,sigma_over_r))
        plt.figure(3)
        plt.scatter(sigma_over_r,qr0_focus)
        plt.plot(sigma_over_r,np.polyval(p_qr0_focus,sigma_over_r))
        plt.figure(4)
        plt.scatter(width_metric,intensity_metric) 
        plt.show()

