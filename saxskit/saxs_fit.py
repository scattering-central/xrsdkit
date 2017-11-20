"""Modules for fitting a measured SAXS spectrum to a scattering equation.

SAXS spectra are described by an n-by-2 array
of scattering vector magnitudes (1/Angstrom)
and intensities (arbitrary units).
Computing the theoretical SAXS spectrum requires a dictionary of populations
(groups of scatterers where each group shares the same scattering model)
and a dictionary of parameters
(parameters will be expected for all of the flagged populations).

The supported populations and parameters are:

    - Common parameters for any and all populations:
      
      - 'I0_floor': magnitude of floor term, flat for all q. 

    - 'guinier_porod': number of Guinier-Porod scatterers 

      - 'G_gp': Guinier prefactor for Guinier-Porod scatterers 
      - 'rg_gp': radius of gyration for Guinier-Porod scatterers 
      - 'D_gp': Porod exponent for Guinier-Porod scatterers 

    - 'spherical_normal': number of populations of spheres 
        with normal (Gaussian) size distribution 

      - 'I0_sphere': spherical form factor scattering intensity scaling factor
      - 'r0_sphere': mean sphere size (Angstrom) 
      - 'sigma_sphere': fractional standard deviation of sphere size 

    - 'diffraction_peaks': number of Psuedo-Voigt 
        diffraction peaks (not yet supported) 

    - 'unidentified': if not zero, the scattering spectrum is unfamiliar. 
        This populations causes all populations and parameters to be ignored.

"""
from __future__ import print_function
import warnings
from collections import OrderedDict
from functools import partial
import copy

import numpy as np
from scipy.optimize import minimize as scipimin

# population types supported:
population_types = [\
    'unidentified',\
    'guinier_porod',\
    'spherical_normal',\
    'diffraction_peaks'])

# parameter limits for fit_spectrum() and MC_anneal_fit():
param_limits = OrderedDict(
    I0_floor = (0.,10.),
    G_gp = (0.,1.E4),
    rg_gp = (1E-6,1.E3),
    D_gp = (0.,4.),
    I0_sphere = (0.,1.E4),
    r0_sphere = (1E-6,1.E3),
    sigma_sphere = (0.,0.5))

def compute_saxs(q,populations,params):
    """Compute a SAXS intensity spectrum given some parameters.

    TODO: Document the equation.

    Parameters
    ----------
    q : array
        Array of q values at which saxs intensity should be computed.
    populations : dict
        Each entry is an integer representing the number 
        of distinct populations of various types of scatterer. 
    params : dict
        Scattering equation parameters. 
        Each entry in the dict may be a float or a list of floats,
        depending on whether there are one or more of the corresponding
        scatterer populations.

    Returns
    ------- 
    I : array
        Array of scattering intensities for each of the input q values
    """
    u_flag = bool(populations['unidentified'])
    pks_flag = bool(populations['diffraction_peaks'])
    I = np.zeros(len(q))
    if not u_flag and not pks_flag:
        n_gp = populations['guinier_porod']
        n_sph = populations['spherical_normal']

        I0_floor = params['I0_floor'] 
        I = I0_floor*np.ones(len(q))

        if n_gp:
            rg_gp = params['rg_gp']
            G_gp = params['G_gp']
            D_gp = params['D_gp']
            if not isinstance(rg_gp,list): rg_gp = [rg_gp]
            if not isinstance(G_gp,list): G_gp = [G_gp]
            if not isinstance(D_gp,list): D_gp = [D_gp]
            for igp in range(n_gp):
                I_gp = guinier_porod(q,rg_gp[igp],D_gp[igp],G_gp[igp])
                I += I_gp

        if n_sph:
            I0_sph = params['I0_sphere']
            r0_sph = params['r0_sphere']
            sigma_sph = params['sigma_sphere']
            if not isinstance(I0_sph,list): I0_sph = [I0_sph]
            if not isinstance(r0_sph,list): r0_sph = [r0_sph]
            if not isinstance(sigma_sph,list): sigma_sph = [sigma_sph]
            for isph in range(n_sph):
                I_sph = spherical_normal_saxs(q,r0_sph[isph],sigma_sph[isph])
                I += I0_sph[isph]*I_sph

    return I

def spherical_normal_saxs(q,r0,sigma):
    """Compute SAXS intensity of a normally-distributed sphere population.

    The returned intensity is normalized 
    such that I(q=0) is equal to 1.
    The current version samples the distribution 
    from r0*(1-5*sigma) to r0*(1+5*sigma) 
    in steps of 0.02*sigma*r0.
    Originally contributed by Amanda Fournier.
    TODO: test distribution sampling, speed up if possible.

    Parameters
    ----------
    q : array
        array of scattering vector magnitudes
    r0 : float
        mean radius of the sphere population
    sigma : float
        fractional standard deviation of the sphere population radii

    Returns
    -------
    I : array
        Array of scattering intensities for each of the input q values
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

def guinier_porod(q,r_g,porod_exponent,guinier_factor):
    """Compute the Guinier-Porod small-angle scattering intensity.
    
    Parameters
    ----------
    q : array
        array of q values
    r_g : float
        radius of gyration
    porod_exponent : float
        high-q Porod's law exponent
    guinier_factor : float
        low-q Guinier prefactor (equal to intensity at q=0)

    Returns
    -------
    I : array
        Array of scattering intensities for each of the input q values

    Reference
    ---------
    B. Hammouda, J. Appl. Cryst. (2010). 43, 716-719.
    """
    # q-domain boundary q_splice:
    q_splice = 1./r_g * np.sqrt(3./2*porod_exponent)
    idx_guinier = (q <= q_splice)
    idx_porod = (q > q_splice)
    # porod prefactor D:
    porod_factor = guinier_factor*np.exp(-1./2*porod_exponent)\
                    * (3./2*porod_exponent)**(1./2*porod_exponent)\
                    * 1./(r_g**porod_exponent)
    I = np.zeros(q.shape)
    # Guinier equation:
    if any(idx_guinier):
        I[idx_guinier] = guinier_factor * np.exp(-1./3*q[idx_guinier]**2*r_g**2)
    # Porod equation:
    if any(idx_porod):
        I[idx_porod] = porod_factor * 1./(q[idx_porod]**porod_exponent)
    return I

def profile_spectrum(q_I):
    """Numerical profiling of a SAXS spectrum.

    Profile a saxs spectrum (n-by-2 array q_I) 
    by taking several fast numerical metrics 
    from the measured data.
    The metrics should be consistent for spectra
    with different intensity scaling 
    or different q domains.   

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
    params : dict
        dictionary of scattering equation parameters,
        for input to compute_saxs() 
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
    #I_sum = np.sum(I)
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

    ### fluctuation analysis
    # array of the difference between neighboring points:
    nn_diff = logI_nz[1:]-logI_nz[:-1]
    # keep indices where the sign of this difference changes.
    # also keep first index
    nn_diff_prod = nn_diff[1:]*nn_diff[:-1]
    idx_keep = np.hstack((np.array([True]),nn_diff_prod<0))
    fluc = np.sum(np.abs(nn_diff[idx_keep]))
    logI_fluctuation = fluc/logI_range

    # TODO: add some pearson metrics.
    # TODO: add qmin, qmax as metrics

    params = OrderedDict()
    params['q_Imax'] = q_Imax
    params['Imax_over_Imean'] = Imax_over_Imean
    params['Imax_sharpness'] = Imax_sharpness
    params['logI_fluctuation'] = logI_fluctuation
    params['logI_max_over_std'] = logI_max_over_std
    return params 

def parameterize_spectrum(q_I,populations):
    """Determine scattering equation parameters for a given spectrum.

    Parameters
    ----------
    q_I : array
        n-by-2 array of q (scattering vector in 1/A) and I (intensity)
    populations : dict
        dictionary of populations, similar to input of compute_saxs()

    Returns
    -------
    params : dict
        dict of scattering equation parameters
        corresponding to `populations`,
        similar to the input of compute_saxs().

    See Also
    --------
    compute_saxs : computes saxs intensity from `populations` and `params`
    """

    d = OrderedDict()
    if bool(populations['unidentified']):
        return d 
    if bool(flags['diffraction_peaks']):
        warnings.warn('{}: diffraction peak parameterization '\
        'is not supported yet: aborting parameterize_spectrum()'\
        .format(__name__))
        return d
    q = q_I[:,0]
    n_q = len(q)
    I = q_I[:,1]
    I_nz = np.invert((I<=0))
    n_gp = populations['guinier_porod']
    n_sph = populations['spherical_normal']

    # PART 1: Get a number for I(q=0)
    # NOTE: This is not designed to handle peaks.
    # TODO: Fit a sensible low-q region
    # when q_Imax is not at low q
    # (i.e. when there are high-q peaks).
    q_Imax = q[np.argmax(I)]
    idx_fit = (q>=q_Imax)&(q<2.*q_Imax)
    if q_Imax > q[int(n_q/2)]:
        idx_fit = (q<q[int(n_q/2)])
    q_fit = q[idx_fit]
    I_fit = I[idx_fit]
    qs_fit = (q_fit-np.mean(q_fit))/np.std(q_fit)
    # attempt a high-order constrained fit
    I_at_0,pI = fit_I0(q_fit,I_fit,4)
    # check the SNR
    I_sig = np.polyval(pI,qs_fit)*np.std(I_fit)+np.mean(I_fit)
    I_bg = I_fit-I_sig
    snr = np.mean(I_fit)/np.std(I_bg)
    if snr < 100:
        # if fit is noisy, try constrained fit on third order
        I_at_0,pI = fit_I0(q_fit,I_fit,3)

    # PART 2: Estimate parameters for flagged populations
    #TODO: add parameters for diffraction peaks
    #TODO: add parameters for non-spherical form factors
    #if n_sph:
    #    #r0_sphere, sigma_sphere = spherical_normal_heuristics(
    #    #    np.array(zip(q,I)),I_at_0)
    #    #d['r0_sphere'] = r0_sphere
    #    #d['sigma_sphere'] = sigma_sphere
    #if n_gp:
    #    rg_pre, G_pre = precursor_heuristics(
    #        np.array(zip(q,I)))
    #    I_pre = guinier_porod(q,rg_pre,4,G_pre)
    #    d['rg_precursor'] = rg_pre 
    #    d['G_precursor'] = G_pre 

    #if n_gp+n_sph > 1:
    #    Ifunc = lambda x: x[0]*np.ones(n_q) \
    #        + guinier_porod(q,rg_pre,4,x[1]) \
    #        + x[2]*spherical_normal_saxs(q,r0_sphere,sigma_sphere) 
    #    I_error = lambda x: np.sum( (np.log(Ifunc(x)[I_nz]) - np.log(I[I_nz]))**2 )
    #    # NOTE: this fit is constrained to hold I(q=0) constant
    #    res = scipimin(I_error,[0.0,G_pre,I_at_0],
    #        bounds=[(0.0,None),(1E-6,None),(1E-6,None)],
    #        constraints=[{'type':'eq','fun':lambda x:np.sum(x)-I_at_0}])
    #    d['I0_floor'] = res.x[0]
    #    d['G_precursor'] = res.x[1]
    #    d['I0_sphere'] = res.x[2]
    #elif f_pre:
    #    d['I0_floor'] = 0
    #elif f_form:
    #    I0_floor = np.mean(I[int(n_q*0.9):])
    #    d['I0_floor'] = I0_floor 
    #    d['I0_sphere'] = I_at_0 - I0_floor

    return d

class SaxsFitter(object):

    def __init__(self,populations,params,q_I,q_range=None):
        self.populations = populations
        self.q = q_I[:,0]
        self.I = q_I[:,1]
        # taking the log may throw a warning- it's ok.
        # any resulting zeros and nans will not be used.
        self.logI = np.log(self.I)
        self.q_range = q_range
        # if we have relatively low intensity at low q,
        # the objective may suffer from experimental errors
        # dominating the low-q region of the spectrum.
        #idx_lowq = (np.arange(n_q)<n_q*1./4)
        #idx_hiq = (np.arange(n_q)>n_q*3./4) 
        #if not np.mean(I[idx_lowq]) > 10.*np.mean(I[idx_hiq]):   
        #    self.idx_fit = ((I>0)&np.invert(idx_lowq))
        if self.q_range is not None:
            self.idx_fit = ((self.I>0)
                &(self.q>self.q_range[0])
                &(self.q<self.q_range[1]))
        else:
            self.idx_fit = (I>0)

    def evaluate(self,test_params):
        param_dict = self.pack_params(test_params)
        I_comp = compute_saxs(self.q,self.populations,param_dict)
        chi2log_total = 0
        for pkey in self.populations.keys():
            if pkey == 'guinier_porod':
                # weight the high-q region 
                chi2log = compute_chi2(I_comp[self.idx_fit],self.logI[idx_fit],self.q)
            if pkey == 'spherical_normal':
                # weight the low-q region
                chi2log = compute_chi2(I_comp[self.idx_fit],self.logI[idx_fit],self.q[::-1])
            #if pkey == 'diffraction_peaks':
            #    chi2log = 0
            #    # window the region surrounding this peak
            #    # idx_pk = ...
            #    for ipk in range(int(self.populations['diffraction_peaks'])):
            #        # get q range of the peak
            #        # evaluate objective
            #        chi2log += compute_chi2(I_comp[(idx_pk&idx_fit)],self.logI[(idx_pk&idx_fit)])
            chi2log_total += chi2log
        return chi2log 

    def pack_params(self,param_array):
        param_dict = OrderedDict()
        param_dict['I0_floor'] = param_array[0]
        idx = 1
        for pkey in self.populations.keys():
            param_dict[pkey] = []
            for ipop in range(int(self.populations[pkey])):
                param_dict['pkey'].append(param_array[idx])
                idx += 1
        return param_dict

    def unpack_params(self,param_dict):
        param_list = []
        if 'I0_floor' in param_dict:
            param_list.append(param_dict['I0_floor'])
        for pkey in self.populations.keys():
            if pkey in param_dict:
                if not isinstance(param_dict[pkey],list):
                    param_list.append(param_dict[pkey])
                else:
                    param_list.extend(param_dict[pkey])
        return np.array(param_list)

def saxs_chi2log(populations,params,q_I):
    """Return a function that evaluates saxs intensity fit error.

    Parameters
    ----------
    populations : dict
        dict of the number of distinct populations 
        for various scatterer types, 
        similar to input of compute_saxs()
    params : dict
        dict of scattering equation parameters,
        similar to the input of compute_saxs()
    q_I : array
        n-by-2 array of scattering vectors q (1/Angstrom)
        and corresponding intensities (arbitrary units)
        against which the computed intensity will be compared

    Returns
    -------
    fit_obj : function
        an anonymous function to be used 
        as a saxs intensity fitting objective

    See Also
    --------
    compute_saxs : computes the saxs intensity given `populations` and `params`
    """
    saxs_fitter = SaxsFitter(populations,params,q_I)
    return saxs_fitter.evaluate

    #n_q = q_I.shape[0]
    #n_gp = populations['guinier_porod']
    #n_sph = populations['spherical_normal']
    #n_pks = populations['diffraction_peaks']
    #saxs_fun = lambda q,x: compute_saxs_with_substitutions(q,populations,params,x)
    #fit_objfuns = []
    # for each population, set up an objective function,
    # such that the parameters for that population
    # are fit to a thoughtfully-chosen q-region 
    #if n_gp:

        # WORK STOPPED HERE Fri 11/17.
        #----------------------------------------------------
        # TODO:
        # Think of a way to separate the input spaces
        # for the nominally separate objective functions,
        # probably by passing a list of (keys,index) tuples 
        # to compute_saxs_with_substitutions
        #q_fit = q_I[idx_fit,0]
        #logI_fit = np.log(q_I[idx_fit,1])
        #logImean_fit = np.mean(logI_fit)
        #logIstd_fit = np.std(logI_fit)
        #logIs_fit = (logI_fit-logImean_fit)/logIstd_fit

        #fit_objfun = lambda x: compute_chi2(
        #(np.log(saxs_fun(q_fit,x))-logImean_fit)/logIstd_fit , logIs_fit)
        #fit_objfuns.append(fit_objfun)

    #fit_obj = lambda x: sum([objf(x) for objf in fit_objfuns])
    #return fit_obj


    # choose a fitting region:
    # if intensities are overall relatively low,
    # experimental errors in the low-q region can 
    # dominate the objective function.
    # in such cases, the low-q region 
    # should be weighed out of the fit. 
    # TODO: consider whether this special case is justified,
    # or come up with a better way to filter high-error low-q points
        #rg_gp = params['rg_gp']
        #G_gp = params['G_gp']
        #D_gp = params['D_gp']
        #if not isinstance(rg_gp,list): rg_gp = [rg_gp]
        #if not isinstance(G_gp,list): G_gp = [G_gp]
        #if not isinstance(D_gp,list): D_gp = [D_gp]
        #for igp in range(n_gp):
        #    q_splice = 1./rg_gp[igp] * np.sqrt(3./2*D_gp[igp])
        #    #I_gp = guinier_porod(q,rg_gp[igp],D_gp[igp],G_gp[igp])

def MC_anneal_fit(q_I,flags,params,stepsize,nsteps,T):
    """Perform a Metropolis-Hastings iteration for spectrum fit refinement.

    Parameters
    ----------
    q_I : array
        n-by-2 array of intensity (arb) versus scattering vector (1/Angstrom)
    flags : dict
        Dict of scattering population flags
    params : dict
        Dict of scattering equation parameters
    stepsize : float
        fractional step size for random walk 
    nsteps : int
        Number of iterations to perform
    T : float
        Temperature employed in Metropolis acceptance decisions.

    Returns
    -------
    p_best : dict
        Dict of best-fit parameters
    p_fin : dict
        Dict of parameters obtained at the final iteration
    rpt : dict
        Report of objective function and Metropolis-Hastings results
    """
    f_bd = flags['bad_data']
    f_pre = flags['precursor_scattering']
    f_form = flags['form_factor_scattering']
    f_pks = flags['diffraction_peaks']
    if f_bd or f_pks: return OrderedDict(),OrderedDict(),OrderedDict()
    q = q_I[:,0]
    n_q = len(q)
    I = q_I[:,1]

    fit_obj = saxs_chi2log(flags,params,q_I) 
    p_init = copy.deepcopy(params) 
    p_current = copy.deepcopy(params) 
    p_best = copy.deepcopy(params) 
    obj_current = fit_obj(p_current.values())
    obj_best = obj_current 
    nrej = 0.

    rpt = OrderedDict()
    all_trials = range(nsteps)
    for imc in all_trials:
        # get trial params 
        p_new = copy.deepcopy(p_current)
        for k,v in p_new.items():
            param_range = param_limits[k][1] - param_limits[k][0]
            if v == 0.:
                p_trial = np.random.rand()*stepsize*param_range 
            else:
                p_trial = v*(1+2*(np.random.rand()-0.5)*stepsize)
            if p_trial < param_limits[k][0]:
                p_trial = param_limits[k][0] 
            if p_trial > param_limits[k][1]:
                p_trial = param_limits[k][1] 
            p_new[k] = p_trial 
        # evaluate objective, determine acceptance
        obj_new = fit_obj(p_new.values())
        if obj_new < obj_current:
            accept = True
            if obj_new < obj_best:
                p_best = p_new
                obj_best = obj_new
        elif T == 0:
            accept = False
        else:
            accept = np.exp(-1.*(obj_new-obj_current)/T) > np.random.rand()
        # act on acceptance decision
        if accept:
            p_current = p_new
            obj_current = obj_new
        else:
            nrej += 1
            p_new = p_current

    rpt['reject_ratio'] = float(nrej)/nsteps
    rpt['objective_init'] = fit_obj(p_init.values())
    rpt['objective_best'] = fit_obj(p_best.values())
    rpt['objective_final'] = fit_obj(p_current.values())

    return p_best,p_current,rpt

def fit_spectrum(q_I,flags,params,fixed_params,objective='chi2log'):
    """Fit a SAXS spectrum, given population flags and parameter guesses.

    Parameters
    ----------
    q_I : array
        n-by-2 array of scattering vector q (1/Angstrom) and intensity.
    flags : dict
        dictionary of flags indicating various scatterer populations.
    params : dict
        scattering equation parameters to use as intial condition for fitting.
    fixed_params : list
        list of keys (strings) indicating which entries in `params` 
        should be held constant during the optimization. 
    objective : str
        choice of objective function for the fitting. supported objectives:
        - 'chi2log': sum of difference of logarithm, squared, across entire q range. 
        - 'chi2log_fixI0': like chi2log, but with I(q=0) constrained. 

    Returns
    -------
    p_opt : dict
        Dict of scattering equation parameters 
        optimized to fit `q_I` under `objective`.
    rpt : dict
        Dict reporting objective function and its values
        at the initial and final points 
    """

    f_bd = flags['bad_data']
    f_pre = flags['precursor_scattering']
    f_form = flags['form_factor_scattering']
    f_pks = flags['diffraction_peaks']
    if f_bd or f_pks: return OrderedDict(),OrderedDict()
    q = q_I[:,0]
    n_q = len(q)
    I = q_I[:,1]

    # index the params into an array 
    params = OrderedDict(params)
    x_init = np.zeros(len(params))
    x_bounds = [] 
    p_idx = OrderedDict() 
    I_idx = OrderedDict() 
    for i,k in zip(range(len(params)),params.keys()):
        p_idx[k] = i 
        x_init[i] = params[k]
        if k in ['G_precursor','I0_floor','I0_sphere']:
            I_idx[k] = i
        x_bounds.append(param_limits[k])
        #if k in ['rg_precursor','r0_sphere']:
        #    x_bounds.append((1E-3,None))
        #elif k in ['G_precursor','I0_sphere','I0_floor']:
        #    x_bounds.append((0.0,None))
        #elif k in ['sigma_sphere']:
        #    x_bounds.append((0.0,1.0))
        #else:
        #    x_bounds.append((None,None))
   
    # --- constraints --- 
    c = []
    if objective in ['chi2log_fixI0']:
        if len(I_idx) > 0:
            # Set up a constraint to keep I(q=0) fixed
            I0_init = np.sum([x_init[i] for i in I_idx.values()])
            cfun = lambda x: np.sum([x[I_idx[k]] for k in I_idx.keys()]) - I0_init
            c.append({'type':'eq','fun':cfun})
    for fixk in fixed_params:
        cfun = lambda x: x[p_idx[fixk]] - params[fixk]
        c.append({'type':'eq','fun':cfun})
    # --- end constraints ---

    p_opt = copy.deepcopy(params) 
    fit_obj = saxs_chi2log(flags,params,q_I)
    rpt = OrderedDict()
    res = scipimin(fit_obj,x_init,
        bounds=x_bounds,
        options={'ftol':1E-3},
        constraints=c)
    for k,xk in zip(params.keys(),res.x):
        p_opt[k] = xk
    #rpt['fixed_params'] = fixed_params
    #rpt['objective'] = objective 
    rpt['objective_value'] = fit_obj(res.x)
    I_opt = compute_saxs(q,flags,p_opt) 
    I_bg = I - I_opt
    snr = np.mean(I_opt)/np.std(I_bg) 
    rpt['fit_snr'] = snr
    return p_opt,rpt

    #I_opt = compute_saxs(q,flags,p_opt) 
    #I_guess = compute_saxs(q,flags,params) 
    #from matplotlib import pyplot as plt
    #plt.figure(2)
    #plt.plot(q,I)
    #plt.plot(q,I_guess,'r')
    #plt.plot(q,I_opt,'g')
    #plt.figure(12)
    #plt.semilogy(q,I)
    #plt.semilogy(q,I_guess,'r')
    #plt.semilogy(q,I_opt,'g')
    #print('flags: \n{}'.format(flags))
    #plt.show()

def precursor_heuristics(q_I):
    """Guess radius of gyration and Guinier prefactor of scatterers.

    Parameters
    ----------
    q_I : array
        n-by-2 array of q (scattering vector magnitude) 
        and I (intensity at q)

    Returns
    -------
    rg_pre : float
        estimated radius of gyration 
    G_pre : float
        estimated Guinier factor
    """
    n_q = len(q_I[:,0])
    ## use the higher-q regions 
    highq_I = q_I[int(n_q*1./2):,:] 
    #highq_I = q_I[int(n_q*3./4):,:] 
    fit_obj = lambda x: fit_guinier_porod(highq_I,x[0],4,x[1])
    idx_nz = highq_I[:,1]>0
    res = scipimin(fit_obj,[1,1],bounds=[(1E-3,10),(1E-3,None)])
    rg_opt, G_opt = res.x
    I_pre = guinier_porod(q_I[:,0],rg_opt,4,G_opt)
    return rg_opt, G_opt

def fit_guinier_porod(q_I,rg,porod_exp,G):
    Igp = guinier_porod(q_I[:,0],rg,porod_exp,G)
    return np.sum( (q_I[:,1]-Igp)**2 )

def spherical_normal_heuristics(q_I,I_at_0):
    """Guess mean and std of radii for spherical scatterers.

    This algorithm was developed and 
    originally contributed by Amanda Fournier.    

    Performs some heuristic measurements on the input spectrum,
    in order to make educated guesses 
    for the parameters of a size distribution
    (mean and standard deviation of radius)
    for a population of spherical scatterers.

    TODO: Document algorithm here.
    """
    m = saxs_Iq4_metrics(q_I)

    width_metric = m['pI_qwidth']/m['q_at_Iqqqq_min1']
    intensity_metric = m['I_at_Iqqqq_min1']/I_at_0
    #######
    #
    # The function spherical_normal_heuristics_setup()
    # (in this same module) should be used to regenerate these polynomials
    # if any part of saxs_Iq4_metrics() is changed.
    # polynomial coefs for qr0 focus: 
    p_f = [16.86239254,8.85709143,-11.10439599,-0.26735688,4.49884714]
    # polynomial coefs for width metric: 
    p_w = [12.42148677,-16.85723287,7.43401497,-0.38234993,0.06203096]
    # polynomial coefs for intensity metric: 
    p_I = [1.19822603,-1.20386273,2.88652860e-01,1.78677430e-02,-2.67888841e-04]
    #
    #######
    # Find the sigma_r/r0 value that gets the extracted metrics
    # as close as possible to p_I and p_w.
    width_error = lambda x: (np.polyval(p_w,x)-width_metric)**2
    intensity_error = lambda x: (np.polyval(p_I,x)-intensity_metric)**2
    # TODO: make the objective function weight all errors equally
    heuristics_error = lambda x: width_error(x) + intensity_error(x)
    res = scipimin(heuristics_error,[0.1],bounds=[(0,0.45)]) 
    sigma_over_r = res.x[0]
    qr0_focus = np.polyval(p_f,sigma_over_r)
    # qr0_focus = x1  ==>  r0 = x1 / q1
    r0 = qr0_focus/m['q_at_Iqqqq_min1']
    return r0,sigma_over_r

def saxs_Iq4_metrics(q_I):
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
    q = q_I[:,0]
    I = q_I[:,1]
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
        raise RuntimeError(ex_msg)
    #######
    # Heuristics 2: Characterize I*q**4 around idxmin1, 
    # by locally fitting a standardized polynomial.


    idx_around_min1 = (q>0.9*q[idxmin1]) & (q<1.1*q[idxmin1])
    # keep only the lower-q side, to encourage upward curvature
    #idx_around_min1 = (q>0.8*q[idxmin1]) & (q<q[idxmin1])


    q_min1_mean = np.mean(q[idx_around_min1])
    q_min1_std = np.std(q[idx_around_min1])
    q_min1_s = (q[idx_around_min1]-q_min1_mean)/q_min1_std
    Iqqqq_min1_mean = np.mean(Iqqqq[idx_around_min1])
    Iqqqq_min1_std = np.std(Iqqqq[idx_around_min1])
    Iqqqq_min1_s = (Iqqqq[idx_around_min1]-Iqqqq_min1_mean)/Iqqqq_min1_std
    #Iqqqq_min1_quad = lambda x: np.sum((x[0]*q_min1_s**2 + x[1]*q_min1_s + x[2] - Iqqqq_min1_s)**2)
    #res = scipimin(Iqqqq_min1_quad,[1E-3,0,0],bounds=[(0,None),(None,None),(None,None)])
    #p_min1 = res.x
    p_min1 = np.polyfit(q_min1_s,Iqqqq_min1_s,2,None,False,np.ones(len(q_min1_s)),False)
    # polynomial vertex horizontal coord is -b/2a
    qs_at_min1 = -1*p_min1[1]/(2*p_min1[0])
    d['q_at_Iqqqq_min1'] = qs_at_min1*q_min1_std+q_min1_mean
    # polynomial vertex vertical coord is poly(-b/2a)
    Iqqqqs_at_min1 = np.polyval(p_min1,qs_at_min1)
    d['Iqqqq_min1'] = Iqqqqs_at_min1*Iqqqq_min1_std+Iqqqq_min1_mean
    d['I_at_Iqqqq_min1'] = d['Iqqqq_min1']*float(1)/(d['q_at_Iqqqq_min1']**4)
    # The focal width of the parabola is 1/a 
    p_min1_fwidth = abs(1./p_min1[0])
    d['pIqqqq_qwidth'] = p_min1_fwidth*q_min1_std
    # The focal point is at -b/2a,poly(-b/2a)+1/(4a)
    p_min1_fpoint = Iqqqqs_at_min1+float(1)/(4*p_min1[0])
    d['pIqqqq_Iqqqqfocus'] = p_min1_fpoint*Iqqqq_min1_std+Iqqqq_min1_mean
    #######
    # Heuristics 2b: Characterize I(q) near min1 of I*q^4.
    I_min1_mean = np.mean(I[idx_around_min1])
    I_min1_std = np.std(I[idx_around_min1])
    I_min1_s = (I[idx_around_min1]-I_min1_mean)/I_min1_std
    #I_min1_error = lambda x: np.sum((x[0]*q_min1_s**2 + x[1]*q_min1_s + x[2] - I_min1_s)**2)
    #res = scipimin(I_min1_error,[0,0,0],bounds=[(0,None),(None,None),(None,None)])
    #pI_min1 = res.x
    pI_min1 = np.polyfit(q_min1_s,I_min1_s,2,None,False,np.ones(len(q_min1_s)),False)
    # polynomial vertex horizontal coord is -b/2a
    qs_vertex = -1*pI_min1[1]/(2*pI_min1[0])
    d['pI_qvertex'] = qs_vertex*q_min1_std+q_min1_mean
    # polynomial vertex vertical coord is poly(-b/2a)
    Is_vertex = np.polyval(pI_min1,qs_vertex)
    d['pI_Ivertex'] = Is_vertex*I_min1_std+I_min1_mean
    # The focal width of the parabola is 1/a 
    pI_fwidth = abs(1./pI_min1[0])
    d['pI_qwidth'] = pI_fwidth*q_min1_std
    # The focal point is at -b/2a,poly(-b/2a)+1/(4a)
    pI_fpoint = Is_vertex+float(1)/(4*pI_min1[0])
    d['pI_Ifocus'] = pI_fpoint*I_min1_std+I_min1_mean
    #######
    return d

#def compute_saxs_with_substitutions(q,populations,params,param_array):
#    subs_params = copy.deepcopy(params)
#    ipar = 0
#    for k in subs_params.keys():
#        if isinstance(subs_params[k],list):
#            for ipop in range(len(subs_params[k])):
#                subs_params[k][ipop] = param_array[ipar]
#                ipar += 1
#        else:
#            subs_params[k] = param_array[ipar] 
#            ipar += 1
#    return compute_saxs(q,populations,subs_params)

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
    #TODO: add a sign constraint, at least
    I_mean = np.mean(I)
    I_std = np.std(I)
    q_mean = np.mean(q)
    q_std = np.std(q)
    I_s = (I-I_mean)/I_std
    q_s = (q-q_mean)/q_std
    p = fit_with_slope_constraint(q_s,I_s,-1*q_mean/q_std,0,order) 
    I_at_0 = np.polyval(p,-1*q_mean/q_std)*I_std+I_mean

    #from matplotlib import pyplot as plt
    #plt.plot(q,I,'bo')
    #plt.plot([0.],[I_at_0],'ro')
    #plt.plot(q,np.polyval(p,q_s)*I_std+I_mean)
    #q_fill = np.arange(0.,q[-1],float(q[-1])/100)
    #q_s_fill = (q_fill-q_mean)/q_std
    #plt.plot(q_fill,np.polyval(p,q_s_fill)*I_std+I_mean)
    #plt.show()

    return I_at_0,p

def fit_with_slope_constraint(q,I,q_cons,dIdq_cons,order,weights=None):
    """
    Perform a polynomial fitting 
    of the low-q region of the spectrum
    with dI/dq(q=0) constrained to be zero.
    This is performed by forming a Lagrangian 
    from a quadratic cost function 
    and the Lagrange-multiplied constraint function.
    
    TODO: Document cost function, constraints, Lagrangian.

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
    # TODO: replace this with a sklearn model 
    r0 = 10
    q = np.arange(0.001/r0,float(20)/r0,0.001/r0)       #1/Angstrom
    # NOTE: algorithm works for sigma/r up to 0.45
    sigma_r_vals = np.arange(0*r0,0.46*r0,0.01*r0)      #Angstrom
    for isig,sigma_r in zip(range(len(sigma_r_vals)),sigma_r_vals):
        I = spherical_normal_saxs(q,r0,sigma_r/r0) 
        print('getting I*q**4 metrics for sigma_r/r0 = {}'.format(sigma_r/r0))
        d = saxs_Iq4_metrics(np.array(zip(q,I)))
        sigma_over_r.append(float(sigma_r)/r0)
        qr0_focus.append(d['q_at_Iqqqq_min1']*r0)
        width_metric.append(d['pI_qwidth']/d['q_at_Iqqqq_min1'])
        I_at_0 = spherical_normal_saxs(np.array([0]),r0,sigma_r/r0)[0] 
        intensity_metric.append(d['I_at_Iqqqq_min1']/I_at_0)
    p_f = np.polyfit(sigma_over_r,qr0_focus,4,None,False,None,False)
    p_w = np.polyfit(sigma_over_r,width_metric,4,None,False,None,False)
    p_I = np.polyfit(sigma_over_r,intensity_metric,4,None,False,None,False)
    print('polynomial coefs for qr0 focus: {}'.format(p_f))
    print('polynomial coefs for width metric: {}'.format(p_w))
    print('polynomial coefs for intensity metric: {}'.format(p_I))
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
        plt.plot(sigma_over_r,np.polyval(p_f,sigma_over_r))
        plt.figure(4)
        plt.scatter(width_metric,intensity_metric) 
        plt.show()



