"""Modules for fitting a measured SAXS spectrum to a scattering equation."""
from __future__ import print_function
import warnings
from collections import OrderedDict
from functools import partial
import copy

from . import saxs_math

import numpy as np
from scipy.optimize import minimize as scipimin

# parameter limits for fit_spectrum() and MC_anneal_fit():
param_limits = OrderedDict(
    I0_floor = (0.,1.E3),
    G_gp = (0.,1.E6),
    rg_gp = (1E-6,1.E3),
    D_gp = (0.,4.),
    I0_sphere = (0.,1.E6),
    r0_sphere = (1E-6,1.E3),
    sigma_sphere = (0.,0.5))

class SaxsFitter(object):
    """Container for handling SAXS spectrum parameter fitting."""

    def __init__(self,q_I,populations):
        """Initialize a SaxsFitter.

        Parameters
        ----------
        q_I : array
            n-by-2 array of scattering vectors q (1/Angstrom)
            and corresponding intensities (arbitrary units)
            against which the computed intensity will be compared
        populations : dict
            dict of the number of distinct populations 
            for each of the various scatterer types. 
            See saxs_math module documentation. 
        """
        self.populations = populations
        self.q = q_I[:,0]
        self.I = q_I[:,1]
        # the following log operation may throw a warning- it's ok.
        # any resulting zeros and nans will not be used.
        self.idx_fit = (self.I>0)
        self.logI = np.empty(self.I.shape)
        self.logI.fill(np.nan)
        self.logI[self.idx_fit] = np.log(self.I[self.idx_fit])
        
    def evaluate_as_array(self,param_keys,param_array):
        param_dict = self.pack_params(param_array,param_keys)
        return self.evaluate(param_dict)

    def fit_objfun(self,params):
        x_init,x_keys,param_idx = self.unpack_params(params) 
        return partial(self.evaluate_as_array,x_keys) 

    def evaluate(self,params):
        """Evaluate the objective for a given dict of params.

        Parameters
        ----------
        params : dict
            Dict of scattering equation parameters.
            See saxs_math module documentation.

        Returns
        -------
        chi2log : float
            sum of difference squared of log(I) 
            between measured intensity and the 
            intensity computed from `param_dict`.
        """
        I_comp = saxs_math.compute_saxs(self.q,self.populations,params)
        chi2log_total = saxs_math.compute_chi2(
                    np.log(I_comp[self.idx_fit]),
                    self.logI[self.idx_fit])
        return chi2log_total 

    def pack_params(self,param_array,param_names):
        param_dict = OrderedDict()
        for pkey,pval in zip(param_names,param_array):
            if not pkey in param_dict.keys():
                param_dict[pkey] = []
            param_dict[pkey].append(pval)
        return param_dict

    def unpack_params(self,param_dict):
        param_list = []
        param_names = []
        param_idx = OrderedDict.fromkeys(param_dict)
        idx=0
        for pkey in saxs_math.all_parameter_keys:
            if pkey in param_dict:
                param_list.extend(param_dict[pkey])
                param_idx[pkey] = []
                for i in range(len(param_dict[pkey])):
                    param_names.append(pkey)
                    param_idx[pkey].append(idx)
                    idx += 1
        return param_list,param_names,param_idx

    def fit(self,params=None,fixed_params=None,objective='chi2log'):
        """Fit the SAXS spectrum, optionally holding some parameters fixed.
    
        Parameters
        ----------
        params : dict
            Dict of scattering equation parameters (initial guess).
            See saxs_math module documentation.
            If not provided, some defaults are chosen.
        fixed_params : dict, optional
            Dict of floats giving values in `params`
            that should be held constant during fitting.
            The structure of this dict should constitute
            a subset of the structure of the `params` dict.
            Entries in `fixed_params` take precedence 
            over the corresponding entries in `params`, so that the 
            initial condition does not violate the constraint.
            Entries in `fixed_params` that are outside 
            the structure of the `params` dict will be ignored.
        objective : string
            Choice of objective function 
            (currently the only option is 'chi2log').

        Returns
        -------
        p_opt : dict
            Dict of optimized SAXS equation parameters,
            with the same shape as the input `params`.
        rpt : dict
            Dict reporting quantities of interest
            pertaining to the fit result.
        """

        if bool(self.populations['unidentified']):
            return OrderedDict(),OrderedDict()

        if params is None:
            params = self.default_params()

        x_init,x_keys,param_idx = self.unpack_params(params) 
        x_bounds = [] 
        for k in x_keys:
            x_bounds.append(param_limits[k])
   
        # --- constraints --- 
        c = []
        if fixed_params is not None:
            for pk,pvals in fixed_params.items():
                if pk in params.keys():
                    for idx,val in enumerate(pvals):
                        if idx < len(params[pk]):
                            params[pk][idx] = val
                            fix_idx = param_idx[pk][idx]
                            cfun = lambda x: x[fix_idx] - x_init[fix_idx]
                            c.append({'type':'eq','fun':cfun})
        # TODO: inequality constraint on I0_floor, G_gp, and I0_sphere,
        # to prevent amplitudes from going to zero
        #if objective in ['chi2log_fixI0']:
        #    if len(I_idx) > 0:
        #        # Set up a constraint to keep I(q=0) fixed
        #        I0_init = np.sum([x_init[i] for i in I_idx.values()])
        #        cfun = lambda x: np.sum([x[I_idx[k]] for k in I_idx.keys()]) - I0_init
        #        c.append({'type':'eq','fun':cfun})
        # --- end constraints ---

        fit_obj = self.fit_objfun(params)
        #fit_obj = saxs_chi2log(flags,params,q_I)
        res = scipimin(fit_obj,x_init,
            bounds=x_bounds,
            options={'ftol':1E-3},
            constraints=c)
        p_opt = self.pack_params(res.x,x_keys) 
        rpt = self.fit_report(p_opt)
        return p_opt,rpt

    def default_params(self):
        pars = OrderedDict()
        defaults = OrderedDict(
            I0_floor = 0.0001,
            G_gp = 0.01,
            rg_gp = 1.,
            D_gp = 4.,
            I0_sphere = 1.,
            r0_sphere = 10.,
            sigma_sphere = 0.1)
        pars['I0_floor'] = [defaults['I0_floor']]
        if 'guinier_porod' in self.populations:
            n_gp = self.populations['guinier_porod']
            if bool(n_gp):
                pars['G_gp'] = []
                pars['rg_gp'] = []
                pars['D_gp'] = []
                for igp in range(n_gp):
                    pars['G_gp'].append(defaults['G_gp'])
                    pars['rg_gp'].append(defaults['rg_gp'])
                    pars['D_gp'].append(defaults['D_gp'])
        if 'spherical_normal' in self.populations:
            n_sn = self.populations['spherical_normal']
            if bool(n_sn):
                pars['I0_sphere'] = []
                pars['r0_sphere'] = []
                pars['sigma_sphere'] = []
                for isn in range(n_sn):
                    pars['I0_sphere'].append(defaults['I0_sphere'])
                    pars['r0_sphere'].append(defaults['r0_sphere'])
                    pars['sigma_sphere'].append(defaults['sigma_sphere'])
        return pars

    def fit_report(self,params):
        rpt = OrderedDict()
        fit_obj = self.fit_objfun(params)
        param_array,param_names,param_idx = self.unpack_params(params) 
        rpt['objective_value'] = fit_obj(param_array)
        I_opt = saxs_math.compute_saxs(self.q,self.populations,params) 
        I_bg = self.I - I_opt
        snr = np.mean(I_opt)/np.std(I_bg) 
        rpt['fit_snr'] = snr
        return rpt 

    def MC_anneal_fit(self,params,stepsize,nsteps,T,fixed_params=None):
        """Perform a Metropolis-Hastings anneal for spectrum fit refinement.

        Parameters
        ----------
        params : dict
            Dict of scattering equation parameters (initial guess).
            See saxs_math module documentation.
        stepsize : float
            fractional step size for random walk 
        nsteps : int
            Number of iterations to perform
        T : float
            Temperature employed in Metropolis acceptance decisions.
        fixed_params : dict 
            Dict indicating fixed values for `params`.
            See documentation of SaxsFitter.fit().

        Returns
        -------
        p_best : dict
            Dict of best-fit parameters
        p_current : dict
            Dict of parameters obtained at the final iteration
        rpt : dict
            Report of objective function and Metropolis-Hastings results
        """
        u_flag = bool(self.populations['unidentified'])
        pks_flag = bool(self.populations['diffraction_peaks'])
        if u_flag or pks_flag: return OrderedDict(),OrderedDict(),OrderedDict()

        # replace any params with the corresponding fixed_params
        if fixed_params is not None:
            for pname,pvals in fixed_params.items():
                if pname in params.keys():
                    for idx,val in enumerate(pvals):
                        if idx < len(params[pname]):
                            params[pname][idx] = val

        fit_obj = self.evaluate
        p_init = copy.deepcopy(params) 
        p_current = copy.deepcopy(params) 
        p_best = copy.deepcopy(params) 
        obj_current = fit_obj(p_current)
        obj_best = obj_current 
        nrej = 0.

        rpt = OrderedDict()
        all_trials = range(nsteps)
        for imc in all_trials:
            # get trial params 
            p_new = copy.deepcopy(p_current)
            x_new,x_keys,param_idx = self.unpack_params(p_new) 
            for idx in range(len(x_new)):
                # TODO: check I0_floor, G_gp, and I0_sphere,
                # to prevent amplitudes from going to zero
                pfix = False
                pkey = x_keys[idx]
                if fixed_params is not None:
                    if pkey in fixed_params.keys():
                        paridx = x_keys[:idx].count(pkey)
                        if paridx < len(fixed_params[pkey]):
                            pfix = True
                if not pfix:
                    xi = x_new[idx]
                    ki = x_keys[idx]
                    param_range = param_limits[ki][1] - param_limits[ki][0]
                    if xi == 0.:
                        xi_trial = np.random.rand()*stepsize*param_range 
                    else:
                        xi_trial = xi*(1+2*(np.random.rand()-0.5)*stepsize)
                    if xi_trial < param_limits[ki][0]:
                        xi_trial = param_limits[ki][0] 
                    if xi_trial > param_limits[ki][1]:
                        xi_trial = param_limits[ki][1] 
                    x_new[idx] = xi_trial 
            p_new = self.pack_params(x_new,x_keys)
            # evaluate objective, determine acceptance
            obj_new = fit_obj(p_new)
            if obj_new < obj_current:
                accept = True
                if obj_new < obj_best:
                    p_best = p_new
                    obj_best = obj_new
            elif T == 0.:
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
        rpt['objective_init'] = fit_obj(p_init)
        rpt['objective_best'] = fit_obj(p_best)
        rpt['objective_final'] = fit_obj(p_current)
        return p_best,p_current,rpt


