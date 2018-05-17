import copy
import re
import os

import numpy as np
import lmfit

from .. import all_params, param_defaults, fixed_param_defaults, param_bound_defaults, update_site_param
from ..scattering import compute_intensity
from ..tools import compute_chi2

class XRSDFitter(object):
    """Class for fitting x-ray scattering and diffraction profiles."""

    def __init__(self,q_I,populations,source_wavelength,dI=None):
        """Initialize a XRSDFitter.

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
        dI : array
            1-dimensional array of error estimates
            for scattering intensities (optional).
            If not provided, error-weighted fitting
            is performed using the square root of the intensity
            as the error estimate. 
        source_wavelength : float
            X-ray source wavelength used for computing diffraction patterns
        """
        self.populations = populations
        self.source_wavelength = source_wavelength
        self.q = q_I[:,0]
        self.I = q_I[:,1]
        self.idx_fit = (self.I>0)
        self.logI = np.empty(self.I.shape)
        self.logI.fill(np.nan)
        self.logI[self.idx_fit] = np.log(self.I[self.idx_fit])
        if dI is not None:
            self.dI = dI
        else:
            self.dI = np.empty(self.I.shape)
            self.dI.fill(np.nan)
            self.dI[self.idx_fit] = np.sqrt(self.I[self.idx_fit])

    def fit(self,\
        fixed_params=None,param_bounds=None,param_constraints=None,\
        error_weighted=True,logI_weighted=True,q_range=[0.,float('inf')]):
        """Fit the self.q_I pattern, return an optimized populations dict.
    
        Parameters
        ----------
        fixed_params : dict
        param_bounds : dict
        param_constraints : dict
        error_weighted : bool
            Flag for whether or not the fit 
            should be weighted by the intensity error estimates.
        logI_weighted : bool
        q_range : list
            Two floats indicating the lower and 
            upper q-limits for objective evaluation
        objective : string
            Choice of objective function 
            (currently the only option is 'chi2log').

        Returns
        -------
        p_opt : dict
            Dict of scattering populations with optimized parameters.
        rpt : dict
            Dict reporting quantities of interest
            about the fit result.
        """
        p_opt = copy.deepcopy(self.populations)
        rpt = {} 

        if 'unidentified' in p_opt.keys():
            return p_opt,rpt 

        obj_init = self.evaluate_residual(p_opt,error_weighted,logI_weighted,q_range)
        #print('INITIAL OBJECTIVE: {}'.format(obj_init))
        #print(p_opt)

        fp = self.empty_params()
        pb = self.empty_params()
        pc = self.empty_params()
        if fixed_params is not None:
            fp = self.update_params(fp,fixed_params)
        if param_bounds is not None:
            pb  = self.update_params(pb,param_bounds)
        if param_constraints is not None:
            pc  = self.update_params(pc,param_constraints)

        lmf_params = self.pack_lmfit_params(p_opt,fp,pb,pc) 
        lmf_res = lmfit.minimize(self.lmf_evaluate,
            lmf_params,method='nelder-mead',
            kws={'error_weighted':error_weighted,'logI_weighted':logI_weighted,'q_range':q_range})
        flat_params = self.unpack_lmfit_params(lmf_res.params)
        p_opt = self.update_params(p_opt,self.unflatten_params(flat_params))

        rpt['success'] = lmf_res.success
        rpt['initial_objective'] = obj_init 
        fit_obj = self.lmf_evaluate(lmf_res.params,error_weighted,logI_weighted,q_range)
        rpt['final_objective'] = fit_obj 
        rpt['error_weighted'] = error_weighted 
        rpt['logI_weighted'] = logI_weighted 
        rpt['q_range'] = q_range 
        I_opt = compute_intensity(self.q,p_opt,self.source_wavelength)
        I_bg = self.I - I_opt
        snr = np.mean(I_opt)/np.std(I_bg) 
        rpt['fit_snr'] = snr

        #print(p_opt)
        #print('FINAL OBJECTIVE: {}'.format(fit_obj))

        I_init = compute_intensity(self.q,self.populations,self.source_wavelength)
        I_opt = compute_intensity(self.q,p_opt,self.source_wavelength)

        #from matplotlib import pyplot as plt
        #plt.figure(1)
        #plt.semilogy(self.q,self.I)
        #plt.semilogy(self.q,I_init,'r-')
        #plt.semilogy(self.q,I_opt,'g-')
        #plt.show()

        return p_opt,rpt

    def empty_params(self):
        ep = {} 
        for pop_name,popd in self.populations.items():
            ep[pop_name] = {} 
            #ep[pop_name]['settings'] = {} 
            ep[pop_name]['parameters'] = dict.fromkeys(popd['parameters'].keys())
            ep[pop_name]['basis'] = dict.fromkeys(popd['basis'].keys())
            #for setting_name in popd['settings'].keys():
            #    ep[pop_name]['settings'][setting_name] = None
            for site_name,site_def in popd['basis'].items():
                ep[pop_name]['basis'][site_name] = {} 
                if 'coordinates' in site_def: 
                    ep[pop_name]['basis'][site_name]['coordinates'] = [None,None,None] 
                if 'coordinates' in site_def: 
                    ep[pop_name]['basis'][site_name]['parameters'] = dict.fromkeys(site_def['parameters'].keys())
        return ep

    def print_report(self,init_pops,fit_pops,report):
        p = 'optimization objective: {} --> {}'.\
        format(report['initial_objective'],report['final_objective'])
        init_flat_params = self.flatten_params(init_pops)
        fit_flat_params = self.flatten_params(fit_pops)
        for k, v in init_flat_params.items():
            p += os.linesep+'{}: {} --> {}'.format(k,v,fit_flat_params[k])
        return p

    @staticmethod
    def update_params(p_base,p_new):
        p_base = copy.deepcopy(p_base)
        for pop_name,popd in p_new.items():
            #if pop_name in p_base.keys():
            if 'parameters' in popd.keys():
                for param_name,param_val in popd['parameters'].items():
                    p_base[pop_name]['parameters'][param_name] = copy.deepcopy(param_val)
            #if 'settings' in popd.keys():
            #    for setting_name,setting_val in popd['settings'].items():
            #        p_base[pop_name]['settings'][setting_name] = copy.deepcopy(setting_val)
            if 'basis' in popd.keys():
                for site_name,site_def in popd['basis'].items():
                    for k,site_item in site_def.items():
                        if k == 'coordinates':
                            for coord_idx,coord_val in enumerate(site_item):
                                p_base[pop_name]['basis'][site_name][k][coord_idx] = copy.deepcopy(coord_val)
                        elif isinstance(site_item,list):
                            for ist,stitm in enumerate(site_item):
                                for ff_param_name, ff_param_val in stitm.items():
                                    p_base[pop_name]['basis'][site_name][k][ist][ff_param_name] = \
                                    copy.deepcopy(ff_param_val)
                        else:
                            for ff_param_name, ff_param_val in site_item.items():
                                update_site_param(p_base,pop_name,site_name,ff_param_name,copy.deepcopy(ff_param_val))
        return p_base

    def pack_lmfit_params(self,populations=None,fixed_params={},param_bounds={},param_constraints={}):
        if populations is None:
            populations=self.populations
        p = self.flatten_params(populations) 
        fp = self.flatten_params(fixed_params) 
        pb = self.flatten_params(param_bounds)
        pc = self.flatten_params(param_constraints)
        lmfp = lmfit.Parameters()
        for pkey,pval in p.items():
            ks = pkey.split('__')
            kdepth = len(ks)
            param_name = ks[-1]
            if re.match('coordinate_.',param_name):
                param_name = 'coordinates'
            p_bounds = param_bound_defaults[param_name] 
            if pkey in pb:
                if pb[pkey] is not None:
                    p_bounds = pb[pkey] 
            vary_flag = not fixed_param_defaults[param_name] 
            if pkey in fp:
                if fp[pkey] is not None:
                    vary_flag = not fp[pkey]
            p_expr = None
            lmfp.add(pkey,value=pval,vary=vary_flag,min=p_bounds[0],max=p_bounds[1])
        for pkey,pval in p.items():
            if pkey in pc:
                if pc[pkey] is not None:
                    p_expr = pc[pkey] 
                    lmfp[pkey].set(expr=p_expr)
        return lmfp

    @staticmethod
    def unpack_lmfit_params(lmfit_params):
        pd = {} 
        for par_name,par in lmfit_params.items():
            pd[par_name] = copy.deepcopy(par.value)
        return pd

    @staticmethod
    def flatten_params(populations):
        pd = {} 
        for pop_name,popd in populations.items():
            if 'parameters' in popd:
                for param_name,param_val in popd['parameters'].items():
                    pd[pop_name+'__'+param_name] = copy.deepcopy(param_val)
            if 'basis' in popd:
                for site_name, site_def in popd['basis'].items():
                    if 'coordinates' in site_def:
                        pd[pop_name+'__'+site_name+'__coordinate_0'] = copy.deepcopy(site_def['coordinates'][0])
                        pd[pop_name+'__'+site_name+'__coordinate_1'] = copy.deepcopy(site_def['coordinates'][1])
                        pd[pop_name+'__'+site_name+'__coordinate_2'] = copy.deepcopy(site_def['coordinates'][2])
                    if 'parameters' in site_def:
                        for ff_param_name, ff_param_val in site_def['parameters'].items():
                            pd[pop_name+'__'+site_name+'__'+ff_param_name] = \
                            copy.deepcopy(ff_param_val)
        return pd

    @staticmethod
    def unflatten_params(flat_params):
        pd = {} 
        for pkey,pval in flat_params.items():
            ks = pkey.split('__')
            kdepth = len(ks)
            pop_name = ks[0]
            if not pop_name in pd:
                pd[pop_name] = {} 
            if kdepth == 2: 
                # a structure parameter 
                if not 'parameters' in pd[pop_name]:
                    pd[pop_name]['parameters'] = {} 
                param_name = ks[1]
                pd[pop_name]['parameters'][param_name] = copy.deepcopy(pval)
            else:
                # a basis or form factor parameter
                site_name = ks[1]
                if not 'basis' in pd[pop_name]:
                    pd[pop_name]['basis'] = {} 
                if not site_name in pd[pop_name]['basis']:
                    pd[pop_name]['basis'][site_name] = {}
                if ks[2] in ['coordinate_0','coordinate_1','coordinate_2']:
                    # a coordinate
                    if not 'coordinates' in pd[pop_name]['basis'][site_name]:
                        pd[pop_name]['basis'][site_name]['coordinates'] = [None,None,None]
                    coord_idx = int(ks[2][-1])
                    pd[pop_name]['basis'][site_name]['coordinates'][coord_idx] = copy.deepcopy(pval) 
                else:
                    # a parameter for a form factor
                    if not 'parameters' in pd[pop_name]['basis'][site_name]:
                        pd[pop_name]['basis'][site_name]['parameters'] = {} 
                    param_name = ks[2]
                    pd[pop_name]['basis'][site_name]['parameters'][param_name] = copy.deepcopy(pval)
        return pd

    def evaluate_residual(self,populations,error_weighted=True,logI_weighted=True,q_range=[0.,float('inf')]):
        """Evaluate the fit residual for a given populations dict.

        Parameters
        ----------
        populations : dict
            Dict of scatterer populations and parameters 
        error_weighted : bool
            Flag for whether or not to weight the result
            by the intensity error estimate.
        q_range : list
            List of two floats for the lower and
            upper q-range for objective evaluation

        Returns
        -------
        res : float
            Value of the residual 
        """
        I_comp = compute_intensity(
            self.q,populations,self.source_wavelength)

        #if q_range[0] is None:
        #    q_range[0] = self.q[0]
        #if q_range[1] is None:
        #    q_range[1] = self.q[-1]

        idx_fit = (self.idx_fit) & (self.q>=q_range[0]) & (self.q<=q_range[1])

        n_q = len(self.q)
        wts = np.ones(n_q)
        if error_weighted:
            wts *= self.dI**2
        if logI_weighted:
            res = compute_chi2(
                np.log(I_comp[idx_fit]),
                self.logI[idx_fit],
                wts[idx_fit])
        else:
            res = compute_chi2(
                I_comp[idx_fit],
                self.I[idx_fit],
                wts[idx_fit])
        return res 

    def lmf_evaluate(self,lmf_params,error_weighted=True,logI_weighted=True,q_range=[None,None]):
        pd = self.unflatten_params(self.unpack_lmfit_params(lmf_params))
        pops = copy.deepcopy(self.populations)
        pops = self.update_params(pops,pd)
        return self.evaluate_residual(pops,error_weighted,logI_weighted,q_range)

    #def fit_intensity_params(self,params):
    #    """Fit the spectrum wrt only the intensity parameters."""
    #    fp = self.default_params()
    #    for k,v in fp.items():
    #        if k in ['I0_floor','I0_sphere','G_gp','I_pkcenter']:
    #            for idx in range(len(v)):
    #                v[idx] = False
    #        else:
    #            for idx in range(len(v)):
    #                v[idx] = True
    #    return self.fit(params,fp)

    #def estimate_peak_params(self,params=None):
    #    if params is None:
    #        params = self.default_params()
    #    if bool(self.populations['diffraction_peaks']):
    #        # 1) walk the spectrum, collect best diff. pk. candidates
    #        pk_idx, pk_conf = peak_finder.peaks_by_window(self.q,self.I,20,0.)
    #        conf_idx = np.argsort(pk_conf)[::-1]
    #        params['q_pkcenter'] = []
    #        params['I_pkcenter'] = []
    #        params['pk_hwhm'] = []
    #        npk = 0
    #        # 2) for each peak (from best candidate to worst),
    #        for idx in conf_idx:
    #            if npk < self.populations['diffraction_peaks']:
    #                # a) record the q value
    #                q_pk = self.q[pk_idx[idx]]
    #                # b) estimate the intensity
    #                I_at_qpk = self.I[pk_idx[idx]]
    #                I_pk = I_at_qpk * 0.1
    #                #I_pk = I_at_qpk - I_nopeaks[pk_idx[idx]] 
    #                # c) estimate the width
    #                idx_around_pk = (self.q>0.95*q_pk) & (self.q<1.05*q_pk)
    #                qs,qmean,qstd = saxs_math.standardize_array(self.q[idx_around_pk])
    #                Is,Imean,Istd = saxs_math.standardize_array(self.I[idx_around_pk])
    #                p_pk = np.polyfit(qs,Is,2,None,False,np.ones(len(qs)),False)
    #                # quadratic vertex horizontal coord is -b/2a
    #                #qpk_quad = -1*p_pk[1]/(2*p_pk[0])
    #                # quadratic focal width is 1/a 
    #                p_pk_fwidth = abs(1./p_pk[0])*qstd
    #                params['q_pkcenter'].append(float(q_pk))
    #                params['I_pkcenter'].append(float(I_pk))
    #                params['pk_hwhm'].append(float(p_pk_fwidth*0.5))
    #                npk += 1    
    #    return params


## TODO: refactor this
#    def MC_anneal_fit(self,params,stepsize,nsteps,T,fixed_params=None):
#        """Perform a Metropolis-Hastings anneal for spectrum fit refinement.
#
#        Parameters
#        ----------
#        params : dict
#            Dict of scattering equation parameters (initial guess).
#            See saxs_math module documentation.
#        stepsize : float
#            fractional step size for random walk 
#        nsteps : int
#            Number of iterations to perform
#        T : float
#            Temperature employed in Metropolis acceptance decisions.
#        fixed_params : dict 
#            Dict indicating fixed values for `params`.
#            See documentation of XRSDFitter.fit().
#
#        Returns
#        -------
#        p_best : dict
#            Dict of best-fit parameters
#        p_current : dict
#            Dict of parameters obtained at the final iteration
#        rpt : dict
#            Report of objective function and Metropolis-Hastings results
#        """
#        u_flag = bool(self.populations['unidentified'])
#        pks_flag = bool(self.populations['diffraction_peaks'])
#        if u_flag or pks_flag: return OrderedDict(),OrderedDict(),OrderedDict()
#
#        # replace any params with the corresponding fixed_params
#        if fixed_params is not None:
#            for pname,pvals in fixed_params.items():
#                if pname in params.keys():
#                    for idx,val in enumerate(pvals):
#                        if idx < len(params[pname]):
#                            params[pname][idx] = val
#
#        fit_obj = self.evaluate
#        p_init = copy.deepcopy(params) 
#        p_current = copy.deepcopy(params) 
#        p_best = copy.deepcopy(params) 
#        obj_current = fit_obj(p_current)
#        obj_best = obj_current 
#        nrej = 0.
#
#        rpt = OrderedDict()
#        all_trials = range(nsteps)
#        for imc in all_trials:
#            # get trial params 
#            p_new = copy.deepcopy(p_current)
#            x_new,x_keys,param_idx = self.unpack_params(p_new) 
#            for idx in range(len(x_new)):
#                # TODO: check I0_floor, G_gp, and I0_sphere,
#                # to prevent amplitudes from going to zero
#                pfix = False
#                pkey = x_keys[idx]
#                if fixed_params is not None:
#                    if pkey in fixed_params.keys():
#                        paridx = x_keys[:idx].count(pkey)
#                        if paridx < len(fixed_params[pkey]):
#                            pfix = True
#                if not pfix:
#                    xi = x_new[idx]
#                    ki = x_keys[idx]
#                    param_range = param_bounds[ki][1] - param_bounds[ki][0]
#                    if xi == 0.:
#                        xi_trial = np.random.rand()*stepsize*param_range 
#                    else:
#                        xi_trial = xi*(1+2*(np.random.rand()-0.5)*stepsize)
#                    if xi_trial < param_bounds[ki][0]:
#                        xi_trial = param_bounds[ki][0] 
#                    if xi_trial > param_bounds[ki][1]:
#                        xi_trial = param_bounds[ki][1] 
#                    x_new[idx] = xi_trial 
#            p_new = self.pack_params(x_new,x_keys)
#            # evaluate objective, determine acceptance
#            obj_new = fit_obj(p_new)
#            if obj_new < obj_current:
#                accept = True
#                if obj_new < obj_best:
#                    p_best = p_new
#                    obj_best = obj_new
#            elif T == 0.:
#                accept = False
#            else:
#                accept = np.exp(-1.*(obj_new-obj_current)/T) > np.random.rand()
#            # act on acceptance decision
#            if accept:
#                p_current = p_new
#                obj_current = obj_new
#            else:
#                nrej += 1
#                p_new = p_current
#
#        rpt['reject_ratio'] = float(nrej)/nsteps
#        rpt['objective_init'] = fit_obj(p_init)
#        rpt['objective_best'] = fit_obj(p_best)
#        rpt['objective_final'] = fit_obj(p_current)
#        return p_best,p_current,rpt
#
#


