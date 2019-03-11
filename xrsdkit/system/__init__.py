import re
import copy

import numpy as np
import lmfit

from .noise import NoiseModel
from .population import Population
from .. import definitions as xrsdefs 
from ..tools import compute_chi2
from ..tools.profiler import profile_keys, profile_pattern

# TODO: when params, settings, etc are changed,
#   ensure all attributes remain valid,
#   wrt constraints as well as wrt supported options.

class System(object):

    # TODO: use caching to speed up repeated compute_intensity() evaluations

    def __init__(self,**kwargs):
        self.populations = {}
        self.fit_report = dict(
            error_weighted=True,
            logI_weighted=True,
            good_fit=False,
            q_range=[0.,float('inf')]
            )
        self.features = dict.fromkeys(profile_keys) 
        src_wl = 0.
        kwargs = copy.deepcopy(kwargs)
        if 'source_wavelength' in kwargs: src_wl = kwargs.pop('source_wavelength')
        # TODO: any other sample metadata items that should be handled from kwargs?
        self.sample_metadata = dict(
            experiment_id='',
            sample_id='',
            data_file='',
            source_wavelength=src_wl,
            time=0.,
            notes=''
            )
        self.noise_model = NoiseModel('flat')
        self.update_from_dict(kwargs)

    def to_dict(self):
        sd = {} 
        for pop_nm,pop in self.populations.items():
            sd[pop_nm] = pop.to_dict()
        sd['noise'] = self.noise_model.to_dict()
        sd['fit_report'] = copy.deepcopy(self.fit_report)
        sd['sample_metadata'] = self.sample_metadata
        sd['features'] = self.features
        return sd

    def update_from_dict(self,d):
        for pop_name,pd in d.items():
            if pop_name == 'noise':
                if not isinstance(pd,dict): pd = pd.to_dict()
                self.update_noise_model(pd)
            elif pop_name == 'features':
                self.features.update(pd)
            elif pop_name == 'fit_report':
                self.fit_report.update(pd)
            elif pop_name == 'sample_metadata':
                self.sample_metadata.update(pd)
            elif not pop_name in self.populations:
                if not isinstance(pd,dict): pd = pd.to_dict()
                self.populations[pop_name] = Population.from_dict(pd) 
            else:
                if not isinstance(pd,dict): pd = pd.to_dict()
                self.populations[pop_name].update_from_dict(pd)

    def update_noise_model(self,noise_dict):
        if 'model' in noise_dict:
            self.noise_model.set_model(noise_dict['model'])
        if 'parameters' in noise_dict:
            self.noise_model.update_parameters(noise_dict['parameters'])

    def update_params_from_dict(self,pd):
        for pop_name, popd in pd.items():
            if pop_name == 'noise':
                self.update_noise_model(popd)
            else:
                if 'parameters' in popd:
                    for param_name, paramd in popd['parameters'].items():
                        self.populations[pop_name].parameters[param_name].update(popd['parameters'][param_name])

    def set_q_range(self,q_min,q_max):
        self.fit_report['q_range'] = [float(q_min),float(q_max)]

    def set_error_weighted(self,err_wtd):
        self.fit_report['error_weighted'] = bool(err_wtd) 

    def set_logI_weighted(self,logI_wtd):
        self.fit_report['logI_weighted'] = bool(logI_wtd) 

    def remove_population(self,pop_nm):
        # TODO: check for violated constraints
        # in absence of this population
        self.populations.pop(pop_nm)

    def add_population(self,pop_nm,structure,form,settings={},parameters={}):
        self.populations[pop_nm] = Population(structure,form,settings,parameters)

    @classmethod
    def from_dict(cls,d):
        return cls(**d)

    def compute_intensity(self,q):
        """Computes scattering/diffraction intensity for some `q` values.

        TODO: Document the equations.

        Parameters
        ----------
        q : array
            Array of q values at which intensities will be computed

        Returns
        ------- 
        I : array
            Array of scattering intensities for each of the input q values
        """
        I = self.noise_model.compute_intensity(q)
        for pop_name,pop in self.populations.items():
            I += pop.compute_intensity(q,self.sample_metadata['source_wavelength'])
        return I

    def evaluate_residual(self,q,I,dI=None,I_comp=None):
        """Evaluate the fit residual for a given populations dict.
    
        Parameters
        ----------
        q : array of float
            1d array of scattering vector magnitudes (1/Angstrom)
        I : array of float
            1d array of intensities corresponding to `q` values
        dI : array of float
            1d array of intensity error estimates for each `I` value 
        error_weighted : bool
            Flag for weighting the objective with the I(q) error estimates
        logI_weighted : bool
            Flag for evaluating the objective on log(I(q)) instead if I(q)
        q_range : list
            Two floats indicating the lower and 
            upper q-limits for objective evaluation
        I_comp : array
            Optional array of computed intensity (for efficiency)- 
            if provided, intensity is not re-computed   
 
        Returns
        -------
        res : float
            Value of the residual 
        """
        q_range = self.fit_report['q_range']
        if I_comp is None:
            I_comp = self.compute_intensity(q)
        idx_nz = (I>0)
        idx_fit = (idx_nz) & (q>=q_range[0]) & (q<=q_range[1])
        wts = np.ones(len(q))
        if self.fit_report['error_weighted']:
            if dI is None:
                dI = np.empty(I.shape)
                dI.fill(np.nan)
                dI[idx_fit] = np.sqrt(I[idx_fit])
            wts *= dI**2
        if self.fit_report['logI_weighted']:
            idx_fit = idx_fit & (I_comp>0)
            # NOTE: returning float('inf') raises a NaN exception within the minimization.
            #if not any(idx_fit):
            #    return float('inf')
            res = compute_chi2(
                np.log(I_comp[idx_fit]),
                np.log(I[idx_fit]),
                wts[idx_fit])
        else:
            res = compute_chi2(
                I_comp[idx_fit],
                I[idx_fit],
                wts[idx_fit])
        return res 

    def lmf_evaluate(self,lmf_params,q,I,dI=None):
        new_params = unpack_lmfit_params(lmf_params)
        old_params = self.flatten_params()
        old_params.update(new_params)
        new_pd = unflatten_params(old_params)
        self.update_params_from_dict(new_pd)
        return self.evaluate_residual(q,I,dI)

    def pack_lmfit_params(self):
        p = self.flatten_params() 
        lmfp = lmfit.Parameters()
        for pkey,pd in p.items():
            ks = pkey.split('__')
            vary_flag = bool(not pd['fixed'])
            p_bounds = copy.deepcopy(pd['bounds'])
            p_expr = copy.copy(pd['constraint_expr'])
            lmfp.add(pkey,value=pd['value'],vary=vary_flag,min=p_bounds[0],max=p_bounds[1])
            if p_expr:
                lmfp[pkey].set(vary=False)
                lmfp[pkey].set(expr=p_expr)
        return lmfp
    
    def flatten_params(self):
        pd = {} 
        for param_name,paramd in self.noise_model.parameters.items():
            pd['noise__'+param_name] = paramd
        for pop_name,pop in self.populations.items():
            for param_name,paramd in pop.parameters.items():
                pd[pop_name+'__'+param_name] = paramd
        return pd

def fit(sys,q,I,dI=None,
    error_weighted=None,logI_weighted=None,q_range=None):
    """Fit the I(q) pattern and return a System with optimized parameters. 

    Parameters
    ----------
    sys : xrsdkit.system.System
        System object defining populations and species,
        as well as settings and bounds/constraints for parameters.
    q : array of float
        1d array of scattering vector magnitudes (1/Angstrom)
    I : array of float
        1d array of intensities corresponding to `q` values
    dI : array of float
        1d array of intensity error estimates for each `I` value 

    Returns
    -------
    sys_opt : xrsdkit.system.System 
        Similar to input `sys`, but with fit-optimized parameters.
    error_weighted : bool
        Flag for weighting the objective with the I(q) error estimates.
    logI_weighted : bool
        Flag for evaluating the objective on log(I(q)) instead if I(q)
    q_range : list
        Two floats indicating the lower and 
        upper q-limits for objective evaluation
    """

    # the System to optimize starts as a copy of the input System
    sys_opt = System.from_dict(sys.to_dict())
    # if inputs were given to control the fit objective,
    # update sys_opt.fit_report with the new settings
    if error_weighted is not None:
        sys_opt.fit_report.update(error_weighted=error_weighted)
    if logI_weighted is not None:
        sys_opt.fit_report.update(logI_weighted=error_weighted)
    if q_range is not None:
        sys_opt.fit_report.update(q_range=q_range)

    obj_init = sys_opt.evaluate_residual(q,I,dI)
    lmf_params = sys_opt.pack_lmfit_params() 
    lmf_res = lmfit.minimize(
        sys_opt.lmf_evaluate,
        lmf_params,method='nelder-mead',
        kws={'q':q,'I':I,'dI':dI}
        )

    fit_obj = sys_opt.evaluate_residual(q,I,dI)
    I_opt = sys_opt.compute_intensity(q)
    I_bg = I - I_opt
    snr = np.mean(I_opt)/np.std(I_bg) 
    sys_opt.fit_report['converged'] = lmf_res.success
    sys_opt.fit_report['initial_objective'] = obj_init 
    sys_opt.fit_report['final_objective'] = fit_obj 
    sys_opt.fit_report['fit_snr'] = snr
    sys_opt.features = profile_pattern(q,I)

    return sys_opt

def unpack_lmfit_params(lmfit_params):
    pd = {} 
    for par_name,par in lmfit_params.items():
        pd[par_name] = {} 
        pd[par_name]['value'] = par.value
        pd[par_name]['bounds'] = [par.min,par.max]
        pd[par_name]['fixed'] = not par.vary
        if par._expr: pd[par_name]['fixed'] = False
        pd[par_name]['constraint_expr'] = par._expr
    return pd

def unflatten_params(flat_params):
    pd = {} 
    for pkey,paramd in flat_params.items():
        ks = pkey.split('__')
        #kdepth = len(ks)
        pop_name = ks[0]
        param_name = ks[1]
        if not pop_name in pd:
            pd[pop_name] = {} 
        if not 'parameters' in pd[pop_name]:
            pd[pop_name]['parameters'] = {} 
        pd[pop_name]['parameters'][param_name] = paramd 
    return pd

