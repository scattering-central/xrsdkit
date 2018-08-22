"""This package provides tools for analysis of scattering and diffraction patterns.

A scattering/diffraction pattern is assumed to represent
a material System composed of one or more Populations,
each of which is composed of one or more Species.
This module outlines a taxonomy of classes and attributes
for describing and manipulating such a System.

Developer note: This is the only module that should require revision
when extending xrsdkit to new kinds of structures and form factors.
"""
from collections import OrderedDict
import re

import numpy as np
import lmfit

from .population import Population
from .specie import Specie
from .. import * 
from ..tools import compute_chi2

def structure_form_exception(structure,form):
    msg = 'structure specification {}'\
        'does not support specie specification {}- '\
        'this specie must be removed from the basis '\
        'before setting this structure'.format(structure,form)
    raise ValueError(msg)

class System(object):

    def __init__(self,populations={}):
        # TODO: polymorphic constructor inputs 
        self.populations = {}
        self.update_from_dict(populations)
        self.fit_report = {}
        #
        # TODO: incorporate noise_model construct in lieu of noise populations
        self.noise_model = {'flat':{'I0':0.}}
        #

    def to_dict(self):
        sd = {} 
        for pop_nm,pop in self.populations.items():
            sd[pop_nm] = pop.to_dict()
        return sd

    def update_from_dict(self,d):
        for pop_name,pd_new in d.items():
            if not pop_name in self.populations:
                self.populations[pop_name] = Population.from_dict(pd_new) 

    def update_params_from_dict(self,pd):
        for pop_name, popd in pd.items():
            if 'parameters' in popd:
                for param_name, paramd in popd['parameters'].items():
                    self.populations[pop_name].parameters[param_name].update(popd['parameters'][param_name])
            if 'basis' in popd:
                for specie_name, specied in popd['basis'].items():
                    if 'coordinates' in specied:
                        for ic in range(3):
                            self.populations[pop_name].basis[specie_name].coordinates[ic].update(
                            specied['coordinates'][ic])
                    if 'parameters' in specied:
                        for param_name, paramd in specied['parameters'].items():
                            self.populations[pop_name].basis[specie_name].parameters[param_name].update(
                            specied['parameters'][param_name])

    @classmethod
    def from_dict(cls,d):
        inst = cls()
        inst.update_from_dict(d)
        return inst

    def compute_intensity(self,q,source_wavelength):
        """Computes scattering/diffraction intensity for some `q` values.

        TODO: Document the equations.

        Parameters
        ----------
        q : array
            Array of q values at which intensities will be computed
        source_wavelength : float 
            Wavelength of radiation source in Angstroms

        Returns
        ------- 
        I : array
            Array of scattering intensities for each of the input q values
        """
        I = np.zeros(len(q))
        for pop_name,pop in self.populations.items():
            I += pop.compute_intensity(q,source_wavelength)
        return I

    def evaluate_residual(self,q,I,source_wavelength,dI=None,
        error_weighted=True,logI_weighted=True,q_range=[0.,float('inf')]):
        """Evaluate the fit residual for a given populations dict.
    
        Parameters
        ----------
        q : array of float
            1d array of scattering vector magnitudes (1/Angstrom)
        I : array of float
            1d array of intensities corresponding to `q` values
        source_wavelength : float
            Wavelength of scattered radiation source
        dI : array of float
            1d array of intensity error estimates for each `I` value 
        error_weighted : bool
            Flag for weighting the objective with the I(q) error estimates
        logI_weighted : bool
            Flag for evaluating the objective on log(I(q)) instead if I(q)
        q_range : list
            Two floats indicating the lower and 
            upper q-limits for objective evaluation
    
        Returns
        -------
        res : float
            Value of the residual 
        """
        I_comp = self.compute_intensity(q,source_wavelength)
        idx_nz = (I>0)
        idx_fit = (idx_nz) & (q>=q_range[0]) & (q<=q_range[1])
        wts = np.ones(len(q))
        if error_weighted:
            if dI is None:
                dI = np.empty(I.shape)
                dI.fill(np.nan)
                dI[idx_fit] = np.sqrt(I[idx_fit])
            wts *= dI**2
        if logI_weighted:
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
    
    def lmf_evaluate(self,lmf_params,q,I,error_weighted=True,logI_weighted=True,q_range=[None,None]):
        new_params = unpack_lmfit_params(lmf_params)
        old_params = self.flatten_params()
        old_params.update(new_params)
        new_pd = unflatten_params(old_params)
        self.update_params_from_dict(new_pd)
        return self.evaluate_residual(q,I,error_weighted,logI_weighted,q_range)

    def pack_lmfit_params(self):
        p = self.flatten_params() 
        lmfp = lmfit.Parameters()
        for pkey,pd in p.items():
            ks = pkey.split('__')
            kdepth = len(ks)
            param_name = ks[-1]
            if re.match('coordinate_.',param_name):
                vary_flag = coord_default['fixed']
                p_bounds = coord_default['bounds'] 
                p_expr = coord_default['constraint_expr'] 
            else:
                vary_flag = not param_defaults[param_name]['fixed']
                if 'fixed' in pd: vary_flag = not pd['fixed']
                p_bounds = param_defaults[param_name]['bounds']
                if 'bounds' in pd: p_bounds = pd['bounds']
                p_expr = None
                if 'constraint_expr' in pd: p_expr = pd['constraint_expr']
            lmfp.add(pkey,value=pd['value'],vary=vary_flag,min=p_bounds[0],max=p_bounds[1])
            if p_expr:
                lmfp[pkey].set(expr=p_expr)
        return lmfp
    
    def flatten_params(self):
        pd = {} 
        for pop_name,pop in self.populations.items():
            for param_name,paramd in pop.parameters.items():
                pd[pop_name+'__'+param_name] = paramd
            for specie_name,specie in pop.basis.items(): 
                if pop.structure in crystalline_structures:
                    pd[pop_name+'__'+specie_name+'__coordinate_0'] = specie.coordinates[0]
                    pd[pop_name+'__'+specie_name+'__coordinate_1'] = specie.coordinates[1] 
                    pd[pop_name+'__'+specie_name+'__coordinate_2'] = specie.coordinates[2]
                for param_name,paramd in specie.parameters.items():
                    pd[pop_name+'__'+specie_name+'__'+param_name] = paramd
        return pd

def fit(sys,q,I,source_wavelength,dI=None,
    error_weighted=True,logI_weighted=True,q_range=[0.,float('inf')]):
    """Fit the I(q) pattern and return a dict of optimized parameters. 

    Parameters
    ----------
    q : array of float
        1d array of scattering vector magnitudes (1/Angstrom)
    I : array of float
        1d array of intensities corresponding to `q` values
    source_wavelength : float
        Wavelength of scattered radiation source
    dI : array of float
        1d array of intensity error estimates for each `I` value 
    error_weighted : bool
        Flag for weighting the objective with the I(q) error estimates.
    logI_weighted : bool
        Flag for evaluating the objective on log(I(q)) instead if I(q)
    q_range : list
        Two floats indicating the lower and 
        upper q-limits for objective evaluation

    Returns
    -------
    p_opt : dict
        Dict of scattering populations with optimized parameters.
    rpt : dict
        Dict reporting quantities of interest
        about the fit result.
    """
    p_save = sys.to_dict()
    p_opt = sys.to_dict()
    sys_opt = System.from_dict(p_opt)
    rpt = {} 

    for pop_name,pd in p_opt.items():
        if pd['structure'] == 'unidentified':
            return p_opt,rpt 

    obj_init = sys_opt.evaluate_residual(q,I,source_wavelength,dI,error_weighted,logI_weighted,q_range)
    lmf_params = sys_opt.pack_lmfit_params() 
    lmf_res = lmfit.minimize(sys_opt.lmf_evaluate,
        lmf_params,method='nelder-mead',
        kws={'q':q,'I':I,'error_weighted':error_weighted,'logI_weighted':logI_weighted,'q_range':q_range})

    p_opt = sys_opt.to_dict()

    fit_obj = sys_opt.evaluate_residual(q,I,source_wavelength,dI,error_weighted,logI_weighted,q_range)
    I_opt = sys_opt.compute_intensity(q,source_wavelength)
    I_bg = I - I_opt
    snr = np.mean(I_opt)/np.std(I_bg) 
    sys_opt.fit_report['success'] = lmf_res.success
    sys_opt.fit_report['initial_objective'] = obj_init 
    sys_opt.fit_report['final_objective'] = fit_obj 
    sys_opt.fit_report['error_weighted'] = error_weighted 
    sys_opt.fit_report['logI_weighted'] = logI_weighted 
    sys_opt.fit_report['q_range'] = q_range 
    sys_opt.fit_report['fit_snr'] = snr

    return sys_opt

def unpack_lmfit_params(lmfit_params):
    pd = {} 
    for par_name,par in lmfit_params.items():
        pd[par_name] = {} 
        pd[par_name]['value'] = par.value
        pd[par_name]['bounds'] = [par.min,par.max]
        pd[par_name]['fixed'] = not par.vary
        pd[par_name]['constraint_expr'] = par._expr
    return pd

def unflatten_params(flat_params):
    pd = {} 
    for pkey,paramd in flat_params.items():
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
            pd[pop_name]['parameters'][param_name] = paramd 
        else:
            # a basis parameter
            specie_name = ks[1]
            if not 'basis' in pd[pop_name]:
                pd[pop_name]['basis'] = {} 
            if not specie_name in pd[pop_name]['basis']:
                pd[pop_name]['basis'][specie_name] = {}
            if re.match('coordinate_.',ks[2]):
                # a coordinate
                if not 'coordinates' in pd[pop_name]['basis'][specie_name]:
                    pd[pop_name]['basis'][specie_name]['coordinates'] = [None,None,None]
                coord_idx = int(ks[2][-1])
                pd[pop_name]['basis'][specie_name]['coordinates'][coord_idx] = paramd 
            else:
                # a parameter for a form factor
                if not 'parameters' in pd[pop_name]['basis'][specie_name]:
                    pd[pop_name]['basis'][specie_name]['parameters'] = {} 
                param_name = ks[2]
                pd[pop_name]['basis'][specie_name]['parameters'][param_name] = paramd
    return pd





# TODO: update convenience constructors to return System objects. 

def fcc_crystal(atom_symbol,a_lat=10.,pk_profile='voigt',I0=1.E-3,q_min=0.,q_max=1.,hwhm_g=0.001,hwhm_l=0.001):
    return dict(
        structure='fcc',
        settings={'q_min':q_min,'q_max':q_max,'profile':pk_profile},
        parameters={'I0':I0,'a':a_lat,'hwhm_g':hwhm_g,'hwhm_l':hwhm_l},
        basis={atom_symbol+'_atom':dict(
            coordinates=[0,0,0],
            form='atomic',
            settings={'symbol':atom_symbol}
            )}
        )

def unidentified_population():
    return dict(
        structure='unidentified',
        settings={}, 
        parameters={},
        basis={}
        )

def empty_site():
    return dict(
        form='diffuse',
        settings={},
        parameters={}
        )
        
def flat_noise(I0=1.E-3):
    return dict(
        structure='diffuse',
        settings={},
        parameters={'I0':I0},
        basis={'flat_noise':{'form':'flat'}}
        )


