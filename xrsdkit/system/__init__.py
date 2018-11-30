"""This subpackage defines classes and attributes
for describing a material system in enough detail
to compute its scattering/diffraction pattern.
"""
from collections import OrderedDict
import re
import copy

import numpy as np
import lmfit
import yaml

from .population import Population
from .specie import Specie
from .. import definitions as xrsdefs 
from ..tools import primitives, compute_chi2

# TODO: when params, settings, etc are changed,
#   ensure all attributes remain valid,
#   wrt constraints as well as wrt supported options.

def save_to_yaml(file_path,sys):
    sd = sys.to_dict()
    with open(file_path, 'w') as yaml_file:
        yaml.dump(primitives(sd),yaml_file)

def load_from_yaml(file_path):
    with open(file_path, 'r') as yaml_file:
        sd = yaml.load(yaml_file)
    return System(sd)

class NoiseModel(object):

    def __init__(self,model=None,params={}):
        if not model:
            model = 'flat' 
        self.model = model
        self.parameters = {}
        for param_nm in xrsdefs.noise_params[model]:
            self.parameters[param_nm] = copy.deepcopy(xrsdefs.noise_param_defaults[param_nm])  
        for param_nm in params:
            self.update_parameter(param_nm,params[param_nm])

    def to_dict(self):
        nd = {} 
        nd['model'] = copy.copy(self.model)
        nd['parameters'] = {}
        for param_nm,param in self.parameters.items():
            nd['parameters'][param_nm] = copy.deepcopy(param)
        return nd

    def set_model(self,new_model):
        self.model = new_model 
        self.update_parameters()

    def update_parameters(self,new_params={}):
        current_param_nms = list(self.parameters.keys())
        valid_param_nms = copy.deepcopy(xrsdefs.noise_params[self.model])
        # remove any non-valid params
        for param_nm in current_param_nms:
            if not param_nm in valid_param_nms:
                self.parameters.pop(param_nm)
        # add any missing params, taking from new_params if available 
        for param_nm in valid_param_nms:
            if not param_nm in self.parameters:
                self.parameters[param_nm] = copy.deepcopy(xrsdefs.noise_param_defaults[param_nm]) 
            if param_nm in new_params:
                self.update_parameter(param_nm,new_params[param_nm])

    def update_parameter(self,param_nm,new_param_dict): 
        self.parameters[param_nm].update(new_param_dict)

class System(object):

    # TODO: implement caching of settings, parameters, intensities,
    # so that redundant calls to compute_intensity
    # are handled instantly 

    def __init__(self,populations={}):
        # TODO: consider polymorphic constructor inputs 
        self.populations = {}
        self.fit_report = {} # this dict gets populated after self.fit() 
        self.noise_model = NoiseModel('flat')
        self.update_from_dict(populations)

    def to_dict(self):
        sd = {} 
        for pop_nm,pop in self.populations.items():
            sd[pop_nm] = pop.to_dict()
        sd['noise'] = self.noise_model.to_dict()
        sd['fit_report'] = copy.deepcopy(self.fit_report)
        return sd

    def update_from_dict(self,d):
        for pop_name,pd in d.items():
            if pop_name == 'noise':
                self.update_noise_model(pd)
            elif pop_name == 'fit_report':
                self.fit_report.update(pd)
            elif not pop_name in self.populations:
                self.populations[pop_name] = Population.from_dict(pd) 
            else:
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

    def remove_population(self,pop_nm):
        # TODO: check for violated constraints
        # in absence of this population
        self.populations.pop(pop_nm)

    def add_population(self,pop_nm,structure,settings={},parameters={},basis={}):
        self.populations[pop_nm] = Population(structure,settings,parameters,basis)

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
        I = self.compute_noise_intensity(q)
        for pop_name,pop in self.populations.items():
            I += pop.compute_intensity(q,source_wavelength)
        return I

    def compute_noise_intensity(self,q):
        I = np.zeros(len(q))
        noise_modnm = self.noise_model.model
        if not noise_modnm in xrsdefs.noise_model_names:
            raise ValueError('unsupported noise specification: {}'.format(noise_modnm))
        if noise_modnm == 'flat':
            I += self.noise_model.parameters['I0']['value'] * np.ones(len(q))
        return I

    def evaluate_residual(self,q,I,source_wavelength,dI=None,
        error_weighted=True,logI_weighted=True,q_range=[0.,float('inf')],I_comp=None):
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
        I_comp : array
            Optional array of computed intensity (for efficiency)- 
            if provided, intensity is not re-computed   
 
        Returns
        -------
        res : float
            Value of the residual 
        """
        if I_comp is None:
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
            idx_fit = idx_fit & (I_comp>0)
            # TODO: returning float('inf') raises a NaN exception within the minimization.
            # Find a way to deal with I_comp = 0.
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
    
    def lmf_evaluate(self,lmf_params,src_wl,q,I,dI=None,error_weighted=True,logI_weighted=True,q_range=[None,None]):
        new_params = unpack_lmfit_params(lmf_params)
        old_params = self.flatten_params()
        old_params.update(new_params)
        new_pd = unflatten_params(old_params)
        self.update_params_from_dict(new_pd)
        return self.evaluate_residual(q,I,src_wl,dI,error_weighted,logI_weighted,q_range)

    def pack_lmfit_params(self):
        p = self.flatten_params() 
        lmfp = lmfit.Parameters()
        for pkey,pd in p.items():
            ks = pkey.split('__')
            kdepth = len(ks)
            param_name = ks[-1]
            if re.match('coord.',param_name):
                default = xrsdefs.coord_default
            else:
                default = xrsdefs.param_defaults[param_name]
            vary_flag = bool(not default['fixed'])
            if 'fixed' in pd: vary_flag = not pd['fixed']
            p_bounds = copy.deepcopy(default['bounds'])
            if 'bounds' in pd: p_bounds = pd['bounds']
            p_expr = copy.copy(default['constraint_expr'])
            if 'constraint_expr' in pd: p_expr = pd['constraint_expr']
            lmfp.add(pkey,value=pd['value'],vary=vary_flag,min=p_bounds[0],max=p_bounds[1])
            if p_expr:
                lmfp[pkey].set(expr=p_expr)
        return lmfp
    
    def flatten_params(self):
        pd = {} 
        for param_name,paramd in self.noise_model.parameters.items():
            pd['noise__'+param_name] = paramd
        for pop_name,pop in self.populations.items():
            for param_name,paramd in pop.parameters.items():
                pd[pop_name+'__'+param_name] = paramd
            for specie_name,specie in pop.basis.items(): 
                if pop.structure == 'crystalline':
                    pd[pop_name+'__'+specie_name+'__coordx'] = specie.coordinates[0]
                    pd[pop_name+'__'+specie_name+'__coordy'] = specie.coordinates[1] 
                    pd[pop_name+'__'+specie_name+'__coordz'] = specie.coordinates[2]
                for param_name,paramd in specie.parameters.items():
                    pd[pop_name+'__'+specie_name+'__'+param_name] = paramd
        return pd

def fit(sys,q,I,source_wavelength,dI=None,
    error_weighted=True,logI_weighted=True,q_range=[0.,float('inf')]):
    """Fit the I(q) pattern and return a System with optimized parameters. 

    Parameters
    ----------
    sys : xrsdkit.system.System
        System object defining populations and species,
        as well as settings and bounds/constraints for parameters.
    source_wavelength : float
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
    sys_opt : xrsdkit.system.System 
        Similar to input `sys`, but with fit-optimized parameters.
    """

    # the System to optimize starts as a copy of the input System
    sys_opt = System.from_dict(sys.to_dict())

    obj_init = sys_opt.evaluate_residual(q,I,source_wavelength,dI,error_weighted,logI_weighted,q_range)
    lmf_params = sys_opt.pack_lmfit_params() 
    lmf_res = lmfit.minimize(sys_opt.lmf_evaluate,
        lmf_params,method='nelder-mead',
        kws={'src_wl':source_wavelength,
            'q':q,'I':I,'dI':dI,
            'error_weighted':error_weighted,
            'logI_weighted':logI_weighted,
            'q_range':q_range})

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
    sys_opt.fit_report['source_wavelength'] = source_wavelength

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
        if pop_name == 'noise':
            # a noise parameter
            if not 'noise' in pd: pd['noise'] = {}
            if not 'parameters' in pd['noise']: pd['noise']['parameters'] = {}
            pd['noise']['parameters'][ks[1]] = paramd
        elif kdepth == 2: 
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
            if re.match('coord.',ks[2]):
                # a coordinate
                if not 'coordinates' in pd[pop_name]['basis'][specie_name]:
                    pd[pop_name]['basis'][specie_name]['coordinates'] = [None,None,None]
                coord_id = ks[2][-1]
                if coord_id == 'x': coord_idx = 0
                if coord_id == 'y': coord_idx = 1
                if coord_id == 'z': coord_idx = 2
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
        
