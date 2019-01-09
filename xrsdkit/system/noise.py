import copy

import numpy as np

from .. import definitions as xrsdefs 
from ..scattering.form_factors import guinier_porod_intensity

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

    def compute_intensity(self,q):
        n_q = len(q)
        I = np.zeros(n_q)
        if not self.model in xrsdefs.noise_model_names:
            raise ValueError('unsupported noise specification: {}'.format(self.model))
        if self.model == 'flat':
            I += self.parameters['I0']['value'] * np.ones(n_q)
        elif self.model == 'low_q_scatter':
            I0_beam = self.parameters['I0']['value'] * (1.-self.parameters['I0_flat_fraction']['value'])
            rg_eff = self.parameters['effective_rg']['value']
            D_eff = self.parameters['effective_D']['value']
            I += I0_beam * guinier_porod_intensity(q,rg_eff,D_eff)
            I += self.parameters['I0']['value'] * self.parameters['I0_flat_fraction']['value'] * np.ones(n_q)
        return I


