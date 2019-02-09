import copy

import numpy as np

from .. import definitions as xrsdefs 
from ..scattering import guinier_porod_intensity

class NoiseModel(object):

    def __init__(self,model=None,parameters={}):
        if not model:
            model = 'flat' 
        self.model = model
        self.parameters = {}
        self.update_parameters(parameters)

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
        valid_params = copy.deepcopy(xrsdefs.noise_params[self.model])
        for param_nm,param_val in new_params.items():
            if not param_nm in valid_params:
                msg = 'Parameter {} is not valid for noise model {}'.format(
                param_nm,self.model)  
                raise ValueError(msg)
        # remove any non-valid params,
        # copy current valid values to valid_params
        current_param_nms = list(self.parameters.keys())
        for param_nm in current_param_nms:
            if param_nm in valid_params.keys():
                valid_params[param_nm].update(self.parameters[param_nm])
            else:
                self.parameters.pop(param_nm)
        # update params-
        # at this point, all new_params are assumed valid
        for param_nm, param_def in new_params.items():
            valid_params[param_nm].update(new_params[param_nm])
        self.parameters.update(valid_params)

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


