import copy
from collections import OrderedDict

import numpy as np

from .. import definitions as xrsdefs 
from .. import scattering as xrsdscat
from ..scattering import diffuse_intensity, disordered_intensity, crystalline_intensity
from ..scattering import space_groups as sgs

class Population(object):

    def __init__(self,structure,form,settings={},parameters={}):
        self.structure = None
        self.form = None
        self.settings = {}
        self.parameters = {}
        self.set_structure(structure)
        self.set_form(form)
        self.update_settings(settings)
        self.update_parameters(parameters)

    def set_structure(self,structure):
        xrsdefs.validate(structure,self.form,self.settings)
        self.structure = structure
        self.update_settings()
        self.update_parameters()

    def set_form(self,form):
        xrsdefs.validate(self.structure,form,self.settings)
        self.form = form
        self.update_settings()
        self.update_parameters()

    def update_settings(self,new_settings={}):
        valid_stgs = xrsdefs.all_settings(self.structure,self.form)
        for stg_nm,stg_val in new_settings.items():
            if not stg_nm in valid_settings:
                msg = 'Setting {} is not valid for current structure ({}) and form factor ({})'.format(
                stg_nm,self.structure,self.form)  
                raise ValueError(msg)
        # copy current setting values to valid_stgs
        current_stg_nms = list(self.settings.keys())
        for stg_nm in current_stg_nms:
            if stg_nm in valid_stgs.keys():
                valid_stgs[stg_nm] = self.settings[stg_nm]
        xrsdefs.validate(self.structure,self.form,valid_stgs)
        # remove any self.settings that are no longer valid
        for stg_nm in current_stg_nms:
            if not stg_nm in valid_settings:
                self.settings.pop(stg_nm)
        # update settings-
        # at this point, all new_settings are assumed valid
        valid_stgs.update(new_settings)
        self.settings.update(valid_settings)
        self.update_parameters()

    def update_parameters(self,new_params={}):
        valid_params = xrsdefs.all_params(self.structure,self.form,self.settings)
        for param_nm,param_val in new_params.items():
            if not param_nm in valid_params:
                msg = 'Parameter {} is not valid for structure: {}, form: {}, settings: {}'.format(
                param_nm,self.structure,self.form,self.settings)  
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

    def to_dict(self):
        pd = {} 
        pd['structure'] = copy.copy(self.structure)
        pd['form'] = copy.copy(self.form)
        pd['settings'] = {}
        for stg_nm,stg in self.settings.items():
            pd['settings'][stg_nm] = copy.copy(stg)
        pd['parameters'] = {}
        for param_nm,param in self.parameters.items():
            pd['parameters'][param_nm] = copy.deepcopy(param)
        return pd

    def update_from_dict(self,d):
        if 'structure' in d:
            self.set_structure(d['structure'])
        if 'form' in d:
            self.set_form(d['form'])
        if 'settings' in d:
            self.update_settings(d['settings'])
        if 'parameters' in d:
            self.update_parameters(d['parameters'])

    @classmethod
    def from_dict(cls,d):
        inst = cls(d['structure'])
        inst.update_from_dict(d)
        return inst

    def compute_intensity(self,q,source_wavelength):
        return xrsdscat.compute_intensity(q,source_wavelength,self.structure,self.form,self.settings,self.parameters)

