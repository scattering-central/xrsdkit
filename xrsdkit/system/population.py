import copy
from collections import OrderedDict

import numpy as np

from .. import definitions as xrsdefs 
from .. import scattering as xrsdscat

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
        current_settings = copy.deepcopy(self.settings)
        primary_settings = copy.deepcopy(xrsdefs.structure_settings[self.structure])
        # edge case: self.form is None during __init__.
        if self.form: primary_settings.update(copy.deepcopy(xrsdefs.form_settings[self.form]))
        # replace any primary_setting values with the new_settings
        for stgnm in primary_settings.keys():
            if stgnm in current_settings:
                primary_settings[stgnm] = current_settings[stgnm]
            if stgnm in new_settings:
                primary_settings[stgnm] = new_settings[stgnm]
        # fetch the secondary settings corresponding to these primary settings
        secondary_settings = xrsdefs.secondary_settings(self.structure,self.form,primary_settings)
        for stgnm in secondary_settings.keys():
            if stgnm in current_settings:
                secondary_settings[stgnm] = current_settings[stgnm]
            if stgnm in new_settings:
                secondary_settings[stgnm] = new_settings[stgnm]
        # form a dict of all settings, primary plus secondary
        all_settings = copy.deepcopy(primary_settings)
        all_settings.update(secondary_settings)
        # ensure validity
        xrsdefs.validate(self.structure,self.form,all_settings)
        # remove any self.settings not in all_settings
        for stg_nm in current_settings.keys():
            if not stg_nm in all_settings:
                self.settings.pop(stg_nm)
        # update self.settings
        self.settings.update(all_settings)
        # update self.parameters to respect the new settings
        self.update_parameters()

    def update_parameters(self,new_params={}):
        valid_params = xrsdefs.all_params(self.structure,self.form,self.settings)
        for param_nm in new_params:
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
            valid_params[param_nm].update(param_def)
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
        stgs = {}
        if 'settings' in d: stgs = d['settings']
        params = {}
        if 'parameters' in d: params = d['parameters']
        inst = cls(d['structure'],d['form'],stgs,params)
        return inst

    def compute_intensity(self,q,source_wavelength):
        return xrsdscat.compute_intensity(q,source_wavelength,self.structure,self.form,self.settings,self.parameters)

