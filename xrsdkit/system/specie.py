import copy

from .. import definitions as xrsdefs 

class Specie(object):

    def __init__(self,form,settings={},parameters={},coordinates=[]):
        self.form = None
        self.settings = {}
        self.parameters = {}
        self.coordinates = [copy.deepcopy(xrsdefs.coord_default),\
        copy.deepcopy(xrsdefs.coord_default),copy.deepcopy(xrsdefs.coord_default)]
        self.set_form(form)
        self.update_settings(settings)
        self.update_parameters(parameters)
        self.update_coordinates(coordinates)

    def set_form(self,form):
        self.form = form
        self.update_settings()
        self.update_parameters()

    def update_settings(self,new_settings={}):
        current_stg_nms = list(self.settings.keys())
        for stg_nm in current_stg_nms:
            if not stg_nm in xrsdefs.form_factor_settings[self.form]:
                self.settings.pop(stg_nm)
        for stg_nm in xrsdefs.form_factor_settings[self.form]:
            if stg_nm in new_settings:
                self.update_setting(stg_nm,new_settings[stg_nm])
            elif not stg_nm in self.settings: 
                self.update_setting(stg_nm,xrsdefs.setting_defaults[stg_nm])

    def update_setting(self,stg_nm,new_val):
        self.settings[stg_nm] = new_val

    def update_parameters(self,new_params={}):
        current_param_nms = list(self.parameters.keys())
        valid_param_nms = copy.deepcopy(xrsdefs.form_factor_params[self.form])
        # remove any non-valid params
        for param_nm in current_param_nms:
            if not param_nm in valid_param_nms:
                self.parameters.pop(param_nm)
        # add any missing params, then update from new_params if available 
        for param_nm in valid_param_nms:
            if not param_nm in self.parameters:
                self.parameters[param_nm] = copy.deepcopy(xrsdefs.param_defaults[param_nm]) 
            if param_nm in new_params:
                self.update_parameter(param_nm,new_params[param_nm])

    def update_parameter(self,param_nm,param_dict):
        self.parameters[param_nm].update(param_dict)

    def to_dict(self):
        sd = {}
        sd['form'] = copy.copy(self.form)
        sd['settings'] = {}
        for stg_nm,stg in self.settings.items():
            sd['settings'][stg_nm] = copy.copy(stg) 
        sd['parameters'] = {}
        for param_nm,param in self.parameters.items():
            sd['parameters'][param_nm] = copy.deepcopy(param)
        sd['coordinates'] = [None,None,None] 
        for ic,cdict in enumerate(self.coordinates):
            sd['coordinates'][ic] = copy.deepcopy(cdict)
        return sd

    @classmethod
    def from_dict(cls,d):
        inst = cls(d['form'])
        inst.update_from_dict(d)
        return inst

    def update_from_dict(self,d):
        if 'form' in d:
            self.set_form(d['form'])
        if 'settings' in d:
            self.update_settings(d['settings'])
        if 'parameters' in d:
            self.update_parameters(d['parameters'])
        if 'coordinates' in d:
            self.update_coordinates(d['coordinates'])

    def update_coordinates(self,coords):
        for ic, cdict in enumerate(coords):
            if cdict is not None:
                self.update_coordinate(ic,cdict)

    def update_coordinate(self,coord_idx,param_dict):
        self.coordinates[coord_idx].update(param_dict)


