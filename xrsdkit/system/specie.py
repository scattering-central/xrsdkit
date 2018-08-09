import copy

from .. import *

class Specie(object):

    def __init__(self,form,settings={},parameters={},coordinates=[]):
        self.form = None
        self.settings = {}
        self.parameters = {}
        make_coord = lambda : {'value':0.,'fixed':True,'bounds':[-1.,1.],'constraint_expr':None}  
        self.coordinates = [make_coord(),make_coord(),make_coord()]
        self.set_form(form)
        self.update_settings(settings)
        self.update_parameters(parameters)
        self.update_coordinates(coordinates)

    def set_form(self,form):
        new_settings = dict.fromkeys(form_factor_settings[form])
        for stg_nm in form_factor_settings[form]:
            if stg_nm in self.settings:
                new_settings[stg_nm] = self.settings[stg_nm]
            else:
                new_settings[stg_nm] = setting_defaults[stg_nm]
        self.settings = new_settings 
        new_params = dict.fromkeys(form_factor_params[form])
        for param_nm in form_factor_params[form]:
            if param_nm in self.parameters:
                new_params[param_nm] = self.parameters[param_nm]
            else:
                new_params[param_nm] = copy.deepcopy((param_defaults[param_nm]))
        self.parameters = new_params
        self.form = form

    def to_dict(self):
        sd = {}
        sd['form'] = self.form
        sd['settings'] = {}
        for stg_nm,stg in self.settings.items():
            sd['settings'][stg_nm] = stg 
        sd['parameters'] = {}
        for param_nm,param in self.parameters.items():
            sd['parameters'][param_nm] = copy.deepcopy(param)
        sd['coordinates'] = [None,None,None] 
        for ic,cparam in enumerate(self.coordinates):
            sd['coordinates'][ic] = copy.deepcopy(cparam)
        return sd

    @classmethod
    def from_dict(cls,d):
        inst = cls(d['form'])
        inst.update_from_dict(d)
        return inst

    def update_from_dict(self,d):
        if 'form' in d:
            self.set_form(d['form'])
        if 'parameters' in d:
            self.update_parameters(d['parameters'])
        if 'settings' in d:
            self.update_settings(d['settings'])
        if 'coordinates' in d:
            self.update_coordinates(d['coordinates'])

    def update_parameters(self,pd):
        for param_nm, paramd in pd.items():
            if param_nm in self.parameters:
                self.parameters[param_nm].update(paramd)

    def update_settings(self,sd):
        for stg_nm, sval in sd.items():
            if stg_nm in self.settings:
                self.settings[stg_nm] = sval

    def update_coordinates(self,coords):
        for ic, cdict in enumerate(coords):
            if cdict is not None:
                self.coordinates[ic].update(cdict)


