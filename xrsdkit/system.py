from collections import OrderedDict

from . import structure_settings, form_factor_settings, \
    structure_params, form_factor_params, \
    param_defaults, setting_defaults, param_bound_defaults, fixed_param_defaults


class System(object):

    def __init__(self,populations={}):
        self.populations = populations
        self.fit_report = {}

    def as_dict(self):
        sd = {} 
        for pop_nm,pop in self.populations.items():
            sd[pop_nm] = pop.as_dict()

    # TODO: incorporate fitting functionality here


class Population(object):

    def __init__(self,structure,settings={},parameters={},basis={}):
        # TODO: check structure and keys of settings and params:
        # warn or raise exception if unexpected values
        self.structure = structure
        stgs = dict.fromkeys(structure_settings[structure])
        for stg_nm in structure_settings[structure]:
            if stg_nm in settings:
                stgs[stg_nm] = settings[stg_nm]
            else:
                stgs[stg_nm] = setting_defaults[stg_nm]
        self.settings = stgs
        params = dict.fromkeys(structure_params[structure])
        for param_nm in structure_params[structure]):
            if param_nm in parameters:
                # TODO: validate parameter entries?
                params[param_nm] = parameters[param_nm]
            else:
                params[param_nm] = Parameter(param_defaults[param_nm],fixed_param_defaults[param_nm],param_bound_defaults[param_nm],None)
        self.parameters = params
        # TODO: validate basis entries?
        self.basis = basis

    def as_dict(self):
        pd = {} 
        pd['structure'] = self.structure
        pd['settings'] = {}
        for stg_nm,stg in self.settings.items():
            pd['settings'][stg_nm] = stg 
        pd['parameters'] = {}
        for param_nm,param in self.parameters.items():
            pd['parameters'][param_nm] = param.as_dict()
        pd['basis'] = {}
        for site_nm,site in self.basis.items():
            pd['basis'][site_nm] = site.as_dict()


class Site(object):

    def __init__(self,form,settings={},parameters={},coordinates=None):
        # TODO: check form and keys of settings and params:
        # warn or raise exception if unexpected values
        self.form = form
        stgs = dict.fromkeys(form_factor_settings[form])
        for stg_nm in form_factor_settings[form]:
            if stg_nm in settings:
                stgs[stg_nm] = settings[stg_nm]
            else:
                stgs[stg_nm] = setting_defaults[stg_nm]
        self.settings = stgs
        params = dict.fromkeys(form_factor_params[form])
        for param_nm in form_factor_params[form]):
            if param_nm in parameters:
                # TODO: validate parameter entries?
                params[param_nm] = parameters[param_nm]
            else:
                params[param_nm] = Parameter(param_defaults[param_nm],fixed_param_defaults[param_nm],param_bound_defaults[param_nm],None)
        self.parameters = params
        if coordinates is None:
            make_coord = lambda : Parameter(param_defaults['coordinates'],fixed_param_defaults['coordinates'],param_bound_defaults['coordinates'],None)  
            self.coordinates = [make_coord(),make_coord(),make_coord()]
        else:
            # TODO: validate coordinate entries?
            self.coordinates = coordinates

    def as_dict(self):
        sd = {}
        sd['form'] = self.form
        sd['settings'] = {}
        for stg_nm,stg in self.settings.items():
            sd['settings'][stg_nm] = stg 
        sd['parameters'] = {}
        for param_nm,param in self.parameters.items():
            sd['parameters'][param_nm] = param.as_dict()
        sd['coordinates'] = [None,None,None] 
        for ic,cparam in enumerate(self.coordinates):
            sd['coordinates'][ic] = cparam.as_dict()


class Parameter(object):
    
    def __init__(self,value,fixed=False,bounds=None,constraint_expr=None):
        self.value = value
        self.fixed = fixed
        self.bounds = bounds
        self.constraint_expr = constraint_expr

    def as_dict(self):
        return dict(
            value=self.value,
            fixed=self.fixed,
            bounds=self.bounds,
            constraint_expr=self.constraint_expr
            ) 


