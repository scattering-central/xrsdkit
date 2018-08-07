"""This package provides tools for analysis of scattering and diffraction patterns.

A scattering/diffraction pattern is assumed to represent
a material System composed of one or more Populations,
each of which is composed of one or more Species.
This module outlines a taxonomy of classes and attributes
for describing and manipulating such a System.

Developer note: This is the only module that should require revision
when extending xrsdkit to new kinds of structures and form factors.
"""
# TODO: add instructions for extending to new structures/form factors

from collections import OrderedDict

# supported structure specifications
structure_names = [\
'unidentified',\
'diffuse',\
'disordered',\
'crystalline']

# supported form factors
form_factor_names = [\
'flat',\
'atomic',\
'standard_atomic',\
'guinier_porod',\
'spherical',\
'spherical_normal']

# supported crystal structures 
crystalline_structures = ['fcc']

# supported disordered structures 
disordered_structures = ['hard_spheres']

# form factors not supported for crystalline structures 
noncrystalline_form_factors = ['spherical_normal','guinier_porod']

# supported parameters for each structure
structure_params = dict(
    unidentified = [],
    diffuse = ['I0'],
    disordered = ['I0'],
    hard_spheres = ['r_hard','v_fraction'],
    crystalline = ['I0','hwhm_g','hwhm_l'],
    fcc = ['a']
    )

# supported settings for each structure
structure_settings = dict(
    unidentified = [],
    diffuse = [],
    disordered = [],
    hard_spheres = [],
    crystalline = ['profile','q_min','q_max'],
    fcc = []
    )

# supported parameters for each form factor 
# NOTE: atomic form factors are described by
# ff = Z - 41.78214 * s**2 * sum_i(a_i*exp(-b_i*s**2)),
# where Z is the atomic number, s = sin(theta)/lambda,
# and a_i, b_i are the form factor parameters.
form_factor_params = dict(
    flat = [],
    atomic = ['Z','a0','a1','a2','a3','b0','b1','b2','b3'],
    standard_atomic = [],
    guinier_porod = ['rg','D'],
    spherical = ['r'],
    spherical_normal = ['r0','sigma']
    )

# supported settings for each form factor 
# NOTE: standard_atomic form factors are specified
# by providing the atomic symbol (as a setting)
# TODO: add settings for spherical_normal sampling resolution 
form_factor_settings = dict(
    flat = [],
    atomic = [],
    standard_atomic = ['symbol'],
    guinier_porod = [],
    spherical = [],
    spherical_normal = []
    )

# all param names
all_params = [\
'I0',\
'a','hwhm_g','hwhm_l',\
'G','rg','D',\
'r',\
'r0','sigma',\
'r_hard','v_fraction',\
'a0','a1','a2','a3','b0','b1','b2','b3'\
]

# params to model with regression:
# intensity-scaling parameters are excluded
regression_params = [
'a','hwhm_g','hwhm_l',\
'rg','D',\
'r',\
'r0','sigma',\
'r_hard','v_fraction'\
]

param_defaults = dict(
    I0 = {'value':1.,'fixed':False,'bounds':[0.,None]},
    rg = {'value':10.,'fixed':False,'bounds':[0.1,None]},
    D = {'value':4.,'fixed':True,'bounds':[0.,4.]},
    r = {'value':20.,'fixed':False,'bounds':[1.E-1,None]},
    r0 = {'value':20.,'fixed':False,'bounds':[1.E-1,None]},
    sigma = {'value':0.05,'fixed':False,'bounds':[0.,2.]},
    r_hard = {'value':20.,'fixed':False,'bounds':[1.E-1,None]},
    v_fraction = {'value':0.5,'fixed':False,'bounds':[0.01,0.7405]},
    hwhm_g = {'value':1.E-3,'fixed':False,'bounds':[1.E-9,None]},
    hwhm_l = {'value':1.E-3,'fixed':False,'bounds':[1.E-9,None]},
    a = {'value':10.,'fixed':False,'bounds':[1.E-1,None]},
    a0 = {'value':1.,'fixed':False,'bounds':[0.,None]},
    a1 = {'value':1.,'fixed':False,'bounds':[0.,None]},
    a2 = {'value':1.,'fixed':False,'bounds':[0.,None]},
    a3 = {'value':1.,'fixed':False,'bounds':[0.,None]},
    b0 = {'value':1.,'fixed':False,'bounds':[0.,None]},
    b1 = {'value':1.,'fixed':False,'bounds':[0.,None]},
    b2 = {'value':1.,'fixed':False,'bounds':[0.,None]},
    b3 = {'value':1.,'fixed':False,'bounds':[0.,None]}
    )

setting_defaults = dict(
    symbol = 'H',
    q_min = 0.,
    q_max = 1.,
    profile = 'voigt'
    )

setting_datatypes = dict(
    symbol = str,
    q_min = float,
    q_max = float,
    profile = str
    )

param_descriptions = dict(
    I0 = 'Intensity prefactor',
    G = 'Guinier-Porod model Guinier factor',
    rg = 'Guinier-Porod model radius of gyration',
    D = 'Guinier-Porod model Porod exponent',
    r = 'Radius of spherical population',
    r0 = 'Mean radius of spherical population with normal distribution of size',
    sigma = 'fractional standard deviation of radius for normally distributed sphere population',
    r_hard = 'Radius of hard-sphere potential for hard sphere (Percus-Yevick) structure factor',
    v_fraction = 'volume fraction of particles in hard sphere (Percus-Yevick) structure factor',
    hwhm_g = 'Gaussian profile half-width at half-max',
    hwhm_l = 'Lorentzian profile half-width at half-max',
    a = 'First lattice parameter',
    a0 = 'atomic form factor coefficient',
    b0 = 'atomic form factor exponent',
    a1 = 'atomic form factor coefficient',
    b1 = 'atomic form factor exponent',
    a2 = 'atomic form factor coefficient',
    b2 = 'atomic form factor exponent',
    a3 = 'atomic form factor coefficient',
    b3 = 'atomic form factor exponent'
    )

parameter_units = dict(
    I0 = 'arbitrary',
    G = 'arbitrary',
    rg = 'Angstrom',
    D = 'unitless',
    r = 'Angstrom',
    r0 = 'Angstrom',
    sigma = 'unitless',
    r_hard = 'Angstrom',
    v_fraction = 'unitless',
    hwhm_g = '1/Angstrom',
    hwhm_l = '1/Angstrom',
    a = 'Angstrom',
    a0 = 'arbitrary',
    b0 = 'arbitrary',
    a1 = 'arbitrary',
    b1 = 'arbitrary',
    a2 = 'arbitrary',
    b2 = 'arbitrary',
    a3 = 'arbitrary',
    b3 = 'arbitrary'
    )


class System(object):

    def __init__(self,populations={}):
        self.populations = populations
        self.fit_report = {}

    def to_dict(self):
        sd = {} 
        for pop_nm,pop in self.populations.items():
            sd[pop_nm] = pop.to_dict()
        return sd

    def to_ordered_dict(self):
        od = OrderedDict()
        ## Step 1: Standardize order of populations by structure and form,
        ## excluding entries for noise or unidentified structures
        for stnm in structure_names:
            for ffnm in form_factor_names:
                for pop_nm,pop in self.populations.items():
                    if not pop_nm == 'noise' \
                    and not pop.structure == 'unidentified':
                        if pop.structure == stnm \
                        and ffnm in [pop.basis[snm]['form'] for snm in pop.basis.keys()]:
                            ## Step 2: Standardize order of species by form factor
                            od[pop_nm] = pop.to_ordered_dict()
        ## Step 3: if noise or unidentified populations, put at the end
        for pop_nm,pop in self.populations.items():
            if pop_nm == 'noise' \
            or pop.structure == 'unidentified':
                od[pop_nm] = pop.to_ordered_dict() 
        return od

    def update_from_dict(self,d):
        for pop_name,pd_new in d.items():
            if not pop_name in self.populations:
                self.populations[pop_name] = Population.from_dict(pd_new) 

    @classmethod
    def from_dict(cls,d):
        inst = cls()
        inst.update_from_dict(d)
        return inst

    # TODO: incorporate fitting functionality

def structure_form_exception(structure,form):
    msg = 'structure specification {}'\
        'does not support specie specification {}- '\
        'this specie must be removed from the basis '\
        'before setting this structure'.format(structure,form)
    raise ValueError(msg)

class Population(object):

    def __init__(self,structure,settings={},parameters={},basis={}):
        # TODO: validate basis entries?
        self.set_structure(structure)
        self.update_basis(basis)
        self.update_settings(settings)
        self.update_parameters(parameters)

    def set_structure(self,structure):
        self.check_structure(structure,self.basis_to_dict())
        new_settings = dict.fromkeys(structure_settings[structure])
        for stg_nm in structure_settings[structure]:
            if stg_nm in self.settings:
                new_settings[stg_nm] = self.settings[stg_nm]
            else:
                new_settings[stg_nm] = setting_defaults[stg_nm]
        self.settings = new_settings
        new_params = dict.fromkeys(structure_params[structure])  
        for param_nm in structure_params[structure]):
            if param_nm in self.parameters:
                new_params[param_nm] = self.parameters[param_nm]
            else:
                new_params[param_nm] = Parameter.from_dict(param_defaults[param_nm])
        self.parameters = new_params

    def add_specie(self,specie_name,ff_name,settings={},parameters={},coordinates=None):
        if self.structure in crystalline_structures and ff_name in noncrystalline_form_factors:
            structure_form_exception(self.structure,ff_name)
        self.basis[specie_name] = Specie(ff_name)
        self.basis[specie_name].update_settings(settings)
        self.basis[specie_name].update_parameters(parameters)
        self.basis[specie_name].update_coordinates(coordinates)

    def to_dict(self):
        pd = {} 
        pd['structure'] = self.structure
        pd['settings'] = {}
        for stg_nm,stg in self.settings.items():
            pd['settings'][stg_nm] = stg 
        pd['parameters'] = {}
        for param_nm,param in self.parameters.items():
            pd['parameters'][param_nm] = param.to_dict()
        pd['basis'] = self.basis_to_dict()
        return pd

    def to_ordered_dict(self):
        opd = OrderedDict()
        opd['structure'] = self.structure
        opd['settings'] = OrderedDict() 
        for stg_nm in structure_settings[self.structure]:
            opd['settings'][stg_nm] = self.settings[stg_nm] 
        opd['parameters'] = OrderedDict() 
        for param_nm in structure_parameters[self.structure]:
            opd['parameters'][param_nm] = self.parameters[param_nm].to_dict()
        opd['basis'] = self.basis_to_ordered_dict()
        return opd

    def basis_to_dict(self):
        bd = {}
        for specie_nm,specie in self.basis.items():
            pd['basis'][specie_nm] = specie.to_dict()
        return bd

    def basis_to_ordered_dict(self):
        obd = OrderedDict()
        for ffnm in form_factor_names:
            for specie_nm,specd in popdef['basis'].items(): 
                # TODO: how should two species of the same form be ordered?
                if specd['form'] == ffnm:
                    obd[specie_nm] = copy.deepcopy(specd)
        return obd

    def update_from_dict(self,d):
        if 'basis' in d:
            self.update_basis(d['basis'])
        if 'structure' in d:
            self.set_structure(d['structure'])
        if 'parameters' in d:
            self.update_parameters(d['parameters'])
        if 'settings' in d:
            self.update_settings(d['settings'])

    def update_parameters(self,pd):
        for param_nm, paramd in pd.items():
            if param_nm in self.parameters:
                self.parameters[param_nm].update_from_dict(paramd)

    def update_settings(self,sd)
        for stg_nm, sval in sd.items():
            if stg_nm in self.settings:
                self.settings[stg_nm] = sval

    def update_basis(self,bd):
        self.check_structure(self.structure,bd)
            for specie_nm,specd in bd.items():
                self.basis[specie_nm] = Specie.from_dict(specd)

    @classmethod
    def from_dict(cls,d):
        inst = cls()
        inst.update_from_dict(d)
        return inst

    @staticmethod
    def check_structure(structure,basis):
        if structure in crystalline_structures:
            for site_nm,specie_def in basis.items():
                if specie_def['form'] in noncrystalline_form_factors:
                    structure_form_exception(structure,specie_def['form'])


class Specie(object):

    def __init__(self,form,settings={},parameters={},coordinates=None):
        self.set_form(form)
        self.update_settings(settings)
        self.update_parameters(parameters)
        make_coord = lambda : Parameter(0.,True,[None,None],None)  
        self.coordinates = [make_coord(),make_coord(),make_coord()]
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
        for param_nm in form_factor_params[form]):
            if param_nm in self.parameters:
                new_params[param_nm] = self.parameters[param_nm]
            else:
                new_params[param_nm] = Parameter.from_dict(param_defaults[param_nm])
        self.parameters = new_params

    def to_dict(self):
        sd = {}
        sd['form'] = self.form
        sd['settings'] = {}
        for stg_nm,stg in self.settings.items():
            sd['settings'][stg_nm] = stg 
        sd['parameters'] = {}
        for param_nm,param in self.parameters.items():
            sd['parameters'][param_nm] = param.to_dict()
        sd['coordinates'] = [None,None,None] 
        for ic,cparam in enumerate(self.coordinates):
            sd['coordinates'][ic] = cparam.to_dict()
        return sd

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
                self.parameters[param_nm].update_from_dict(paramd)

    def update_settings(self,sd)
        for stg_nm, sval in sd.items():
            if stg_nm in self.settings:
                self.settings[stg_nm] = sval

    def update_coordinates(self,coords)
        for ic, cdict in enumerate(coords):
            if cdict is not None:
                self.coordinates[ic].update_from_dict(cdict)


class Parameter(object):
    
    def __init__(self,value,fixed=False,bounds=[None,None],constraint_expr=None):
        self.value = value
        self.fixed = fixed
        self.bounds = bounds
        self.constraint_expr = constraint_expr

    def to_dict(self):
        return dict(
            value=self.value,
            fixed=self.fixed,
            bounds=self.bounds,
            constraint_expr=self.constraint_expr
            ) 
    
    def update_from_dict(self,paramd):
        if 'value' in paramd:
            self.value = paramd['value']
        if 'fixed' in paramd:
            self.fixed = paramd['fixed']
        if 'bounds' in paramd:
            self.bounds = paramd['bounds']
        if 'constraint_expr' in paramd:
            self.constraint_expr = paramd['constraint_expr']



