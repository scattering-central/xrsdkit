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

from .population import Population
from .specie import Specie
from .parameter import Parameter

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

# form factors not supported for crystalline structures 
noncrystalline_form_factors = ['spherical_normal','guinier_porod']

# supported parameters for each structure
structure_params = dict(
    unidentified = [],
    diffuse = ['I0'],
    disordered = ['I0'],
    crystalline = ['I0','hwhm_g','hwhm_l']
    )

# supported settings for each structure
structure_settings = dict(
    unidentified = [],
    diffuse = [],
    disordered = ['interaction'],
    crystalline = ['lattice','profile','q_min','q_max'],
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

# supported disordered structures 
disordered_structures = ['hard_spheres']

# supported crystal structures 
crystalline_structures = ['fcc']

# supported disordered and crystalline structure params
disordered_structure_params = dict(
    hard_spheres = ['r_hard','v_fraction']
    )
crystalline_structure_params = dict(
    fcc = ['a']
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
    lattice = 'fcc',
    interaction = 'hard_spheres',
    symbol = 'H',
    q_min = 0.,
    q_max = 1.,
    profile = 'voigt'
    )

setting_datatypes = dict(
    lattice = str,
    interaction = str,
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

def structure_form_exception(structure,form):
    msg = 'structure specification {}'\
        'does not support specie specification {}- '\
        'this specie must be removed from the basis '\
        'before setting this structure'.format(structure,form)
    raise ValueError(msg)


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


