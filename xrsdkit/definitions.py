"""This module defines the systems and parameters handled by xrsdkit.

When a new definition is added,
it must also be handled appropriately throughout the package,
and added to the database on which the models are built.
TODO: include instructions on how to support new definitions.
"""
import copy

from .scattering import space_groups as sgs
from .scattering import form_factors as xrff 

# supported structure specifications, form factors, and noise models
structure_names = [\
'diffuse',\
'disordered',\
'crystalline']
form_factor_names = [\
'atomic',\
'guinier_porod',\
'spherical',\
'spherical_normal']
noise_model_names = ['flat']
# list of form factors that do not support crystalline arrangements 
noncrystalline_form_factors = ['spherical_normal','guinier_porod']

# supported settings for each structure, form factor
structure_settings = dict(
    diffuse = [],
    disordered = ['interaction'],
    crystalline = ['lattice','centering','space_group','texture','profile',
        'structure_factor_mode','integration_mode','q_min','q_max']
    )
form_factor_settings = dict(
    atomic = ['symbol'],
    guinier_porod = [],
    spherical = [],
    spherical_normal = []   # TODO: add setting for sampling resolution
    )

# default values for all settings- set to None if no default
setting_defaults = dict(
    lattice = 'cubic',
    centering = 'P',
    space_group = '',
    texture = 'random',
    profile = 'voigt',
    structure_factor_mode = 'local',
    integration_mode = 'spherical',
    q_min = 0.,
    q_max = 1.,
    interaction = 'hard_spheres',
    symbol = 'H'
    )

# datatypes and descriptions for all settings (gui tooling)
setting_datatypes = dict(
    lattice = str,
    centering = str,
    space_group = str,
    texture = str,
    profile = str,
    structure_factor_mode = str,
    integration_mode = str,
    q_min = float,
    q_max = float,
    interaction = str,
    symbol = str
    )
setting_descriptions = dict(
    lattice = 'Name of the lattice family for crystalline populations',
    centering = 'Crystalline lattice centering specifier',
    space_group = 'Crystalline space group specification (International symbol)',
    texture = 'Distribution of orientations for crystalline populations',
    profile = 'Selection of peak profile for broadening diffraction peaks',
    structure_factor_mode = 'Strategy for computing off-peak structure factors',
    integration_mode = 'Strategy for integrating over the reciprocal lattice',
    q_min = 'minimum q-value for reciprocal space integration',
    q_max = 'maximum q-value for reciprocal space integration', 
    interaction = 'Interaction potential describing disordered populations',
    symbol = 'Atomic symbol'
    )

# all possible options for all settings
# TODO: descriptions of each option
setting_selections = dict(
    lattice = ['cubic','hexagonal','rhombohedral','tetragonal','orthorhombic','monoclinic','triclinic'],
    centering = ['P','F','I','C','HCP'],
    space_group = ['']+sgs.all_space_groups,
    texture = ['random'],               # TODO: implement 'textured', 'single_crystal'
    integration_mode = ['spherical'],   # TODO: non-spherical integration modes for sections of q-space 
    interaction = ['hard_spheres'],     # TODO: coulombic sphere sf, any others that are analytical
    symbol = xrff.atomic_params.keys(),
    profile = ['gaussian','lorentzian','voigt'],
    structure_factor_mode = ['local','radial']
    )

# numerical parameters for each structure, form factor, and noise model
structure_params = dict(
    diffuse = ['I0'],
    disordered = ['I0'],
    crystalline = ['I0','hwhm_g','hwhm_l']
    )
form_factor_params = dict(
    atomic = [],
    guinier_porod = ['rg','D'],
    spherical = ['r'],
    spherical_normal = ['r0','sigma']
    )
noise_params = dict(flat = ['I0'])

# params whose existence depends on a setting selection
setting_params = dict(
    interaction = dict(hard_spheres=['r_hard','v_fraction']),
    lattice = dict( 
        cubic = ['a'],
        hexagonal = ['a','c'],
        rhombohedral = ['a','alpha'],
        tetragonal = ['a','c'],
        orthorhombic = ['a','b','c'],
        monoclinic = ['a','b','c','beta'],
        triclinic = ['a','b','c','alpha','beta','gamma']
        ) 
    )

param_defaults = dict(
    I0 = {'value':1.,'fixed':False,'bounds':[0.,None],'constraint_expr':None},
    rg = {'value':10.,'fixed':False,'bounds':[0.1,None],'constraint_expr':None},
    D = {'value':4.,'fixed':True,'bounds':[0.,4.],'constraint_expr':None},
    r = {'value':20.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},
    r0 = {'value':20.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},
    sigma = {'value':0.05,'fixed':False,'bounds':[0.,2.],'constraint_expr':None},
    r_hard = {'value':20.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},
    v_fraction = {'value':0.5,'fixed':False,'bounds':[0.01,0.7405],'constraint_expr':None},
    hwhm_g = {'value':1.E-3,'fixed':False,'bounds':[1.E-9,None],'constraint_expr':None},
    hwhm_l = {'value':1.E-3,'fixed':False,'bounds':[1.E-9,None],'constraint_expr':None},
    a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},
    b = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},
    c = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},
    alpha = {'value':90.,'fixed':False,'bounds':[0.,180.],'constraint_expr':None},
    beta = {'value':90.,'fixed':False,'bounds':[0.,180.],'constraint_expr':None},
    gamma = {'value':90.,'fixed':False,'bounds':[0.,180.],'constraint_expr':None}
    )

noise_param_defaults = dict(
    I0 = {'value':1.E-6,'fixed':False,'bounds':[1.E-12,None],'constraint_expr':None},
    )

coord_default = {'value':0.,'fixed':True,'bounds':[-1.,1.],'constraint_expr':None}

parameter_descriptions = dict(
    I0 = 'Intensity prefactor',
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
    b = 'Second lattice parameter',
    c = 'Third lattice parameter',
    alpha = 'Angle between second and third lattice vectors',
    beta = 'Angle between first and third lattice vectors',
    gamma = 'Angle between first and second lattice vectors'
    )

parameter_units = dict(
    I0 = 'arbitrary',
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
    b = 'Angstrom',
    c = 'Angstrom',
    alpha = 'degree',
    beta = 'degree',
    gamma = 'degree'
    )

