"""This package provides tools for analysis of scattering and diffraction patterns."""

from collections import OrderedDict
import copy

import numpy as np
# TODO: add instructions for extending to new structures/form factors

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

# form factors not supported for crystalline structures 
noncrystalline_form_factors = ['spherical_normal','guinier_porod']

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
    a0 = {'value':1.,'fixed':False,'bounds':[0.,None],'constraint_expr':None},
    a1 = {'value':1.,'fixed':False,'bounds':[0.,None],'constraint_expr':None},
    a2 = {'value':1.,'fixed':False,'bounds':[0.,None],'constraint_expr':None},
    a3 = {'value':1.,'fixed':False,'bounds':[0.,None],'constraint_expr':None},
    b0 = {'value':1.,'fixed':False,'bounds':[0.,None],'constraint_expr':None},
    b1 = {'value':1.,'fixed':False,'bounds':[0.,None],'constraint_expr':None},
    b2 = {'value':1.,'fixed':False,'bounds':[0.,None],'constraint_expr':None},
    b3 = {'value':1.,'fixed':False,'bounds':[0.,None],'constraint_expr':None}
    )

coord_default = {'value':0.,'fixed':True,'bounds':[-1.,1.],'constraint_expr':None}

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

