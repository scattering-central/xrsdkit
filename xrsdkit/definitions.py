"""This module defines the systems and parameters handled by xrsdkit.

When a new definition is added,
it must also be handled appropriately throughout the package,
and added to the database on which the models are built.
TODO: include instructions on how to support new definitions.
"""
from .scattering import space_groups as sgs
from .scattering import form_factors as xrff 

# supported structures, forms, and noise models
structures = dict(
    diffuse = 'disordered, non-interacting particles',
    disordered = 'disordered, interacting particles',
    crystalline = 'particles arranged in a lattice'
    )
forms = dict( 
    atomic = 'Single atom',
    polyatomic = 'Multiple atoms',
    guinier_porod = 'Scatterer described by Guinier-Porod equations',
    spherical = 'Spherical particle'
    )
noise_models = dict(
    flat = 'Flat noise floor for all q',
    low_q_scatter = 'Flat noise floor plus a Porod-like contribution'
    )

structure_names = list(structures.keys())
form_names = list(forms.keys())
noise_model_names = list(noise_models.keys())

# supported settings for each structure, form factor,
# along with default values
structure_settings = dict(
    diffuse = {},
    disordered = {'interaction':'hard_spheres'},
    crystalline = dict(
        lattice = 'cubic',
        centering = 'P',
        space_group = '',
        structure_factor_mode = 'local',
        integration_mode = 'spherical',
        texture = 'random',
        profile = 'voigt'
        )
    )
form_settings = dict(
    atomic = {'symbol':'C'},
    polyatomic = {'n_atoms':2},
    guinier_porod = {'rg_distribution':'rg_single'},
    spherical = {'r_distribution':'r_single'}
    )

# default parameters for each structure, form factor, and noise model
structure_params = dict(
    diffuse = {'I0':{'value':1.,'fixed':False,'bounds':[0.,None],'constraint_expr':None}},
    disordered = {'I0':{'value':100.,'fixed':False,'bounds':[0.,None],'constraint_expr':None}},
    crystalline = {'I0':{'value':1.E-5,'fixed':False,'bounds':[0.,None],'constraint_expr':None}}  
    )
form_factor_params = dict(
    atomic = {},
    polyatomic = {},
    guinier_porod = dict(
        rg = {'value':10.,'fixed':False,'bounds':[0.1,None],'constraint_expr':None},
        D = {'value':4.,'fixed':True,'bounds':[0.,4.],'constraint_expr':None} 
        )
    spherical = {'r':{'value':20.,'fixed':False,'bounds':[0.1,None],'constraint_expr':None}}
    )
noise_params = dict(
    flat = {'I0':{'value':1.E-3,'fixed':False,'bounds':[0.,None],'constraint_expr':None}},
    low_q_scatter = dict(
        I0 = {'value':100,'fixed':False,'bounds':[0.,None],'constraint_expr':None},
        I0_flat_fraction = {'value':0.01,'fixed':False,'bounds':[0.,1.],'constraint_expr':None},
        effective_rg = {'value':100.,'fixed':True,'bounds':[0.,None],'constraint_expr':None},
        effective_D = {'value':4.,'fixed':True,'bounds':[0.,4.],'constraint_expr':None} 
        )
    )

# generate any additional default settings that depend on setting selections
def additional_settings(setting_name,setting_value):
    if setting_name == 'n_atoms':
        return dict([('symbol{}'.format(iat),'H') for iat in range(setting_value)])
    if setting_name == 'integration_mode':
        if setting_value == 'spherical':
            return {'q_min':0.,'q_max':1.}
    if setting_name == 'r_distribution':
        if setting_value == 'r_normal':
            return {'n_samples':80,'sampling_width':3.5}
    if setting_name == 'rg_distribution':
        if setting_value == 'rg_normal':
            return {'n_samples':40,'sampling_width':2.0}
    return {} 

# all possible options for all settings (empty list if not enumerable)
# TODO: descriptions of each option
setting_selections = dict(
    lattice = ['cubic','hexagonal',\
        'rhombohedral','tetragonal','orthorhombic',\
        'monoclinic','triclinic',\
        'diamond','hcp'\
        ],
    centering = ['P','F','I','C'],
    space_group = ['']+sgs.all_space_groups,
    # TODO: implement textures 'textured', 'single_crystal' 
    texture = ['random'],
    profile = ['gaussian','lorentzian','voigt'],
    structure_factor_mode = ['local','radial'],
    # TODO: non-spherical integration modes for sections of q-space  
    integration_mode = ['spherical'],
    q_min = [],
    q_max = [],
    # TODO: coulombic and square-well interactions
    interaction = ['hard_spheres'],
    symbol = xrff.atomic_params.keys(),
    n_atoms = [],
    r_distribution = ['r_single','r_normal'],
    rg_distribution = ['rg_single','rg_normal'],
    n_samples = []
    sampling_width = []
    )

# generate any additional default parameters that depend on setting selections
def additional_params(setting_name,setting_value):
    if setting_name == 'interaction':
        if setting_value == 'hard_spheres':
            return dict(
                r_hard = {'value':20.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},
                v_fraction = {'value':0.5,'fixed':False,'bounds':[0.01,0.7405],'constraint_expr':None}
                ) 
    if setting_name == 'lattice':
        if setting_value in ['cubic','diamond','hcp']:
            return {'a':{'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None}}
        if setting_value in ['hexagonal','tetragonal']:
            return dict(
                a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                c = {'value':20.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None}  
                ) 
        if setting_value == 'rhombohedral':
            return dict(
                a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                alpha = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None}  
                ) 
        if setting_value == 'orthorhombic':
            return dict(
                a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                b = {'value':12.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                c = {'value':15.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None}  
                ) 
        if setting_value == 'monoclinic':
            return dict(
                a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                b = {'value':12.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                c = {'value':15.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None}, 
                beta = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None}  
                )
        if setting_value == 'triclinic':
            return dict(
                a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                b = {'value':12.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                c = {'value':15.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None}, 
                alpha = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None},  
                beta = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None}, 
                gamma = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None} 
                )
    if setting_name == 'profile':
        if setting_name == 'voigt':
            return dict(
                hwhm_g = {'value':1.E-3,'fixed':False,'bounds':[1.E-9,None],'constraint_expr':None},
                hwhm_l = {'value':1.E-3,'fixed':False,'bounds':[1.E-9,None],'constraint_expr':None}
                )
        if setting_name in ['gaussian','lorentzian']:
            return {'hwhm':'value':1.E-3,'fixed':False,'bounds':[1.E-9,None],'constraint_expr':None}
    if setting_name == 'n_atoms':
        coord_params = {} 
        for iat in range(setting_value):
            coord_params.update({
                'x'+str(iat): {'value':0.1*iat,'fixed':True,'bounds':[-1.,1.],'constraint_expr':None},
                'y'+str(iat): {'value':0.1*iat,'fixed':True,'bounds':[-1.,1.],'constraint_expr':None},
                'z'+str(iat): {'value':0.1*iat,'fixed':True,'bounds':[-1.,1.],'constraint_expr':None} 
                })
        for iat in range(setting_value):
            coord_params.update( 
                {'occupancy'+str(iat):{'value':1.,'fixed':True,'bounds':[0.,1.],'constraint_expr':None}}
                )
        return coord_params
    if setting_name == 'rg_distribution':
        if setting_value == 'normal':
            return {'sigma_rg':{'value':0.05,'fixed':False,'bounds':[0.,2.],'constraint_expr':None}}
    if setting_name == 'r_distribution']:
        if setting_value == 'normal':
            return {'sigma_r':{'value':0.05,'fixed':False,'bounds':[0.,2.],'constraint_expr':None}}
    return {} 

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
    symbol = str,
    n_atoms = int,
    rg_distribution = str,
    r_distribution = str,
    sampling_width = float, 
    n_samples = int
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
    symbol = 'Atomic symbol',
    n_atoms = 'Number of atoms',
    r_distribution = 'Probability distribution for parameter r',
    rg_distribution = 'Probability distribution for parameter rg',
    n_samples = 'Number of values to sample from distribution'
    sampling_width = 'Number of standard deviations sample from distribution'
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

