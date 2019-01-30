"""This module defines the settings and parameters handled by xrsdkit."""
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
    low_q_scatter = 'Flat noise floor plus a Guinier-Porod-like contribution'
    )

def validate(structure,form,settings):
    if structure in ['diffuse','disordered']:
        # polyatomic forms are not allowed- everything else is ok
        if form == 'polyatomic':
            raise ValueError('{} structure does not support polyatomic forms'.format(structure))
    elif structure == 'crystalline':
        # guinier_porod forms are not allowed
        if form == 'guinier_porod':
            raise ValueError('crystalline structure does not support guinier_porod forms')
        elif form == 'spherical':
            # crystalline structure with spherical form factor:
            # distributions of size are not allowed
            if not settings['distribution'] == 'single':
                msg = 'crystalline structure does not support size distribution {}'\
                .format(settings['distribution']))
                raise ValueError(msg)

def valid_form_factors(structure,prior_settings={}):
    diffuse = ['atomic','guinier_porod','spherical'],
    disordered = ['atomic','guinier_porod','spherical'],
    crystalline = ['atomic','spherical','polyatomic']
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
        lattice = 'P_cubic',
        space_group = '',
        structure_factor_mode = 'local',
        integration_mode = 'spherical',
        texture = 'random',
        profile = 'voigt'
        )
    )
modelable_structure_settings = dict(
    diffuse = [],
    disordered = ['interaction'],
    crystalline = ['lattice']
    )
form_settings = dict(
    atomic = {'symbol':'C'},
    polyatomic = {'n_atoms':2},
    guinier_porod = {'distribution':'single'},
    spherical = {'distribution':'single'}
    )
modelable_form_settings = dict(
    atomic = ['symbol'],
    polyatomic = ['n_atoms'],
    guinier_porod = ['distribution'],
    spherical = ['distribution']
    )

def all_settings(structure,form=None):
    """Return all valid settings, along with sensible default values.

    Parameters
    ----------
    structure : str
        Population structure designation, for fetching valid structure settings
    form : str
        Population form factor designation, for fetching valid form factor settings

    Returns
    -------
    stgs : dict
        Dict of all possible settings along with sensible default values
    """
 
    stgs = {}
    stgs.update(copy.deepcopy(structure_settings[structure]))
    if form:
        stgs.update(copy.deepcopy(form_settings[form]))
    for stg_nm,stg_val in stgs.items():
        if stg_nm == 'n_atoms':
            stgs.update(
                dict([('symbol_{}'.format(iat),'H') for iat in range(setting_value)]) 
                )
        if stg_nm == 'integration_mode':
            if stg_val == 'spherical':
                stgs.update({'q_min':0.,'q_max':1.})
        if stg_nm == 'distribution' and form == 'spherical':
            if setting_value == 'r_normal':
                stgs.update({'sampling_width':3.5,'sampling_step':0.05})
        if stg_nm == 'distribution' and form == 'guinier_porod':
            if setting_value == 'rg_normal':
                return {'sampling_width':2.0,'sampling_step':0.1}
    return stgs 

# all possible options for all settings (empty list if not enumerable)
def setting_selections(structure,form=None,prior_settings={}):
    stg_sel = {}
    if structure == 'crystalline':
        stg_sel['lattice'] = sgs.lattices
        #valid_sgs = sgs.all_space_groups.values()
        if 'lattice' in prior_settings:
            valid_sgs = sgs.lattice_space_groups[prior_settings['lattice']]
        stg_sel['space_group'] = ['']+valid_sgs
        # TODO: implement 'textured', 'single_crystal' 
        stg_sel['texture'] = ['random']
        stg_sel['profile'] = ['gaussian','lorentzian','voigt']
        stg_sel['structure_factor_mode'] = ['local','radial']
        # TODO: non-spherical integration modes for sections of q-space  
        stg_sel['integration_mode'] = ['spherical']
        stg_sel['q_min'] = []
        stg_sel['q_max'] = []
    if structure == 'disordered':
        # TODO: coulombic and square-well interactions
        stg_sel['interaction'] = ['hard_spheres']
    if form == 'atomic':
        stg_sel['symbol'] = xrff.atomic_params.keys()
    if form == 'polyatomic':
        stg_sel['n_atoms'] = [],
    if form == 'spherical':
        stg_sel['distribution'] = ['single','r_normal'],
        stg_sel['sampling_width'] = [],
        stg_sel['sampling_step'] = [],
    if form == 'guinier_porod':
        stg_sel['distribution'] = ['single','rg_normal'],
        stg_sel['sampling_width'] = []
        stg_sel['sampling_step'] = [],
    return stg_sel

# default parameters for each structure, form factor, and noise model
form_factor_params = dict(
    atomic = {},
    polyatomic = {},
    guinier_porod = dict(
        rg = {'value':10.,'fixed':False,'bounds':[0.1,None],'constraint_expr':None},
        D = {'value':4.,'fixed':True,'bounds':[0.,4.],'constraint_expr':None} 
        ),
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

# generate any additional parameters that depend on setting selections
def additional_structure_params(structure,prior_settings):
    if structure == 'disordered':
        if 'interaction' in prior_settings:
            if prior_settings['interaction'] == 'hard_spheres':
                return dict(
                    r_hard = {'value':20.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},
                    v_fraction = {'value':0.5,'fixed':False,'bounds':[0.01,0.7405],'constraint_expr':None}
                    ) 
    if structure == 'crystalline':
        if 'lattice' in prior_settings:
            if prior_settings['lattice'] in ['P_cubic','I_cubic','F_cubic','diamond','hcp']:
                return {'a':{'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None}}
            if prior_settings['lattice'] in ['hexagonal','P_tetragonal','I_tetragonal']:
                return dict(
                    a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    c = {'value':20.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None}  
                    ) 
            if prior_settings['lattice'] == 'rhombohedral':
                return dict(
                    a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    alpha = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None}  
                    ) 
            if prior_settings['lattice'] in ['P_orthorhombic','C_orthorhombic','I_orthorhombic','F_orthorhombic']:
                return dict(
                    a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    b = {'value':12.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    c = {'value':15.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None}  
                    ) 
            if prior_settings['lattice'] in ['P_monoclinic','C_monoclinic']:
                return dict(
                    a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    b = {'value':12.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    c = {'value':15.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None}, 
                    beta = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None}  
                    )
            if prior_settings['lattice'] == 'triclinic':
                return dict(
                    a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    b = {'value':12.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    c = {'value':15.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None}, 
                    alpha = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None},  
                    beta = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None}, 
                    gamma = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None} 
                    )
        if 'profile' in prior_settings:
            if prior_settings['profile'] == 'voigt':
                return dict(
                    hwhm_g = {'value':1.E-3,'fixed':False,'bounds':[1.E-9,None],'constraint_expr':None},
                    hwhm_l = {'value':1.E-3,'fixed':False,'bounds':[1.E-9,None],'constraint_expr':None}
                    )
            if prior_settings['profile'] in ['gaussian','lorentzian']:
                return {'hwhm':{'value':1.E-3},'fixed':False,'bounds':[1.E-9,None],'constraint_expr':None}

def additional_form_factor_params(form,prior_settings):
    if form == 'polyatomic':
        if 'n_atoms' in prior_settings:
            coord_params = {} 
            for iat in range(prior_settings['n_atoms']):
                coord_params.update({
                    'ra_'+str(iat): {'value':0.1*iat,'fixed':True,'bounds':[-1.,1.],'constraint_expr':None},
                    'rb_'+str(iat): {'value':0.1*iat,'fixed':True,'bounds':[-1.,1.],'constraint_expr':None},
                    'rc_'+str(iat): {'value':0.1*iat,'fixed':True,'bounds':[-1.,1.],'constraint_expr':None} 
                    })
            for iat in range(prior_settings['n_atoms']):
                coord_params.update( 
                    {'occupancy_'+str(iat):{'value':1.,'fixed':True,'bounds':[0.,1.],'constraint_expr':None}}
                    )
            return coord_params
    if form == 'guinier_porod':
        if 'distribution' in prior_settings:
            if prior_settings['distribution'] == 'rg_normal':
                return {'sigma_rg':{'value':0.05,'fixed':False,'bounds':[0.,2.],'constraint_expr':None}}
    if form == 'spherical':
        if 'distribution' in prior_settings:
            if prior_settings['distribution'] == 'r_normal':
                return {'sigma_r':{'value':0.05,'fixed':False,'bounds':[0.,2.],'constraint_expr':None}}
    return {} 

def all_params(structure,form,prior_settings):
    all_pars = {}
    if structure=='diffuse': all_pars['I0']={'value':1.,'fixed':False,'bounds':[0.,None],'constraint_expr':None}
    if structure=='disordered': all_pars['I0']={'value':100.,'fixed':False,'bounds':[0.,None],'constraint_expr':None}
    if structure=='crystalline': all_pars['I0']={'value':1.E-5,'fixed':False,'bounds':[0.,None],'constraint_expr':None}
    all_pars.update(copy.deepcopy(form_factor_params[form]))
    all_pars.update(additional_structure_params(structure,prior_settings))
    all_pars.update(additional_form_factor_params(form,prior_settings))
    return all_pars

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
    sampling_step = float
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
    sampling_width = 'Number of standard deviations sample from distribution',
    sampling_step = 'Resolution of sampling, in units of standard deviations'
    )

parameter_units = dict(
    I0 = 'arbitrary',
    rg = 'Angstrom',
    D = 'unitless',
    r = 'Angstrom',
    r0 = 'Angstrom',
    sigma_r = 'unitless',
    sigma_rg = 'unitless',
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
    sigma_r = 'fractional standard deviation of radius',
    sigma_rg = 'fractional standard deviation of radius of gyration',
    r_hard = 'Radius of hard-sphere potential for hard sphere (Percus-Yevick) structure factor',
    v_fraction = 'volume fraction of particles in hard sphere (Percus-Yevick) structure factor',
    hwhm_g = 'Gaussian profile half-width at half-max',
    hwhm_l = 'Lorentzian profile half-width at half-max',
    a = 'First lattice parameter',
    b = 'Second lattice parameter',
    c = 'Third lattice parameter',
    alpha = 'Angle between second and third lattice vectors',
    beta = 'Angle between first and third lattice vectors',
    gamma = 'Angle between first and second lattice vectors',
    ra = 'fractional coordinate along first lattice vector',
    rb = 'fractional coordinate along second lattice vector',
    rc = 'fractional coordinate along third lattice vector',
    occupancy = 'likelihood of finding an atomic specie at its lattice site'
    )

