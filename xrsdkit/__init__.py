"""Computation and analysis of scattering and diffraction patterns.

A scattering/diffraction pattern is assumed to represent
one or more populations of scattering objects.
The populations are described by a dict,
where each population has a name (key)
and a sub-dict of parameters (value).
Each population sub-dict should have the following entries: 

    - 'structure' : the structure of the population 
        (e.g. 'diffuse', 'fcc'). 

    - 'settings' : dict defining settings for 
        the computational treatment of the population,
        such as peak profile specifications, 
        atomic symbols, and q-limits for reciprocal space analysis.
        Settings are not intended to vary during optimization.

        - 'q_min' : minimum q-value for reciprocal lattice analysis 
        - 'q_max' : maximum q-value for reciprocal lattice analysis 
        - 'profile' : 'gaussian', 'lorentzian', or 'voigt' 

    - 'parameters' : dict describing the structure (lattice parameters, etc)
        as well as any other scalar parameters used for the computation.
        Parameters are intended to be varied continuously,
        e.g. during optimization problems.
        Some keys are used for parameterizing intensities and diffraction peaks:

        - 'I0' : the scattering or diffraction computed for each population 
            is multiplied by this intensity prefactor,
            assumed equal to 1 if not provided
        - 'hwhm_g' : half-width at half max of Gaussian functions 
        - 'hwhm_l' : half-width at half max of Lorentzian functions 
        - 'q_center' : center q-value for describing single 'disordered' peaks
    
        Other keys are used for structural parameters:

        - 'a', 'b', 'c' : a, b, and c lattice parameters
        - 'alpha' : angle between b and c lattice vectors
        - 'beta' : angle between a and c lattice vectors
        - 'gamma' : angle between a and b lattice vectors

    - 'basis' : dict containing site names (as keys)
        and dicts specifying site location and content (as values).
        The site content dicts are structured as:

        - 'coordinates' : list of three floating point numbers,
            the fractional coordinates relative to a lattice site.

        - The remaining entries are form factor specifiers,
            referring to dicts or lists of dicts 
            containing parameter names and values for that form factor.
            A list of dicts is used to specify 
            multiple scatterers of the same type,
            e.g. for implementing fractional occupancies.

The supported structure factors and their parameters are: 

    - 'diffuse' : a diffuse (or dilute), 
        non-interfering scattering ensemble.
        This structure factor has no parameters,
        except for its intensity prefactor.

    - 'hard_spheres' : condensed (interacting) hard spheres,
        computed via the Percus-Yevick
        solution to the Ornstein-Zernike equation. 
        TODO: Cite
        (Ornstein-Zernike, 1918; Percus-Yevick, 1958; Hansen-McDonald, 1986, Hammouda)

        - 'r_hard' : hard sphere radius 
        - 'v_fraction' : volume fraction of spheres

    - 'disordered' : condensed, disordered material, 
        computed as a single peak, whose profile
        is specified by the 'profile' setting.

        - 'q_center' : q-location of the single peak 
        - 'hwhm_g' : Gaussian half-width at half-max
        - 'hwhm_l' : Lorentzian half-width at half-max

    - 'fcc' : crystalline fcc lattice; 
        peaks are computed according to 
        the settings 'profile', 'q_min', and 'q_max'.

        - 'a' : cubic lattice parameter 
        - 'hwhm_g' : Gaussian half-width at half-max
        - 'hwhm_l' : Lorentzian half-width at half-max
        
    - all structures:

      - 'I0': intensity prefactor 


The supported form factors and their parameters are:

    - 'flat': a flat form factor for all q,
        implemented for simulating a noise floor.
        This form factor introduces no parameters.

    - 'guinier_porod': scatterer populations described 
        by the Guinier-Porod equations.

      - 'G': Guinier prefactor 
      - 'rg': radius of gyration 
      - 'D': Porod exponent 

    - 'spherical': solid spheres 

      - 'r': sphere radius (Angstrom) 

    - 'spherical_normal': solid spheres 
        with a normal size distribution.

      - 'r0': mean sphere radius (Angstrom) 
      - 'sigma': fractional standard deviation of radius

    - 'atomic': atomic form factors described by
        ff = Z - 41.78214 * s**2 * sum_i(a_i*exp(-b_i*s**2)),
        where Z is the atomic number,
        s = sin(theta)/lambda,
        and a_i, b_i are the form factor parameters.
        These scatterers respect the following settings:

      - 'symbol': Atomic symbol (as on the periodic table),
        for using the standard scattering parameters
        (see atomic_scattering_parameters.yaml).
        Atomic sites outside the standard parameter set
        must provide a setting for the atomic number (Z),
        and up to four pairs of scattering parameters (a, b).
      - 'Z': atomic number

      The parameters for atomic scatterers are:

      - 'a0', 'a1', 'a2', 'a3': a coefficients 
      - 'b0', 'b1', 'b2', 'b3': b coefficients 

    - all form factors:

      - 'occupancy': occupancy fraction, used for basis sites 
        with multiple fractional occupancies.
        If not specified, 
        an occupancy of 1 is assumed.

For example, a single flat scatterer 
is placed in a 40-Angstrom fcc lattice,
with peaks from q=0.1 to q=1.0 
included in the summation:

my_populations = dict(
    my_fcc_population = dict(
        structure='fcc',
        settings=dict(
            q_min=0.1,
            q_max=1.,
            profile='voigt'
            ),
        parameters=dict(
            a=40.,
            hwhm_g=0.01,
            hwhm_l=0.01
            ),
        basis=dict(
            my_flat_scatterer=dict(
                coordinates=[0,0,0],
                flat={}
                )
            )
        )
    )
"""
from collections import OrderedDict

# list of allowed structure specifications
structure_names = [\
'unidentified',\
'hard_spheres',\
'diffuse',\
'disordered',\
'fcc']

# list of allowed form factors:
form_factor_names = [\
'flat',\
'guinier_porod',\
'spherical',\
'spherical_normal',\
'atomic']

# list of structures that are crystalline
crystalline_structure_names = ['fcc']

# list of form factors that are non-crystalline
noncrystalline_ff_names = ['spherical_normal','guinier_porod']

# list of allowed parameters for each structure
structure_params = OrderedDict.fromkeys(structure_names)
for nm in structure_names: structure_params[nm] = ['I0']
structure_params['unidentified'] = []
structure_params['hard_spheres'].extend(['r_hard','v_fraction'])
structure_params['disordered'].extend(['q_center','hwhm_g','hwhm_l'])
structure_params['fcc'].extend(['a','hwhm_g','hwhm_l'])

# list of allowed settings for each structure
structure_settings = OrderedDict.fromkeys(structure_names)
for nm in structure_settings: structure_settings[nm] = []
structure_settings['fcc'].extend(['profile','q_min','q_max'])
structure_settings['disordered'].extend(['profile'])

# list of allowed parameters for each form factor 
form_factor_params = OrderedDict.fromkeys(form_factor_names)
for nm in form_factor_names: form_factor_params[nm] = ['occupancy']
form_factor_params['guinier_porod'].extend(['G','rg','D'])
form_factor_params['spherical'].extend(['r'])
form_factor_params['spherical_normal'].extend(['r0','sigma'])
form_factor_params['atomic'].extend(['a0','a1','a2','a3','b0','b1','b2','b3'])

# list of allowed settings for each form factor 
form_factor_settings = OrderedDict.fromkeys(form_factor_names)
for nm in form_factor_settings: form_factor_settings[nm] = []
form_factor_settings['atomic'].extend(['symbol','Z'])

all_params = [\
'I0','occupancy',\
'coordinates',\
'a',\
'G','rg','D',\
'r',\
'r0', 'sigma',\
'r_hard','v_fraction',\
'hwhm_g','hwhm_l','q_center',\
'a0','a1','a2','a3','b0','b1','b2','b3']

param_defaults = OrderedDict(
    I0 = 1.E-3,
    occupancy = 1.,
    coordinates = 0.,
    G = 1.,
    rg = 10.,
    D = 4.,
    r = 20.,
    r0 = 20.,
    sigma = 0.05,
    r_hard = 20.,
    v_fraction = 0.1,
    hwhm_g = 1.E-3,
    hwhm_l = 1.E-3,
    q_center = 1.E-1,
    a = 10.,
    Z = 1.,
    a0=1.,a1=1.,a2=1.,a3=1.,
    b0=1.,b1=1.,b2=1.,b3=1.)

setting_defaults = OrderedDict(
    Z = 1,
    symbol = 'H',
    q_min = 0.,
    q_max = 1.,
    profile = 'voigt')

setting_datatypes = OrderedDict(
    Z = int,
    symbol = str,
    q_min = float,
    q_max = float,
    profile = str)

param_bound_defaults = OrderedDict(
    I0 = [0.,None],
    occupancy = [0.,1.],
    coordinates = [None,None],
    G = [0.,None],
    rg = [1.E-1,None],
    D = [0.,4.],
    r = [1.E-1,None],
    r0 = [1.E-1,None],
    sigma = [0.,0.5],
    r_hard = [1.E-1,None],
    v_fraction = [0.05,0.7405],
    hwhm_g = [1.E-6,None],
    hwhm_l = [1.E-6,None],
    q_center = [0.,None],
    a = [0.,None],
    Z = [0.,120],
    a0=[0.,None],a1=[0.,None],a2=[0.,None],a3=[0.,None],
    b0=[0.,None],b1=[0.,None],b2=[0.,None],b3=[0.,None])

fixed_param_defaults = OrderedDict(
    I0 = False,
    occupancy = True,
    coordinates = True,
    G = False,
    rg = False,
    D = False,
    r = False,
    r0 = False,
    sigma = False,
    r_hard = False,
    v_fraction = False,
    hwhm_g = False,
    hwhm_l = False,
    q_center = False,
    a = False,
    Z = True, 
    a0=True, a1=True, a2=True, a3=True,
    b0=True, b1=True, b2=True, b3=True)

def contains_param(populations,pop_nm,param_nm):
    if pop_nm in populations:
        if 'parameters' in populations[pop_nm]:
            if param_nm in populations[pop_nm]['parameters']:
                return True
    return False 

def update_param(populations,pop_nm,param_nm,param_val):
    if not pop_nm in populations:
        populations[pop_nm] = {}
    if not 'parameters' in populations[pop_nm]:
        populations[pop_nm]['parameters'] = {}
    populations[pop_nm]['parameters'][param_nm] = param_val

def contains_setting(populations,pop_nm,setting_nm):
    if pop_nm in populations:
        if 'settings' in populations[pop_nm]:
            if setting_nm in populations[pop_nm]['settings']:
                return True
    return False 

def update_setting(populations,pop_nm,setting_nm,setting_val):
    if not pop_nm in populations:
        populations[pop_nm] = {}
    if not 'settings' in populations[pop_nm]:
        populations[pop_nm]['settings'] = {}
    populations[pop_nm]['settings'][setting_nm] = setting_val

def fcc_crystal(atom_symbol,a_lat=10.,pk_profile='voigt',I0=1.E-3,q_min=0.,q_max=1.,hwhm_g=0.001,hwhm_l=0.001):
    return dict(
        structure='fcc',
        settings={'q_min':q_min,'q_max':q_max,'profile':pk_profile},
        parameters={'I0':I0,'a':a_lat,'hwhm_g':hwhm_g,'hwhm_l':hwhm_l},
        basis={'{}_atom'.format(atom_symbol):dict(
            coordinates=[0,0,0],
            atomic={'symbol':atom_symbol}
            )}
        )

def flat_noise(I0=1.E-3):
    return dict(
        structure='diffuse',
        settings={},
        parameters={'I0':I0},
        basis={'flat_noise':{'flat':{}}}
        )

# TODO: more convenience constructors for various populations



