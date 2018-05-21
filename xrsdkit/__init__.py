"""Computation and analysis of scattering and diffraction patterns.

A scattering/diffraction pattern is assumed to represent
one or more populations of scattering objects.
The populations are described by a dict,
where each population has a name (key)
and a sub-dict of parameters (value).
Each population sub-dict should have the following entries: 

    - 'structure' : structure identifier (e.g. 'diffuse', 'fcc'). 

    - 'settings' : any settings needed for computing the structure factor
        
    - 'parameters' : computational and physical parameters of the structure 

    - 'basis' : dict containing site names (as keys)
        and dicts specifying site location and content (as values).
        The site content dicts are structured as:

        - 'coordinates' : list of three floating point numbers,
            the fractional coordinates relative to a lattice site,
            required only for crystalline structures.

        - 'form' : form factor identifier
    
        - 'settings' : any settings needed for computing the form factor

        - 'parameters' : form factor parameters 


The supported structure factors and their settings and parameters are: 

    - all structures:

      - parameter 'I0': intensity prefactor 

    - 'diffuse' : a diffuse (or dilute), 
        non-interfering scattering ensemble.
        This structure factor has no parameters,
        except for its intensity prefactor.

    - 'hard_spheres' : condensed (interacting) hard spheres,
        computed via the Percus-Yevick
        solution to the Ornstein-Zernike equation. 
        TODO: Cite
        (Ornstein-Zernike, 1918; Percus-Yevick, 1958; Hansen-McDonald, 1986, Hammouda)

        - parameter 'r_hard' : hard sphere radius 
        - parameter 'v_fraction' : volume fraction of spheres

    - 'fcc' : crystalline fcc lattice; 
        peaks are computed according to 
        the settings 'profile', 'q_min', and 'q_max'.

        - setting 'profile' : choice of peak profile ('gaussian','lorentzian','voigt')
        - setting 'q_min' : minimum q-value for reciprocal space summation 
        - setting 'q_max' : minimum q-value for reciprocal space summation 
        - parameter 'a' : cubic lattice parameter 
        - parameter 'hwhm_g' : Gaussian half-width at half-max
        - parameter 'hwhm_l' : Lorentzian half-width at half-max
        

The supported form factors and their settings and parameters are:

    - 'flat': a flat form factor for all q,
        implemented for simulating a noise floor.
        This form factor introduces no parameters.

    - 'guinier_porod': scatterer populations described 
        by the Guinier-Porod equations.

      - parameter 'G': Guinier prefactor 
      - parameter 'rg': radius of gyration 
      - parameter 'D': Porod exponent 

    - 'spherical': solid spheres 

      - parameter 'r': sphere radius (Angstrom) 

    - 'spherical_normal': solid spheres 
        with a normal size distribution.

      - parameter 'r0': mean sphere radius (Angstrom) 
      - parameter 'sigma': fractional standard deviation of radius

    - 'atomic': atomic form factors described by
        ff = Z - 41.78214 * s**2 * sum_i(a_i*exp(-b_i*s**2)),
        where Z is the atomic number,
        s = sin(theta)/lambda,
        and a_i, b_i are the form factor parameters.
        These scatterers respect the following settings:

      - setting 'symbol': Atomic symbol (as on the periodic table),
        for using the standard scattering parameters
        (see atomic_scattering_parameters.yaml).
        Atomic sites outside the standard parameter set
        must provide a setting for the atomic number (Z),
        and up to four pairs of scattering parameters (a, b).
      - setting 'Z': atomic number
      - parameters 'a0', 'a1', 'a2', 'a3' coefficients 
      - parameters 'b0', 'b1', 'b2', 'b3' exponents 

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
                form='flat'
                )
            )
        )
    )
"""
from collections import OrderedDict

import numpy as np

# list of allowed structure specifications
structure_names = [\
'unidentified',\
'hard_spheres',\
'diffuse',\
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
structure_params['fcc'].extend(['a','hwhm_g','hwhm_l'])

# list of allowed settings for each structure
structure_settings = OrderedDict.fromkeys(structure_names)
for nm in structure_settings: structure_settings[nm] = []
structure_settings['fcc'].extend(['profile','q_min','q_max'])

# list of allowed parameters for each form factor 
form_factor_params = OrderedDict.fromkeys(form_factor_names)
for nm in form_factor_params: form_factor_params[nm] = []
form_factor_params['guinier_porod'].extend(['G','rg','D'])
form_factor_params['spherical'].extend(['r'])
form_factor_params['spherical_normal'].extend(['r0','sigma'])
form_factor_params['atomic'].extend(['a0','a1','a2','a3','b0','b1','b2','b3'])

# list of allowed settings for each form factor 
form_factor_settings = OrderedDict.fromkeys(form_factor_names)
for nm in form_factor_settings: form_factor_settings[nm] = []
form_factor_settings['atomic'].extend(['symbol','Z'])

all_params = [\
'I0',\
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

def contains_coordinates(populations,pop_nm,site_nm):
    if pop_nm in populations:
        if 'basis' in populations[pop_nm]:
            if site_nm in populations[pop_nm]['basis']:
                if 'coordinates' in populations[pop_nm]['basis'][site_nm]:
                    return True
    return False    

def contains_site_param(populations,pop_nm,site_nm,param_nm):
    if pop_nm in populations:
        if 'basis' in populations[pop_nm]:
            if site_nm in populations[pop_nm]['basis']:
                site_def = populations[pop_nm]['basis'][site_nm]
                if 'parameters' in site_def: 
                    if param_nm in site_def['parameters']: 
                        return True

def update_site_param(populations,pop_nm,site_nm,param_nm,new_value):
    if not pop_nm in populations:
        populations[pop_nm] = {}
    if not 'basis' in populations[pop_nm]:
        populations[pop_nm]['basis'] = {}
    if not site_nm in populations[pop_nm]['basis']:
        populations[pop_nm]['basis'][site_nm] = {}
    if not 'parameters' in populations[pop_nm]['basis'][site_nm]:
        populations[pop_nm]['basis'][site_nm]['parameters'] = {} 
    populations[pop_nm]['basis'][site_nm]['parameters'][param_nm] = new_value

def update_coordinates(populations,pop_nm,site_nm,new_values):
    if not pop_nm in populations:
        populations[pop_nm] = {}
    if not 'basis' in populations[pop_nm]:
        populations[pop_nm]['basis'] = {}
    if not site_nm in populations[pop_nm]['basis']:
        populations[pop_nm]['basis'][site_nm] = {}
    populations[pop_nm]['basis'][site_nm]['coordinates'] = new_values

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

# TODO: more convenience constructors for various populations

def fcc_crystal(atom_symbol,a_lat=10.,pk_profile='voigt',I0=1.E-3,q_min=0.,q_max=1.,hwhm_g=0.001,hwhm_l=0.001):
    return dict(
        structure='fcc',
        settings={'q_min':q_min,'q_max':q_max,'profile':pk_profile},
        parameters={'I0':I0,'a':a_lat,'hwhm_g':hwhm_g,'hwhm_l':hwhm_l},
        basis={atom_symbol+'_atom':dict(
            coordinates=[0,0,0],
            form='atomic',
            settings={'symbol':atom_symbol}
            )}
        )

def unidentified_population():
    return dict(
        structure='unidentified',
        settings={}, 
        parameters={},
        basis={}
        )

def empty_site():
    return dict(
        form='diffuse',
        settings={},
        parameters={}
        )
        
def flat_noise(I0=1.E-3):
    return dict(
        structure='diffuse',
        settings={},
        parameters={'I0':I0},
        basis={'flat_noise':{'form':'flat'}}
        )

def new_site(pop_dict,pop_name,site_name,ff_name):
    structure_name = pop_dict[pop_name]['structure']
    pd = OrderedDict()
    pd[pop_name] = OrderedDict()
    pd[pop_name]['basis'] = OrderedDict()
    pd[pop_name]['basis'][site_name] = OrderedDict()
    sd = pd[pop_name]['basis'][site_name]
    fp = OrderedDict()
    pb = OrderedDict()
    pc = OrderedDict()
    sd['form'] = ff_name
    sd['settings'] = OrderedDict.fromkeys(form_factor_settings[ff_name])
    sd['parameters'] = OrderedDict.fromkeys(form_factor_params[ff_name])
    if structure_name in crystalline_structure_names:
        cdef = param_defaults['coordinates']
        sd['coordinates'] = [float(cdef),float(cdef),float(cdef)]
    for snm in form_factor_settings[ff_name]:
        sd['settings'][snm] = setting_defaults[snm] 
    for pnm in form_factor_params[ff_name]:
        sd['parameters'][pnm] = param_defaults[pnm] 
    if structure_name == 'fcc' and ff_name == 'spherical':
        expr = pop_name+'__'+'a'+'*sqrt(2)/4'
        rval = pop_dict[pop_name]['parameters']['a']*np.sqrt(2)/4
        sd['parameters']['r'] = rval
        update_site_param(pc,pop_name,site_name,'r',expr)
    if structure_name == 'hard_spheres' and ff_name == 'spherical':
        expr = pop_name+'__'+'r_hard'
        rval = pop_dict[pop_name]['parameters']['r_hard']
        sd['parameters']['r'] = rval
        update_site_param(pc,pop_name,site_name,'r',expr)
    if structure_name == 'hard_spheres' and ff_name == 'spherical_normal':
        expr = pop_name+'__'+'r_hard'
        rval = pop_dict[pop_name]['parameters']['r_hard']
        sd['parameters']['r0'] = rval
        update_site_param(pc,pop_name,site_name,'r0',expr)
    # NOTE: any more default bounds or constraints should be inserted here
    return pd,fp,pb,pc

def update_populations(pops,new_pops):
    for pop_name,pd_new in new_pops.items():
        if not pop_name in pops:
            pops[pop_name] = {}
        pd = pops[pop_name]
        if 'structure' in pd_new:
            if 'structure' in pd:
                if not pd_new['structure'] == pd['structure']:
                    pd['parameters'] = OrderedDict()
                    pd['settings'] = OrderedDict()
            pd['structure'] = pd_new['structure']
        if 'parameters' in pd_new:
            if not 'parameters' in pd:
                pd['parameters'] = OrderedDict()
            for param_nm, pval in pd_new['parameters'].items():
                pd['parameters'][param_nm] = pval
        if 'settings' in pd_new:
            if not 'settings' in pd:
                pd['settings'] = OrderedDict()
            for stg_nm, sval in pd_new['settings'].items():
                pd['settings'][stg_nm] = sval
        if 'basis' in pd_new:
            if not 'basis' in pd:
                pd['basis'] = OrderedDict()
            for site_nm,sd_new in pd_new['basis'].items():
                if not site_nm in pd['basis']:
                    pd['basis'][site_nm] = OrderedDict() 
                sd = pd['basis'][site_nm]
                if 'form' in sd_new:
                    if 'form' in sd:
                        if not sd_new['form'] == sd['form']:
                            sd['parameters'] = OrderedDict()
                            sd['settings'] = OrderedDict()
                            if pd['structure'] not in crystalline_structure_names:
                                if 'coordinates' in sd:
                                    sd.pop('coordinates')
                    sd['form'] = sd_new['form']
                if 'parameters' in sd_new:
                    if not 'parameters' in sd:
                        sd['parameters'] = OrderedDict() 
                    for param_nm, pval in sd_new['parameters'].items():
                        sd['parameters'][param_nm] = pval
                if 'settings' in sd_new:
                    if not 'settings' in sd:
                        sd['settings'] = OrderedDict() 
                    for stg_nm, sval in sd_new['settings'].items():
                        sd['settings'][stg_nm] = sval
                if 'coordinates' in sd_new:
                    if not 'coordinates' in sd:
                        sd['coordinates'] = [None,None,None] 
                    for cidx, cval in enumerate(sd_new['coordinates']):
                        if cval is not None: 
                            sd['coordinates'][cidx] = cval

