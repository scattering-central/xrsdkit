"""This module defines the settings and parameters handled by xrsdkit."""
from collections import OrderedDict
import copy
import os

import numpy as np
import yaml

structure_names = ['diffuse','disordered','crystalline']
structures = OrderedDict.fromkeys(structure_names)
form_factor_names = ['atomic','polyatomic','guinier_porod','spherical']
form_factors = OrderedDict.fromkeys(form_factor_names)
noise_model_names = ['flat','low_q_scatter'] 
noise_models = OrderedDict.fromkeys(noise_model_names)

# supported structures, forms, and noise models
structures.update(
    diffuse = 'disordered, non-interacting particles',
    disordered = 'disordered, interacting particles',
    crystalline = 'particles arranged in a lattice'
    )
form_factors.update( 
    atomic = 'Single atom',
    polyatomic = 'Multiple atoms',
    guinier_porod = 'Scatterer described by Guinier-Porod equations',
    spherical = 'Spherical particle'
    )
noise_models.update(
    flat = 'Flat noise floor for all q',
    low_q_scatter = 'Flat noise floor plus a Guinier-Porod-like contribution'
    )

# load parameters for all atomic scattering form factors
fpath = os.path.join(os.path.dirname(__file__),'scattering','atomic_scattering_params.yml')
f = open(fpath,'r')
atomic_params = yaml.load(f)
f.close()

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
            if ('distribution' in settings) and (not settings['distribution'] == 'single'):
                msg = 'crystalline structure does not support size distribution {}'\
                .format(settings['distribution'])
                raise ValueError(msg)

# top-level settings, along with default values:
# these settings must exist for the corresponding
# structures and form factors
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
form_settings = dict(
    atomic = {'symbol':'C'},
    polyatomic = {'n_atoms':2},
    guinier_porod = {'distribution':'single'},
    spherical = {'distribution':'single'}
    )

# top-level parameters, along with default definitions:
# these parameters are always valid for the corresponding
# form factors and noise models, 
# regardless of settings, etc.
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
        effective_rg = {'value':100.,'fixed':True,'bounds':[0.1,None],'constraint_expr':None},
        effective_D = {'value':4.,'fixed':True,'bounds':[0.,4.],'constraint_expr':None} 
        )
    )

# crystal system and lattice definitions
crystal_systems = ['triclinic','monoclinic','orthorhombic','tetragonal','trigonal','hexagonal','cubic']
bravais_lattices = [\
            'triclinic','P_monoclinic','C_monoclinic',\
            'P_orthorhombic','C_orthorhombic','A_orthorhombic',\
            'I_orthorhombic','F_orthorhombic',\
            'P_tetragonal','I_tetragonal',\
            'rhombohedral','hexagonal',\
            'P_cubic','I_cubic','F_cubic',\
            ]
all_lattices = bravais_lattices+['hcp','diamond']

# TODO: deprecate this, 
# and find a better way to deal with modelable settings:
# consider that after classifying lattice,
# the next layer will be to classify space group,
# and after classifying n_atoms, the next layer should attempt 
# to classify the atom symbols for all n_atoms species.
modelable_structure_settings = dict(
    diffuse = [],
    disordered = ['interaction'],
    crystalline = ['lattice']
    )
modelable_form_factor_settings = dict(
    atomic = ['symbol'],
    polyatomic = ['n_atoms'],
    guinier_porod = ['distribution'],
    spherical = ['distribution']
    )

def secondary_settings(structure,form,primary_settings):
    """Return secondary settings, along with sensible default values.

    Secondary settings depend on the structure, form,
    and possibly primary setting values.
    Primary settings are defined by xrsdkit.definitions.structure_settings
    and xrsdkit.definitions.form_factor_settings.

    Parameters
    ----------
    structure : str
        Population structure designation, for fetching valid structure settings
    form : str
        Population form factor designation, for fetching valid form factor settings
    primary_settings : dict
        Dict of primary settings

    Returns
    -------
    stgs : dict
        Dict of all possible settings along with sensible default values
    """
    #stgs = {}
    #stgs.update(copy.deepcopy(structure_settings[structure]))
    #if form:
    #    stgs.update(copy.deepcopy(form_settings[form]))
    #stgs.update(prior_settings)
    #for stg_nm in stgs.keys():
    #    if (stg_nm in prior_settings) \
    #    and (prior_settings[stg_nm] in setting_selections(structure,form,prior_settings)):
    #        stgs[stg_nm] = prior_settings[stg_nm]
    sec_stgs = {}
    for stg_nm,stg_val in primary_settings.items():
        if stg_nm == 'n_atoms':
            sec_stgs.update(
                dict([('symbol_{}'.format(iat),'H') for iat in range(stg_val)]) 
                )
        if stg_nm == 'integration_mode':
            if stg_val == 'spherical':
                sec_stgs.update({'q_min':0.,'q_max':1.})
        if stg_nm == 'distribution' and form == 'spherical':
            if stg_val == 'r_normal':
                sec_stgs.update({'sampling_width':3.5,'sampling_step':0.05})
        #if stg_nm == 'distribution' and form == 'guinier_porod':
        #    if stg_val == 'rg_normal':
        #        sec_stgs.update({'sampling_width':2.0,'sampling_step':0.1})
    return sec_stgs 

# datatypes for all settings 
def setting_datatypes(stg_nm):
    if stg_nm in ['lattice','space_group',\
    'texture','profile','structure_factor_mode',\
    'integration_mode','interaction','distribution']: 
        return str
    if stg_nm in ['q_min','q_max','sampling_width','sampling_step']:
        return float
    if stg_nm == 'n_atoms': return int
    if 'symbol' in stg_nm: return str

# all possible options for all settings (empty list if not enumerable)
def setting_selections(stg_nm,structure=None,form=None,prior_settings={}):
    if stg_nm == 'lattice': return all_lattices
    if stg_nm == 'space_group':
        if 'lattice' in prior_settings:
            valid_sgs = lattice_space_groups[prior_settings['lattice']]
            return ['']+valid_sgs
        else:
            return ['']
    if stg_nm == 'texture': return ['random']
    if stg_nm == 'profile': return ['gaussian','lorentzian','voigt']
    if stg_nm == 'structure_factor_mode': return ['local','radial']
    if stg_nm == 'integration_mode': return ['spherical']
    if stg_nm == 'interaction': return ['hard_spheres']
    if stg_nm == 'distribution':
        if form == 'guinier_porod':
            return ['single']#,'rg_normal']
        if form == 'spherical':
            return ['single','r_normal']
    if stg_nm in ['q_min','q_max','sampling_width','sampling_step']: return []
    if stg_nm == 'n_atoms': return []
    if 'symbol' in stg_nm: return list(atomic_params.keys())

# generate any additional parameters that depend on setting selections
def structure_params(structure,prior_settings):
    params = {}
    if structure == 'disordered':
        if 'interaction' in prior_settings:
            if prior_settings['interaction'] == 'hard_spheres':
                params.update(
                    r_hard = {'value':20.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},
                    v_fraction = {'value':0.5,'fixed':False,'bounds':[0.01,0.7405],'constraint_expr':None}
                    ) 
    if structure == 'crystalline':
        if 'lattice' in prior_settings:
            if prior_settings['lattice'] in ['P_cubic','I_cubic','F_cubic','diamond','hcp']:
                params.update(a={'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None})
            if prior_settings['lattice'] in ['hexagonal','P_tetragonal','I_tetragonal']:
                params.update(
                    a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    c = {'value':20.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None}  
                    ) 
            if prior_settings['lattice'] == 'rhombohedral':
                params.update(
                    a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    alpha = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None}  
                    ) 
            if prior_settings['lattice'] in ['P_orthorhombic','C_orthorhombic','I_orthorhombic','F_orthorhombic']:
                params.update(
                    a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    b = {'value':12.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    c = {'value':15.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None}  
                    ) 
            if prior_settings['lattice'] in ['P_monoclinic','C_monoclinic']:
                params.update(
                    a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    b = {'value':12.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    c = {'value':15.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None}, 
                    beta = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None}  
                    )
            if prior_settings['lattice'] == 'triclinic':
                params.update(
                    a = {'value':10.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    b = {'value':12.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None},  
                    c = {'value':15.,'fixed':False,'bounds':[1.E-1,None],'constraint_expr':None}, 
                    alpha = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None},  
                    beta = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None}, 
                    gamma = {'value':90.,'fixed':False,'bounds':[0,180.],'constraint_expr':None} 
                    )
        if 'profile' in prior_settings:
            if prior_settings['profile'] == 'voigt':
                params.update(
                    hwhm_g = {'value':1.E-3,'fixed':False,'bounds':[1.E-9,None],'constraint_expr':None},
                    hwhm_l = {'value':1.E-3,'fixed':False,'bounds':[1.E-9,None],'constraint_expr':None}
                    )
            if prior_settings['profile'] in ['gaussian','lorentzian']:
                params.update(hwhm={'value':1.E-3,'fixed':False,'bounds':[1.E-9,None],'constraint_expr':None})
    return params

def additional_form_factor_params(form,prior_settings):
    if form == 'polyatomic':
        if 'n_atoms' in prior_settings:
            coord_params = {} 
            for iat in range(prior_settings['n_atoms']):
                coord_params.update({
                    'u_'+str(iat): {'value':0.1*iat,'fixed':True,'bounds':[-1.,1.],'constraint_expr':None},
                    'v_'+str(iat): {'value':0.1*iat,'fixed':True,'bounds':[-1.,1.],'constraint_expr':None},
                    'w_'+str(iat): {'value':0.1*iat,'fixed':True,'bounds':[-1.,1.],'constraint_expr':None} 
                    })
            for iat in range(prior_settings['n_atoms']):
                coord_params.update( 
                    {'occupancy_'+str(iat):{'value':1.,'fixed':True,'bounds':[0.,1.],'constraint_expr':None}}
                    )
            return coord_params
    #if form == 'guinier_porod':
    #    if 'distribution' in prior_settings:
    #        if prior_settings['distribution'] == 'rg_normal':
    #            return {'sigma':{'value':0.05,'fixed':False,'bounds':[0.,2.],'constraint_expr':None}}
    if form == 'spherical':
        if 'distribution' in prior_settings:
            if prior_settings['distribution'] == 'r_normal':
                return {'sigma':{'value':0.05,'fixed':False,'bounds':[0.,2.],'constraint_expr':None}}
    return {} 

def all_params(structure,form=None,prior_settings={}):
    all_pars = {'I0':{'value':1.,'fixed':False,'bounds':[0.,None],'constraint_expr':None}}
    all_pars.update(structure_params(structure,prior_settings))
    if form: all_pars.update(copy.deepcopy(form_factor_params[form]))
    if form: all_pars.update(additional_form_factor_params(form,prior_settings))
    return all_pars

# point groups associated with each crystal system 
crystal_point_groups = dict(
    triclinic = ['1','-1'],
    monoclinic = ['2','2/m','222','m','mm2','mmm'],
    orthorhombic = ['2','2/m','222','m','mm2','mmm'],
    tetragonal = ['4','-4','4/m','422','4mm','-42m','4/mmm'],
    trigonal = ['3','-3','32','3m','-3m'],
    hexagonal = ['6','-6','6/m','622','6mm','-6m2','6/mmm'],
    cubic = ['23','m-3','432','-43m','m-3m']  
    )
all_point_groups = []
for xtlsys in crystal_systems: all_point_groups.extend(crystal_point_groups[xtlsys])

# space groups associated with each lattice 
lattice_space_groups = dict(
    triclinic = {
        1:'P1',2:'P-1'
        },
    P_monoclinic = {
        3:'P2',4:'P2(1)',6:'Pm',7:'Pc',10:'P2/m',
        11:'P2(1)/m',13:'P2/c',14:'P2(1)/c'
        },
    C_monoclinic = {
        5:'C2',8:'Cm',9:'Cc',12:'C2/m',15:'C2/c'
        },
    P_orthorhombic = {
        16:'P222',17:'P222(1)',18:'P2(1)2(1)2',19:'P2(1)2(1)2(1)',
        25:'Pmm2',26:'Pmc2(1)',27:'Pcc2',28:'Pma2',29:'Pca2(1)',
        30:'Pnc2',31:'Pmn2(1)',32:'Pba2',33:'Pna2(1)',34:'Pnn2',
        47:'Pmmm',48:'Pnnn',49:'Pccm',50:'Pban',51:'Pmma',52:'Pnna',
        53:'Pmna',54:'Pcca',55:'Pbam',56:'Pccn',57:'Pbcm',58:'Pnnm',
        59:'Pmmn',60:'Pbcn',61:'Pbca',62:'Pnma'
        },
    C_orthorhombic = {
        20:'C222(1)',21:'C222',35:'Cmm2',36:'Cmc2(1)',37:'Ccc2',   
        63:'Cmcm',64:'Cmce',65:'Cmmm',66:'Cccm',67:'Cmme',68:'Ccce' 
        },
    A_orthorhombic = {
        38:'Amm2',39:'Aem2',40:'Ama2',41:'Aea2' 
        },
    I_orthorhombic = {
        23:'I222',24:'I2(1)2(1)2(1)',44:'Imm2',45:'Iba2',
        46:'Ima2',71:'Immm',72:'Ibam',73:'Ibca',74:'Imma'
        },
    F_orthorhombic = {
        22:'F222',42:'Fmm2',43:'Fdd2',69:'Fmmm',70:'Fddd' 
        },
    P_tetragonal = {
        75:'P4',76:'P4(1)',77:'P4(2)',78:'P4(3)',81:'P-4',
        83:'P4/m',84:'P4(2)/m',85:'P4/n',86:'P4(2)/n',
        89:'P422',90:'P42(1)2',91:'P4(1)22',92:'P4(1)2(1)2',
        93:'P4(2)22',94:'P4(2)2(1)2',95:'P4(3)22',96:'P4(3)2(1)2',
        99:'P4mm',100:'P4bm',101:'P4(2)cm',102:'P4(2)nm',
        103:'P4cc',104:'P4nc',105:'P4(2)mc',106:'P4(2)bc',
        111:'P-42m',112:'P-42c',113:'P-42(1)m',114:'P-42(1)c',
        115:'P-4m2',116:'P-4c2',117:'P-4b2',118:'P-4n2',123:'P4/mmm',
        124:'P4/mcc',125:'P4/nbm',126:'P4/nnc',127:'P4/mbm',
        128:'P4/mnc',129:'P4/nmm',130:'P4/ncc',131:'P4(2)/mmc',
        132:'P4(2)/mcm',133:'P4(2)/nbc',134:'P4(2)/nnm',
        135:'P4(2)/mbc',136:'P4(2)/mnm',137:'P4(2)/nmc',138:'P4(2)/ncm'
        },
    I_tetragonal = {
        79:'I4',80:'I4(1)',82:'I-4',87:'I4/m',88:'I4(1)/a',97:'I422', 
        98:'I4(1)22',107:'I4mm',108:'I4cm',109:'I4(1)md',110:'I4(1)cd', 
        119:'I-4m2',120:'I-4c2',121:'I-42m',122:'I-42d',139:'I4/mmm',
        140:'I4/mcm',141:'I4(1)/amd',142:'I4(1)/acd'
        },
    hexagonal = {
        143:'P3',144:'P3(1)',145:'P3(2)',147:'P-3',
        149:'P312',150:'P321',151:'P3(1)12',152:'P3(1)21',153:'P3(2)12',
        154:'P3(2)21',156:'P3m1',157:'P31m',158:'P3c1',159:'P31c',
        162:'P-31m',163:'P-31c',164:'P-3m1',165:'P-3c1',
        168:'P6',169:'P6(1)',170:'P6(5)',171:'P6(2)',172:'P6(4)',
        173:'P6(3)',174:'P-6',175:'P6/m',176:'P6(3)/m',177:'P622',
        178:'P6(1)22',179:'P6(5)22',180:'P6(2)22',181:'P6(4)22',
        182:'P6(3)22',183:'P6mm',184:'P6cc',185:'P6(3)cm',186:'P6(3)mc',
        187:'P-6m2',188:'P-6c2',189:'P-62m',190:'P-62c',191:'P6/mmm',
        192:'P6/mcc',193:'P6(3)/mcm',194:'P6(3)/mmc'
        },
    rhombohedral = {
        146:'R3',148:'R-3',155:'R32',160:'R3m',
        161:'R3c',166:'R-3m',167:'R-3c'
        },
    P_cubic = {
        195:'P23',198:'P2(1)3',200:'Pm-3',201:'Pn-3',205:'Pa-3',
        207:'P432',208:'P4(2)32',212:'P4(3)32',213:'P4(1)32',215:'P-43m',
        218:'P-43n',221:'Pm-3m',222:'Pn-3n',223:'Pm-3n',224:'Pn-3m'
        },
    I_cubic = {
        197:'I23',199:'I2(1)3',204:'Im-3',206:'Ia-3',211:'I432',
        214:'I4(1)32',217:'I-43m',220:'I-43d',229:'Im-3m',230:'Ia-3d'
        },
    F_cubic = {
        196:'F23',202:'Fm-3',203:'Fd-3',209:'F432',210:'F4(1)32',216:'F4-3m',
        219:'F-43c',225:'Fm-3m',226:'Fm-3c',227:'Fd-3m',228:'Fd-3c' 
        },
    hcp = {194:'P6(3)/mmc'},
    diamond = {227:'Fd-3m'}
    )
all_space_groups = {}
for lat in bravais_lattices: 
    for isg in lattice_space_groups[lat].keys():
        all_space_groups[isg] = lattice_space_groups[lat][isg]

# point groups associated with each space group 
sg_point_groups = OrderedDict()
sg_point_groups[all_space_groups[1]] = '1' 
sg_point_groups[all_space_groups[2]] = '-1' 
for isg in range(3,6): sg_point_groups[all_space_groups[isg]] = '2'
for isg in range(6,10): sg_point_groups[all_space_groups[isg]] = 'm'
for isg in range(10,16): sg_point_groups[all_space_groups[isg]] = '2/m'
for isg in range(16,25): sg_point_groups[all_space_groups[isg]] = '222'
for isg in range(25,47): sg_point_groups[all_space_groups[isg]] = 'mm2'
for isg in range(47,75): sg_point_groups[all_space_groups[isg]] = 'mmm'
for isg in range(75,81): sg_point_groups[all_space_groups[isg]] = '4'
sg_point_groups[all_space_groups[81]] = '-4'
sg_point_groups[all_space_groups[82]] = '-4'
for isg in range(83,89): sg_point_groups[all_space_groups[isg]] = '4/m'
for isg in range(89,99): sg_point_groups[all_space_groups[isg]] = '422'
for isg in range(99,111): sg_point_groups[all_space_groups[isg]] = '4mm'
for isg in range(111,123): sg_point_groups[all_space_groups[isg]] = '-42m'
for isg in range(123,143): sg_point_groups[all_space_groups[isg]] = '4/mmm'
for isg in range(143,147): sg_point_groups[all_space_groups[isg]] = '3'
sg_point_groups[all_space_groups[147]] = '-3'
sg_point_groups[all_space_groups[148]] = '-3'
for isg in range(149,156): sg_point_groups[all_space_groups[isg]] = '32'
for isg in range(156,162): sg_point_groups[all_space_groups[isg]] = '3m'
for isg in range(162,168): sg_point_groups[all_space_groups[isg]] = '-3m'
for isg in range(168,174): sg_point_groups[all_space_groups[isg]] = '6'
sg_point_groups[all_space_groups[174]] = '-6'
sg_point_groups[all_space_groups[175]] = '6/m'
sg_point_groups[all_space_groups[176]] = '6/m'
for isg in range(177,183): sg_point_groups[all_space_groups[isg]] = '622'
for isg in range(183,187): sg_point_groups[all_space_groups[isg]] = '6mm'
for isg in range(187,191): sg_point_groups[all_space_groups[isg]] = '-6m2'
for isg in range(191,195): sg_point_groups[all_space_groups[isg]] = '6/mmm'
for isg in range(195,200): sg_point_groups[all_space_groups[isg]] = '23'
for isg in range(200,207): sg_point_groups[all_space_groups[isg]] = 'm-3'
for isg in range(207,215): sg_point_groups[all_space_groups[isg]] = '432'
for isg in range(215,221): sg_point_groups[all_space_groups[isg]] = '-43m'
for isg in range(221,231): sg_point_groups[all_space_groups[isg]] = 'm-3m'

def lattice_coords(lattice):
    if lattice in ['triclinic','P_monoclinic','P_orthorhombic','P_tetragonal','rhombohedral','hexagonal','P_cubic']:
        return np.array([[0.,0.,0.]])
    elif lattice in ['C_monoclinic','C_orthorhombic']:
        return np.array([
            [0.,0.,0.],
            [0.,0.5,0.5]
            ])
    elif lattice in ['A_orthorhombic']:
        return np.array([
            [0.,0.,0.],
            [0.5,0.5,0.]
            ])
    elif lattice in ['I_orthorhombic','I_tetragonal','I_cubic']:
        return np.array([
            [0.,0.,0.],
            [0.5,0.5,0.5]
            ])
    elif lattice in ['F_orthorhombic','F_cubic']:
        return np.array([
            [0.,0.,0.],
            [0.5,0.5,0.],
            [0.5,0.,0.5],
            [0.,0.5,0.5]
            ])
    elif lattice == 'hcp':
        return np.array([
            [0.,0.,0.],
            [2./3,1./3,0.5]
            ])
    elif lattice == 'diamond':
        return np.array([
            [0.,0.,0.],
            [0.5,0.5,0.],
            [0.5,0.,0.5],
            [0.,0.5,0.5],
            [0.25,0.25,0.25],
            [0.75,0.75,0.25],
            [0.75,0.25,0.75],
            [0.25,0.75,0.75]
            ])

def lattice_vectors(lattice,a=None,b=None,c=None,alpha=None,beta=None,gamma=None):
    if lattice in ['P_cubic','I_cubic','F_cubic','diamond']:
        a1 = [a, 0., 0.]
        a2 = [0., a, 0.]
        a3 = [0., 0., a]
    elif lattice in ['hcp']:
        a1 = [a, 0., 0.]
        a2 = [0.5*a, np.sqrt(3.)/2*a, 0.]
        a3 = [0., 0., np.sqrt(8./3.)*a]
    elif lattice in ['hexagonal']:
        a1 = [a, 0., 0.]
        a2 = [0.5*a, np.sqrt(3.)/2*a, 0.]
        a3 = [0., 0., c]
    elif lattice in ['P_orthorhombic','C_orthorhombic',\
        'A_orthorhombic','I_orthorhombic','F_orthorhombic']:
        a1 = [a, 0., 0.]
        a2 = [0., b, 0.]
        a3 = [0., 0., c]
    elif lattice in ['P_tetragonal','I_tetragonal']:
        a1 = [a, 0., 0.]
        a2 = [0., a, 0.]
        a3 = [0., 0., c]
    elif lattice in ['P_monoclinic','C_monoclinic']:
        beta_rad = float(beta)*np.pi/180.
        a1 = [a, 0., 0.]
        a2 = [0., b, 0.]
        a3 = [c*np.cos(beta_rad), 0., c*np.sin(beta_rad)]
    elif lattice in ['rhombohedral']:
        alpha_rad = float(alpha)*np.pi/180.
        omega = a**3*np.sqrt(1.-np.cos(alpha_rad)**2-np.cos(alpha_rad)**2-np.cos(alpha_rad)**2
                +2*np.cos(alpha_rad)*np.cos(alpha_rad)*np.cos(alpha_rad))
        cy = a*(np.cos(alpha_rad)-np.cos(alpha_rad)**2)/np.sin(alpha_rad)
        cz = omega/(a**2*np.sin(alpha_rad)) 
        a1 = [a, 0., 0.]
        a2 = [a*np.cos(alpha_rad), a*np.sin(alpha_rad), 0.]
        a3 = [a*np.cos(alpha_rad), cy, cz]
    elif lattice in ['triclinic']:
        alpha_rad = float(alpha)*np.pi/180.
        beta_rad = float(beta)*np.pi/180.
        gamma_rad = float(gamma)*np.pi/180.
        omega = a*b*c*np.sqrt(1.-np.cos(alpha_rad)**2-np.cos(beta_rad)**2-np.cos(gamma_rad)**2
                +2*np.cos(alpha_rad)*np.cos(beta_rad)*np.cos(gamma_rad))
        cy = c*(np.cos(alpha_rad)-np.cos(beta_rad)*np.cos(gamma_rad))/np.sin(gamma_rad)
        cz = omega/(a*b*np.sin(gamma_rad)) 
        a1 = [a, 0., 0.]
        a2 = [b*np.cos(gamma_rad), b*np.sin(gamma_rad), 0.]
        a3 = [c*np.cos(beta_rad),cy,cz]
    return a1,a2,a3

def reciprocal_lattice_vectors(lat1, lat2, lat3, crystallographic=True):
    """Compute the reciprocal lattice vectors.

    If not `crystallographic`, the computation includes
    the factor of 2*pi that is commmon in solid state physics
    """
    rlat1_xprod = np.cross(lat2,lat3)
    rlat2_xprod = np.cross(lat3,lat1)
    rlat3_xprod = np.cross(lat1,lat2)
    cellvol = np.dot(lat1,rlat1_xprod)
    rlat1 = rlat1_xprod/cellvol
    rlat2 = rlat2_xprod/cellvol
    rlat3 = rlat3_xprod/cellvol
    if not crystallographic:
        rlat1 *= 2*np.pi
        rlat2 *= 2*np.pi
        rlat3 *= 2*np.pi
    return rlat1, rlat2, rlat3

setting_descriptions = dict(
    lattice = 'Lattice identifier for crystalline populations',
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
    distribution = 'Specifies a distribution for parameter values over a population',
    sampling_width = 'Number of standard deviations to sample from distribution',
    sampling_step = 'Resolution of sampling, in units of standard deviations'
    )

parameter_units = dict(
    I0 = 'arbitrary',
    rg = 'Angstrom',
    D = 'unitless',
    r = 'Angstrom',
    sigma = 'unitless',
    r_hard = 'Angstrom',
    v_fraction = 'unitless',
    hwhm = '1/Angstrom',
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
    sigma = 'standard deviation of normally distributed parameter divided by the parameter mean',
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
    u = 'fractional coordinate along first lattice vector',
    v = 'fractional coordinate along second lattice vector',
    w = 'fractional coordinate along third lattice vector',
    occupancy = 'likelihood of finding an atomic specie at its lattice site'
    )

