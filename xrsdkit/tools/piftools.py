from collections import OrderedDict

import numpy as np
import pypif.obj as pifobj

from . import profiler
from .. import structure_names
from .. import form_factor_names
from .. import crystalline_structure_names 

parameter_description = OrderedDict()
parameter_description['I0'] = 'flat background intensity'
parameter_description['G'] = 'Guinier-Porod model Guinier factor'
parameter_description['rg'] = 'Guinier-Porod model radius of gyration'
parameter_description['D'] = 'Guinier-Porod model Porod exponent'
parameter_description['r0'] = 'Mean radius of normally distributed sphere population'
parameter_description['sigma'] = 'fractional standard deviation of normally distributed sphere population'
parameter_description['q_center'] = 'peak center in q'
parameter_description['hwhm_l'] = 'Lorentzian profile half-width at half-max'
parameter_description['hwhm_g'] = 'Gaussian profile half-width at half-max'

parameter_units = OrderedDict()
parameter_units['I0'] = 'arb'
parameter_units['G'] = 'arb'
parameter_units['rg'] = 'Angstrom'
parameter_units['D'] = 'unitless'
parameter_units['r0'] = 'Angstrom'
parameter_units['sigma'] = 'unitless'
parameter_units['q_center'] = '1/Angstrom'
parameter_units['hwhm_l'] = '1/Angstrom'
parameter_units['hwhm_g'] = '1/Angstrom'

#population_keys = ['unidentified','guinier_porod','spherical_normal','diffraction_peaks']

# property names for all of the modeling "outputs"
model_output_names = list([
    'unidentified_structure_flag',
    'crystalline_structure_flag',
    'diffuse_structure_flag',
    'fcc_structure_count',
    'guinier_porod_population_count',
    'spherical_normal_population_count',
     'rg_0', # for samples with guinier_porod_population_count == 1
    'sigma_0', # for samples with spherical_normal_population_count == 1
     'r0_0'# for samples with spherical_normal_population_count == 1])
    ])

cl_model_output_names = list([
    #'unidentified_structure_flag', # crystalline and diffuse models are used to predict "unidentified" flag
    'crystalline_structure_flag',
    'diffuse_structure_flag',
    #'fcc_structure_count',
    'guinier_porod_population_count',
    'spherical_normal_population_count',
    ])

reg_model_output_names = list([
     'rg_0', # for samples with guinier_porod_population_count == 1
    'sigma_0', # for samples with spherical_normal_population_count == 1
     'r0_0'# for samples with spherical_normal_population_count == 1])
    ])

def make_pif(uid,expt_id=None,t_utc=None,q_I=None,temp_C=None,populations=None):
    """Make a pypif.obj.ChemicalSystem object describing XRSD data.

    Parameters
    ----------
    uid : str
        record id, should be unique wrt the record's destination dataset
    expt_id : str
        experiment id, should be the same as other records in the experiment
    t_utc : int
        UTC time in seconds
    q_I : array
        n-by-2 array of q (1/Angstrom) and intensity (arb)
    temp_C : float
        temperature of the sample in degrees C
    populations : dict
        dict defining sample populations and parameters 

    Returns
    -------
    csys : pypif.obj.ChemicalSystem
        pypif.obj.ChemicalSystem representation of the PIF record
    """
    csys = pifobj.ChemicalSystem()
    csys.uid = uid
    csys.ids = []
    csys.tags = []
    csys.properties = []
    if expt_id is not None:
        csys.ids.append(id_tag('EXPERIMENT_ID',expt_id))
    if t_utc is not None:
        csys.tags.append('time (utc): '+str(int(t_utc)))
    if q_I is not None:
        csys.properties.extend(q_I_properties(q_I,temp_C))
    if populations is not None:
        csys.properties.extend(populations_properties(populations))
    if q_I is not None:
        csys.properties.extend(profile_properties(q_I))
    return csys

def id_tag(idname,idval,tags=None):
    return pifobj.Id(idname,idval,tags)

def populations_properties(populations):
    properties = []
    unidentified_flag = 0
    if any([popd['structure'] == 'unidentified' 
    for pop_name,popd in populations.items()]):
        unidentified_flag = 1
    crystalline_flag = 0
    if any([popd['structure'] in crystalline_structure_names 
    for pop_name,popd in populations.items()]):
        crystalline_flag = 1
    diffuse_flag = 0
    if any([popd['structure'] == 'diffuse' and not pop_name == 'noise' 
    for pop_name,popd in populations.items()]):
        diffuse_flag = 1
    properties.append(scalar_property(
        'unidentified_structure_flag',unidentified_flag,
        'unidentified structure flag','EXPERIMENTAL'))
    properties.append(scalar_property(
        'crystalline_structure_flag',crystalline_flag,
        'crystalline structure flag','EXPERIMENTAL'))
    properties.append(scalar_property(
        'diffuse_structure_flag',diffuse_flag,
        'diffuse structure flag','EXPERIMENTAL'))
# TODO: refactor
#    structure_params = OrderedDict.fromkeys(structure_names)
#    structure_params.pop('unidentified')
#    ff_params = OrderedDict.fromkeys(form_factor_names)
#    for k in structure_names: structure_params[k] = [] 
#    for k in form_factor_names: ff_params[k] = [] 
#    for pop_name,popd in populations.items():
#        if not pop_name == 'noise' and not popd['structure'] == 'unidentified':
#            structure_params[popd['structure']].append(popd['parameters'])
#        if popd['structure'] == 'diffuse':
#            #if 'basis' in popd:
#            for site_name,site_def in popd['basis'].items():
#                for site_item_tag, site_item in site_items.items():
#                    #if site_item_tag in diffuse_form_factor_names:
#                    if isinstance(site_item,list): 
#                        ff_params[site_item_tag].extend(site_item)
#                    else:
#                        ff_params[site_item_tag].append(site_item)
#    structure_properties = []
#    for structure_name,param_list in structure_params.items():
#        structure_properties.append(scalar_property(
#            '{}_structure_count'.format(structure_name),len(param_list),
#            'number of {} structures'.format(structure_name),
#            'EXPERIMENTAL'))
#    for specie_name,param_list in ff_params.items():
#        structure_properties.append(scalar_property(
#            '{}_population_count'.format(specie_name),len(param_list),
#            'number of {} populations'.format(specie_name),
#            'EXPERIMENTAL'))
#    properties.extend(structure_properties)
#
#    param_properties = []
#    for specie_name,specie_params in diffuse_params.items():
#        for specie_idx,p in enumerate(specie_params):
#            for param_name, param_val in p.items():
#                param_properties.append(scalar_property(
#                '{}_{}'.format(param_name,specie_idx),param_val,
#                'parameter {} for {} population {}'.format(param_name,specie_name,specie_idx)))
#    if 'noise' in populations and not unidentified_flag \
#    and not crystalline_flag:
#        # TODO: remove the crystalline_flag condition
#        # after we have noise floors fitted for crystalline populations
#        I0_noise = populations['noise']['parameters']['I0']
#        #flat_amplitude = populations['noise']['basis']['flat_noise']['flat']['amplitude']
#        param_properties.append(scalar_property(
#        'I0_noise',I0_noise,
#        'magnitude of flat noise intensity'))
#    properties.extend(param_properties)

    return properties

def q_I_properties(q_I,temp_C=None):
    properties = []
    # Process measured q_I into a property
    pI = q_I_property(q_I)
    if temp_C is not None:
        pI.conditions.append(pifobj.Value('temperature',
        [pifobj.Scalar(temp_C)],None,None,None,'degrees Celsius'))
    properties.append(pI)
    return properties

def q_I_property(q_I,qunits='1/Angstrom',Iunits='arb',propname='Intensity'):
    pI = pifobj.Property()
    n_qpoints = q_I.shape[0]
    pI.scalars = [pifobj.Scalar(q_I[i,1]) for i in range(n_qpoints)]
    pI.units = Iunits 
    pI.conditions = []
    pI.conditions.append( pifobj.Value('scattering vector magnitude', 
                        [pifobj.Scalar(q_I[i,0]) for i in range(n_qpoints)],
                        None,None,None,qunits) )
    pI.name = propname 
    return pI 

def profile_properties(q_I):
    prof = profiler.full_profile(q_I)
    props = []
    for fnm,fval in prof.items():
        if fval is not None:
            props.append(scalar_property(
            fnm,fval,'spectrum profiling quantity'))
    return props



def ml_population_properties(ml_pops):
    props = []
    for popname,pop in ml_pops.items():
        if pop is not None:
            props.append(scalar_property(
                popname+'_ML',int(pop[0]),
                'number of {} populations by ML'.format(popname),
                'MACHINE_LEARNING'))
            props.append(scalar_property(
                popname+'_ML_certainty',float(pop[1]),
                'certainty in number of {} populations by ML'.format(popname),
                'MACHINE_LEARNING'))
    return props

def ground_truth_population_properties(populations):
    props = []
    for popname,pop in populations.items():
        if not bool(pop):
            pop = 0
        props.append(scalar_property(
            popname,int(pop),'number of {} populations'.format(popname),
            'EXPERIMENTAL'))
    return props

def param_properties(params):
    props = []
    for pname,pvals in params.items():
        if pvals is not None:
            if len(pvals)>0:
                props.append(scalar_property(
                pname,pvals,parameter_description[pname],
                'FIT',parameter_units[pname]))
    return props

def fitreport_properties(rpt):
    props = []
    for rptnm,rptval in rpt.items():
        if rptval is not None:
            #if isinstance(rptval,float):
            props.append(scalar_property(
            rptnm,rptval,'spectrum fitting quantity','FIT'))
    return props

def scalar_property(fname,fval,desc=None,data_type=None,funits=None):
    pf = pifobj.Property()
    pf.name = fname
    if isinstance(fval,list):
        pf.scalars = [pifobj.Scalar(v) for v in fval]
    else:
        pf.scalars = [pifobj.Scalar(fval)]
    if desc:
        pf.tags = [desc]
    if data_type:
        pf.dataType = data_type 
    if funits:
        pf.units = funits
    return pf

def unpack_pif(pp): # TODO: refactor
    expt_id = None
    t_utc = None
    q_I = None
    temp = None
    features = OrderedDict()
    if pp.ids is not None:
        for iidd in pp.ids:
            if iidd.name == 'EXPERIMENT_ID':
                expt_id = iidd.value
    if pp.tags is not None:
        for ttgg in pp.tags:
            if 'time (utc): ' in ttgg:
                t_utc = float(ttgg.replace('time (utc): ',''))
    if pp.properties is not None: 
        for prop in pp.properties:
            if prop.name == 'SAXS intensity':
                I = [float(sca.value) for sca in prop.scalars]
                for val in prop.conditions:
                    if val.name == 'scattering vector':
                        q = [float(sca.value) for sca in val.scalars]
                    if val.name == 'temperature':
                        temp = float(val.scalars[0].value)
                q_I = np.vstack([q,I]).T
            elif prop.tags is not None:
                if 'spectrum profiling quantity' in prop.tags:
                    features[prop.name] = float(prop.scalars[0].value) 
    return expt_id,t_utc,q_I,temp,features

def get_model_outputs(pp):
    model_outputs = OrderedDict.fromkeys(model_output_names)
    for pr in pp.properties:
        if pr.name in model_output_names:
            model_outputs[pr.name] = pr.scalars[0].value
    return model_outputs
    #return expt_id,t_utc,q_I,temp,feats,pops,par,rpt

#### obsolete functions below this line ####

population_keys=['unidentified','guinier_porod','spherical_normal','diffraction_peaks']
all_parameter_keys=['I0_floor','G_gp','rg_gp','D_gp','I0_sphere','r0_sphere','sigma_sphere']
def unpack_old_pif(pp):
    expt_id = None
    t_utc = None
    q_I = None
    temp = None
    
    feats = OrderedDict()
    pops = OrderedDict()
    par = OrderedDict()
    rpt = OrderedDict() 

    if pp.properties is not None:
        for prop in pp.properties:
            if prop.name == 'SAXS intensity':
                I = [float(sca.value) for sca in prop.scalars]
                for val in prop.conditions:
                    if val.name == 'scattering vector':
                        q = [float(sca.value) for sca in val.scalars]
                    if val.name == 'temperature':
                        temp = float(val.scalars[0].value)
                q_I = np.vstack([q,I]).T
            elif prop.name in population_keys:
                pops[prop.name] = int(prop.scalars[0].value)
            elif prop.name in all_parameter_keys:
                par[prop.name] = [float(s.value) for s in prop.scalars]
            elif prop.tags is not None:
                if 'spectrum fitting quantity' in prop.tags:
                    rpt[prop.name] = float(prop.scalars[0].value)
                if 'spectrum profiling quantity' in prop.tags:
                    feats[prop.name] = float(prop.scalars[0].value) 

    if pp.ids is not None:
        for iidd in pp.ids:
            if iidd.name == 'EXPERIMENT_ID':
                expt_id = iidd.value

    if pp.tags is not None:
        for ttgg in pp.tags:
            if 'time (utc): ' in ttgg:
                t_utc = float(ttgg.replace('time (utc): ',''))

    return expt_id,t_utc,q_I,temp,feats,pops,par,rpt



