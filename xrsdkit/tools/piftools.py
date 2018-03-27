from collections import OrderedDict

import numpy as np
import pypif.obj as pifobj

from . import profiler
from ..scattering import diffuse_form_factor_names
from ..diffraction import crystalline_structure_names 

#parameter_description = OrderedDict.fromkeys(all_parameter_keys)
#parameter_description['I0_floor'] = 'flat background intensity'
#parameter_description['G_gp'] = 'guinier_porod Guinier factor'
#parameter_description['rg_gp'] = 'guinier_porod radius of gyration'
#parameter_description['D_gp'] = 'guinier_porod Porod exponent'
#parameter_description['I0_sphere'] = 'spherical_normal scattering intensity'
#parameter_description['r0_sphere'] = 'spherical_normal mean radius'
#parameter_description['sigma_sphere'] = 'spherical_normal polydispersity'
#parameter_description['I_pkcenter'] = 'diffraction peak center intensity'
#parameter_description['q_pkcenter'] = 'diffraction peak center in q'
#parameter_description['pk_hwhm'] = 'diffraction peak half-width at half-max'

#parameter_units = OrderedDict.fromkeys(all_parameter_keys)
#parameter_units['I0_floor'] = 'arb'
#parameter_units['G_gp'] = 'arb'
#parameter_units['rg_gp'] = 'Angstrom'
#parameter_units['D_gp'] = 'unitless'
#parameter_units['I0_sphere'] = 'arb'
#parameter_units['r0_sphere'] = 'Angstrom'
#parameter_units['sigma_sphere'] = 'unitless'
#parameter_units['I_pkcenter'] = 'arb'
#parameter_units['q_pkcenter'] = '1/Angstrom'
#parameter_units['pk_hwhm'] = '1/Angstrom'

#population_keys = ['unidentified','guinier_porod','spherical_normal','diffraction_peaks']

# property names for all of the modeling "outputs"
model_output_names = list([
    'crystalline_structure_flag',
    'diffuse_structure_flag',
    'guinier_porod_population_count',
    'spherical_normal_population_count'])

def make_pif(uid,expt_id=None,t_utc=None,q_I=None,temp_C=None,populations=None):
    """Make a pypif.obj.ChemicalSystem object describing a SAXS experiment.

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
    #csys.classifications = []
    csys.properties = []
    if expt_id is not None:
        csys.ids.append(id_tag('EXPERIMENT_ID',expt_id))
    if t_utc is not None:
        csys.tags.append('time (utc): '+str(int(t_utc)))
    if q_I is not None:
        csys.properties.extend(q_I_properties(q_I,temp_C))
    if populations is not None:
        csys.properties.extend(structure_properties(populations))
    if q_I is not None:
        csys.properties.extend(profile_properties(q_I))
    return csys

def id_tag(idname,idval,tags=None):
    return pifobj.Id(idname,idval,tags)

def structure_properties(populations):
    properties = []
    crystalline_flag = 0
    if any([popd['structure'] in crystalline_structure_names 
    for pop_name,popd in populations.items()]):
        crystalline_flag = 1
    disordered_flag = 0
    if any([popd['structure'] == 'disordered' 
    for pop_name,popd in populations.items()]):
        disordered_flag = 1
    diffuse_flag = 0
    if any([popd['structure'] == 'diffuse' and not pop_name == 'noise' 
    for pop_name,popd in populations.items()]):
        diffuse_flag = 1

        # ASIDE: properties describing diffuse populations
        # TODO: make this better
        diffuse_population_properties = []
        n_diffuse = OrderedDict.fromkeys(diffuse_form_factor_names)
        for ff_name in diffuse_form_factor_names:
            n_diffuse[ff_name] = 0
        for pop_name,popd in populations.items():
            if popd['structure'] == 'diffuse':
                for coord, species in popd['basis'].items():
                    for specie_name, specie_params in species.items():
                        if specie_name in diffuse_form_factor_names:
                            n_diffuse[specie_name] = 1
                            if isinstance(specie_params,list):
                                n_diffuse[specie_name] = len(specie_params)
        for specie_name, ns in n_diffuse.items():
            diffuse_population_properties.append(scalar_property(
                '{}_population_count'.format(specie_name),ns,
                'number of diffuse {} populations'.format(specie_name),
                'EXPERIMENTAL'))

    properties.append(scalar_property(
        'crystalline_structure_flag',crystalline_flag,
        'crystalline structure flag','EXPERIMENTAL'))
    properties.append(scalar_property(
        'diffuse_structure_flag',diffuse_flag,
        'diffuse structure flag','EXPERIMENTAL'))
    properties.append(scalar_property(
        'disordered_structure_flag',disordered_flag,
        'disordered structure flag','EXPERIMENTAL'))
    if diffuse_flag:
        properties.append(diffuse_population_properties)
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

#        if populations is not None:
#            # population-specific featurizations
#            det_profiles = saxs_math.detailed_profile(q_I,populations)
#            det_profile_props = profile_properties(det_profiles)
#            props.extend(det_profile_props)
#        # ML flags for this featurization
#        sxc = saxs_classify.SaxsClassifier()
#        ml_pops = sxc.classify(np.array(list(prof.values())).reshape(1,-1))
#        ml_pop_props = ml_population_properties(ml_pops)
#        props.extend(ml_pop_props)

#    if q_I is not None and params is not None and populations is not None:
#        if not bool(populations['unidentified']):
#            qcomp = np.arange(0.,q_I[-1,0],0.001)
#            I_computed = saxs_math.compute_saxs(qcomp,populations,params)
#            pI_computed = q_I_property(
#                np.array([qcomp,I_computed]).T,
#                propname='computed SAXS intensity')
#            props.append(pI_computed)
#            # add properties for the fit report
#            sxf = saxs_fit.SaxsFitter(q_I,populations)
#            report = sxf.fit_report(params) 
#            rprops = fitreport_properties(report)
#            props.extend(rprops)
#
#
#    if populations is not None:
#        fprops = ground_truth_population_properties(populations)
#        props.extend(fprops)
#
#    if params is not None:
#        pprops = param_properties(params)
#        props.extend(pprops)
#
#    return props

#def structure_classifications(populations):
#    c = []
#    if not isinstance(populations,list):
#        populations = [populations]
#    for popd in populations:
#        if not popd['name'] == 'noise':
#            c_struct = pifobj.Classification('{}_structure'.format(popd['name']),popd['structure'])
#            c.append(c_struct)        
#    return c


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

def unpack_pif(pp): # I need to work on it!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
    populations = []
    if pp.classifications is not None:
        for cls in pp.classifications:
            popd = OrderedDict()
            popd['name'] = cls.name[:cls.name.rfind('_')]
            popd['structure'] = cls.value 
            populations.append(popd)
    return expt_id,t_utc,q_I,temp,features,populations

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



