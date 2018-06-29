from collections import OrderedDict
import copy

import numpy as np
from pypif.obj import ChemicalSystem, Property, Classification, Id, Value, Scalar

from . import profiler
from .. import structure_names, form_factor_names, crystalline_structure_names, regression_params
from .. import ordered_populations

def make_pif(uid,expt_id=None,t_utc=None,q_I=None,temp_C=None,src_wl=None,populations=None):
    """Make a pypif.obj.ChemicalSystem object describing XRSD data.

    Parameters
    ----------
    uid : str
        record id, should be unique across the dataset
    expt_id : str
        experiment id, should be shared across the experiment 
    t_utc : int
        UTC time in seconds
    q_I : array
        n-by-2 array of q (1/Angstrom) and intensity (arb)
    temp_C : float
        temperature of the sample in degrees C
    src_wl : float
        wavelength of light source in Angstroms 
    populations : dict
        dict defining sample populations and parameters 

    Returns
    -------
    csys : pypif.obj.ChemicalSystem
        pypif.obj.ChemicalSystem representation of the PIF record
    """
    csys = ChemicalSystem()
    csys.uid = uid
    csys.ids = []
    csys.tags = []
    csys.properties = []
    csys.classifications = []
    if expt_id is not None:
        csys.ids.append(id_tag('EXPERIMENT_ID',expt_id))
    if t_utc is not None:
        csys.tags.append('time (utc): '+str(int(t_utc)))
    if q_I is not None:
        csys.properties.extend(q_I_properties(q_I,temp_C,src_wl))
    if populations is not None:
        opd = ordered_populations(populations)
        csys.classifications.extend(system_classifications(opd))
        csys.properties.extend(system_properties(opd))
    if q_I is not None:
        csys.properties.extend(profile_properties(q_I))
    return csys

def unpack_pif(pp):
    expt_id = None
    t_utc = None
    q_I = None
    temp = None
    features = OrderedDict()
    populations = OrderedDict()
    reg_pp_outputs={}
    cls_dict = {}
    for cl in pp.classifications:
        cls_dict[cl.name] = cl.value

    # NOTE: take the system_classifiation out of the cls_dict, or else label it as noise
    cl_pp_output = 'noise'
    if 'system_classification' in cls_dict:
        cl_pp_output = cls_dict.pop('system_classification')

    # NOTE: assume the remaining classifications define the populations
    all_pops_found = False
    ip = 0
    while not all_pops_found:
        pop_name_label = 'pop{}_name'.format(ip)
        pop_structure_label = 'pop{}_structure'.format(ip)
        if pop_name_label in cls_dict:
            pop_name = cls_dict[pop_name_label]
            populations[pop_name] = {} 
            populations[pop_name]['structure'] = cls_dict[pop_structure_label]
            if 'pop{}_site0_name'.format(ip) in cls_dict:
                populations[pop_name]['basis'] = {} 
                all_sites_found = False
                ist = 0
                while not all_sites_found:
                    site_name_label = 'pop{}_site{}_name'
                    site_form_label = 'pop{}_site{}_form'
                    if site_name_label in cls_dict:
                        site_name = cls_dict[site_name_label]
                        populations[pop_name]['basis'][site_name] = {}
                        populations[pop_name]['basis'][site_name]['form'] = cls_dict[site_form_label]
                        ist += 1
                    else:
                        all_sites_found = True
            ip += 1 
        else:
            all_pops_found = True

    if pp.ids is not None:
        for iidd in pp.ids:
            if iidd.name == 'EXPERIMENT_ID':
                expt_id = iidd.value
    if pp.tags is not None:
        for ttgg in pp.tags:
            if 'time (utc): ' in ttgg:
                t_utc = float(ttgg.replace('time (utc): ',''))

    # TODO: pack properties and settings into the populations dict
    if pp.properties is not None:
        for prop in pp.properties:
            if prop.name == 'Intensity':
                I = [float(sca.value) for sca in prop.scalars]
                for val in prop.conditions:
                    if val.name == 'scattering vector magnitude':
                        q = [float(sca.value) for sca in val.scalars]
                    if val.name == 'temperature':
                        temp = float(val.scalars[0].value)
                    if val.name == 'source wavelength':
                        src_wl = float(val.scalars[0].value) #TODO check if we need it
                q_I = np.vstack([q,I]).T
            elif prop.name in profiler.profile_keys:
                features[prop.name] = float(prop.scalars[0].value)
            elif any([rp == prop.name[-1*len(rp):] for rp in regression_params]):
                reg_pp_outputs[prop.name]= prop.scalars[0].value

    return pp.uid,expt_id,t_utc,q_I,temp,src_wl,populations,features,cl_pp_output,reg_pp_outputs

def id_tag(idname,idval,tags=None):
    return Id(idname,idval,tags)

def q_I_properties(q_I,temp_C=None,src_wl=None):
    properties = []
    # Process measured q_I into a property
    pI = q_I_property(q_I)
    if temp_C is not None:
        pI.conditions.append(Value('temperature',
        [Scalar(temp_C)],None,None,None,'degrees Celsius'))
    if src_wl is not None:
        pI.conditions.append(Value('source wavelength',
        [Scalar(src_wl)],None,None,None,'Angstroms'))
    properties.append(pI)
    return properties

def q_I_property(q_I,qunits='1/Angstrom',Iunits='arb',propname='Intensity'):
    pI = Property()
    n_qpoints = q_I.shape[0]
    pI.scalars = [Scalar(q_I[i,1]) for i in range(n_qpoints)]
    pI.units = Iunits 
    pI.conditions = []
    pI.conditions.append( Value('scattering vector magnitude', 
                        [Scalar(q_I[i,0]) for i in range(n_qpoints)],
                        None,None,None,qunits) )
    pI.name = propname 
    return pI 

def profile_properties(q_I):
    prof = profiler.full_profile(q_I)
    pp = []
    for fnm,fval in prof.items():
        if fval is not None:
            pp.append(Property(fnm,fval))
            #props.append(scalar_property(
            #fnm,fval,'spectrum profiling quantity'))
    return pp

def system_classifications(opd):
    clss = []
    main_cls = ''
    for ip,pop_nm in enumerate(opd.keys()):
        popd = opd[pop_nm]
        if not pop_nm == 'noise':
        #and not popd['structure'] == 'unidentified': 
            main_cls += 'pop{}_{}__'.format(ip,popd['structure'])
            popnm_cls = Classification('pop{}_name'.format(ip),pop_nm)
            pop_cls = Classification('pop{}_structure'.format(ip),popd['structure'])
            clss.extend([popnm_cls,pop_cls])
            for ist,site_nm in enumerate(popd['basis'].keys()):
                sited = popd['basis'][site_nm]
                main_cls += 'site{}_{}__'.format(ist,sited['form'])
                sitenm_cls = Classification('pop{}_site{}_name'.format(ip,ist),site_nm)
                site_cls = Classification('pop{}_site{}_form'.format(ip,ist),sited['form'])
                clss.extend([sitenm_cls,site_cls])
    if main_cls[-2:] == '__':
        main_cls = main_cls[:-2]
    clss.append(Classification('system_classification',main_cls))
    return clss

def system_properties(opd):
    properties = []
    for ip,popnm in enumerate(opd.keys()):
        if popnm == 'noise' and not opd[popnm]['structure'] == 'unidentified':
            properties.append(Property('noise_intensity',opd[popnm]['parameters']['I0']))
        else:
            popd = opd[popnm]
            if 'settings' in popd:
                properties.extend(setting_properties(ip,popd))
            if 'parameters' in popd:
                properties.extend(param_properties(ip,popd))
            if 'basis' in popd:
                for ist, stnm in enumerate(popd['basis'].keys()):
                    stdef = popd['basis'][stnm]
                    if 'settings' in stdef:
                        properties.extend(site_setting_properties(ip,ist,stdef))
                    if 'parameters' in stdef:
                        properties.extend(site_param_properties(ip,ist,stdef))
    return properties

def setting_properties(ip,popd):
    pps = []
    for stgnm,stgval in popd['settings'].items():
        pp = Property('pop{}_{}'.format(ip,stgnm))
        pp.tags = str(stgval)
        pps.append(pp)
    return pps

def param_properties(ip,popd):
    # TODO: add tags for bounds, constraints, vary/fix flag
    pps = []
    for pnm,pval in popd['parameters'].items():
        pp = Property('pop{}_{}'.format(ip,pnm),pval)
        pps.append(pp)
    return pps

def site_setting_properties(ip,ist,stdef):
    pps = []
    for stgnm,stgval in stdef['settings'].items():
        pp = Property('pop{}_site{}_{}'.format(ip,ist,stgnm))
        pp.tags = str(stgval)
        pps.append(pp)
    return pps

def site_param_properties(ip,ist,stdef):
    # TODO: add tags for bounds, constraints, vary/fix flag
    pps = []
    for pnm,pval in stdef['parameters'].items():
        pp = Property('pop{}_site{}_{}'.format(ip,ist,pnm),pval)
        pps.append(pp)
    return pps

def scalar_property(fname,fval,desc=None,data_type=None,funits=None):
    pf = Property()
    pf.name = fname
    if isinstance(fval,list):
        pf.scalars = [Scalar(v) for v in fval]
    else:
        pf.scalars = [Scalar(fval)]
    if desc:
        pf.tags = [desc]
    if data_type:
        pf.dataType = data_type 
    if funits:
        pf.units = funits
    return pf


