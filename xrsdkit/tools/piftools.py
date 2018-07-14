from collections import OrderedDict

import numpy as np
from pypif.obj import ChemicalSystem, Property, Classification, Id, Value, Scalar

from . import profiler
from .. import contains_params, contains_site_params, regression_params, structure_params, \
    form_factor_params, ordered_populations, structure_settings, \
    form_factor_settings, setting_datatypes, update_populations

# TODO: pack and unpack fitting objective, q-range, settings
# (add input/output for fit_report dict on make_pif/unpack_pif)

def make_pif(uid,expt_id=None,t_utc=None,q_I=None,temp_C=None,src_wl=None,
    populations=None,fixed_params=None,param_bounds=None,param_constraints=None):
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
    fixed_params : dict
        dict defining fixed parameters, if any
    param_bounds : dict
        dict defining parameter bounds, if any 
    param_constraints : dict
        dict defining parameter equality constraints, if any 

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
    if populations is not None: # is not empty Dict
        opd = ordered_populations(populations)
        csys.classifications.extend(system_classifications(opd))
        csys.properties.extend(system_properties(opd,fixed_params,param_bounds,param_constraints))
    if q_I is not None:
        csys.properties.extend(profile_properties(q_I))
    return csys


def unpack_pif(pp):
    #print(pif.dumps(pp, indent=4))
    expt_id = None
    t_utc = None
    q_I = None
    temp = None
    features = OrderedDict()
    populations = OrderedDict()
    fp, pb, pc = {}, {}, {}
    reg_pp_outputs={}

    # pack classification labels into a dict
    cls_dict = {}
    if pp.classifications is not None:
        for cl in pp.classifications:
            cls_dict[cl.name] = cl.value

    # pack properties into a dict
    props_dict = {}
    if pp.properties is not None:
        for prop in pp.properties:
            props_dict[prop.name] = prop

    # take the system_classification out of the cls_dict,
    # label it as noise if not otherwise labeled
    cl_pp_output = 'noise'
    #if 'system_classification' in cls_dict:   # changed
    if cls_dict['system_classification'] is not None:
        cl_pp_output = cls_dict.pop('system_classification')

    # assume the remaining classifications are labels for the populations
    all_pops_found = False
    ip = 0
    while not all_pops_found:
        pop_name_label = 'pop{}_name'.format(ip)
        pop_structure_label = 'pop{}_structure'.format(ip)
        if pop_name_label in cls_dict:
            pop_name = cls_dict[pop_name_label]
            structure_name = cls_dict[pop_structure_label]
            populations[pop_name] = {} 
            populations[pop_name]['structure'] = structure_name
            param_labels = ['pop{}_{}'.format(ip,param_nm) for param_nm in structure_params[structure_name]]
            if any([pl in props_dict for pl in param_labels]):
                populations[pop_name]['parameters'] = {} ########
                for pl,param_nm in zip(param_labels,structure_params[structure_name]): 
                    if pl in props_dict:
                        populations[pop_name]['parameters'][param_nm] = \
                        props_dict[pl].scalars[0].value
                        if props_dict[pl].tags is not None: # added
                            for tg in props_dict[pl].tags():
                                if 'fixed value: ' in tg:
                                    fp_val = bool(tg.strip('fixed value: '))
                                    update_populations(fp,{pop_name:{'parameters':{param_nm:fp_val}}})
                                if 'bounds: ' in tg:
                                    bds = tg.strip('bounds: ','[',']').split(',')
                                    lbnd, ubnd = float(bds[0]), float(bds[1])
                                    update_populations(pb,{pop_name:{'parameters':{param_nm:[lbnd,ubnd]}}})
                                if 'constraint expression: ' in tg:
                                    cexpr = tg.strip('constraint expression: ')
                                    update_populations(pc,{pop_name:{'parameters':{param_nm:cexpr}}})
                            #update_populations(pb,param_bounds_from_tags(props_dict[pl].tags)) # TODO imlement param_bounds_from_tags
                            #update_populations(pc,param_constraints_from_tags(props_dict[pl].tags))
            stg_labels = ['pop{}_{}'.format(ip,stg_nm) for stg_nm in structure_settings[structure_name]]
            if any([sl in props_dict for sl in stg_labels]):
                populations[pop_name]['settings'] = {}
                for sl,stg_nm in zip(stg_labels,structure_settings[structure_name]): 
                    tp = setting_datatypes[stg_nm]
                    populations[pop_name]['settings'][stg_nm] = \
                    tp(props_dict[sl].tags[0])

            if 'pop{}_site0_name'.format(ip) in cls_dict:
                populations[pop_name]['basis'] = {} 
                all_sites_found = False
                ist = 0
                while not all_sites_found:
                    site_name_label = 'pop{}_site{}_name'.format(ip,ist)
                    site_form_label = 'pop{}_site{}_form'.format(ip,ist)
                    if site_name_label in cls_dict:
                        site_name = cls_dict[site_name_label]
                        site_form = cls_dict[site_form_label]
                        populations[pop_name]['basis'][site_name] = {}
                        populations[pop_name]['basis'][site_name]['form'] = site_form 
                        param_labels = ['pop{}_site{}_{}'.format(ip,ist,param_nm) for param_nm in form_factor_params[site_form]]
                        if any([pl in props_dict for pl in param_labels]):
                            populations[pop_name]['basis'][site_name]['parameters'] = {} 
                            for pl,param_nm in zip(param_labels,form_factor_params[site_form]): 
                                if pl in props_dict:
                                    populations[pop_name]['basis'][site_name]['parameters'][param_nm] = \
                                    props_dict[pl].scalars[0].value
                                    if props_dict[pl].tags is not None: # added
                                        for tg in props_dict[pl].tags():
                                            if 'fixed value: ' in tg:
                                                fp_val = bool(tg.strip('fixed value: '))
                                                update_populations(fp,{pop_name:{'basis':{site_name:{'parameters':{param_nm:fp_val}}}}})
                                            if 'bounds: ' in tg:
                                                bds = tg.strip('bounds: ','[',']').split(',')
                                                lbnd, ubnd = float(bds[0]), float(bds[1])
                                                update_populations(pb,{pop_name:{'basis':{site_name:{'parameters':{param_nm:[lbnd,ubnd]}}}}})
                                            if 'constraint expression: ' in tg:
                                                cexpr = tg.strip('constraint expression: ')
                                                update_populations(pc,{pop_name:{'basis':{site_name:{'parameters':{param_nm:cexpr}}}}})
                        stg_labels = ['pop{}_site{}_{}'.format(ip,ist,stg_nm) for stg_nm in form_factor_settings[site_form]]
                        if any([sl in props_dict for sl in stg_labels]):
                            populations[pop_name]['basis'][site_name]['settings'] = {}
                            for sl,stg_nm in zip(stg_labels,form_factor_settings[site_form]): 
                                if sl in props_dict:
                                    tp = setting_datatypes[stg_nm]
                                    populations[pop_name]['basis'][site_name]['settings'][stg_nm] = \
                                    tp(props_dict[sl].tags[0])
                        coord_labels = ['pop{}_site{}_coordinate{}'.format(ip,ist,ic) for ic in [0,1,2]]
                        if all([cl in props_dict for cl in coord_labels]):
                            c0 = float(props_dict['pop{}_site{}_coordinate0'.format(ip,ist)])
                            c1 = float(props_dict['pop{}_site{}_coordinate1'.format(ip,ist)])
                            c2 = float(props_dict['pop{}_site{}_coordinate2'.format(ip,ist)])
                            populations[pop_name]['basis'][site_name]['coordinates'] = [c0,c1,c2]
                            # TODO: deal with fixed_params, param_bounds, param_constraints on coordinates
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

    for prop_nm, prop in props_dict.items():
        if prop_nm == 'Intensity':
            I = [float(sca.value) for sca in prop.scalars]
            for val in prop.conditions:
                if val.name == 'scattering vector magnitude':
                    q = [float(sca.value) for sca in val.scalars]
                if val.name == 'temperature':
                    temp = float(val.scalars[0].value)
                if val.name == 'source wavelength':
                    src_wl = float(val.scalars[0].value)
            q_I = np.vstack([q,I]).T
        elif prop_nm in profiler.profile_keys:
            features[prop_nm] = float(prop.scalars[0].value)
        elif any([rp == prop_nm[-1*len(rp):] for rp in regression_params]):
            reg_pp_outputs[prop_nm]= prop.scalars[0].value
    return pp.uid,expt_id,t_utc,q_I,temp,src_wl,populations,fp,pb,pc,features,cl_pp_output,reg_pp_outputs

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
            if "basis" in popd.keys():
                for ist,site_nm in enumerate(popd['basis'].keys()):
                    sited = popd['basis'][site_nm]
                    main_cls += 'site{}_{}__'.format(ist,sited['form'])
                    sitenm_cls = Classification('pop{}_site{}_name'.format(ip,ist),site_nm)
                    site_cls = Classification('pop{}_site{}_form'.format(ip,ist),sited['form'])
                    clss.extend([sitenm_cls,site_cls])
    if main_cls[-2:] == '__':
        main_cls = main_cls[:-2]
    if main_cls == '':
        main_cls = 'noise'
    clss.append(Classification('system_classification',main_cls))
    return clss

def system_properties(opd,fixed_params,param_bounds,param_constraints):
    properties = []
    for ip,popnm in enumerate(opd.keys()):
        if popnm == 'noise' and not opd[popnm]['structure'] == 'unidentified':
            properties.append(Property('noise_intensity',opd[popnm]['parameters']['I0']))
        else:
            popd = opd[popnm]
            popfp, poppb, poppc = None, None, None
            if contains_params(fixed_params,popnm): popfp = fixed_params[popnm]
            if contains_params(param_bounds,popnm): poppb = param_bounds[popnm]
            if contains_params(param_constraints,popnm): poppc = param_constraints[popnm]
            if 'settings' in popd:
                properties.extend(setting_properties(ip,popd))
            if 'parameters' in popd:
                properties.extend(param_properties(ip,popd,popfp,poppb,poppc))
            if 'basis' in popd:
                for ist, stnm in enumerate(popd['basis'].keys()):
                    stdef = popd['basis'][stnm]
                    stfp, stpb, stpc = None, None, None
                    if contains_site_params(fixed_params,popnm,stnm):
                        stfp = fixed_params[popnm]['basis'][stnm]
                    if contains_site_params(param_bounds,popnm,stnm):
                        stpb = param_bounds[popnm]['basis'][stnm]
                    if contains_site_params(param_constraints,popnm,stnm):
                        stpc = param_constraints[popnm]['basis'][stnm]
                    if 'coordinates' in stdef:
                        properties.extend(site_coord_properties(ip,ist,stdef))
                    if 'settings' in stdef:
                        properties.extend(site_setting_properties(ip,ist,stdef))
                    if 'parameters' in stdef:
                        properties.extend(site_param_properties(ip,ist,stdef,stfp,stpb,stpc))
    return properties

def setting_properties(ip,popd):
    pps = []
    for stgnm,stgval in popd['settings'].items():
        pp = Property('pop{}_{}'.format(ip,stgnm))
        pp.tags = str(stgval)
        pps.append(pp)
    return pps

def param_properties(ip,popd,popfp=None,poppb=None,poppc=None):
    pps = []
    for pnm,pval in popd['parameters'].items():
        pp = Property('pop{}_{}'.format(ip,pnm),pval)
        pp.tags = []
        if popfp is not None:
            if pnm in popfp['parameters']:
                pp.tags.append('fixed value: {}'.format(bool(pval)))
        if poppb is not None:
            if pnm in poppb['parameters']:
                lbnd = poppb['parameters'][pnm][0]
                ubnd = poppb['parameters'][pnm][1]
                pp.tags.append('bounds: [{},{}]'.format(lbnd,ubnd))
        if poppc is not None:
            if pnm in poppc['parameters']:
                cexpr = poppc['parameters'][pnm]
                pp.tags.append('constraint expression: {}'.format(cexpr))
        pps.append(pp)
    return pps
                                    
def site_setting_properties(ip,ist,stdef):
    pps = []
    for stgnm,stgval in stdef['settings'].items():
        pp = Property('pop{}_site{}_{}'.format(ip,ist,stgnm))
        pp.tags = str(stgval)
        pps.append(pp)
    return pps

# TODO: deal with fixed_params, param_bounds, param_constraints on site coordinates
def site_coord_properties(ip,ist,stdef):
    pps = []
    for ic,cval in enumerate(stdef['coordinates']):
        pp = Property('pop{}_site{}_coordinate{}'.format(ip,ist,ic),cval)
        pps.append(pp)
    return pps

def site_param_properties(ip,ist,stdef,stfp,stpb,stpc):
    pps = []
    for pnm,pval in stdef['parameters'].items():
        pp = Property('pop{}_site{}_{}'.format(ip,ist,pnm),pval)
        pps.append(pp)
        pp.tags = []
        if stfp is not None:
            if pnm in stfp['parameters']:
                pp.tags.append('fixed value: {}'.format(bool(pval)))
        if stpb is not None:
            if pnm in stpb['parameters']:
                lbnd = stpb['parameters'][pnm][0]
                ubnd = stpb['parameters'][pnm][1]
                pp.tags.append('bounds: [{},{}]'.format(lbnd,ubnd))
        if stpc is not None:
            if pnm in stpc['parameters']:
                cexpr = stpc['parameters'][pnm]
                pp.tags.append('constraint expression: {}'.format(cexpr))
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


