from collections import OrderedDict
import re
import copy

import pandas as pd
import numpy as np
from pypif.obj import ChemicalSystem, Property, Classification, Id, Value, Scalar
from citrination_client import PifSystemReturningQuery, DatasetQuery, DataQuery, Filter

from . import profiler
from .. import definitions as xrsdefs 
from ..scattering.form_factors import atomic_params
from ..system import System

_reg_params = list(xrsdefs.param_defaults.keys())
_reg_params[_reg_params.index('I0')] = 'I0_fraction'

def make_pif(uid,sys=None,q_I=None,expt_id=None,t_utc=None,temp_C=None,src_wl=None):
    """Make a pypif.obj.ChemicalSystem object describing XRSD data.

    Parameters
    ----------
    uid : str
        record id, should be unique across the dataset
    sys : xrsdkit.system.System 
        System object describing populations and parameters
    q_I : array
        n-by-2 array of q (1/Angstrom) and intensity (arb)
    expt_id : str
        experiment id for the system, used for dataset grouping
    t_utc : int
        UTC time in seconds
    temp_C : float
        temperature of the sample in degrees C
    src_wl : float
        wavelength of light source in Angstroms 

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
    if sys is not None and src_wl is not None:
        sys_clss, sys_props = pack_system_objects(sys,src_wl)
        csys.classifications.extend(sys_clss)
        csys.properties.extend(sys_props)
    if q_I is not None:
        csys.properties.extend(profile_properties(q_I[:,0],q_I[:,1]))
    return csys

def unpack_pif(pp):
    
    expt_id = None
    t_utc = None
    q_I = None
    temperature = None
    features = OrderedDict()
    regression_outputs = {}
    classification_labels = {}

    # pack classification labels into a dict
    cls_dict = {}
    if pp.classifications is not None:
        for cl in pp.classifications:
            cls_dict[cl.name] = cl

    # pack properties into a dict
    props_dict = {}
    if pp.properties is not None:
        for prop in pp.properties:
            props_dict[prop.name] = prop

    # unpack system classification
    classification_labels['system_classification'] = cls_dict.pop('system_classification').value

    # unpack noise classification 
    #if 'noise_classification' in cls_dict:
    noise_cls = cls_dict.pop('noise_classification')
    classification_labels['noise_classification'] = noise_cls.value
    noise_model = {'model':noise_cls.value,'parameters':{}} 
    for param_nm in xrsdefs.noise_params[noise_cls.value]:
        noise_param_name = 'noise_'+param_nm
        noise_model['parameters'][param_nm] = param_from_pif_property(param_nm,props_dict.pop(noise_param_name)) 

    # unpack fit report
    fit_rpt = {}
    if 'fit_report' in props_dict:
        prop = props_dict.pop('fit_report')
        for tg in prop.tags:
            if 'success: ' in tg: fit_rpt['success'] = bool(tg.strip('success: '))
            if 'initial_objective: ' in tg: fit_rpt['initial_objective'] = float(tg.strip('initial_objective: '))
            if 'final_objective: ' in tg: fit_rpt['final_objective'] = float(tg.strip('final_objective: '))
            if 'error_weighted: ' in tg: fit_rpt['error_weighted'] = bool(tg.strip('error_weighted: '))
            if 'logI_weighted: ' in tg: fit_rpt['logI_weighted'] = bool(tg.strip('logI_weighted: '))
            if 'q_range: ' in tg: 
                bds = tg.strip('q_range: []').split(',')
                fit_rpt['q_range'] = [float(bds[0]), float(bds[1])]
            if 'fit_snr: ' in tg: fit_rpt['fit_snr'] = float(tg.strip('fit_snr: '))

    # use the remaining cls_dict entries to rebuild the System  
    sysd = OrderedDict()
    # identify population names and structure specifications 
    ip = 0
    pops_found = False
    while not pops_found:
        if 'pop{}_structure'.format(ip) in cls_dict:
            cls = cls_dict['pop{}_structure'.format(ip)] 
            sysd[cls.tags[0]] = dict(structure=cls.value,
            basis={},settings={},parameters={}) 
            ip += 1
        else:
            pops_found = True
    for ip,popnm in enumerate(sysd.keys()):
        # identify any specie names and form factor specifications
        isp = 0
        species_found = False
        while not species_found:
            if 'pop{}_specie{}_form'.format(ip,isp) in cls_dict:
                cls = cls_dict['pop{}_specie{}_form'.format(ip,isp)]
                sysd[popnm]['basis'][cls.tags[0]] = dict(form=cls.value,
                settings={},parameters={},coordinates=[None,None,None])
                isp += 1
            else:
                species_found = True
            if not sysd[popnm]['structure'] == 'unidentified':
                # unpack the basis classification for this population
                basis_cls_label = 'pop{}_basis_classification'.format(ip)
                classification_labels[basis_cls_label] = cls_dict[basis_cls_label].value
        # moving forward, we assume the structure has been assigned,
        # all species in the basis have been identified,
        # and all settings and params exist in props_dict
        for param_nm in xrsdefs.structure_params[sysd[popnm]['structure']]:
            pl = 'pop{}_{}'.format(ip,param_nm) 
            sysd[popnm]['parameters'][param_nm] = \
            param_from_pif_property(param_nm,props_dict[pl])
        for stg_nm in xrsdefs.structure_settings[sysd[popnm]['structure']]:  
            tp = xrsdefs.setting_datatypes[stg_nm]
            stg_label = 'pop{}_{}'.format(ip,stg_nm) 
            stg_val = tp(props_dict[stg_label].tags[0])
            sysd[popnm]['settings'][stg_nm] = stg_val
            # unpack classification labels for structure settings
            if stg_nm in ['lattice','interaction']:
                classification_labels[stg_label] = stg_val
        for isp, specie_nm in enumerate(sysd[popnm]['basis'].keys()):
            # TODO (later): unpack classification label for atom symbol, if atomic
            specie_form = sysd[popnm]['basis'][specie_nm]['form']
            for param_nm in xrsdefs.form_factor_params[specie_form]:
                pl = 'pop{}_specie{}_{}'.format(ip,isp,param_nm) 
                sysd[popnm]['basis'][specie_nm]['parameters'][param_nm] = \
                param_from_pif_property(param_nm,props_dict[pl])
            for stg_nm in xrsdefs.form_factor_settings[specie_form]:  
                tp = xrsdefs.setting_datatypes[stg_nm]
                sysd[popnm]['basis'][specie_nm]['settings'][stg_nm] = \
                tp(props_dict['pop{}_specie{}_{}'.format(ip,isp,stg_nm)].tags[0])
            for ic,coord_id in enumerate(['x','y','z']):
                # TODO: (later): coordinates should be regression outputs,
                # if more than one specie exists in the population
                pl = 'pop{}_specie{}_coord{}'.format(ip,isp,coord_id)
                sysd[popnm]['basis'][specie_nm]['coordinates'][ic] = \
                param_from_pif_property(param_nm,props_dict[pl])

    # sysd should now contain all system information: build the System
    sysd['noise'] = noise_model
    sysd['fit_report'] = fit_rpt
    sys = System(sysd)

    # unpack remaining properties not related to the system structure 
    for prop_nm, prop in props_dict.items():
        if prop_nm == 'Intensity':
            I = [float(sca.value) for sca in prop.scalars]
            for val in prop.conditions:
                if val.name == 'scattering vector magnitude':
                    q = [float(sca.value) for sca in val.scalars]
                if val.name == 'temperature':
                    temperature = float(val.scalars[0].value)
                if val.name == 'source wavelength':
                    src_wl = float(val.scalars[0].value)
            q_I = np.vstack([q,I]).T
        elif prop_nm in profiler.profile_keys:
            features[prop_nm] = float(prop.scalars[0].value)
        elif prop_nm == 'noise_I0_fraction':
            regression_outputs[prop_nm] = prop.scalars[0].value
        elif any([rp == prop_nm[-1*len(rp):] for rp in _reg_params]):
            regression_outputs[prop_nm] = prop.scalars[0].value

    # unpack the experiment_id
    if pp.ids is not None:
        for iidd in pp.ids:
            if iidd.name == 'EXPERIMENT_ID':
                expt_id = iidd.value

    # unpack the time in seconds utc
    if pp.tags is not None:
        for ttgg in pp.tags:
            if 'time (utc): ' in ttgg:
                t_utc = float(ttgg.replace('time (utc): ',''))

    return pp.uid,sys,q_I,expt_id,t_utc,temperature,src_wl,\
        features,classification_labels,regression_outputs

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

def profile_properties(q,I):
    prof = profiler.profile_pattern(q,I)
    pp = []
    for fnm,fval in prof.items():
        if fval is not None:
            pp.append(Property(fnm,fval))
    return pp

def pack_system_objects(sys,src_wl):
    """Return pypif.obj objects describing System attributes"""
    all_props = []
    all_clss = [] 
    sys_cls = ''
    ipop = 0
    if sys.fit_report:
        all_props.append(fit_report_property(sys.fit_report))
    if any([p.structure=='unidentified' for pnm,p in sys.populations.items()]):
        sys_cls = 'unidentified'
    else:
        I0 = sys.compute_intensity(np.array([0.]),src_wl)[0]
        I0_noise = sys.compute_noise_intensity(np.array([0.]))[0]
        if I0 == 0.: 
            all_props.append(Property('noise_I0_fraction',0.))
        else:
            all_props.append(Property('noise_I0_fraction',I0_noise/I0))
        all_clss.append(Classification('noise_classification',sys.noise_model.model,None))
        for param_nm,pd in sys.noise_model.parameters.items():
            all_props.append(pif_property_from_param('noise_'+param_nm,pd))
        for struct_nm in xrsdefs.structure_names: # use structure_names to impose order 
            struct_pops = OrderedDict() 
            for pop_nm,pop in sys.populations.items():
                if pop.structure == struct_nm:
                    struct_pops[pop_nm] = pop
            # sort any populations with same structure
            struct_pops = _sort_populations(struct_nm,struct_pops)
            for pop_nm,pop in struct_pops.items():
                if I0 == 0.:
                    all_props.append(Property('pop{}_I0_fraction'.format(ipop),0.))
                else:
                    all_props.append(Property('pop{}_I0_fraction'.format(ipop),
                    pop.compute_intensity(np.array([0.]),src_wl)[0]/I0))
                all_props.extend(param_properties(ipop,pop))
                all_props.extend(setting_properties(ipop,pop))
                if sys_cls: sys_cls += '__'
                sys_cls += 'pop{}_{}'.format(ipop,pop.structure)
                all_clss.append(Classification('pop{}_structure'.format(ipop),pop.structure,[pop_nm]))
                bas_cls = ''
                ispec = 0
                for ff_nm in xrsdefs.form_factor_names: # use form_factor_names to impose order
                    ff_species = OrderedDict() 
                    for specie_nm,specie in pop.basis.items():
                        if specie.form == ff_nm:
                            ff_species[specie_nm] = specie
                    # sort any species with same form
                    ff_species = _sort_species(ff_nm,ff_species)
                    for specie_nm,specie in ff_species.items():
                        all_props.extend(specie_param_properties(ipop,ispec,specie))
                        all_props.extend(specie_setting_properties(ipop,ispec,specie))
                        if bas_cls: bas_cls += '__'
                        bas_cls += 'specie{}_{}'.format(ispec,specie.form)
                        all_clss.append(Classification('pop{}_specie{}_form'.format(ipop,ispec),specie.form,[specie_nm]))
                        ispec += 1
                all_clss.append(Classification('pop{}_basis_classification'.format(ipop),bas_cls,None))
                ipop += 1
    all_clss.append(Classification('system_classification',sys_cls,None))
    return all_clss, all_props

def fit_report_property(fit_report):
    prop_val = None
    prop = Property('fit_report',fit_report['final_objective'])
    prop.tags = [rpt_key+': '+str(rpt_val) for rpt_key,rpt_val in fit_report.items()]
    return prop

def _sort_populations(struct_nm,pops_dict):
    """Sort a set of populations (all with the same structure)"""
    if struct_nm == 'unidentified' or len(pops_dict) < 2: 
        return pops_dict
    new_pops = OrderedDict()

    # get a list of the population labels
    pop_labels = list(pops_dict.keys())

    # collect params for each population
    param_vals = dict.fromkeys(pop_labels)
    for l in pop_labels: param_vals[l] = []
    param_labels = []
    dtypes = {}
    if struct_nm == 'crystalline': 
        # order crystalline structures primarily according to their lattice
        for l in pop_labels: param_vals[l].append(
        xrsdefs.setting_selections['lattice'].index(pops_dict[l].settings['lattice']))
        param_labels.append('lattice')
        dtypes['lattice']='int'
        for param_nm in xrsdefs.setting_params['lattice'][pops_dict[l].settings['lattice']]:
            for l in pop_labels: param_vals[l].append(pops_dict[l].parameters[param_nm]['value'])
            param_labels.append(param_nm)
            dtypes[param_nm]='float'
    if struct_nm == 'disordered': 
        for l in pop_labels: param_vals[l].append(
        xrsdefs.setting_selections['interaction'].index(pops_dict[l].settings['interaction']))
        param_labels.append('interaction')
        dtypes['interaction']='int'
        for param_nm in xrsdefs.setting_params['interaction'][pops_dict[l].settings['interaction']]:
            for l in pop_labels: param_vals[l].append(pops_dict[l].parameters[param_nm]['value'])
            param_labels.append(param_nm)
            dtypes[param_nm]='float'

    # for diffuse structures, order primarily by the first specie in the basis
    if struct_nm == 'diffuse':
        for l in pop_labels: 
            bk0 = list(pops_dict[l].basis.keys())[0]
            param_vals[l].append(xrsdefs.form_factor_names.index(pops_dict[l].basis[bk0].form))
        param_labels.append('form')
        dtypes['form']='int'
    # for all structures, order by their structure_params,
    # from highest to lowest priority
    for param_nm in xrsdefs.structure_params[struct_nm]:
        for l in pop_labels: param_vals[l].append(pops_dict[l].parameters[param_nm]['value'])
        param_labels.append(param_nm)
        dtypes[param_nm]='float'
    param_ar = np.array(
        [tuple([l]+param_vals[l]) for l in pop_labels], 
        dtype = [('pop_name','U32')]+[(pl,dtypes[pl]) for pl in param_labels]
        )

    # TODO: ensure the sort results are correct
    param_ar.sort(axis=0,order=param_labels)
    for ip,p in enumerate(param_ar): new_pops[p[0]] = pops_dict[p[0]]

    return new_pops

def _sort_species(ff_nm,species_dict):
    """Sort a set of species (all with the same form)"""
    if ff_nm == 'flat' or len(species_dict)<2:
        return species_dict
    new_species = OrderedDict()

    # get list of specie labels
    specie_labels = list(species_dict.keys())

    # collect parameter values, labels, dtypes
    param_vals = dict.fromkeys(specie_labels)
    for l in specie_labels: param_vals[l] = []
    param_labels = []
    dtypes = {}
    if ff_nm == 'atomic':
        for l in specie_labels: param_vals[l].append(atomic_params[specie.settings['symbol']]['Z'])
        param_labels.append('Z') 
        dtypes['Z'] = 'float'
    for param_nm in xrsdefs.form_factor_params[ff_nm]:
        for l in specie_labels: param_vals[l].append(species_dict[l].parameters[param_nm]['value'])
        param_labels.append(param_nm)
        dtypes[param_nm]='float'
    
    param_ar = np.array(
        [tuple([l]+param_vals[l]) for l in specie_labels], 
        dtype = [('specie_name','U32')]+[(pl,dtypes[pl]) for pl in param_labels]
        )
    # TODO: make tests that ensure the sort results are correct
    param_ar.sort(axis=0,order=param_labels)

    for ip,p in enumerate(param_ar):
        new_species[p[0]] = species_dict[p[0]]
    return new_species

def setting_properties(ip,pop):
    pps = []
    for stgnm in xrsdefs.structure_settings[pop.structure]:
        stgval = xrsdefs.setting_defaults[stgnm]
        if stgnm in pop.settings:
            stgval = pop.settings[stgnm]
        pp = Property('pop{}_{}'.format(ip,stgnm))
        pp.tags = [str(stgval)]
        pps.append(pp)
    return pps

def param_properties(ip,pop):
    pps = []
    param_nms = copy.deepcopy(xrsdefs.structure_params[pop.structure])
    if pop.structure == 'crystalline':
        param_nms.extend(xrsdefs.setting_params['lattice'][pop.settings['lattice']])
    if pop.structure == 'disordered':
        param_nms.extend(xrsdefs.setting_params['interaction'][pop.settings['interaction']])
    for param_nm in param_nms:
        pd = copy.deepcopy(xrsdefs.param_defaults[param_nm])
        if param_nm in pop.parameters:
            pd = pop.parameters[param_nm]
        pnm = 'pop{}_{}'.format(ip,param_nm)
        pps.append(pif_property_from_param(pnm,pd))
    return pps

def specie_setting_properties(ip,isp,specie):
    pps = []
    for stgnm,stgval in specie.settings.items():
        pp = Property('pop{}_specie{}_{}'.format(ip,isp,stgnm))
        pp.tags = [str(stgval)]
        pps.append(pp)
    return pps

def specie_param_properties(ip,isp,specie):
    pps = []
    for ic,cd in enumerate(specie.coordinates):
        if ic == 0: coord_id = 'x'
        if ic == 1: coord_id = 'y'
        if ic == 2: coord_id = 'z'
        pnm = 'pop{}_specie{}_coord{}'.format(ip,isp,coord_id)
        pps.append(pif_property_from_param(pnm,cd))
    for param_nm,pd in specie.parameters.items():
        pnm = 'pop{}_specie{}_{}'.format(ip,isp,param_nm)
        pps.append(pif_property_from_param(pnm,pd))
    return pps

def param_from_pif_property(param_nm,prop):
    pdict = copy.deepcopy(xrsdefs.param_defaults[param_nm])
    pdict['value'] = float(prop.scalars[0].value)
    if prop.tags is not None:
        for tg in prop.tags:
            if 'fixed: ' in tg:
                pdict['fixed'] = bool(tg[7:])
            if 'bounds: ' in tg:
                bds = tg[8:].strip('[]').split(',')
                try: 
                    lbd = float(bds[0]) 
                except:
                    lbd = None
                try:
                    ubd = float(bds[1]) 
                except:
                    ubd = None
                pdict['bounds'] = [lbd, ubd]
            if 'constraint_expr: ' in tg:
                expr = tg[17:]
                if expr == 'None': expr = None
                pdict['constraint_expr'] = expr
    return pdict

def pif_property_from_param(param_nm,paramd):
    pp = Property(param_nm,paramd['value'])
    pp.tags = []
    pp.tags.append('fixed: {}'.format(bool(paramd['fixed'])))
    pp.tags.append('bounds: [{},{}]'.format(paramd['bounds'][0],paramd['bounds'][1]))
    pp.tags.append('constraint_expr: {}'.format(paramd['constraint_expr']))
    return pp

def get_data_from_Citrination(client, dataset_id_list):
    """Get data from Citrination and create a dataframe.

    Parameters
    ----------
    client : citrination_client.CitrinationClient
        A python CitrinationClient for fetching data
    dataset_id_list : list of int
        List of dataset ids (integers) for fetching xrsdkit PIF records

    Returns
    -------
    df_work : pandas.DataFrame
        dataframe containing features and labels built from xrsdkit PIFs
    """
    pifs = get_pifs_from_Citrination(client,dataset_id_list)

    data = []
    # reg_labels and cls_labels are lists of dicts,
    # containing regression and classification outputs for each sample
    reg_labels = []
    cls_labels = []
    # all_reg_labels and all_cls_labels are sets of all
    # unique labels over the provided datasets 
    all_reg_labels = set()
    all_cls_labels = set()
    for i,pp in enumerate(pifs):
        pif_uid, sys, q_I, expt_id, t_utc, temp, src_wl, \
        features, classification_labels, regression_outputs = unpack_pif(pp)
        for k,v in regression_outputs.items():
            all_reg_labels.add(k)
        reg_labels.append(regression_outputs)
        for k,v in classification_labels.items():
            all_cls_labels.add(k)
        cls_labels.append(classification_labels)
        # NOTE: the index `i` is added at the end of each data row,
        # to index the pif that was originally packed there 
        data_row = [expt_id] + list(features.values()) + [i]
        data.append(data_row)

    reg_labels_list = list(all_reg_labels)
    reg_labels_list.sort()
    cls_labels_list = list(all_cls_labels)
    cls_labels_list.sort()

    for datai,rli,cli in zip(data,reg_labels,cls_labels):
        orl = OrderedDict.fromkeys(reg_labels_list)
        ocl = OrderedDict.fromkeys(cls_labels_list)
        orl.update(rli)
        ocl.update(cli)
        datai.extend(list(orl.values()))
        datai.extend(list(ocl.values()))

    colnames = ['experiment_id'] + \
            copy.deepcopy(profiler.profile_keys) + \
            ['local_id'] + \
            reg_labels_list + \
            cls_labels_list
    d = pd.DataFrame(data=data, columns=colnames)
    d['system_classification'] = d['system_classification'].where(
        (pd.notnull(d['system_classification'])),'unidentified')
    df_work = d.where((pd.notnull(d)), None) # replace all NaN by None

    return df_work

def get_pifs_from_Citrination(client, dataset_id_list):
    all_hits = []
    print('fetching PIF records from datasets: {}...'.format(dataset_id_list))
    for dataset in dataset_id_list:
        query = PifSystemReturningQuery(
            from_index=0,
            size=100,
            query=DataQuery(
                dataset=DatasetQuery(
                    id=Filter(
                    equal=dataset))))
        current_result = client.search.pif_search(query)
        while current_result.hits!=[]:
            all_hits.extend(current_result.hits)
            n_current_hits = len(current_result.hits)
            #n_hits += n_current_hits
            query.from_index += n_current_hits 
            current_result = client.search.pif_search(query)
    pifs = [x.system for x in all_hits]
    print('done - found {} records'.format(len(pifs)))
    return pifs


