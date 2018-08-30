from collections import OrderedDict
import re

import numpy as np
from pypif.obj import ChemicalSystem, Property, Classification, Id, Value, Scalar

from . import profiler
from .. import * 
from ..scattering.form_factors import atomic_params
from ..system import System

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
    if sys is not None: 
        sys_clss, sys_props = pack_system_objects(sys)
        csys.classifications.extend(sys_clss)
        csys.properties.extend(sys_props)
    if q_I is not None:
        csys.properties.extend(profile_properties(q_I))
    return csys
    # TODO: update invocations of make_pif() to take System objects as input

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

    # unpack noise model
    noise_model = {} 
    if 'noise_classification' in cls_dict:
        noise_cls = cls_dict.pop('noise_classification')
        noise_ids = noise_cls.value.split('__')
        for noise_id in noise_ids:
            noise_model[noise_id] = {}
            for param_nm in noise_params[noise_id]:
                noise_param_name = 'noise__'+noise_id+'__'+param_nm
                noise_model[noise_id][param_nm] = param_from_pif_property(param_nm,props_dict.pop(noise_param_name)) 

    # use the remaining cls_dict entries to rebuild the System  
    popd = OrderedDict()
    # identify population names and structure specifications 
    ip = 0
    pops_found = False
    while not pops_found:
        if 'pop{}_structure'.format(ip) in cls_dict:
            cls = cls_dict['pop{}_structure'.format(ip)] 
            popd[cls.tags[0]] = dict(structure=cls.value,
            basis={},settings={},parameters={}) 
            ip += 1
        else:
            pops_found = True
    for ip,popnm in enumerate(popd.keys()):
        # identify any specie names and form factor specifications
        isp = 0
        species_found = False
        while not species_found:
            if 'pop{}_specie{}_form'.format(ip,isp) in cls_dict:
                cls = cls_dict['pop{}_specie{}_form'.format(ip,isp)]
                popd[popnm]['basis'][cls.tags[0]] = dict(form=cls.value,
                settings={},parameters={},coordinates=[None,None,None])
                isp += 1
            else:
                species_found = True
            if not popd[popnm]['structure'] == 'unidentified':
                # unpack the basis classification for this population
                basis_cls_label = 'pop{}_basis_classification'.format(ip)
                classification_labels[basis_cls_label] = cls_dict[basis_cls_label].value
        # moving forward, we assume the structure has been assigned,
        # all species in the basis have been identified,
        # and all settings and params exist in props_dict
        for param_nm in structure_params[popd[popnm]['structure']]:
            pl = 'pop{}_{}'.format(ip,param_nm) 
            popd[popnm]['parameters'][param_nm] = \
            param_from_pif_property(param_nm,props_dict[pl])
        for stg_nm in structure_settings[popd[popnm]['structure']]:  
            tp = setting_datatypes[stg_nm]
            stg_label = 'pop{}_{}'.format(ip,stg_nm) 
            stg_val = tp(props_dict[stg_label].tags[0])
            popd[popnm]['settings'][stg_nm] = stg_val
            # unpack classification labels for structure settings
            if stg_nm in ['lattice','interaction']:
                classification_labels[stg_label] = stg_val
        for isp, specie_nm in enumerate(popd[popnm]['basis'].keys()):
            specie_form = popd[popnm]['basis'][specie_nm]['form']
            for param_nm in form_factor_params[specie_form]:
                pl = 'pop{}_specie{}_{}'.format(ip,isp,param_nm) 
                popd[popnm]['basis'][specie_nm]['parameters'][param_nm] = \
                param_from_pif_property(param_nm,props_dict[pl])
            for stg_nm in form_factor_settings[specie_form]:  
                tp = setting_datatypes[stg_nm]
                popd[popnm]['basis'][specie_nm]['settings'][stg_nm] = \
                tp(props_dict['pop{}_specie{}_{}'.format(ip,isp,stg_nm)].tags[0])
            for ic in range(3):
                pl = 'pop{}_specie{}_coordinate{}'.format(ip,isp,ic)
                popd[popnm]['basis'][specie_nm]['coordinates'][ic] = \
                param_from_pif_property(param_nm,props_dict[pl])

    # popd should now contain all system information: build the System
    sys = System(popd)
    sys.fit_report = fit_rpt
    sys.noise_model = noise_model

    # unpack remaining properties not related to the system definition
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
        elif any([rp == prop_nm[-1*len(rp):] for rp in regression_params]):
            regression_outputs[prop_nm]= prop.scalars[0].value

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

def profile_properties(q_I):
    prof = profiler.profile_spectrum(q_I)
    pp = []
    for fnm,fval in prof.items():
        if fval is not None:
            pp.append(Property(fnm,fval))
            #props.append(scalar_property(
            #fnm,fval,'spectrum profiling quantity'))
    return pp

def pack_system_objects(sys):
    """Return pypif.obj objects describing System attributes"""
    all_props = []
    all_clss = [] 
    sys_cls = ''
    ipop = 0
    all_clss.append(noise_classification(sys.noise_model))
    all_props.extend(noise_properties(sys.noise_model))
    if sys.fit_report:
        all_props.append(fit_report_property(sys.fit_report))
    if any([p.structure=='unidentified' for pnm,p in sys.populations.items()]):
        sys_cls = 'unidentified'
    else:
        for struct_nm in structure_names: # use structure_names to impose order 
            struct_pops = OrderedDict() 
            for pop_nm,pop in sys.populations.items():
                if pop.structure == struct_nm:
                    struct_pops[pop_nm] = pop
            # sort any populations with same structure
            struct_pops = _sort_populations(struct_nm,struct_pops)
            for pop_nm,pop in struct_pops.items():
                all_props.extend(param_properties(ipop,pop))
                all_props.extend(setting_properties(ipop,pop))
                if sys_cls: sys_cls += '__'
                sys_cls += 'pop{}_{}'.format(ipop,pop.structure)
                all_clss.append(Classification('pop{}_structure'.format(ipop),pop.structure,[pop_nm]))
                bas_cls = ''
                ispec = 0
                for ff_nm in form_factor_names: # use form_factor_names to impose order
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

def noise_classification(noise_model):
    noise_cls = ''
    for nm in noise_model_names:
        if nm in noise_model:
            if noise_cls: noise_cls += '__'
            noise_cls += nm
    return Classification('noise_classification',noise_cls,None)

def noise_properties(noise_model):
    props = []
    for noise_nm,params in noise_model.items():
        for pnm,pd in params.items():
            props.append(pif_property_from_param('noise__'+noise_nm+'__'+pnm,pd))
    return props

def fit_report_property(fit_report):
    prop_val = None
    prop = Property('fit_report',fit_report['final_objective'])
    prop.tags = [rpt_key+': '+str(rpt_val) for rpt_key,rpt_val in fit_report.items()]
    return prop

# TODO: consider basis content when sorting populations?
# currently only sorts on structure params
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
        # order crystalline structures primarily according to the list xrsdkit.crystalline_structures
        for l in pop_labels: param_vals[l].append(crystalline_structures.index(pops_dict[l].settings['lattice']))
        param_labels.append('lattice')
        dtypes['lattice']='int'
        for param_nm in crystalline_structure_params[pops_dict[l].settings['lattice']]:
            for l in pop_labels: param_vals[l].append(pops_dict[l].parameters[param_nm]['value'])
            param_labels.append(param_nm)
            dtypes[param_nm]='float'
    # likewise for xrsdkit.disordered_structures 
    if struct_nm == 'disordered': 
        for l in pop_labels: param_vals[l].append(disordered_structures.index(pops_dict[l].settings['interaction']))
        param_labels.append('interaction')
        dtypes['interaction']='int'
        for param_nm in disordered_structure_params[pops_dict[l].settings['interaction']]:
            for l in pop_labels: param_vals[l].append(pops_dict[l].parameters[param_nm]['value'])
            param_labels.append(param_nm)
            dtypes[param_nm]='float'
    # for all structures, order by their structure_params,
    # from highest to lowest priority
    for param_nm in structure_params[struct_nm]:
        for l in pop_labels: param_vals[l].append(pops_dict[l].parameters[param_nm]['value'])
        param_labels.append(param_nm)
        dtypes[param_nm]='float'
    param_ar = np.array(
        [tuple([l]+param_vals[l]) for l in pop_labels], 
        dtype = [('pop_name','U32')]+[(pl,dtypes[pl]) for pl in param_labels]
        )

    # TODO: make tests that ensure the sort results are correct
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
        for l in specie_labels: param_vals[l].append(species_dict[l].parameters['Z'])
        param_labels.append('Z') 
        dtypes['Z'] = 'float'
    if ff_nm == 'standard_atomic':
        for l in specie_labels: param_vals[l].append(atomic_params[specie.settings['symbol']['Z']])
        param_labels.append('Z') 
        dtypes['Z'] = 'float'
    for param_nm in form_factor_params[ff_nm]:
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
    for stgnm in structure_settings[pop.structure]:
        stgval = setting_defaults[stgnm]
        if stgnm in pop.settings:
            stgval = pop.settings[stgnm]
        pp = Property('pop{}_{}'.format(ip,stgnm))
        pp.tags = [str(stgval)]
        pps.append(pp)
    return pps

def param_properties(ip,pop):
    pps = []
    param_nms = copy.deepcopy(structure_params[pop.structure])
    if pop.structure == 'crystalline':
        param_nms.extend(crystalline_structure_params[pop.settings['lattice']])
    if pop.structure == 'disordered':
        param_nms.extend(disordered_structure_params[pop.settings['interaction']])
    for param_nm in param_nms:
        pd = copy.deepcopy(param_defaults[param_nm])
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
        pnm = 'pop{}_specie{}_coordinate{}'.format(ip,isp,ic)
        pps.append(pif_property_from_param(pnm,cd))
    for param_nm,pd in specie.parameters.items():
        pnm = 'pop{}_specie{}_{}'.format(ip,isp,param_nm)
        pps.append(pif_property_from_param(pnm,pd))
    return pps

def param_from_pif_property(param_nm,prop):
    pdict = copy.deepcopy(param_defaults[param_nm])
    pdict['value'] = prop.scalars[0].value
    if prop.tags is not None:
        for tg in prop.tags:
            if 'fixed: ' in tg:
                pdict['fixed'] = bool(tg.strip('fixed: '))
            if 'bounds: ' in tg:
                bds = tg.strip('bounds: []').split(',')
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
                pdict['constraint_expr'] = tg.strip('constraint_expr: ')
    return pdict

def pif_property_from_param(param_nm,paramd):
    pp = Property(param_nm,paramd['value'])
    pp.tags = []
    pp.tags.append('fixed: {}'.format(bool(paramd['fixed'])))
    pp.tags.append('bounds: [{},{}]'.format(paramd['bounds'][0],paramd['bounds'][1]))
    pp.tags.append('constraint_expr: {}'.format(paramd['constraint_expr']))
    return pp

#def scalar_property(fname,fval,desc=None,data_type=None,funits=None):
#    pf = Property()
#    pf.name = fname
#    if isinstance(fval,list):
#        pf.scalars = [Scalar(v) for v in fval]
#    else:
#        pf.scalars = [Scalar(fval)]
#    if desc:
#        pf.tags = [desc]
#    if data_type:
#        pf.dataType = data_type 
#    if funits:
#        pf.units = funits
#    return pf


