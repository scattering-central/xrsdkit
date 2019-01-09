from collections import OrderedDict
import re
import copy

import pandas as pd
import numpy as np
from pypif.obj import ChemicalSystem, Property, Classification, Id, Value, FileReference, Scalar
from citrination_client import PifSystemReturningQuery, DatasetQuery, DataQuery, Filter

from . import profiler
from .. import definitions as xrsdefs 
from ..scattering.form_factors import atomic_params
from ..system import System

_reg_params = list(xrsdefs.param_defaults.keys())
_reg_params[_reg_params.index('I0')] = 'I0_fraction'

def make_pif(sys=None,q_I=None):
    """Make a pypif.obj.ChemicalSystem object describing XRSD data.

    Parameters
    ----------
    sys : xrsdkit.system.System 
        System object describing populations and parameters
    q_I : array
        n-by-2 array of q (1/Angstrom) and intensity (arb)

    Returns
    -------
    csys : pypif.obj.ChemicalSystem
        pypif.obj.ChemicalSystem representation of the PIF record
    """
    expt_id = sys.sample_metadata['experiment_id']
    t = sys.sample_metadata['time']
    uid = sys.sample_metadata['sample_id']
    src_wl = sys.sample_metadata['source_wavelength']
    q_I_file = sys.sample_metadata['data_file']
    if not uid:
        uid = 'tmp'
        if expt_id:
            uid = expt_id
        if t:
            uid = uid+'_'+str(int(t))
    csys = ChemicalSystem()
    csys.uid = uid
    csys.ids = []
    csys.tags = []
    csys.properties = []
    csys.classifications = []
    if expt_id:
        csys.ids.append(id_tag('experiment_id',expt_id))
    if t is not None:
        csys.tags.append('time (seconds): '+str(int(t)))
    if sys is not None and src_wl is not None:
        sys_clss, sys_props = pack_system_objects(sys)
        csys.classifications.extend(sys_clss)
        csys.properties.extend(sys_props)
    if q_I_file is not None:
        csys.properties.append(Property('q_I_file',files=[FileReference(q_I_file)]))
    if q_I is not None:
        csys.properties.extend(profile_properties(q_I[:,0],q_I[:,1]))
    return csys

def id_tag(idname,idval,tags=None):
    return Id(idname,idval,tags)

def profile_properties(q,I):
    prof = profiler.profile_pattern(q,I)
    pp = []
    for fnm,fval in prof.items():
        if fval is not None:
            pp.append(Property(fnm,fval))
    return pp

def pack_system_objects(sys,temp_C=None):
    """Return pypif.obj objects describing System attributes"""
    all_props = []
    all_clss = [] 
    sys_cls = ''
    ipop = 0
    src_wl = sys.sample_metadata['source_wavelength']
    if sys.fit_report:
        all_props.append(fit_report_property(sys.fit_report))
    if src_wl is not None:
        all_props.append(Property('source_wavelength',src_wl))
    if temp_C is not None:
        all_props.append(Property('temperature',temp_C,units='degrees C'))
    if any([p.structure=='unidentified' for pnm,p in sys.populations.items()]):
        sys_cls = 'unidentified'
    else:
        I0 = sys.compute_intensity(np.array([0.]))[0]
        I0_noise = sys.noise_model.compute_intensity(np.array([0.]))[0]
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
    # TODO: ensure the sort results are correct
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

def pif_property_from_param(param_nm,paramd):
    pp = Property(param_nm,paramd['value'])
    pp.tags = []
    pp.tags.append('fixed: {}'.format(bool(paramd['fixed'])))
    pp.tags.append('bounds: [{},{}]'.format(paramd['bounds'][0],paramd['bounds'][1]))
    pp.tags.append('constraint_expr: {}'.format(paramd['constraint_expr']))
    return pp

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

