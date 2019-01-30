from collections import OrderedDict
import os
import sys
import copy

import pandas as pd
import numpy as np
import yaml

from . import profiler, primitives
from ..system import System
from .. import definitions as xrsdefs

def save_sys_to_yaml(file_path,sys):
    sd = sys.to_dict()
    with open(file_path, 'w') as yaml_file:
        yaml.dump(primitives(sd),yaml_file)

def load_sys_from_yaml(file_path):
    with open(file_path, 'r') as yaml_file:
        sd = yaml.load(yaml_file)
    return System(**sd)

def read_all_experiments(path_to_dir):
    """Load xrsdkit data from a directory.

    The directory should contain subdirectories,
    one for each experiment in the dataset.
    Each subdirectory should contain .yml files describing 
    the xrsdkit.system.System objects from the experiment.

    Parameters
    ----------
    path_to_dir : str
        absolute path to the root directory of the dataset

    Returns
    -------
    sys_dicts : list
        list of dictionaries loaded from any .yml files 
        that were found in the experiment subdirectories
    """
    sys_dicts = []
    for experiment in os.listdir(path_to_dir):
        exp_data_dir = os.path.join(path_to_dir,experiment)
        if os.path.isdir(exp_data_dir):
            for s_data_file in os.listdir(exp_data_dir):
                if s_data_file.endswith('.yml'):
                    file_path = os.path.join(exp_data_dir, s_data_file)
                    sys = load_sys_from_yaml(file_path)
                    #if bool(int(sys.fit_report['good_fit'])):
                    sys_dicts.append(sys.to_dict())
    return sys_dicts


def gather_dataset(path_to_dir,output_csv=False):
    """Build a DataFrame for a dataset saved in a local directory.

    The data directory should contain one or more subdirectories,
    where each subdirectory contains the .yml files for an experiment,
    where each .yml file describes one sample,
    as created by save_sys_to_yaml().
    If `output_csv` is True,
    the dataset is saved to dataset.csv in `path_to_dir`.

    Parameters
    ----------
    path_to_dir : str
        absolute path to the folder with the training set
        Precondition: dataset directory includes subdirectories 
        for each of the experiments in the dataset; 
        each experiment directory contains the .yml files
        describing xrsdkit.system.System objects that were fit 
        to scattering data from the experiment. 

    Returns
    -------
    df_work : pandas.DataFrame
        dataframe containing features and labels
        exctracted from the dataset.
    """
    data = []
    reg_labels = []
    cls_labels = []
    all_reg_labels = set()
    all_cls_labels = set()

    all_sys = read_xrsd_system_data(path_to_dir)

    for sys in all_sys:
        expt_id, sample_id, features, classification_labels, regression_outputs = \
            unpack_sample(sys)

        for k,v in regression_outputs.items():
            all_reg_labels.add(k)
        reg_labels.append(regression_outputs)

        for k,v in classification_labels.items():
            all_cls_labels.add(k)
        cls_labels.append(classification_labels)

        data_row = [expt_id] + [sample_id] + list(features.values())
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

    colnames = ['experiment_id'] + ['sample_id'] +\
            copy.copy(profiler.profile_keys) + \
            reg_labels_list + \
            cls_labels_list

    df_work = pd.DataFrame(data=data, columns=colnames)
    if output_csv:
        df_work.to_csv(os.path.join(path_to_dir,'dataset.csv'))
    return df_work


def unpack_sample(sys_dict):
    """Extract features and labels from the dict describing the sample.

    Parameters
    ----------
    sys_dict : dict
        dict containing description of xrsdkit.system.System.
        Includes fit_report, sample_metadata, features,
        noise_model, and one dict for each of the populations.

    Returns
    -------
    expt_id : str
        name of the experiment (must be unique for a training set)
    features : OrderedDict
        OrderedDict of features with their values,
        similar to output of xrsdkit.tools.profiler.profile_pattern()
    classification_labels : dict
        dict of all classification labels with their values for given sample
    regression_labels : dict
        dict of all regression labels with their values for given sample
    sample_id : str
        uinque sample id (includes experiment id)
    """
    expt_id = sys_dict['sample_metadata']['experiment_id']
    sample_id = sys_dict['sample_metadata']['sample_id']
    features = sys_dict['features']
    sys = System(**sys_dict)

    regression_labels = {}
    classification_labels = {}
    sys_cls = ''
    ipop = 0

    I0 = sys.noise_model.parameters['I0']['value']
    for k, v in sys.populations.items():
        I0 += v.parameters['I0']['value']

    I0_noise = sys.noise_model.parameters['I0']['value']
    if I0 == 0.:
        regression_labels['noise_I0_fraction'] = 0.
    else:
        regression_labels['noise_I0_fraction'] = I0_noise/I0

    classification_labels['noise_model'] =sys.noise_model.model
    for param_nm,pd in sys.noise_model.parameters.items():
        regression_labels['noise_'+param_nm] = pd['value']
    # use xrsdefs.structure_names to index the populations 
    for struct_nm in xrsdefs.structure_names:
        struct_pops = OrderedDict()
        for pop_nm,pop in sys.populations.items():
            if pop.structure == struct_nm:
                struct_pops[pop_nm] = pop
        # sort any populations with same structure
        struct_pops = sort_populations(struct_nm,struct_pops)
        for pop_nm, pop in struct_pops.items():
            if I0 == 0.:
                regression_labels['pop{}_I0_fraction'.format(ipop)] = 0.
            else:
                pop_I0 = pop.parameters['I0']['value']
                regression_labels['pop{}_I0_fraction'.format(ipop)] = pop_I0/I0
            #classification_labels['pop{}_structure'.format(ipop)] = pop.structure
            classification_labels['pop{}_form'.format(ipop)] = pop.form
            for param_nm, param_def in pop.parameters.items():
                regression_labels['pop{}_{}'.format(ipop,param_nm)] = param_def['value']
            for stg_nm, stg_val in pop.settings.items():
                if stg_nm in xrsdefs.modelable_structure_settings[pop.structure] \
                or stg_nm in xrsdefs.modelable_form_factor_settings[pop.form]:
                    classification_labels['pop{}_{}'.format(ipop,stg_nm)] = str(stg_val)
            if sys_cls: sys_cls += '__'
            sys_cls += pop.structure
            ipop += 1
    if sys_cls == '':
        sys_cls = 'unidentified'
    classification_labels['system_class'] =sys_cls
    return expt_id, sample_id, features, classification_labels, regression_labels


def sort_populations(struct_nm,pops_dict):
    """Sort a set of populations (all with the same structure)"""
    if len(pops_dict) < 2: 
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
        # order crystalline structures primarily by lattice,
        # secondly by form factor
        for l in pop_labels: 
            param_vals[l].append(sgs.lattices.index(pops_dict[l].settings['lattice']))
        param_labels.append('lattice')
        dtypes['lattice']='int'
        for l in pop_labels: 
            param_vals[l].append(xrsdefs.form_factor_names.index(pops_dict[l].form))
        param_labels.append('form')
        dtypes['form']='int'
        # NOTE: the following only works if the previous two categories were all-same
        #for param_nm in xrsdefs.structure_params(struct_nm,pops_dict[l].settings):
        #    for l in pop_labels: param_vals[l].append(pops_dict[l].parameters[param_nm]['value'])
        #    param_labels.append(param_nm)
        #    dtypes[param_nm]='float'
    if struct_nm == 'disordered': 
        # order disordered structures primarily by interaction,
        # secondly by form factor
        intxns = xrsdefs.setting_selections(struct_nm)['interaction']
        for l in pop_labels: 
            param_vals[l].append(intxns.index(pops_dict[l].settings['interaction']))
        param_labels.append('interaction')
        dtypes['interaction']='int'
        for l in pop_labels: 
            param_vals[l].append(xrsdefs.form_factor_names.index(pops_dict[l].form))
        param_labels.append('form')
        dtypes['form']='int'
        # NOTE: the following only works if the previous two categories were all-same
        #for param_nm in xrsdefs.structure_params(struct_nm,pops_dict[l].settings):
        #    for l in pop_labels: param_vals[l].append(pops_dict[l].parameters[param_nm]['value'])
        #    param_labels.append(param_nm)
        #    dtypes[param_nm]='float'
    # for diffuse structures, order primarily by form,
    # secondly by form factor params
    if struct_nm == 'diffuse':
        for l in pop_labels: 
            param_vals[l].append(xrsdefs.form_factor_names.index(pops_dict[l].form))
        param_labels.append('form')
        dtypes['form']='int'
        ff = pops_dict[pop_labels[0]].form
        if all([pops_dict[ll].form == ff for ll in pop_labels]):  
            for param_nm in xrsdefs.form_factor_params(ff):
                for l in pop_labels: param_vals[l].append(pops_dict[l].parameters[param_nm]['value'])
            param_labels.append(param_nm)
            dtypes[param_nm]='float'
    param_ar = np.array(
        [tuple([l]+param_vals[l]) for l in pop_labels], 
        dtype = [('pop_name','U32')]+[(pl,dtypes[pl]) for pl in param_labels]
        )
    param_ar.sort(axis=0,order=param_labels)
    for ip,p in enumerate(param_ar): new_pops[p[0]] = pops_dict[p[0]]
    return new_pops

