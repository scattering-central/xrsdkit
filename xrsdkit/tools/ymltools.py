from collections import OrderedDict
import os
import sys
import copy

from sklearn import preprocessing
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

def read_all_experiments(dataset_dir):
    """Load xrsdkit data from a directory.

    The directory should contain subdirectories,
    one for each experiment in the dataset.
    Each subdirectory should contain .yml files describing 
    the xrsdkit.system.System objects from the experiment.

    Parameters
    ----------
    dataset_dir : str
        absolute path to the root directory of the dataset

    Returns
    -------
    sys_dicts : list
        list of dictionaries loaded from any .yml files 
        that were found in the experiment subdirectories
    """
    sys_dicts = []
    for experiment in os.listdir(dataset_dir):
        exp_data_dir = os.path.join(dataset_dir,experiment)
        if os.path.isdir(exp_data_dir):
            for s_data_file in os.listdir(exp_data_dir):
                if s_data_file.endswith('.yml'):
                    file_path = os.path.join(exp_data_dir, s_data_file)
                    sys = load_sys_from_yaml(file_path)
                    #if bool(int(sys.fit_report['good_fit'])):
                    sys_dicts.append(sys.to_dict())
    return sys_dicts


def gather_dataset(dataset_dir,downsampling_distance=None,output_csv=False):
    """Build a DataFrame for a dataset saved in a local directory.

    The data directory should contain one or more subdirectories,
    where each subdirectory contains the .yml files for an experiment,
    where each .yml file describes one sample,
    as created by save_sys_to_yaml().
    If `downsampling_distance` is not None,
    the dataset will be downsampled with downsample_by_group().
    If `output_csv` is True,
    the dataset is saved to dataset.csv in `dataset_dir`.

    Parameters
    ----------
    dataset_dir : str
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
    cls_labels = []
    reg_labels = []
    feat_labels = []
    all_reg_labels = set()
    all_cls_labels = set()

    all_sys = read_all_experiments(dataset_dir)

    for sys in all_sys:
        expt_id, sample_id, feature_labels, classification_labels, regression_outputs = \
            unpack_sample(sys)

        for k,v in regression_outputs.items():
            all_reg_labels.add(k)
        reg_labels.append(regression_outputs)

        for k,v in classification_labels.items():
            all_cls_labels.add(k)
        cls_labels.append(classification_labels)

        feat_labels.append(feature_labels)
        data.append([expt_id,sample_id])

    reg_labels_list = list(all_reg_labels)
    reg_labels_list.sort()

    cls_labels_list = list(all_cls_labels)
    cls_labels_list.sort()

    for datai,cli,rli,featsi in zip(data,cls_labels,reg_labels,feat_labels):
        ocl = OrderedDict.fromkeys(cls_labels_list)
        ocl.update(cli)
        orl = OrderedDict.fromkeys(reg_labels_list)
        orl.update(rli)
        ofl = OrderedDict.fromkeys(profiler.profile_keys)
        ofl.update(featsi)
        datai.extend(list(ocl.values()))
        datai.extend(list(orl.values()))
        datai.extend(list(ofl.values()))

    colnames = ['experiment_id'] + ['sample_id'] +\
            cls_labels_list + \
            reg_labels_list + \
            copy.copy(profiler.profile_keys)

    df_work = pd.DataFrame(data=data, columns=colnames)
    if downsampling_distance:
        df_work = downsample_by_group(df_work,downsampling_distance)
    if output_csv:
        df_work.to_csv(os.path.join(dataset_dir,'dataset.csv'))
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
    features : dict 
        dict of features with their values,
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

    classification_labels['noise_model'] = sys.noise_model.model
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
    classification_labels['system_class'] = sys_cls
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

def downsample_by_group(df,min_distance=1.):
    """Group and down-sample a DataFrame of xrsd records.
        
    Parameters
    ----------
    df : pandas.DataFrame
        dataframe containing xrsd samples 
    min_distance : float
        the minimum allowed nearest-neighbor distance 
        for continuing to downsample after 10 or more samples
        have been selected 

    Returns
    -------
    data_sample : pandas.DataFrame
        DataFrame containing all of the down-sampled data from 
        each group in the input dataframe.
        Features in this DataFrame are not scaled:
        the correct scaler should be applied before training models.
    """
    data_sample = pd.DataFrame(columns=df.columns)
    group_cols = ['experiment_id','system_class']
    all_groups = df.groupby(group_cols)
    # downsample each group independently
    for group_labels,grp in all_groups.groups.items():
        group_df = df.iloc[grp].copy()
        print('Downsampling data for group: {}'.format(group_labels))
        #lbl_df = _filter_by_labels(data,lbls)
        dsamp = downsample(df.iloc[grp].copy(), min_distance)
        print('Finished downsampling: kept {}/{}'.format(len(dsamp),len(group_df)))
        data_sample = data_sample.append(dsamp)
    return data_sample

def downsample(df, min_distance):
    """Downsample records from one DataFrame.

    Transforms the DataFrame feature arrays 
    (scaling by the columns in profiler.profile_keys),
    before collecting at least 10 samples.
    If the size of `df` is <= 10, it is returned directly.
    If it is larger than 10, the first point is chosen
    based on greatest nearest-neighbor distance.
    Subsequent points are chosen  
    in order of decreasing nearest-neighbor distance
    to the already-sampled points. 

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe containing xrsd samples 
    min_distance : float
        the minimum allowed nearest-neighbor distance 
        for continuing to downsample after 10 or more samples
        have been selected 

    Returns
    -------
    sample : pandas.DataFrame
        dataframe containing downsampled rows
    """
    df_size = len(df)
    sample = pd.DataFrame(columns=df.columns)
    if df_size <= 10:
        sample = sample.append(df)
    else:
        scaler = preprocessing.StandardScaler()
        scaler.fit(df[profiler.profile_keys])

        # get distance matrix between samples in scaled feature space
        features_matr = scaler.transform(df[profiler.profile_keys]) 
        dist_func = lambda i,j: np.sum(
            np.linalg.norm(features_matr[i]
            - features_matr[j]))
        # TODO: compute only the upper or lower triangle of this matrix
        dist_matrix = np.array([[dist_func(i,j) for i in range(df_size)] for j in range(df_size)])

        # artificially inflate self-distance,
        # so that samples are not their own nearest neighbors
        for i in range(df_size):
            dist_matrix[i,i] = float('inf')

        # samples are taken in order of greatest nearest-neighbor distance
        nn_distance_array = np.min(dist_matrix,axis=1)
        sample_order = np.argsort(nn_distance_array)[::-1]
        keep_samples = np.array([idx<10 or nn_distance_array[sample_idx]>min_distance for idx,sample_idx in enumerate(sample_order)])
        sample_order = sample_order[keep_samples]

        sample = sample.append(df.iloc[sample_order])
    return sample
