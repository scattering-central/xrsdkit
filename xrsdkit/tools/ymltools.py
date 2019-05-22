from __future__ import print_function
from collections import OrderedDict
import os
import sys
import copy
from distutils.dir_util import copy_tree
import shutil

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

def read_local_dataset(dataset_dir,downsampling_distance=None,message_callback=print):
    """Load xrsdkit data from a directory.

    The directory should contain subdirectories,
    one for each experiment in the dataset.
    Each subdirectory should contain .yml files describing 
    the xrsdkit.system.System objects from the experiment.
    Each .yml file should have a corresponding .dat file in the same directory,
    where the .dat file contains the integrated scattering pattern.
    The name of the .dat file should be specified in the .yml file,
    as the 'data_file' from the sample_metadata dictionary.
    TODO: move this dataset description to the main documentation,  
    then refer to it from here.

    Parameters
    ----------
    dataset_dir : str
        absolute path to the root directory of the dataset

    Returns
    -------
    df : pandas.DataFrame 
        modeling DataFrame built from dataset files
    """
    sys_dicts = OrderedDict() 
    for experiment in os.listdir(dataset_dir):
        exp_data_dir = os.path.join(dataset_dir,experiment)
        if os.path.isdir(exp_data_dir):
            for s_data_file in os.listdir(exp_data_dir):
                if s_data_file.endswith('.yml'):
                    message_callback('loading data from {}'.format(s_data_file))
                    file_path = os.path.join(exp_data_dir, s_data_file)
                    #sys = load_sys_from_yaml(file_path)
                    #if bool(int(sys.fit_report['good_fit'])):
                    sys_dicts[s_data_file] = yaml.load(open(file_path,'r')) 
    df = create_modeling_dataset(list(sys_dicts.values()),
                downsampling_distance=downsampling_distance,
                message_callback=message_callback)
    return df 

def migrate_features(data_dir):
    """Update features for all yml files in a local directory.

    Parameters
    ----------
    data_dir : str
        absolute path to the directory containing yml data 
    """
    print('BEGINNING FEATURE MIGRATION FOR DIRECTORY: {}'.format(data_dir))
    for s_data_file in os.listdir(data_dir):
        if s_data_file.endswith('.yml'):
            print('loading data from {}'.format(s_data_file))
            file_path = os.path.join(data_dir, s_data_file)
            sys = load_sys_from_yaml(file_path)
            q_I = np.loadtxt(os.path.join(data_dir,sys.sample_metadata['data_file']))
            sys.features = profiler.profile_pattern(q_I[:,0],q_I[:,1])
            save_sys_to_yaml(file_path,sys)
    print('FINISHED FEATURE MIGRATION')


def create_modeling_dataset(xrsd_system_dicts, downsampling_distance=None, message_callback=print):
    """Build a modeling DataFrame from xrsdkit.system.System objects.

    If `downsampling_distance` is not None, the dataset will be 
    downsampled with downsample_by_group(downsampling_distance).

    Parameters
    ----------
    xrsd_system_dicts: list of dict
        Dicts describing all xrsdkit.system.System 
        objects in the dataset. Each of these dicts should be 
        similar to the output of xrsdkit.system.System.to_dict().

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

    for sys in xrsd_system_dicts:
        expt_id, sample_id, good_fit, feature_labels, \
            classification_labels, regression_outputs = unpack_sample(sys)
        if good_fit:
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
        df_work = downsample_by_group(df_work,downsampling_distance,message_callback)
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
        id of the experiment (should be unique across all experiments)
    sample_id : str
        id of the sample (must be unique across all samples)
    good_fit : bool 
        True if this sample's fit is good enough to train models on it
    features : dict 
        dict of features with their values,
        similar to output of xrsdkit.tools.profiler.profile_pattern()
    classification_labels : dict
        dict of all classification labels with their values for given sample
    regression_labels : dict
        dict of all regression labels with their values for given sample
    """
    expt_id = sys_dict['sample_metadata']['experiment_id']
    sample_id = sys_dict['sample_metadata']['sample_id']
    features = sys_dict['features']
    good_fit = bool(sys_dict['fit_report']['good_fit'])
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
    return expt_id, sample_id, good_fit, features, classification_labels, regression_labels


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

def downsample_by_group(df,min_distance=1.,message_callback=print):
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
        message_callback('Downsampling data for group: {}'.format(group_labels))
        #lbl_df = _filter_by_labels(data,lbls)
        dsamp = downsample(df.iloc[grp].copy(), min_distance)
        message_callback('Finished downsampling: kept {}/{}'.format(len(dsamp),len(group_df)))
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

