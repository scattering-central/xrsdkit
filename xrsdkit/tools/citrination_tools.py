from collections import OrderedDict
import os

import pandas as pd
import numpy as np
import yaml
from sklearn import preprocessing
from pypif import pif
from citrination_client import PifSystemReturningQuery, DatasetQuery, DataQuery, Filter

from ..tools import profiler
from ..tools import piftools


def get_data_from_Citrination(client, dataset_id_list):
    """Get data from Citrination and create a dataframe.

    Parameters
    ----------
    client : citrination_client.CitrinationClient
        A python CitrinationClient for fetching data
    dataset_id_list : list of int
        List of dataset ids (integers) for fetching SAXS records

    Returns
    -------
    df_work : pandas.DataFrame
        dataframe containing features and labels
        obtained through `client` from the Citrination datasets
        listed in `dataset_id_list`
    pifs : list
        list of pif objects. Each of them contains data about one sample.
    """
    data = []
    # reg_labels is a list of dicts of regression model outputs for each sample
    reg_labels = []
    # all_reg_labels will be the set of all
    # unique regression labels over the provided datasets 
    all_reg_labels = set()

    pifs = get_pifs_from_Citrination(client,dataset_id_list)

    for i,pp in enumerate(pifs):
        pif_uid, expt_id, t_utc, q_I, temp, src_wl, populations,fp,pb,pc, \
            pp_feats, cl_model_outputs, reg_model_outputs = piftools.unpack_pif(pp)
        feats = OrderedDict.fromkeys(profiler.profile_keys)
        feats.update(pp_feats)

        # TODO: explain why "i" is included in the data row 
        data_row = [expt_id]+list(feats.values())+[cl_model_outputs]+[i]

        data.append(data_row)
        for k,v in reg_model_outputs.items():
            all_reg_labels.add(k)
        reg_labels.append(reg_model_outputs)

    reg_labels_list = list(all_reg_labels)
    reg_labels_list.sort()

    for i,rl in enumerate(reg_labels):
        # create a dict of labels for all possible regression models
        lb = OrderedDict.fromkeys(reg_labels_list)
        # fill in values that were found in the record
        lb.update(rl)
        # add the regression labels to the end of the data row
        data[i] = data[i] + list(lb.values())

    colnames = ['experiment_id']
    colnames.extend(profiler.profile_keys)
    colnames.extend(['system_class'])
    # TODO: explain the local_id
    colnames.extend(['local_id'])
    colnames.extend(reg_labels_list)

    d = pd.DataFrame(data=data, columns=colnames)
    df_work = d.where((pd.notnull(d)), None) # replace all NaN by None

    return df_work, pifs

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


def downsample_Citrination_datasets(client, dataset_id_list, save_samples=True, train_hyperparameters=False):
    """Down-sample one or more datasets, and optionally save the samples.
        
    Down-sampled datasets are (optionally) saved to their datasets as assigned
    in the index file at xrsdkit/models/modeling_data/dataset_ids.yml. 

    Parameters
    ----------
    client : citrination_client.CitrinationClient
        A python Citrination client for managing datasets
    dataset_id_list : list of int
        List of dataset ids (integers) that will be down-sampled 
    save_samples : bool
        if True, the down-sampled data will be saved to a dataset
    train_hyperparameters : bool
        if True, the models will be optimized
        over a grid of hyperparameters during training

    Returns
    -------
    full_downsampled_dataset : pandas.DataFrame
        dataframe containing all of the down-sampled data from 
        all of the datasets in `dataset_id_list`. 
        Features in this DataFrame are not scaled:
        the correct scaler should be applied before training models.
    """

    data, pifs = get_data_from_Citrination(client, dataset_id_list)

    #### create data_sample ########################
    data_sample = pd.DataFrame(columns=data.columns)
    expt_samples = {}
    expt_local_ids = {} # local ids of samples to save by exp
    all_exp = data.experiment_id.unique()

    features = []
    features.extend(profiler.profile_keys_1)

    scaler = preprocessing.StandardScaler()
    scaler.fit(data[features])

    transformed_data = pd.DataFrame(columns=data.columns, data=data[data.columns])
    transformed_data[features] = scaler.transform(data[features])

    for exp_id in all_exp:
        df = transformed_data[transformed_data['experiment_id']==exp_id]
        dsamp = downsample_one_experiment(df, 1.0)
        data_sample = data_sample.append(dsamp)
        if save_samples:
            expt_samples[exp_id] = data_sample
            expt_local_ids[exp_id] = data_sample.local_id.tolist()
    ################################################

    # store references to unscaled data for all samples in data_sample
    samples_to_save = data_sample.local_id.tolist()
    unscaled_data = pd.DataFrame(columns=data.columns)
    for samp_id in samples_to_save:
        unscaled_data = unscaled_data.append(data.iloc[samp_id])

    if save_samples:
        p = os.path.abspath(__file__)
        d2 = os.path.dirname(os.path.dirname(p))
        ds_map_filepath = os.path.join(d2,'models','modeling_data','dataset_ids.yml')
        dataset_ids = yaml.load(open(ds_map_filepath,'rb'))
        sys_classifier_dsid = dataset_ids['system_classifier']

        for expt_id in all_exp:
            # sort sample of pifs by classes
            all_sys_classes = expt_samples[expt_id].system_class.unique()
            pifs_by_sys_class = {}
            for cl in all_sys_classes:
                pifs_by_sys_class[cl] = []

            for pif_local_id in expt_local_ids[expt_id]:
                cl = data.iloc[pif_local_id].system_class
                pifs_by_sys_class[cl].append(pifs[pif_local_id])

            d = os.path.dirname(os.path.dirname(os.path.dirname(p)))
            for cl,pp in pifs_by_sys_class.items():
                # check if this system class has an assigned dataset id 
                if cl in dataset_ids:
                    ds_id = dataset_ids[cl]
                # if not, create a new one and add it to the index
                else:
                    ds = client.data.create_dataset(cl, 
                    'Downsampled modeling data for system class {}'.format(cl))
                    ds_id = ds.id
                    dataset_ids[cl] = ds_id
                jsf = os.path.join(d, cl+'_'+expt_id+'.json')
                pif.dump(pp, open(jsf,'w'))
                client.data.upload(ds_id, jsf)
                # upload into the large sample for the main classifier:
                client.data.upload(sys_classifier_dsid, jsf)
        with open(ds_map_filepath, 'w') as yaml_file:
            yaml.dump(dataset_ids, yaml_file)
    return unscaled_data 

def downsample_one_experiment(data_fr, min_distance):
    """Downsample records from one experimental dataset.

    Parameters
    ----------
    data_fr : pandas.DataFrame
        dataframe containing the samples from one experiment.
    min_distance : float
        the minimal allowed distance between the samples.

    Returns
    -------
    sample : pandas.DataFrame
        dataframe containing subset of rows
        that was chosen using distance between the samples
    """
    expt_id = data_fr['experiment_id'].iloc[0] 
    groups_by_class = data_fr.groupby('system_class')
    sample = pd.DataFrame(columns=data_fr.columns)
    print('downsampling records from experiment: {}'.format(expt_id))
    print('total sample size: {}'.format(len(data_fr)))
    for name, group in groups_by_class:
        group_size = len(group)
        sys_cls = group['system_class'].iloc[0] 
        print('- number of samples for system class {}: {}'.format(sys_cls,group_size))
        if group_size >= 10:
            df = pd.DataFrame(columns=data_fr.columns)
            # define the distance between two samples in feature space 
            group_dist_func = lambda i,j: sum(
                (group.iloc[i][profiler.profile_keys_1]
                - group.iloc[j][profiler.profile_keys_1]).abs())

            print('- building inter-sample distance matrix...')
            group_dist_matrix = np.array([[group_dist_func(i,j) for i in range(group_size)] for j in range(group_size)])
            # get the most isolated sample first:
            # this should be the sample with the greatest minimum distance
            # between itself and all other samples
            min_distance_array = np.array([min(group_dist_matrix[i,:]) for i in range(group_size)])
            best_idx = np.argmax(min_distance_array)
            print('- best sample: {} (min distance: {})'.format(best_idx,min_distance_array[best_idx]))
            df = df.append(group.iloc[best_idx])
            sampled_idxs = [best_idx] 
            continue_downsampling = True
            while(continue_downsampling):

                # find the next best index to sample:
                # the sample with the greatest minimum distance
                # between itself and the downsampled samples
                sample_size = len(df)
                sample_dist_matrix = np.array([group_dist_matrix[i,:] for i in sampled_idxs])
                #sample_dist_matrix = np.array([[sample_dist_func(i,j) 
                #for i in range(group_size)] for j in range(sample_size)])
                min_distance_array = np.array([min(sample_dist_matrix[:,j]) for j in range(group_size)])
                best_idx = np.argmax(min_distance_array)
                best_min_distance = min_distance_array[best_idx]
                print('- next best sample: {} (min distance: {})'.format(best_idx,best_min_distance))

                # if we have at least 10 samples,
                # and all remaining samples are close to current data set,
                # down-sampling can stop here.
                if sample_size >= 10 and best_min_distance < min_distance: 
                    continue_downsampling = False
                else:
                    sampled_idxs.append(best_idx)
                    df = df.append(group.iloc[best_idx])

            print('- down-sampled to: {}'.format(len(df)))
            sample = sample.append(df)
        else:
            print('- skipped downsampling (insufficient data)')
            sample = sample.append(group)
    print('number of samples retained for {}: {}'.format(expt_id,len(sample)))
    return sample

