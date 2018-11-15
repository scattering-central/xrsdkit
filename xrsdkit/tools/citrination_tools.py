from collections import OrderedDict
import os
import copy
import re

import pandas as pd
import numpy as np
import yaml
from sklearn import preprocessing
from pypif import pif
from citrination_client import PifSystemReturningQuery, DatasetQuery, DataQuery, Filter

from . import profiler, piftools

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
    """
    data = []
    # reg_labels and cls_labels are lists of dicts,
    # containing regression and classification outputs for each sample
    reg_labels = []
    cls_labels = []
    # all_reg_labels and all_cls_labels are sets of all
    # unique labels over the provided datasets 
    all_reg_labels = set()
    all_cls_labels = set()

    pifs = get_pifs_from_Citrination(client,dataset_id_list)

    for i,pp in enumerate(pifs):
        pif_uid, sys, q_I, expt_id, t_utc, temp, src_wl, \
        features, classification_labels, regression_outputs = piftools.unpack_pif(pp)

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
    d['system_classification'] = \
        d['system_classification'].where((pd.notnull(d['system_classification'])),
                                         'unidentified')
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
        dataframe containing subset of rows
        that was chosen using distance between the samples
    """
    df_size = len(df)
    sample = pd.DataFrame(columns=df.columns)
    print('total DataFrame size: {}'.format(df_size))
    if df_size <= 10:
        sample = sample.append(df)
    else:
        features = profiler.profile_keys
        scaler = preprocessing.StandardScaler()
        scaler.fit(df[features])

        features_matr = scaler.transform(df[features]) # features_matr is a np arraly
        # define the distance between two samples in feature space

        dist_func = lambda i,j: np.sum(
            np.abs(features_matr[i]
            - features_matr[j]))
        dist_matrix = np.array([[dist_func(i,j) for i in range(df_size)] for j in range(df_size)])
        # get the most isolated sample first:
        # this should be the sample with the greatest 
        # nearest-neighbor distance 
        nn_distance_array = np.array([min(dist_matrix[i,:]) for i in range(df_size)])
        best_idx = np.argmax(nn_distance_array)
        sample = sample.append(df.iloc[best_idx])
        sampled_idxs = [best_idx] 
        continue_downsampling = True
        while(continue_downsampling):
            # find the next best index to sample:
            # the sample with the greatest minimum distance
            # between itself and the downsampled samples
            sample_size = len(sample)
            sample_dist_matrix = np.array([dist_matrix[i,:] for i in sampled_idxs])
            nn_distance_array = np.array([min(sample_dist_matrix[:,j]) for j in range(df_size)])
            best_idx = np.argmax(nn_distance_array)
            best_nn_distance = nn_distance_array[best_idx]
            # if we have at least 10 samples,
            # and all remaining samples are close to the current sample,
            # down-sampling can stop here.
            if sample_size >= 10 and best_nn_distance < min_distance: 
                continue_downsampling = False
            else:
                sampled_idxs.append(best_idx)
                sample = sample.append(df.iloc[best_idx])
        print('downsampled DataFrame size: {}/{}'.format(len(sampled_idxs),df_size))
    return sample

