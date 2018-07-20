from collections import OrderedDict
import os

import pandas as pd
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
    # all_reg_labels will be a list of all unique regression labels for the set of pifs
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
    return pifs


def downsample_Citrination_datasets(client, dataset_id_list, save_sample=True):
    """Down-sample one or more datasets, save the downsampled data to another dataset. 

    Parameters
    ----------
    client : citrination_client.CitrinationClient
        A python Citrination client for managing datasets
    dataset_id_list : list of int
        List of dataset ids (integers) that will be down-sampled 
    save_sample : bool
        if True, the down-sampled data will be saved to a dataset

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
        if save_sample:
            expt_samples[exp_id] = sample
            expt_local_ids[exp_id] = sample.local_id.tolist()
    ################################################

    # store references to unscaled data for all samples in data_sample
    samples_to_save = data_sample.local_id.tolist()
    unscaled_data = pd.DataFrame(columns=data.columns)
    for samp_id in samples_to_save:
        unscaled_data = unscaled_data.append(data.iloc[samp_id])

    if save_sample:
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
                if not cl in dataset_ids:
                    ds_id = dataset_ids[cl]
                # if not, create a new one and add it to the index
                else:
                    ds = client.data.create_dataset(cl, 
                    'Downsampled modeling data for system class {}'.format(cl))
                    ds_id = ds.id
                    dataset_ids[cl] = ds_id
                jsf = os.path.join(d, cl+'_'+ex+'.json')
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
    groups_by_class = data_fr.groupby('system_class')
    sample = pd.DataFrame(columns=data_fr.columns)
    for name, group in groups_by_class:
        df = pd.DataFrame(columns=data_fr.columns)
        # keep the first sample in the group
        df = df.append(group.iloc[0])
        # test all remaining samples and add them to the sample
        # if their distance from other samples is sufficiently large
        dist_func = lambda i,j: sum((group.iloc[i][profiler.profile_keys_1] 
            - group.iloc[j][profiler.profile_keys_1]).abs()) 
        for i in range(1, group.shape[0]):
            add_row = all( [dist_func(i,j) > min_distance for j in range(0,group.shape[0])] )
            if add_row: df = df.append(group.iloc[i])
        sample = sample.append(df)
    return sample

