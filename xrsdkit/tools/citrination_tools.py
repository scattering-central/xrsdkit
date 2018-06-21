from collections import OrderedDict
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from pypif import pif
from citrination_client import PifSystemReturningQuery, DatasetQuery, DataQuery, Filter

from ..tools import profiler
from ..tools import piftools
#import paws
#from pypaws.operations.IO.CITRINATION.UploadPIF import UploadPIF


def get_data_from_Citrination(client, dataset_id_list):
    """Get data from Citrination and create a dataframe.

    Parameters
    ----------
    client : citrination_client.CitrinationClient
        A python Citrination client for fetching data
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
    reg_labels = []
    all_reg_labels = set()

    pifs = get_pifs_from_Citrination(client,dataset_id_list)

    for pp in pifs:
        expt_id,t_utc,q_I,temp,pp_feats, cl_model_outputs, reg_model_outputs = piftools.unpack_pif(pp)
        #print(reg_model_outputs)
        feats = OrderedDict.fromkeys(profiler.profile_keys)
        feats.update(pp_feats)
        
        data_row = [expt_id]+list(feats.values())+[cl_model_outputs]
        data.append(data_row)
        for k,v in reg_model_outputs.items():
            all_reg_labels.add(k)
        reg_labels.append(reg_model_outputs)

    reg_labels_list = list(all_reg_labels)
    reg_labels_list.sort()

    for i in range(len(reg_labels)):
        lb = OrderedDict.fromkeys(reg_labels_list)
        lb.update(reg_labels[i])
        data[i] = data[i] + list(lb.values())

    colnames = ['experiment_id']
    colnames.extend(profiler.profile_keys)
    colnames.extend(['system_class'])
    colnames.extend(reg_labels_list)

    d = pd.DataFrame(data=data, columns=colnames)
    d = d.where((pd.notnull(d)), None) # replace all NaN by None
    shuffled_rows = np.random.permutation(d.index)
    df_work = d.loc[shuffled_rows]

    return df_work

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

def sampl_data_on_Citrination(client, data_cl, dataset_id_list):
    """Create a sample of data and ship it on Citrination.

    Parameters
    ----------
    client : citrination_client.CitrinationClient
        A python Citrination client for fetching data
    data_cl : citrination_client.data.client.DataClient
        A python Citrination client encapsulating data management behavior.
    dataset_id_list : list of int
        List of dataset ids (integers) for fetching SAXS records

    Returns
    -------
    new_datase_id : int
        id of new dataset
    """
    my_list = ','.join(map(str, dataset_id_list))
    sample_dataset_name = "sample_from_"+my_list
    ds = data_cl.create_dataset(sample_dataset_name, "Test sampling of data") # TODO name for new dataset sould be unique
    new_datase_id = ds.id

    p = os.path.abspath(__file__)
    d = os.path.dirname(os.path.dirname(os.path.dirname(p)))

    data = []
    pifs = get_pifs_from_Citrination(client,dataset_id_list)

    for i in range(len(pifs)):
        pp = pifs[i]
        expt_id,_,_,_,pp_feats, cl_model_outputs, _ = piftools.unpack_pif(pp)
        feats = OrderedDict.fromkeys(profiler.profile_keys)
        feats.update(pp_feats)

        data_row = [expt_id]+list(feats.values())+[cl_model_outputs] + [i] # i will be the local_id
        data.append(data_row)

    colnames = ['experiment_id']
    colnames.extend(profiler.profile_keys)
    colnames.extend(['system_class'])
    colnames.extend(['local_id'])

    df = pd.DataFrame(data=data, columns=colnames)
    data = df.where((pd.notnull(df)), None) # replace all NaN by None

    data_sample = pd.DataFrame(columns=data.columns)
    all_exp = data.experiment_id.unique()

    features = []
    features.extend(profiler.profile_keys_1)

    scaler = preprocessing.StandardScaler()
    scaler.fit(data[features])
    data[features] = scaler.transform(data[features])

    for exp_id in all_exp:
        df = data[data['experiment_id']==exp_id]
        sample = make_sample_one_experiment(df, 1.0)
        data_sample = data_sample.append(sample)

    #print(data_sample.shape)
    count = data_sample.shape[0]

    samples_to_save = data_sample.local_id.tolist()

    pifs_to_save = []
    for samp_id in samples_to_save:
        pifs_to_save.append(pifs[samp_id])

    pif_file = os.path.join(d, 'test.json')

    pif.dump(pifs_to_save, open(pif_file,'w'))
    r = client.data.upload(new_datase_id, pif_file)

    return new_datase_id, count


def make_sample_one_experiment(data_fr, min_distance):
    """make a sample from ONE experiment.
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
        df = df.append(group.iloc[0])
        for i in range(1, group.shape[0]):
            add_row = True
            for j in range(0, df.shape[0]):
                s = sum((group.iloc[i][profiler.profile_keys_1] - group.iloc[j][profiler.profile_keys_1]).abs())
                if s < min_distance:
                    add_row = False
            if add_row:
                df = df.append(group.iloc[i])
        sample = sample.append(df)
    return sample
