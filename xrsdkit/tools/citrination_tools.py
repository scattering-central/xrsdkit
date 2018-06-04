from collections import OrderedDict

import pandas as pd
import numpy as np
from citrination_client import CitrinationClient
from citrination_client import PifSystemReturningQuery, DatasetQuery, DataQuery, Filter

from ..tools import profiler
from ..tools import piftools


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
    colnames.extend(['populations'])
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


