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

    pifs = get_pifs_from_Citrination(client,dataset_id_list)

    for pp in pifs:
        expt_id,t_utc,q_I,temp,pp_feats = piftools.unpack_pif(pp)
        feats = OrderedDict.fromkeys(profiler.profile_keys)
        feats.update(pp_feats)
        model_outputs = piftools.get_model_outputs(pp)
        
        data_row = [expt_id]+list(feats.values())+list(model_outputs.values())
        data.append(data_row)

    colnames = ['experiment_id']
    colnames.extend(profiler.profile_keys)
    colnames.extend(piftools.model_output_names)

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

        current_result = client.search(query)
        while current_result.hits is not None:
            all_hits.extend(current_result.hits)
            n_current_hits = len(current_result.hits)
            #n_hits += n_current_hits
            query.from_index += n_current_hits 
            current_result = client.search(query)

    pifs = [x.system for x in all_hits]
    return pifs


