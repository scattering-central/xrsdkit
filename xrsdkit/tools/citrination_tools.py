from collections import OrderedDict
import copy

import pandas as pd
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
            copy.deepcopy(profiler.profile_defs.keys()) + \
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

