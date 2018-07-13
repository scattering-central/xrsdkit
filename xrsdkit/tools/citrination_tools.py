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


def sampl_data_on_Citrination(client, data_cl, dataset_id_list, save_sample=True, ids_to_reuse = []):
    """Create a sample of data and ship it on Citrination.

    Parameters
    ----------
    client : citrination_client.CitrinationClient
        A python Citrination client for fetching data
    data_cl : citrination_client.data.client.DataClient
        A python Citrination client encapsulating data management behavior.
    dataset_id_list : list of int
        List of dataset ids (integers) for fetching SAXS records
    save_sample : bool
        if True, the sample of data will be save on Citrination
    ids_to_reuse : list of int
        list of ids for Citrination datasets that we want to reuse;
        it will be used only id save_sample=True.

    Returns
    -------
    data_sample_not_transf : pandas.DataFrame
        dataframe containing subset of rows
        that was chosen using distance between the samples;
        the data was not transformed.
    """

    data, pifs = get_data_from_Citrination(client, dataset_id_list)

    #### create data_sample ########################
    data_sample = pd.DataFrame(columns=data.columns)
    samples = {}
    samples_ids = {} # local ids of samples to save by exp
    all_exp = data.experiment_id.unique()

    features = []
    features.extend(profiler.profile_keys_1)

    scaler = preprocessing.StandardScaler()
    scaler.fit(data[features])

    transformed_data = pd.DataFrame(columns=data.columns, data= data[data.columns])
    transformed_data[features] = scaler.transform(data[features])

    for exp_id in all_exp:
        df = transformed_data[transformed_data['experiment_id']==exp_id]
        sample = make_sample_one_experiment(df, 1.0)
        data_sample = data_sample.append(sample)
        if save_sample:
            samples[exp_id] = sample
            samples_ids[exp_id] = sample.local_id.tolist()
    ################################################

    samples_to_save = data_sample.local_id.tolist()
    data_sample_not_transf = pd.DataFrame(columns=data.columns)
    for samp_id in samples_to_save:
        data_sample_not_transf=data_sample_not_transf.append(data.iloc[samp_id])

    if save_sample:
        p = os.path.abspath(__file__)
        d2 = os.path.dirname(os.path.dirname(p))
        yml_file_path = os.path.join(d2,'models','modeling_data','datasamples_ids.yml')
        try:
            existing_samples = open(yml_file_path,'rb')
            sys_class_sample_ids = yaml.load(existing_samples)
            sys_classifier_sample_id = sys_class_sample_ids["Sample of data for system classification"]
        except:
            sys_class_sample_ids = {}
            if len(ids_to_reuse)>0:
                sys_classifier_sample_id = ids_to_reuse.pop()
                data_cl.create_dataset_version(sys_classifier_sample_id)
                data_cl.update_dataset(sys_classifier_sample_id,
                                                "Sample of data for system classification",
                                                "Sample of data for system classification")
            else:
                ds_sys = data_cl.create_dataset("Sample of data for system classification",
                                                "Sample of data for system classification")
                sys_classifier_sample_id = ds_sys.id
            sys_class_sample_ids["Sample of data for system classification"] = sys_classifier_sample_id

        '''
        datasets_for_saxskit = list(range(90, 121))
        datasets_in_usage = list(sys_class_sample_ids.values())
        avalible_ids = list(set(datasets_for_saxskit)-set(datasets_in_usage))
        print('sys_class_sample_ids', sys_class_sample_ids)
        print('avalible_ids', avalible_ids)
        '''
        for ex in all_exp:
            # sort sample of pifs by classes
            all_sys_classes = samples[ex].system_class.unique()
            pifs_by_classes = {}
            for cl in all_sys_classes:
                pifs_by_classes[cl] = []

            for samp_id in samples_ids[ex]:
                cl = data.iloc[samp_id].system_class
                pifs_by_classes[cl].append(pifs[samp_id])

            d = os.path.dirname(os.path.dirname(os.path.dirname(p)))
            for k,v in pifs_by_classes.items():
                # to check if we alredy have a sample for this class:
                if k in sys_class_sample_ids:
                    ds_id = sys_class_sample_ids[k]
                    print('found', ds_id, k)
                else:
                    if len(ids_to_reuse)>0:
                        ds_id = ids_to_reuse.pop()
                        data_cl.create_dataset_version(ds_id)
                        ds = data_cl.update_dataset(ds_id, k, "Sample of data for: "+ k)
                        print('updated', ds_id, k)
                    else:
                        ds = data_cl.create_dataset(k, "Sample of data for: "+ k)
                        ds_id = ds.id
                        print('created', ds_id, k)
                    sys_class_sample_ids[k] = ds_id

                pif_file = os.path.join(d, k+ex+'.json')
                pif.dump(v, open(pif_file,'w'))
                client.data.upload(ds_id, pif_file)
                # upload into the large sample for the main classifier:
                client.data.upload(sys_classifier_sample_id, pif_file)

            with open(yml_file_path, 'w') as yaml_file:
                    yaml.dump(sys_class_sample_ids, yaml_file)

    return data_sample_not_transf

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
