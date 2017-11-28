import pandas as pd
import numpy as np
from pypif.pif import dumps
import json

from collections import OrderedDict

import sklearn
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import linear_model
import yaml
import os

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeavePGroupsOut

from citrination_client import PifSystemReturningQuery
from citrination_client import DatasetQuery
from citrination_client import DataQuery
from citrination_client import Filter

from . import saxs_math

def train_classifiers(all_data, yaml_filename=None, hyper_parameters_search=False):
    """Train and save SAXS classification models as a YAML file.

    Parameters
    ----------
    all_data : pandas.DataFrame
        dataframe containing features and labels
    yaml_filename : str
        File where scalers and models will be saved.
        If None, the default file is used.
    """
    p = os.path.abspath(__file__)
    d = os.path.dirname(p)
    if yaml_filename is None:
        yaml_filename = os.path.join(d,'modeling_data','scalers_and_models.yml')
    else:
        yaml_filename = os.path.join(d,'modeling_data',yaml_filename)

    # TODO: Put model accuracy in the .yml file?
    accuracy_txt = os.path.join(d,'modeling_data','accuracy.txt')
    current_version = list(map(int,sklearn.__version__.split('.')))

    scalers = {}
    models = {}
    accuracy = {}
    scalers_and_models = OrderedDict(
        version=current_version,
        scalers=scalers, 
        models=models, 
        accuracy=accuracy)
    #features = ['q_Imax', 'Imax_over_Imean', 'Imax_sharpness','logI_fluctuation', 'logI_max_over_std']
    features = saxs_math.profile_keys
    possible_models = check_labels(all_data)

    # using leaveTwoGroupOut makes sense when we have at least 5 groups
    if len(all_data.experiment_id.unique()) > 4:
        leaveTwoGroupOut = True
    else:
        # use 5-fold cross validation
        leaveTwoGroupOut = False 

    # unidentified scatterer population model
    if possible_models['unidentified'] == True:
        scaler = preprocessing.StandardScaler()
        scaler.fit(all_data[features])
        transformed_data = scaler.transform(all_data[features])
        if hyper_parameters_search == True:
            penalty, alpha, l1_ratio = hyperparameters_search(
                transformed_data, all_data[['unidentified']],
                all_data['experiment_id'], leaveTwoGroupOut)
        else:
            penalty = 'l1'
            alpha = 0.001
            l1_ratio = 1.0

        logsgdc = linear_model.SGDClassifier(
            alpha=alpha, loss='log', penalty=penalty, l1_ratio=l1_ratio)
        logsgdc.fit(transformed_data, all_data['unidentified'])

        # save the scaler and model for "bad_data"
        scalers['unidentified'] = scaler.__dict__
        models['unidentified'] = logsgdc.__dict__

        # save the accuracy
        if leaveTwoGroupOut:
            accuracy['unidentified'] = testing_by_experiments(
                all_data, 'unidentified', features, alpha, l1_ratio, penalty)
        else:
            accuracy['unidentified'] = testing_using_crossvalidation(
                all_data, 'unidentified', features, alpha, l1_ratio, penalty)
    else:
        scalers['unidentified'] = None
        models['unidentified'] = None
        accuracy['unidentified'] = None

    # For the rest of the models, 
    # we will use only data with
    # identifiable scattering populations 
    all_data = all_data[all_data['unidentified']==False]

    # spherical_normal scatterer population model
    if possible_models['spherical_normal'] == True:
        scaler = preprocessing.StandardScaler()
        scaler.fit(all_data[features])
        transformed_data = scaler.transform(all_data[features])
        if hyper_parameters_search == True:
            penalty, alpha, l1_ratio = hyperparameters_search(
                transformed_data, all_data[['spherical_normal']],
                all_data['experiment_id'], leaveTwoGroupOut)
        else:
            penalty = 'l1'
            alpha = 0.001
            l1_ratio = 1.0

        logsgdc = linear_model.SGDClassifier(
            alpha=alpha, loss='log', penalty=penalty, l1_ratio=l1_ratio)
        logsgdc.fit(transformed_data, all_data['spherical_normal'])

        scalers['spherical_normal'] = scaler.__dict__
        models['sperical_normal'] = logsgdc.__dict__
        if leaveTwoGroupOut:
            accuracy['spherical_normal'] = testing_by_experiments(
                all_data, 'spherical_normal', features, alpha, l1_ratio, penalty)
        else:
            accuracy['spherical_normal'] = testing_using_crossvalidation(
                all_data, 'spherical_normal', features, alpha, l1_ratio, penalty)
    else:
        scalers['spherical_normal'] = None
        models['spherical_normal'] = None
        accuracy['spherical_normal'] = None

    # guinier_porod scatterer population model
    if possible_models['guinier_porod'] == True:
        scaler = preprocessing.StandardScaler()
        scaler.fit(all_data[features])
        transformed_data = scaler.transform(all_data[features])

        if hyper_parameters_search == True:
            penalty, alpha, l1_ratio = hyperparameters_search(
                transformed_data, all_data[['guinier_porod']],
                all_data['experiment_id'], leaveTwoGroupOut)
        else:
            penalty = 'elasticnet'
            alpha = 0.01
            l1_ratio = 0.85

        logsgdc = linear_model.SGDClassifier(
            alpha=alpha, loss='log', penalty=penalty, l1_ratio=l1_ratio)
        logsgdc.fit(transformed_data, all_data['guinier_porod'])

        scalers['guinier_porod'] = scaler.__dict__
        models['guinier_porod'] = logsgdc.__dict__
        if leaveTwoGroupOut:
            accuracy['guinier_porod'] = testing_by_experiments(
                all_data, 'guinier_porod', features, alpha, l1_ratio, penalty)
        else:
            accuracy['guinier_porod'] = testing_using_crossvalidation(
                all_data, 'guinier_porod', features, alpha, l1_ratio, penalty)
    else:
        scalers['guinier_porod'] = None
        models['guinier_porod'] = None
        accuracy['guinier_porod'] = None

    # diffraction peak population model
    if possible_models['diffraction_peaks'] == True:
        scaler = preprocessing.StandardScaler()
        scaler.fit(all_data[features])
        transformed_data = scaler.transform(all_data[features])

        if hyper_parameters_search == True:
            penalty, alpha, l1_ratio = hyperparameters_search(
                transformed_data, all_data[['diffraction_peaks']],
                all_data['experiment_id'], leaveTwoGroupOut)
        else:
            penalty = 'elasticnet'
            alpha = 0.001
            l1_ratio = 0.85

        logsgdc = linear_model.SGDClassifier(
            alpha=alpha, loss='log', penalty=penalty, l1_ratio=l1_ratio)
        logsgdc.fit(transformed_data, all_data['diffraction_peaks'])

        scalers['diffraction_peaks'] = scaler.__dict__
        models['diffraction_peaks'] = logsgdc.__dict__
        if leaveTwoGroupOut:
            accuracy['diffraction_peaks'] = testing_by_experiments(
                all_data,'diffraction_peaks',features, alpha, l1_ratio, penalty)
        else:
            accuracy['diffraction_peaks'] = testing_using_crossvalidation(
                all_data,'diffraction_peaks', features, alpha, l1_ratio, penalty)
    else:
        scalers['diffraction_peaks'] = None
        models['diffraction_peaks'] = None
        accuracy['diffraction_peaks'] = None

    # save scalers and models
    with open(yaml_filename, 'w') as yaml_file:
        yaml.dump(scalers_and_models, yaml_file)

    # TODO: Is this not already saved in scalers_and_models.yml?
    # save accuracy
    #with open (accuracy_txt, 'w') as txt_file:
    #    txt_file.write(str(accuracy))

def train_regressors(dataframe, yaml_filename=None):
    """Train and save SAXS regression models as a YAML file.

    Parameters
    ----------
    dataframe : pandas dataframe of features
        and labels
    yaml_filename : str
        File where scalers and models will be saved.
        If None, the default file is used.
    """
    pass

def hyperparameters_search(data_features, data_labels, group_by, leaveTwoGroupOut):
    """Grid search for alpha, penalty, and l1 ratio

    Parameters
    ----------
    data_features : array
        2D numpy array of features, one row for each sample
    data_labels : array
        array of labels (as a DataFrame column), one label for each sample
    group_by: string
        DataFrame column header for LeavePGroupsOut(groups=group_by)
    leaveTwoGroupOut: boolean 
        Indicated whether or not we have enough experimental data
        to cross-validate by the leave-two-groups-out approach

    Returns
    -------
    penalty : string
        TODO- document
    alpha : float
        TODO- document
    l1_ratio : string
        TODO- document
    """

    parameters = {'penalty':('none', 'l2', 'l1', 'elasticnet'), #default l2
              'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1], #regularisation coef, default 0.0001
             'l1_ratio': [0, 0.15, 0.5, 0.85, 1.0]} #using with elasticnet only; default 0.15

    if leaveTwoGroupOut == True:
        cv=LeavePGroupsOut(n_groups=2).split(data_features, np.ravel(data_labels), groups=group_by)
    else:
        cv = 5 # five folders cross validation

    svc = linear_model.SGDClassifier(loss='log')
    clf = GridSearchCV(svc, parameters, cv=cv)
    clf.fit(data_features, np.ravel(data_labels))

    penalty = clf.best_params_['penalty']
    alpha = clf.best_params_['alpha']
    l1_ratio = clf.best_params_['l1_ratio']

    return penalty, alpha, l1_ratio

def check_labels(dataframe):
    """Test whether or not `dataframe` has True and False values for each label

    Because a model requires a distribution of training samples,
    this function checks `dataframe` to ensure that its
    labels are not all the same.
    For a model where the label is always the same,
    this function returns False,
    indicating that this `dataframe`
    cannot be used to train that model.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        dataframe of sample features and corresponding labels

    Returns
    -------
    possible_models : dict
        dictionary with booleans indicating whether or not
        True and False labels were found
        for each of the possible models. 
    """
    possible_models = {}
    if len(dataframe.unidentified.unique()) == 2:
        possible_models['unidentified'] = True
    else:
        possible_models['unidentified'] = False
    # we will use only samples with identifiable 
    # scattering popoulations for the other models
    dataframe = dataframe[dataframe['unidentified']==False]
    for l in ['spherical_normal', 'guinier_porod', 'diffraction_peaks']:
        if len(dataframe[l].unique()) == 2:
            possible_models[l] = True
        else:
            possible_models[l] = False
    return possible_models

def testing_using_crossvalidation(df, label, features, alpha, l1_ratio, penalty):
    """Fit a model, then test it using 5-fold crossvalidation

    Parameters
    ----------
    df : pandas.DataFrame
        pandas dataframe of features and labels
    features : list of strings
        list of feature labels to use in model training
    alpha : float 
        weighting of the regularization term
    l1_ratio : float 
        the Elastic Net mixing parameter, 0 <= l1_ratio <= 1
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1
    penalty : string
        penalty specification, 'none', 'l2', 'l1', or 'elasticnet'

    Returns
    -------
    float 
        average crossvalidation score (accuracy)
    """
    scaler = preprocessing.StandardScaler()
    scaler.fit(df[features])
    logsgdc = linear_model.SGDClassifier(
        alpha=alpha, loss='log', l1_ratio=l1_ratio, penalty=penalty)
    scores = cross_val_score(
        logsgdc, scaler.transform(df[features]), df[label], cv=5)
    return scores.mean()

def testing_by_experiments(df, label, features, alpha, l1_ratio, penalty):
    """Fit a model, then test it by leaveTwoGroupsOut cross-validation 

    Parameters
    ----------
    df : pandas.DataFrame
        pandas dataframe of features and labels
    features : list of strings
        specifies which features to use
    alpha : float 
        weighting of the regularization term
    l1_ratio : float
        the Elastic Net mixing parameter, 0 <= l1_ratio <= 1
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1
    penalty : string
        penalty specification, 'none', 'l2', 'l1', or 'elasticnet'

    Returns
    -------
    float
        average crossvalidation score (accuracy)
    """
    experiments = df.experiment_id.unique()# we have at least 5 experiments
    test_scores_by_ex = []
    count = 0
    for i in range(len(experiments)):
        for j in range(i+1, len(experiments)):
            tr = df[(df['experiment_id']!= experiments[i]) \
                & (df['experiment_id']!= experiments[j])]
            test = df[(df['experiment_id']== experiments[i]) \
                | (df['experiment_id']== experiments[j])]
            scaler = preprocessing.StandardScaler()
            scaler.fit(tr[features])
            logsgdc = linear_model.SGDClassifier(
                alpha=alpha, loss='log', l1_ratio=l1_ratio, penalty=penalty)
            logsgdc.fit(scaler.transform(tr[features]), tr[label])
            test_score = logsgdc.score(
                scaler.transform(test[features]), test[label])
            test_scores_by_ex.append(test_score)
            count +=1
    acc =  sum(test_scores_by_ex)/count
    return acc

def get_data_from_Citrination(client, dataset_id_list):
    """Get data from Citrination and create a dataframe

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

    for dataset in dataset_id_list:
        query = PifSystemReturningQuery(
            from_index=0,
            size=100,
            query=DataQuery(
                dataset=DatasetQuery(
                    id=Filter(
                    equal=dataset))))

        all_hits = []
        #n_hits = 0
        current_result = client.search(query)
        while current_result.hits is not None:
            all_hits.extend(current_result.hits)
            n_current_hits = len(current_result.hits)
            #n_hits += n_current_hits
            query.from_index += n_current_hits 
            current_result = client.search(query)

        pifs = [x.system for x in all_hits]

        for pp in pifs:
            expt_id,q_I,temp,feats,pops,par,rpt = unpack_pif(pp)
            data_row = [expt_id]+list(feats.values())+list(pops.values())+list(par.values())
            data.append(data_row)

    # TODO: make sure the column names are in the right order,
    # i.e. in the same order as the columns in `data`.
    colnames = ['experiment_id']
    colnames.extend(saxs_math.profile_keys)
    colnames.extend(saxs_math.population_keys)
    colnames.extend(saxs_math.parameter_keys)
    d = pd.DataFrame(data=data, columns=colnames) 
    shuffled_rows = np.random.permutation(d.index)
    df_work = d.loc[shuffled_rows]

    return df_work

            
        # TODO: use the pif records directly. 
        # Should not need to dump them to strings.
        #for line in all_hits: # every line of pifs is one sample; we need to extract labels and features from it
        #    try:
        #        my_str = dumps(line)
        #        obj = json.loads(my_str) # to transform the string to dictionary

        #        # default values for labels
        #        bad_data = False
        #        form = False
        #        precursor = False
        #        structure = False

        #        ro_sphere = None
        #        sigma_sphere = None
        #        g_precursor = None
        #        rg_precursor = None

        #        experiment_id = None

        #        for i in obj['system']['ids']:
        #            if i['name'] == 'EXPERIMENT_ID':
        #                experiment_id = i['value']

        #        for pr in obj['system']['properties']:

        #            # extract features
        #            if pr['name'] == 'q_Imax':
        #                q_Imax = np.float32(pr['scalars'][0]['value'])
        #            if pr['name'] == 'Imax_over_Imean':
        #                Imax_over_Imean = np.float32(pr['scalars'][0]['value'])
        #            if pr['name'] == 'Imax_sharpness':
        #                Imax_sharpness = np.float32(pr['scalars'][0]['value'])
        #            if pr['name'] == 'logI_fluctuation':
        #                logI_fluctuation = np.float32(pr['scalars'][0]['value'])
        #            if pr['name'] == 'logI_max_over_std':
        #                logI_max_over_std = np.float32(pr['scalars'][0]['value'])

        #            # extract labels
        #            if pr['name'] == 'bad_data_flag':
        #                bad_data = np.float32(pr['scalars'][0]['value'])
        #                if bad_data == True:
        #                    continue
        #            if pr['name'] == 'form_factor_scattering_flag':
        #                form = np.float32(pr['scalars'][0]['value'])
        #            if pr['name'] == 'diffraction_peaks_flag':
        #                structure = np.float32(pr['scalars'][0]['value'])
        #            if pr['name'] == 'precursor_scattering_flag':
        #                precursor = np.float32(pr['scalars'][0]['value'])

        #            if pr['name'] == 'r0_sphere':
        #                ro_sphere = np.float32(pr['scalars'][0]['value'])
        #            if pr['name'] == 'sigma_sphere':
        #                sigma_sphere = np.float32(pr['scalars'][0]['value'])
        #            if pr['name'] == 'G_precursor':
        #                g_precursor = np.float32(pr['scalars'][0]['value'])
        #            if pr['name'] == 'rg_precursor':
        #                rg_precursor = np.float32(pr['scalars'][0]['value'])

        #        data.append([experiment_id, q_Imax, Imax_over_Imean, Imax_sharpness, logI_fluctuation,
        #                                   logI_max_over_std, bad_data, form, precursor, structure,
        #                              ro_sphere, sigma_sphere, g_precursor, rg_precursor])
        #    except:
        #        # May be in PAWS we need to put a custom exeption here
        #        my_str = dumps(line)
        #        obj = json.loads(my_str) # to transform the string to dictionary
        #        print(obj)
        #        continue

    #['experiment_id', 'q_Imax', 'Imax_over_Imean', 
    #'Imax_sharpness','logI_fluctuation', 'logI_max_over_std', 
    #'bad_data', 'form', 'precursor', 'structure',
    #'ro_shpere', 'sigma_sphere', 'g_precursor', 'rg_precursor'])

def unpack_pif(pp):
    expt_id = None
    q_I = None
    temp = None
    feats = OrderedDict()
    pops = OrderedDict() 
    par = OrderedDict() 
    rpt = OrderedDict() 
    for prop in pp.properties:
        if prop.name == 'SAXS intensity':
            I = [float(sca.value) for sca in prop.scalars]
            for val in prop.conditions:
                if val.name == 'scattering vector':
                    q = [float(sca.value) for sca in val.scalars]
                if val.name == 'temperature':
                    temp = float(val.scalars[0].value)
            q_I = np.array(zip(q,I))
        elif prop.name in saxs_math.population_keys:
            pops[prop.name] = bool(float(prop.scalars[0].value))
        elif prop.name in saxs_math.parameter_keys:
            par[prop.name] = float(prop.scalars[0].value)
        elif prop.tags is not None:
            if 'spectrum fitting quantity' in prop.tags:
                rpt[prop.name] = float(prop.scalars[0].value)
            if 'spectrum profiling quantity' in prop.tags:
                feats[prop.name] = float(prop.scalars[0].value) 
    for iidd in pp.ids:
        if iidd.name == 'EXPERIMENT_ID':
            expt_id = iidd.value
    return expt_id,q_I,temp,feats,pops,par,rpt


