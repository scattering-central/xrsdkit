from collections import OrderedDict
import os

import pandas as pd
import numpy as np
import sklearn
import yaml
from citrination_client import PifSystemReturningQuery, DatasetQuery, DataQuery, Filter
from sklearn import model_selection, preprocessing, linear_model
from sklearn.metrics import mean_absolute_error

from . import saxs_math
from . import saxs_piftools
from . import population_keys, parameter_keys, profile_keys
from . import all_profile_keys, all_parameter_keys

def train_classifiers(all_data, hyper_parameters_search=False):
    """Train SAXS classification models, optionally searching for optimal hyperparameters.

    Parameters
    ----------
    all_data : pandas.DataFrame
        dataframe containing features and labels
    hyper_parameters_search : bool
        If true, grid-search model hyperparameters
        to seek high cross-validation accuracy.

    Returns
    -------
    scalers : dict
        Dictionary of sklearn standard scalers (one scaler per model).
    models : dict
        Dictionary of sklearn models.
    accuracy : dict
        Dictionary of accuracies for each model.
    """
    scalers = {}
    models = {}
    accuracy = {}

    # use the "unidentified" profiling for all classification models 
    features = profile_keys['unidentified']
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
                all_data['experiment_id'], leaveTwoGroupOut, 2)
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
                all_data['experiment_id'], leaveTwoGroupOut, 2)
        else:
            penalty = 'l1'
            alpha = 0.001
            l1_ratio = 1.0

        logsgdc = linear_model.SGDClassifier(
            alpha=alpha, loss='log', penalty=penalty, l1_ratio=l1_ratio)
        logsgdc.fit(transformed_data, all_data['spherical_normal'])

        scalers['spherical_normal'] = scaler.__dict__
        models['spherical_normal'] = logsgdc.__dict__
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
                all_data['experiment_id'], leaveTwoGroupOut, 2)
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
                all_data['experiment_id'], leaveTwoGroupOut, 2)
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

    return scalers, models, accuracy

def train_regressors(all_data, hyper_parameters_search=False):
    """Train SAXS parameter regression models, optionally searching for optimal hyperparameters.

    Parameters
    ----------
    all_data : pandas.DataFrame
        dataframe containing features and labels
    hyper_parameters_search : bool
        If true, grid-search model hyperparameters
        to seek high cross-validation accuracy.

    Returns
    -------
    scalers : dict
        Dictionary of sklearn standard scalers (one scaler per model).
    models : dict
        Dictionary of sklearn models.
    accuracy : dict
        Dictionary of accuracies for each model.
    """

    scalers = {}
    models = {}
    accuracy = {}

    possible_models = check_labels_regression(all_data)

    # r0_sphere model
    if possible_models['r0_sphere'] == True:
        features = []
        features.extend(profile_keys['unidentified'])

        scaler, reg, acc = train(all_data, features, 'r0_sphere', hyper_parameters_search)

        scalers['r0_sphere'] = scaler.__dict__
        models['r0_sphere'] = reg.__dict__
        accuracy['r0_sphere'] = acc
    else:
        scalers['r0_sphere'] = None
        models['r0_sphere'] = None
        accuracy['r0_sphere'] = None


    # sigma_shpere model
    if possible_models['sigma_sphere'] == True:
        features = []
        features.extend(profile_keys['unidentified'])
        features.extend(profile_keys['spherical_normal'])

        scaler, reg, acc = train(all_data, features, 'sigma_sphere', hyper_parameters_search)

        scalers['sigma_sphere'] = scaler.__dict__
        models['sigma_sphere'] = reg.__dict__
        accuracy['sigma_sphere'] = acc
    else:
        scalers['sigma_sphere'] = None
        models['sigma_sphere'] = None
        accuracy['sigma_sphere'] = None

    # rg_gp model
    if possible_models['rg_gp'] == True:
        features = []
        features.extend(profile_keys['unidentified'])
        features.extend(profile_keys['guinier_porod'])

        scaler, reg, acc = train(all_data, features, 'rg_gp', hyper_parameters_search)

        scalers['rg_gp'] = scaler.__dict__
        models['rg_gp'] = reg.__dict__
        accuracy['rg_gp'] = acc
    else:
        scalers['rg_gp'] = None
        models['rg_gp'] = None
        accuracy['rg_gp'] = None

    return scalers, models, accuracy

def train(all_data, features, target, hyper_parameters_search):
    """Helper function for training regression models.

    Parameters
    ----------
    all_data : pandas.DataFrame
        dataframe containing features and labels
    features : list of str
        list of columns to use as features
    target : str
        name of target column
    hyper_parameters_search : bool
        if "false", we will use default parameters

    Returns
    -------
    scaler : StandardScaler
        scaler used to scale the data
    reg : SGDRegressor
        trained model
    accuracy : float
        average crossvalidation score
    """
    d = all_data[all_data[target].isnull() == False]
    data = d.dropna(subset=features)
    if len(data.experiment_id.unique()) > 4:
        leaveNGroupOut = True
    else:
        leaveNGroupOut = False
    scaler = preprocessing.StandardScaler()
    scaler.fit(data[features])
    data.loc[ : , features] = scaler.transform(data[features])
    if hyper_parameters_search == True:
        penalty, alpha, l1_ratio, loss, \
        epsilon = hyperparameters_search_regression(data[features],
            data[target], data['experiment_id'], leaveNGroupOut, 1)
    else: # default parametrs from sklern
        penalty =  'elasticnet'  #'l2'
        alpha = 0.01 #0.0001
        l1_ratio = 0.5 # 0.15
        loss = 'huber'
        epsilon = 0.1

    reg = linear_model.SGDRegressor(alpha= alpha, loss= loss,
                                        penalty = penalty,l1_ratio = l1_ratio,
                                        epsilon = epsilon, max_iter=1000)
    reg.fit(data[features], data[target])

    # accuracy
    label_std = data[target].std()
    if leaveNGroupOut:
        acc = testing_by_experiments_regression(
            data, target, features, alpha, l1_ratio, penalty, loss,
            epsilon, label_std)
    else:
        acc = testing_using_crossvalidation_regression(
            data, target, features, alpha, l1_ratio, penalty,  loss, epsilon, label_std)

    return scaler, reg, acc


def hyperparameters_search(data_features, data_labels, group_by, leaveNGroupOut, n):
    """Grid search for optimal alpha, penalty, and l1 ratio hyperparameters.

    Parameters
    ----------
    data_features : array
        2D numpy array of features, one row for each sample
    data_labels : array
        array of labels (as a DataFrame column), one label for each sample
    group_by: string
        DataFrame column header for LeavePGroupsOut(groups=group_by)
    leaveNGroupOut: boolean
        Indicated whether or not we have enough experimental data
        to cross-validate by the leave-two-groups-out approach
    n: integer
        number of groups to leave out

    Returns
    -------
    penalty : string
        The penalty (aka regularization term) to be used.
        Options are  'none', 'l2', 'l1', or 'elasticnet'.
    alpha : float
        Constant that multiplies the regularization term.
        Defaults to 0.0001 Also used to compute learning_rate when set to 'optimal'.
    l1_ratio : string
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15.
    """

    parameters = {'penalty':('none', 'l2', 'l1', 'elasticnet'), #default l2
              'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1], #regularisation coef, default 0.0001
             'l1_ratio': [0, 0.15, 0.5, 0.85, 1.0]} #using with elasticnet only; default 0.15

    if leaveNGroupOut == True:
        cv=model_selection.LeavePGroupsOut(n_groups=n).split(
            data_features, np.ravel(data_labels), groups=group_by)
    else:
        cv = 5 # five folders cross validation

    svc = linear_model.SGDClassifier(loss='log')
    clf = model_selection.GridSearchCV(svc, parameters, cv=cv)
    clf.fit(data_features, np.ravel(data_labels))

    penalty = clf.best_params_['penalty']
    alpha = clf.best_params_['alpha']
    l1_ratio = clf.best_params_['l1_ratio']

    return penalty, alpha, l1_ratio

def hyperparameters_search_regression(data_features, data_labels, group_by, leaveNGroupOut, n):
    """Grid search for alpha, penalty, l1 ratio, loss, and epsilon.

    Parameters
    ----------
    data_features : array
        2D numpy array of features, one row for each sample
    data_labels : array
        array of labels (as a DataFrame column), one label for each sample
    group_by: string
        DataFrame column header for LeavePGroupsOut(groups=group_by)
    leaveNGroupOut: boolean
        Indicated whether or not we have enough experimental data
        to cross-validate by the leave-two-groups-out approach
    n: interer
        number of groups to leave out

    Returns
    -------
    penalty : string
        The penalty (aka regularization term) to be used.
        Options are  'none', 'l2', 'l1', or 'elasticnet'.
    alpha : float
        Constant that multiplies the regularization term.
        Defaults to 0.0001. Also used to compute learning_rate when set to 'optimal'.
    l1_ratio : string
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15.
    loss: string 'huber' or 'squared_loss'
        The loss function to be used.
    epsilon: float
        For 'huber' loss, epsilon determines the threshold at which it becomes less
        important to get the prediction exactly right.
    """

    parameters = {'loss':('huber', 'squared_loss'), # huber with epsilon = 0 gives us abs error (MAE)
              'epsilon': [1, 0.1, 0.01, 0.001, 0],
              'penalty':['none', 'l2', 'l1', 'elasticnet'], #default l2
              'alpha':[0.0001, 0.001, 0.01], #default 0.0001
             'l1_ratio': [0, 0.15, 0.5, 0.95], #default 0.15
             }

    if leaveNGroupOut == True:
        cv=model_selection.LeavePGroupsOut(n_groups=n).split(
            data_features, np.ravel(data_labels), groups=group_by)
    else:
        cv = 5 # five folders cross validation

    reg = linear_model.SGDRegressor(max_iter=1000)
    clf = model_selection.GridSearchCV(reg, parameters, cv=cv)
    clf.fit(data_features, np.ravel(data_labels))

    penalty = clf.best_params_['penalty']
    alpha = clf.best_params_['alpha']
    l1_ratio = clf.best_params_['l1_ratio']
    loss = clf.best_params_['loss']
    epsilon = clf.best_params_['epsilon']

    return penalty, alpha, l1_ratio, loss, epsilon


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

def check_labels_regression(dataframe):
    """Test whether or not `dataframe` has at least five non-null value for each regresison label

    For a model where the label is always None,
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
    model_names = all_parameter_keys
    possible_models = {}
    for mnm in model_names:
        data = dataframe[dataframe[mnm].isnull() == False]
        if data.shape[0] > 4:
            possible_models[mnm] = True
        else:
            possible_models[mnm] = False
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
    scores = model_selection.cross_val_score(
        logsgdc, scaler.transform(df[features]), df[label], cv=5)
    return scores.mean()


def testing_using_crossvalidation_regression(df, label, features, alpha,
                                    l1_ratio, penalty, loss, epsilon, label_std):
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
    reg = linear_model.SGDRegressor(alpha= alpha, loss= loss,
                                        penalty = penalty,l1_ratio = l1_ratio,
                                        epsilon = epsilon, max_iter=1000)
    scores = model_selection.cross_val_score(
        reg, df[features], df[label], cv=5, scoring = 'neg_mean_absolute_error')
    return -1.0 * scores.mean()/label_std


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

            # The number of class labels must be greater than one
            if len(tr[label].unique()) < 2:
                continue

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

def testing_by_experiments_regression(df, label, features, alpha, l1_ratio,
                                      penalty, loss, epsilon, label_std):
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

            reg = linear_model.SGDRegressor(alpha= alpha, loss= loss,
                                        penalty = penalty,l1_ratio = l1_ratio,
                                        epsilon = epsilon, max_iter=1000)
            reg.fit(tr[features], tr[label])
            pr = reg.predict(test[features])
            test_score = mean_absolute_error(pr, test[label])
            test_scores_by_ex.append(test_score/label_std)
            count +=1
    normalized_error =  sum(test_scores_by_ex)/count
    return normalized_error

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
        feats = OrderedDict.fromkeys(all_profile_keys)
        pops = OrderedDict.fromkeys(population_keys)
        par = OrderedDict.fromkeys(all_parameter_keys)
        expt_id,t_utc,q_I,temp,pif_feats,pif_pops,pif_par,rpt = saxs_piftools.unpack_pif(pp)
        feats.update(saxs_math.profile_spectrum(q_I))
        feats.update(saxs_math.detailed_profile(q_I,pif_pops))
        pops.update(pif_pops)
        par.update(pif_par)
        param_list = []
        for k in par.keys():
            if par[k] is not None:
                val = par[k][0]
            else:
                val = None
            param_list.append(val)

        data_row = [expt_id]+list(feats.values())+list(pops.values())+param_list
        data.append(data_row)

    colnames = ['experiment_id']
    colnames.extend(all_profile_keys)
    colnames.extend(population_keys)
    colnames.extend(all_parameter_keys)

    d = pd.DataFrame(data=data, columns=colnames)
    d = d.where((pd.notnull(d)), None) # replace all NaN by None
    shuffled_rows = np.random.permutation(d.index)
    df_work = d.loc[shuffled_rows]

    return df_work

def train_classifiers_partial(new_data, file_path=None, all_training_data=None):
    """Read SAXS classification models from a YAML file, then update them with new data.

    Parameters
    ----------
    new_data : pandas.DataFrame
        dataframe containing features and labels for updating models
    file_path : str (optional)
        Full path to YAML file where scalers and models are saved.
        If None, the default saxskit models are used.
    all_training_data : pandas.DataFrame (optional)
        dataframe containing all of the original training data,
        for computing cross-validation errors of the updated models.

    Returns
    -------
    scalers : dict
        Dictionary of sklearn standard scalers (one scaler per model).
    models : dict
        Dictionary of sklearn models.
    cv_errors : dict
        Dictionary of cross-validation errors for each model.
    """
    if file_path is None:
        p = os.path.abspath(__file__)
        d = os.path.dirname(p)
        file_path = os.path.join(d,'modeling_data','scalers_and_models.yml')
    s_and_m_file = open(file_path,'rb')
    s_and_m = yaml.load(s_and_m_file)

    models = s_and_m['models']
    scalers = s_and_m['scalers']
    cv_errors = s_and_m['accuracy']

    possible_models = check_labels(all_training_data)
    features = profile_keys['unidentified']

    # unidentified scatterer population model
    if possible_models['unidentified'] == True:
        scaler, model, cverr = train_partial(True, new_data, features, 'unidentified',
                                           models, scalers, all_training_data)
        if scaler:
            scalers['unidentified'] = scaler.__dict__
        if model:
            models['unidentified'] = model.__dict__
        if cverr:
            cv_errors['unidentified'] = cverr 

    # For the rest of the models,
    # we will use only data with
    # identifiable scattering populations
    new_data = new_data[new_data['unidentified']==False]

    for k, v in possible_models.items():
        if v == True and k != 'unidentified':
            scaler, model, cverr = train_partial(True, new_data, features, k,
                                           models, scalers, all_training_data)
            if scaler:
                scalers[k] = scaler.__dict__
            if model:
                models[k] = model.__dict__
            if cverr:
                cv_errors[k] = cverr
    if all_training_data is None:
        cv_errors['NOTE'] = 'Cross-validation errors '\
        'were not re-computed after partial model training'
    return scalers, models, cv_errors 

def train_regressors_partial(new_data, file_path=None, all_training_data=None):
    """Read SAXS regression models from a YAML file, then update them with new data.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels for updating models
    file_path : str (optional)
        Full path to YAML file where scalers and models are saved.
        If None, the default saxskit models are used.
    all_training_data : pandas.DataFrame (optional)
        dataframe containing all of the original training data
        for computing the accuracy of the updated models.

    Returns
    -------
    scalers : dict
        Dictionary of sklearn standard scalers (one scaler per model).
    models : dict
        Dictionary of sklearn models.
    accuracy : dict
        Dictionary of accuracies for each model.
    """
    if file_path is None:
        p = os.path.abspath(__file__)
        d = os.path.dirname(p)
        file_path = os.path.join(d,'modeling_data','scalers_and_models_regression.yml')

    s_and_m_file = open(file_path,'rb')
    s_and_m = yaml.load(s_and_m_file)
    models = s_and_m['models']
    scalers = s_and_m['scalers']
    cv_errors = s_and_m['accuracy']

    possible_models = check_labels_regression(new_data)
    # r0_sphere model
    if possible_models['r0_sphere'] == True:
        features = []
        features.extend(profile_keys['unidentified'])
        scaler, model, cverr = train_partial(False, new_data, features, 'r0_sphere',
                                           models, scalers, all_training_data)
        if scaler:
            scalers['r0_sphere'] = scaler.__dict__
        if model:
            models['r0_sphere'] = model.__dict__
        if cverr:
            cv_errors['r0_sphere'] = cverr 

    # sigma_shpere model
    if possible_models['sigma_sphere'] == True:
        features = []
        features.extend(profile_keys['unidentified'])
        features.extend(profile_keys['spherical_normal'])
        scaler, model, cverr = train_partial(False, new_data, features, 'sigma_sphere',
                                           models, scalers, all_training_data)
        if scaler:
            scalers['sigma_sphere'] = scaler.__dict__
        if model:
            models['sigma_sphere'] = model.__dict__
        if cverr:
            cv_errors['sigma_sphere'] = cverr 

    # rg_gp model
    if possible_models['rg_gp'] == True:
        features = []
        features.extend(profile_keys['unidentified'])
        features.extend(profile_keys['guinier_porod'])
        scaler, model, cverr = train_partial(False, new_data, features, 'rg_gp',
                                           models, scalers, all_training_data)
        if scaler:
            scalers['rg_gp'] = scaler.__dict__
        if model:
            models['rg_gp'] = model.__dict__
        if cverr:
            cv_errors['rg_gp'] = cverr 
    if all_training_data is None:
        cv_errors['NOTE'] = 'Cross-validation errors '\
        'were not re-computed after partial model training'
    return scalers, models, cv_errors 

# helper function - to set parametrs for scalers and models
def set_param(m_s, param):
        for k, v in param.items():
            if isinstance(v, list):
                setattr(m_s, k, np.array(v))
            else:
                setattr(m_s, k, v)

def train_partial(classifier, data, features, target, reg_models_dict, scalers_dict, testing_data):
    model_params = reg_models_dict[target]
    scaler_params = scalers_dict[target]
    if scaler_params is not None:
        scaler = preprocessing.StandardScaler()
        set_param(scaler,scaler_params)
        if classifier == True:
            model = linear_model.SGDClassifier()
        else:
            model = linear_model.SGDRegressor()
        set_param(model,model_params)
        d = data[data[target].isnull() == False]
        data2 = d.dropna(subset=features)
        scaler.fit(data2[features])
        data2.loc[ : , features] = scaler.transform(data2[features])
        model.partial_fit(data2[features], data2[target])
        if testing_data is None:
            accuracy = None
        else: # calculate training accuracy using all provided data
            d = testing_data[testing_data[target].isnull() == False]
            data = d.dropna(subset=features)
            if len(data.experiment_id.unique()) > 4:
                leaveNGroupOut = True
            else:
                leaveNGroupOut = False
            label_std = data[target].std()
            if leaveNGroupOut:
                if classifier == True:
                    accuracy = testing_by_experiments(
                        data, target, features, model_params['alpha'], model_params['l1_ratio'],
                        model_params['penalty'])
                else:
                    accuracy = testing_by_experiments_regression(
                        data, target, features, model_params['alpha'], model_params['l1_ratio'],
                        model_params['penalty'], model_params['loss'],
                        model_params['epsilon'], label_std)
            else:
                if classifier == True:
                    accuracy = testing_using_crossvalidation(
                        data, target, features,  model_params['alpha'], model_params['l1_ratio'],
                        model_params['penalty'])
                else:
                    accuracy = testing_using_crossvalidation_regression(
                        data, target, features,  model_params['alpha'], model_params['l1_ratio'],
                        model_params['penalty'], model_params['loss'],
                        model_params['epsilon'], label_std)
    else:
        scaler = None
        model = None
        accuracy = None
    return scaler, model, accuracy

def save_models(scalers, models, cv_errors, file_path=None):
    """Save model parameters and CV errors in YAML and .txt files.

    Parameters
    ----------
    scalers : dict
        Dictionary of sklearn standard scalers (one scaler per model).
    models : dict
        Dictionary of sklearn models.
    cv_errors : dict
        Dictionary of normalized cross-validation errors each model.
    file_path : str
        Full path to the YAML file where the models will be saved. 
        Scalers, models, sklearn version, and cross-validation errors 
        will be saved at this path, and the cross-validation errors 
        are also saved in a .txt file of the same name, in the same directory. 
    """
    if file_path is None:
        p = os.path.abspath(__file__)
        d = os.path.dirname(p) 
        suffix = 0
        file_path = os.path.join(d,'modeling_data',
            'custom_models_'+str(suffix)+'.yml')
        while os.path.exists(file_path):
            suffix += 1
            file_path = os.path.join(d,'modeling_data',
                'custom_models_'+str(suffix)+'.yml')
    if not os.path.splitext(file_path)[1] == '.yml':
        file_path = file_path+'.yml'
    cverr_txt_path = os.path.splitext(file_path)[0]+'.txt'
    scalers_and_models = OrderedDict(
        version=list(map(int,sklearn.__version__.split('.'))),
        scalers=scalers,
        models=models,
        accuracy=cv_errors)
    # save scalers and models
    with open(file_path, 'w') as yaml_file:
        yaml.dump(scalers_and_models, yaml_file)
    # save accuracy
    with open(cverr_txt_path, 'w') as txt_file:
        txt_file.write(str(cv_errors))


