
def train_classifiers(cit_client,dsid_list=[],yaml_filename=None):
    """Train and save SAXS classification models as a YAML file.

    Parameters
    ----------
    cit_client : citrination_client.CitrinationClient
        A python Citrination client for fetching data
    dsid_list : list of int
        List of dataset id (integers) for fetching SAXS records
    yaml_filename : str
        File where scalers and models will be saved.
        If None, the default file is used.
    """


    current_version = list(map(int,sklearn.__version__.split('.')))
    major,minor,patch = current_version

    scalers = {}
    models = {}
    scalers_and_models = {'version':current_version, 'scalers' : scalers, 'models': models}

    all_data = get_data_from_Citrination(cit_client, dsid_list)
    features = ['q_Imax', 'Imax_over_Imean', 'Imax_sharpness','logI_fluctuation', 'logI_max_over_std']

    # bad_data model
    scaler = preprocessing.StandardScaler()
    scaler.fit(all_data[features])
    transformed_data = scaler.transform(all_data[features])

    if hyper_parameters_search == True:
        penalty, alpha, l1_ratio = hyperparameters_search(transformed_data, all_data[['bad_data']], all_data['experiment'])
    else:
        penalty = 'l1'
        alpha = 0.001
        l1_ratio = 1.0

    log = linear_model.SGDClassifier(alpha= alpha,loss= 'log', penalty= penalty, l1_ratio = l1_ratio)
    log.fit(transformed_data, all_data['bad_data'])

    # save the scaler and model for "bad_data"
    scalers['bad_data'] = scaler.__dict__
    models['bad_data'] = log.__dict__

    # we will use only data with "bad_data" = Fasle for the other models
    all_data = all_data[all_data['bad_data']==False]

    # form_scattering model
    scaler = preprocessing.StandardScaler()
    scaler.fit(all_data[features])
    transformed_data = scaler.transform(all_data[features])

    if hyper_parameters_search == True:
        penalty, alpha, l1_ratio = hyperparameters_search(transformed_data, all_data[['form']], all_data['experiment'])
    else:
        penalty = 'l1'
        alpha = 0.001
        l1_ratio = 1.0

    log = linear_model.SGDClassifier(alpha= alpha,loss= 'log', penalty= penalty, l1_ratio = l1_ratio)
    log.fit(transformed_data, all_data['form'])

    scalers['form_factor_scattering'] = scaler.__dict__
    models['form_factor_scattering'] = log.__dict__

    # precursor_scattering model
    scaler = preprocessing.StandardScaler()
    scaler.fit(all_data[features])
    transformed_data = scaler.transform(all_data[features])

    if hyper_parameters_search == True:
        penalty, alpha, l1_ratio = hyperparameters_search(transformed_data, all_data[['precursor']], all_data['experiment'])
    else:
        penalty = 'elasticnet'
        alpha = 0.01
        l1_ratio = 0.85

    log = linear_model.SGDClassifier(alpha= alpha,loss= 'log', penalty= penalty, l1_ratio = l1_ratio)
    log.fit(transformed_data, all_data['precursor'])

    scalers['precursor_scattering'] = scaler.__dict__
    models['precursor_scattering'] = log.__dict__

    # diffraction_peaks model
    scaler = preprocessing.StandardScaler()
    scaler.fit(all_data[features])
    transformed_data = scaler.transform(all_data[features])

    if hyper_parameters_search == True:
        penalty, alpha, l1_ratio = hyperparameters_search(transformed_data, all_data[['structure']], all_data['experiment'])
    else:
        penalty = 'elasticnet'
        alpha = 0.001
        l1_ratio = 0.85

    log = linear_model.SGDClassifier(alpha= alpha,loss= 'log', penalty= penalty, l1_ratio = l1_ratio)
    log.fit(transformed_data, all_data['structure'])

    scalers['diffraction_peaks'] = scaler.__dict__
    models['diffraction_peaks'] = log.__dict__

    # save scalers and models
    with open('modeling_data/scalers_and_models.yml', 'w') as yaml_file:
        yaml.dump(scalers_and_models, yaml_file)






def train_regressors(cit_client,dsid_list=[],yaml_filename=None):
    """Train and save SAXS regression models as a YAML file.

    Parameters
    ----------
    cit_client : citrination_client.CitrinationClient
        A python Citrination client for fetching data
    dsid_list : list of int
        List of dataset id (integers) for fetching SAXS records
    yaml_filename : str
        File where scalers and models will be saved.
        If None, the default file is used.
    """
    pass

