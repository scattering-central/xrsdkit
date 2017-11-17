
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



def hyperparameters_search(data_features, data_labels, group_by):
    """Grid search for alpha, penalty, and l1 ratio

    Parameters
    ----------
    data_features : 2D numpy array of features
    data_labels : a column of dataframe with labels
    group_by: a column of dataframe we want to group_by

    Returns
        -------
        string penalty, float alpha, string l1_ratio
    """

    parameters = {'penalty':('none', 'l2', 'l1', 'elasticnet'), #default l2
              'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1], #regularisation koef, default 0.0001
             'l1_ratio': [0, 0.15, 0.5, 0.85, 1.0]} #using with elasticnet only; default 0.15

    cv=LeavePGroupsOut(n_groups=2).split(data_features, data_labels, groups=group_by)

    svc = linear_model.SGDClassifier(loss = 'log')
    clf = GridSearchCV(svc, parameters, cv=cv)
    clf.fit(data_features, data_labels)

    penalty = clf.best_score_['penalty']
    alpha = clf.best_score_['alpha']
    l1_ratio = clf.best_score_['l1_ratio']

    return penalty, alpha, l1_ratio


def get_data_from_Citrination(client, dataset_id_list):
    """Get data from Citrination and create a dataframe


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



    df = pd.DataFrame(columns= [['experiment_id', 'q_Imax', 'Imax_over_Imean', 'Imax_sharpness','logI_fluctuation',
                                 'logI_max_over_std', 'bad_data', 'form', 'precursor', 'structure',
                                 'ro_shpere', 'sigma_sphere', 'g_precursor', 'rg_precursor']])
    for dataset in dataset_id_list:
        query = PifSystemReturningQuery(
            from_index=0,
            size=100,
            query=DataQuery(
                dataset=DatasetQuery(
                    id=Filter(
                    equal=dataset))))


        all_hits = []
        while len(all_hits) < 1200: # max for one dataset
            current_result = client.search(query)
            if current_result.hits is None:
                break
            all_hits.extend(current_result.hits)
            query.from_index += len(current_result.hits)

        for line in all_hits: # every line of pifs is one sample; we need to extract labels and features from it
            try:
                my_str = dumps(line)
                obj = json.loads(my_str) # to transform the string to dictionary

                # default values for labels
                bad_data = False
                form = False
                precursor = False
                structure = False

                ro_sphere = None
                sigma_sphere = None
                g_precursor = None
                rg_precursor = None

                experiment_id = None

                for i in obj['system']['ids']:
                    if i['name'] == 'EXPERIMENT_ID':
                        experiment_id = i['value']

                for pr in obj['system']['properties']:

                    # extract features
                    if pr['name'] == 'q_Imax':
                        q_Imax = np.float32(pr['scalars'][0]['value'])
                    if pr['name'] == 'Imax_over_Imean':
                        Imax_over_Imean = np.float32(pr['scalars'][0]['value'])
                    if pr['name'] == 'Imax_sharpness':
                        Imax_sharpness = np.float32(pr['scalars'][0]['value'])
                    if pr['name'] == 'logI_fluctuation':
                        logI_fluctuation = np.float32(pr['scalars'][0]['value'])
                    if pr['name'] == 'logI_max_over_std':
                        logI_max_over_std = np.float32(pr['scalars'][0]['value'])

                    # extract labels
                    if pr['name'] == 'bad_data_flag':
                        bad_data = bool(float(pr['scalars'][0]['value']))
                        if bad_data == True:
                            continue
                    if pr['name'] == 'form_factor_scattering_flag':
                        form = bool(float(pr['scalars'][0]['value']))
                    if pr['name'] == 'diffraction_peaks_flag':
                        structure = bool(float(pr['scalars'][0]['value']))
                    if pr['name'] == 'precursor_scattering_flag':
                        precursor = bool(float(pr['scalars'][0]['value']))

                    if pr['name'] == 'r0_sphere':
                        ro_sphere = np.float32(pr['scalars'][0]['value'])
                    if pr['name'] == 'sigma_sphere':
                        sigma_sphere = np.float32(pr['scalars'][0]['value'])
                    if pr['name'] == 'G_precursor':
                        g_precursor = np.float32(pr['scalars'][0]['value'])
                    if pr['name'] == 'rg_precursor':
                        rg_precursor = np.float32(pr['scalars'][0]['value'])

                df.loc[df.shape[0]] = [experiment_id, q_Imax, Imax_over_Imean, Imax_sharpness, logI_fluctuation,
                                           logI_max_over_std, bad_data, form, precursor, structure,
                                      ro_sphere, sigma_sphere, g_precursor, rg_precursor]
            except:
                # May be in PAWS we need to put a custom exeption here
                my_str = dumps(line)
                obj = json.loads(my_str) # to transform the string to dictionary
                print(obj)
                continue

    d = df.convert_objects(convert_numeric=True)
    #d = pd.to_numeric(df)
    shuffled_rows = np.random.permutation(d.index)
    df_work = d.loc[shuffled_rows]

    return df_work

