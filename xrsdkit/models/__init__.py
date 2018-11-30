import os
import re
from collections import OrderedDict

import yaml
import pandas as pd
import numpy as np
from citrination_client import CitrinationClient
from sklearn import preprocessing

from .. import definitions as xrsdefs 
from .regressor import Regressor
from .classifier import Classifier
from ..tools import primitives, profiler, piftools
from ..system import System

file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(file_path))
root_dir = os.path.dirname(src_dir)

# find directory containing packaged modeling data
modeling_data_dir = os.path.join(src_dir,'models','modeling_data')
regression_models_dir = os.path.join(modeling_data_dir,'regressors')
classification_models_dir = os.path.join(modeling_data_dir,'classifiers')

# find directory containing test modeling data
testing_data_dir = os.path.join(src_dir,'models','modeling_data','test')
if not os.path.exists(testing_data_dir): os.mkdir(testing_data_dir)
test_regression_models_dir = os.path.join(testing_data_dir,'regressors')
test_classification_models_dir = os.path.join(testing_data_dir,'classifiers')

# read api key from file if present
# TODO (later): look for user's api key in their home directory?
# NOTE: this will be platform-dependent
api_key_file = os.path.join(root_dir, 'api_key.txt')
citcl = None
if os.path.exists(api_key_file):
    a_key = open(api_key_file, 'r').readline().strip()
    citcl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)

# read index of Citrination dataset ids 
src_dsid_file = os.path.join(src_dir,'models','modeling_data','source_dataset_ids.yml')
src_dsid_list = yaml.load(open(src_dsid_file,'r'))
model_dsid_file = os.path.join(src_dir,'models','modeling_data','modeling_dataset_ids.yml')
model_dsids = yaml.load(open(model_dsid_file,'r'))

# --- LOAD READY-TRAINED MODELS --- #
# for any models currently saved as yml files,
# the model parameters are loaded and the model 
# is added to one of the dicts of saved models 
# (either regression_models or classification_models).
# For models not currently saved as yml files, 
# the models must first be created by train_regression_models()
    
_reg_params = list(xrsdefs.param_defaults.keys())
_reg_params[_reg_params.index('I0')] = 'I0_fraction'

# --- LOAD REGRESSION MODELS --- #
def load_regression_models(model_root_dir=regression_models_dir):
    model_dict = OrderedDict()
    if not os.path.exists(model_root_dir): os.mkdir(model_root_dir)
    for sys_cls in os.listdir(model_root_dir):
        model_dict[sys_cls] = {}
        sys_cls_dir = os.path.join(model_root_dir,sys_cls)
        # the directory for the system class has subdirectories
        # for each population in the system
        for pop_id in os.listdir(sys_cls_dir):
            model_dict[sys_cls][pop_id] = {}
            pop_dir = os.path.join(sys_cls_dir,pop_id)
            # the directory for the population has model parameter files,
            # subdirectories for parameters of each modellable structure,
            # and subdirectories for each modellable basis class...
            # or, if pop_id == 'noise', there will be a subdirectory for the noise model
            for pop_itm in os.listdir(pop_dir):
                if pop_itm in xrsdefs.noise_model_names + \
                xrsdefs.setting_selections['lattice'] + xrsdefs.setting_selections['interaction']:
                    # this is a subdirectory for parameters of noise, or of crystalline or disordered structures
                    model_dict[sys_cls][pop_id][pop_itm] = {}
                    pop_subdir = os.path.join(pop_dir,pop_itm)
                    for pop_subitm in os.listdir(pop_subdir):
                        if pop_subitm.endswith('.yml'):
                            param_nm = pop_subitm.split('.')[0]
                            yml_path = os.path.join(pop_subdir,pop_subitm)
                            model_dict[sys_cls][pop_id][pop_itm][param_nm] = Regressor(param_nm,yml_path) 
                elif pop_itm.endswith('.yml'):
                    # model parameters
                    param_nm = pop_itm.split('.')[0]
                    yml_path = os.path.join(pop_dir,pop_itm)
                    model_dict[sys_cls][pop_id][param_nm] = Regressor(param_nm,yml_path) 
                elif not pop_itm.endswith('.txt'):
                    # subdirectory for a basis class
                    bas_cls = pop_itm.split('.')[0]
                    model_dict[sys_cls][pop_id][bas_cls] = {} 
                    bas_cls_dir = os.path.join(pop_dir,bas_cls)
                    # the directory for the basis class has subdirectories
                    # for each specie in the basis
                    for specie_id in os.listdir(bas_cls_dir):
                        model_dict[sys_cls][pop_id][bas_cls][specie_id] = {}
                        specie_dir = os.path.join(bas_cls_dir,specie_id)
                        # the directory for the specie should contain only model parameter files 
                        for specie_itm in os.listdir(specie_dir):
                            if specie_itm.endswith('.yml'):
                                param_nm = specie_itm.split('.')[0]
                                yml_path = os.path.join(specie_dir,specie_itm)
                                model_dict[sys_cls][pop_id][bas_cls][specie_id][param_nm] = \
                                Regressor(param_nm,yml_path)
    return model_dict
regression_models = load_regression_models(regression_models_dir)
test_regression_models = load_regression_models(test_regression_models_dir) 


# --- LOAD CLASSIFICATION MODELS --- #
def load_classification_models(model_root_dir=classification_models_dir):  
    model_dict = OrderedDict()
    if not os.path.exists(model_root_dir): os.mkdir(model_root_dir)
    yml_path = os.path.join(model_root_dir,'system_classification.yml')
    if os.path.exists(yml_path):
        model_dict['system_classification'] = Classifier('system_classification',yml_path)
    for sys_cls in os.listdir(model_root_dir):
        # if this is not a yml or text file, it is a subdirectory for a system class
        if not sys_cls.endswith('.yml') and not sys_cls.endswith('.txt'):
            model_dict[sys_cls] = {}
            sys_cls_dir = os.path.join(model_root_dir,sys_cls)
            noise_yml_path = os.path.join(sys_cls_dir,'noise_classification.yml')
            if os.path.exists(noise_yml_path):
                model_dict[sys_cls]['noise'] = Classifier('noise_classification',noise_yml_path)
            # the directory for the system class has subdirectories
            # for each population in the system
            for pop_id in os.listdir(sys_cls_dir):
                if not pop_id.endswith('.yml') and not pop_id.endswith('.txt'):
                    model_dict[sys_cls][pop_id] = {}
                    pop_dir = os.path.join(sys_cls_dir,pop_id)
                    # the directory for the population class contains
                    # yml files with model parameters
                    # for the population's basis classifier
                    # and other structure-specific classifiers if applicable
                    for pop_itm in os.listdir(pop_dir):
                        if pop_itm.endswith('.yml'):
                            yml_path = os.path.join(pop_dir,pop_itm)
                            model_type = os.path.splitext(pop_itm)[0]
                            model_label = pop_id+'_'+model_type
                            model_dict[sys_cls][pop_id][model_type] = Classifier(model_label,yml_path)
                        # else (currently not implemented):
                        # if not a yml file, then it would be a sub-directory
                        # containing classifiers applicable to a specific basis class,
                        # e.g. specie classifiers for atomic form factors
    return model_dict
classification_models = load_classification_models(classification_models_dir) 
test_classification_models = load_classification_models(test_classification_models_dir)

def downsample_and_train(
    source_dataset_ids=src_dsid_list,
    citrination_client=citcl,
    save_samples=False,
    save_models=False,
    train_hyperparameters=False,
    test=False):
    """Downsample datasets and use the samples to train xrsdkit models.

    This is a developer tool for building models 
    from a set of Citrination datasets.
    It is used by the package developers to deploy
    a standard set of models with xrsdkit.

    Parameters
    ----------
    source_dataset_ids : list of int
        Dataset ids for downloading source data
    save_samples : bool
        If True, downsampled datasets will be saved to their own datasets,
        according to xrsdkit/models/modeling_data/dataset_ids.yml
    save_models : bool
        If True, the models will be saved to yml files 
        in xrsdkit/models/modeling_data/
    train_hyperparameters : bool
        if True, the models will be optimized during training,
        for best cross-validation performance
        over a grid of hyperparameters 
    test : bool
        if True, the downsampling statistics and models will be
        saved in modeling_data/testing_data dir
    """
    df = piftools.get_data_from_Citrination(citrination_client,source_dataset_ids)
    df_sample = downsample_by_group(df)
    train_from_dataframe(df_sample,train_hyperparameters,save_models,test)

def downsample_by_group(df):
    """Group and down-sample a DataFrame of xrsd records.
        
    Parameters
    ----------
    df : pandas.DataFrame
        dataframe containing xrsd samples 

    Returns
    -------
    data_sample : pandas.DataFrame
        DataFrame containing all of the down-sampled data from 
        all of the datasets in `dataset_id_list`. 
        Features in this DataFrame are not scaled:
        the correct scaler should be applied before training models.
    """
    #### create data_sample ########################
    data_sample = pd.DataFrame(columns=df.columns)
    #expt_samples = {}
    #expt_local_ids = {} # local ids of samples to save by exp
    #all_exp = data.experiment_id.unique()
    group_cols, all_groups = group_by_labels(df)

    #for exp_id in all_exp:
    # downsample each group independently
    for group_labels,grp in all_groups.groups.items():
        group_df = df.iloc[grp].copy()
        print('Downsampling data for group: {}'.format(group_labels))
        #lbl_df = _filter_by_labels(data,lbls)
        dsamp = downsample(df.iloc[grp].copy(), 1.0)
        print('Finished downsampling: kept {}/{}'.format(len(dsamp),len(group_df)))
        data_sample = data_sample.append(dsamp)
    return data_sample

def downsample(df, min_distance):
    """Downsample records from one DataFrame.

    Transforms the DataFrame feature arrays 
    (scaling by the columns in profiler.profile_keys),
    before collecting at least 10 samples.
    If the size of `df` is <= 10, it is returned directly.
    If it is larger than 10, the first point is chosen
    based on greatest nearest-neighbor distance.
    Subsequent points are chosen  
    in order of decreasing nearest-neighbor distance
    to the already-sampled points. 

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe containing xrsd samples 
    min_distance : float
        the minimum allowed nearest-neighbor distance 
        for continuing to downsample after 10 or more samples
        have been selected 

    Returns
    -------
    sample : pandas.DataFrame
        dataframe containing subset of rows
        that was chosen using distance between the samples
    """
    df_size = len(df)
    sample = pd.DataFrame(columns=df.columns)
    if df_size <= 10:
        sample = sample.append(df)
    else:
        scaler = preprocessing.StandardScaler()
        scaler.fit(df[profiler.profile_keys])

        features_matr = scaler.transform(df[profiler.profile_keys]) 
        # define the distance between two samples in feature space

        dist_func = lambda i,j: np.sum(
            np.abs(features_matr[i]
            - features_matr[j]))
        dist_matrix = np.array([[dist_func(i,j) for i in range(df_size)] for j in range(df_size)])
        # get the most isolated sample first:
        # this should be the sample with the greatest 
        # nearest-neighbor distance 
        nn_distance_array = np.array([min(dist_matrix[i,:]) for i in range(df_size)])
        best_idx = np.argmax(nn_distance_array)
        sample = sample.append(df.iloc[best_idx])
        sampled_idxs = [best_idx] 
        continue_downsampling = True
        while(continue_downsampling):
            # find the next best index to sample:
            # the sample with the greatest minimum distance
            # between itself and the downsampled samples
            sample_size = len(sample)
            sample_dist_matrix = np.array([dist_matrix[i,:] for i in sampled_idxs])
            nn_distance_array = np.array([min(sample_dist_matrix[:,j]) for j in range(df_size)])
            best_idx = np.argmax(nn_distance_array)
            best_nn_distance = nn_distance_array[best_idx]
            # if we have at least 10 samples,
            # and all remaining samples are close to the current sample,
            # down-sampling can stop here.
            if sample_size >= 10 and best_nn_distance < min_distance: 
                continue_downsampling = False
            else:
                sampled_idxs.append(best_idx)
                sample = sample.append(df.iloc[best_idx])
    return sample


def train_from_dataframe(data,train_hyperparameters=False,save_models=False,test=False):
    # regression models:
    reg_models = train_regression_models(data, hyper_parameters_search=train_hyperparameters)
    # classification models: 
    cls_models = train_classification_models(data, hyper_parameters_search=train_hyperparameters)
    # optionally, save the models:
    # this adds/updates yml files and also adds the models
    # to the regression_models and classification_models dicts.
    if save_models:
        save_regression_models(reg_models, test=test)
        save_classification_models(cls_models, test=test)

def train_regression_models(data, hyper_parameters_search=False):
    """Train all trainable regression models from `data`.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels
    hyper_parameters_search : bool
        If true, grid-search model hyperparameters
        to seek high cross-validation R^2 score.

    Returns
    -------
    models : dict
        the dict keys are system class names, and the values are
        dictionaries of regression models for the system class
    """
    reg_models = trainable_regression_models(data)
    for sys_cls,sys_models in reg_models.items():
        print(os.linesep+'Training regressors for system class: ')
        print(sys_cls)
        sys_cls_data = data[(data['system_classification']==sys_cls)]
        for pop_id,pop_models in sys_models.items():
            print('population id: {}'.format(pop_id))
            for k in pop_models.keys():
                if k in _reg_params:
                    print('    parameter: {}'.format(k))
                    # train reg_models[sys_cls][pop_id][k]
                    target = pop_id+'_'+k
                    reg_model = Regressor(target, None)

                    try: # check if we already have a trained model for this label
                        old_pars = regression_models[sys_cls][pop_id][k].model.get_params()
                        reg_model.model.set_params(alpha=old_pars['alpha'], l1_ratio=old_pars['l1_ratio'],
                                                   epsilon=old_pars['epsilon'])
                    except:
                        pass
                    reg_model.train(sys_cls_data, hyper_parameters_search)
                    if not reg_model.trained:
                        print('    insufficient data or zero variance: using default value')
                    pop_models[k] = reg_model
                elif k in xrsdefs.setting_selections['lattice']:
                    print('    structure: {}'.format(k))
                    for param_nm in xrsdefs.setting_params['lattice'][k]:
                        print('        parameter: {}'.format(param_nm))
                        # train reg_models[sys_cls][pop_id][k][param_nm]
                        lattice_label = pop_id+'_lattice'
                        sub_cls_data = sys_cls_data[(sys_cls_data[lattice_label]==k)]
                        target = pop_id+'_'+param_nm
                        reg_model = Regressor(target, None)
                        try:
                            old_pars = regression_models[sys_cls][pop_id][k][param_nm].model.get_params()
                            reg_model.model.set_params(alpha=old_pars['alpha'], l1_ratio=old_pars['l1_ratio'],
                                                       epsilon=old_pars['epsilon'])
                        except:
                            pass
                        reg_model.train(sub_cls_data, hyper_parameters_search)
                        if not reg_model.trained:
                            print('        insufficient data or zero variance: using default value')
                        pop_models[k][param_nm] = reg_model
                elif k in xrsdefs.setting_selections['interaction']:
                    print('    interaction: {}'.format(k))
                    for param_nm in xrsdefs.setting_params['interaction'][k]:
                        print('        parameter: {}'.format(param_nm))
                        # train reg_models[sys_cls][pop_id][k][param_nm]
                        interxn_label = pop_id+'_interaction'
                        sub_cls_data = sys_cls_data[(sys_cls_data[interxn_label]==k)]
                        target = pop_id+'_'+param_nm
                        reg_model = Regressor(target, None)
                        try:
                            old_pars = regression_models[sys_cls][pop_id][k][param_nm].model.get_params()
                            reg_model.model.set_params(alpha=old_pars['alpha'], l1_ratio=old_pars['l1_ratio'],
                                                       epsilon=old_pars['epsilon'])
                        except:
                            pass
                        reg_model.train(sub_cls_data, hyper_parameters_search)
                        if not reg_model.trained:
                            print('        insufficient data or zero variance: using default value')
                        pop_models[k][param_nm] = reg_model
                elif k in xrsdefs.noise_model_names:
                    print('    noise model: {}'.format(k))
                    noise_param_models = pop_models[k]
                    for param_nm in noise_param_models.keys():
                        print('            parameter: {}'.format(param_nm))
                        target = pop_id+'_'+param_nm
                        reg_model = Regressor(target, None)
                        try:
                            old_pars = regression_models[sys_cls][pop_id][k][param_nm].model.get_params()
                            reg_model.model.set_params(alpha=old_pars['alpha'], l1_ratio=old_pars['l1_ratio'],
                                                           epsilon=old_pars['epsilon'])
                        except:
                            pass
                        reg_model.train(sys_cls_data, hyper_parameters_search)
                        if not reg_model.trained:
                            print('            insufficient data or zero variance: using default value')
                        noise_param_models[param_nm] = reg_model
                else:
                    # k is a basis classification
                    print('    basis class: {}'.format(k))
                    bas_cls = k
                    bas_models = pop_models[k]
                    bas_cls_label = pop_id+'_basis_classification'
                    bas_cls_data = sys_cls_data[(sys_cls_data[bas_cls_label]==bas_cls)]
                    for specie_id,specie_models in bas_models.items():
                        print('        specie id: {}'.format(specie_id))
                        for param_nm in specie_models.keys():
                            print('            parameter: {}'.format(param_nm))
                            target = pop_id+'_'+specie_id+'_'+param_nm
                            reg_model = Regressor(target, None)
                            try:
                                old_pars = regression_models[sys_cls][pop_id][k][specie_id][param_nm].model.get_params()
                                reg_model.model.set_params(alpha=old_pars['alpha'], l1_ratio=old_pars['l1_ratio'],
                                                           epsilon=old_pars['epsilon'])
                            except:
                                pass
                            reg_model.train(bas_cls_data, hyper_parameters_search)
                            if not reg_model.trained:
                                print('            insufficient data or zero variance: using default value')
                            specie_models[param_nm] = reg_model
    return reg_models

def trainable_regression_models(data):
    """Get a data structure for all regression models trainable from `data`. 

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels

    Returns
    -------
    reg_models : dict
        embedded dictionary with blank spaces for all trainable regression models 
        (a model is trainable if there are sufficient samples in the provided `data`)
    """
    
    # TODO: fix re.match args to scale to infinity
    # NOTE: see how this is done in predict() using re.search and re.sub

    # extract all unique system classifications:
    # the system classification is the top level of the dict of labels
    sys_cls_labels = list(data['system_classification'].unique())
    if 'unidentified' in sys_cls_labels: sys_cls_labels.pop(sys_cls_labels.index('unidentified'))
    reg_models = dict.fromkeys(sys_cls_labels)
    for sys_cls in sys_cls_labels:
        reg_models[sys_cls] = {}
        # get the slice of `data` that is relevant for this sys_cls
        sys_cls_data = data.loc[data['system_classification']==sys_cls].copy()
        # drop the columns where all values are None:
        sys_cls_data.dropna(axis=1,how='all',inplace=True)
        # every system class has one or more models for noise parameters 
        reg_models[sys_cls]['noise'] = {}
        noise_tp_labels = list(sys_cls_data['noise_classification'].unique())
        for ntp in noise_tp_labels:
            reg_models[sys_cls]['noise'][ntp] = {}
            for pk in xrsdefs.noise_params[ntp]+['I0_fraction']:
                if pk in _reg_params:
                    reg_models[sys_cls]['noise'][ntp][pk] = None 
        # use the sys_cls to identify the populations and their structures
        pop_struct_specifiers = sys_cls.split('__')
        for pop_struct in pop_struct_specifiers:
            pop_id = pop_struct[:re.compile('pop._').match(pop_struct).end()-1]
            structure_id = pop_struct[re.compile('pop._').match(pop_struct).end():]
            reg_models[sys_cls][pop_id] = {}
            for param_nm in xrsdefs.structure_params[structure_id]+['I0_fraction']:
                reg_label = pop_id+'_'+param_nm
                if reg_label in sys_cls_data.columns and param_nm in _reg_params:
                    reg_models[sys_cls][pop_id][param_nm] = None
            # if the structure is crystalline or disordered, add sub-dicts
            # for the parameters of the relevant lattices or interactions
            if structure_id == 'crystalline':
                lattice_header = pop_id+'_lattice'
                lattice_labels = sys_cls_data[lattice_header].unique()
                for ll in lattice_labels:
                    reg_models[sys_cls][pop_id][ll] = {} 
                    for param_nm in xrsdefs.setting_params['lattice'][ll]:
                        reg_label = pop_id+'_'+param_nm
                        if reg_label in sys_cls_data.columns and param_nm in _reg_params: 
                            reg_models[sys_cls][pop_id][ll][param_nm] = None
            elif structure_id == 'disordered':
                interxn_header = pop_id+'_interaction'
                interxn_labels = sys_cls_data[interxn_header].unique() 
                for il in interxn_labels:
                    reg_models[sys_cls][pop_id][il] = {} 
                    for param_nm in xrsdefs.setting_params['interaction'][il]:
                        reg_label = pop_id+'_'+param_nm
                        if reg_label in sys_cls_data.columns and param_nm in _reg_params:
                            reg_models[sys_cls][pop_id][il][param_nm] = None
            # add entries for modellable parameters of any species found in this class 
            bas_cls_header = pop_id+'_basis_classification'
            bas_cls_labels = list(sys_cls_data[bas_cls_header].unique())
            for bas_cls in bas_cls_labels:
                reg_models[sys_cls][pop_id][bas_cls] = {}
                # get the slice of `data` that is relevant for this bas_cls
                bas_cls_data = sys_cls_data.loc[sys_cls_data[bas_cls_header]==bas_cls].copy()
                # drop the columns where all values are None:
                bas_cls_data.dropna(axis=1,how='all',inplace=True)
                # use the bas_cls to identify the species and their forms 
                specie_form_specifiers = bas_cls.split('__')
                for specie_form in specie_form_specifiers:
                    specie_id = specie_form[:re.compile('specie._').match(specie_form).end()-1]
                    form_id = specie_form[re.compile('specie._').match(specie_form).end():]
                    # add a dict of models for this specie 
                    reg_models[sys_cls][pop_id][bas_cls][specie_id] = {}
                    # determine all modellable params for this form
                    for param_nm in xrsdefs.form_factor_params[form_id]:
                        reg_label = pop_id+'_'+specie_id+'_'+param_nm
                        if reg_label in bas_cls_data.columns and param_nm in _reg_params:
                            reg_models[sys_cls][pop_id][bas_cls][specie_id][param_nm] = None 
    return reg_models

def train_classification_models(data, hyper_parameters_search=False):
    """Train all trainable classification models from `data`. 

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels
    hyper_parameters_search : bool
        if True, the models will be optimized
        over a grid of hyperparameters during training

    Returns
    -------
    cls_models : dict
        Embedded dicts containing all possible classification models
        (as instances of sklearn.SGDClassifier)
        trained on the given dataset `data`.
    """
    cls_models = trainable_classification_models(data)
    if 'system_classification' in cls_models:
        print(os.linesep+'Training main system classifier')
        model = Classifier('system_classification',None)

        if 'system_classification'in classification_models.keys() \
        and classification_models['system_classification'].trained: 
            old_pars = classification_models['system_classification'].model.get_params()
            model.model.set_params(alpha=old_pars['alpha'], l1_ratio=old_pars['l1_ratio'])

        model.train(data, hyper_parameters_search=hyper_parameters_search)
        if not model.trained: 
            print('insufficient or uniform training data: using default value')
        cls_models['system_classification'] = model

    for sys_cls_lbl, sys_models in cls_models.items():
        if not sys_cls_lbl == 'system_classification':
            sys_cls_data = data[data['system_classification']==sys_cls_lbl]
            print('Training classifiers for system: ')
            print(sys_cls_lbl)
            for pop_id,pop_models in sys_models.items():
                print('population id: {}'.format(pop_id))

                if pop_id == 'noise':
                    print('    Training noise classifier for system class {}'.format(sys_cls_lbl))
                    model = Classifier('noise_classification',None)
                    if sys_cls_lbl in classification_models.keys():
                        if 'noise' in classification_models[sys_cls_lbl].keys() and \
                        classification_models[sys_cls_lbl]['noise'].trained: # we have a trained model
                            old_pars = classification_models[sys_cls_lbl]['noise'].model.get_params()
                            model.model.set_params(alpha=old_pars['alpha'], l1_ratio=old_pars['l1_ratio'])
                    model.train(sys_cls_data, hyper_parameters_search=hyper_parameters_search)
                    if not model.trained:
                        print('    insufficient or uniform training data: using default value')
                    sys_models[pop_id] = model

                else:
                    # pop_models will be classifiers for population `pop_id`
                    for cls_label in pop_models.keys():
                        model_label = pop_id+'_'+cls_label
                        print('    classifier: {}'.format(model_label))
                        m = Classifier(model_label,None)
                        try: # check if we alredy have a trained model for this label
                            old_pars = classification_models[sys_cls_lbl][pop_id][cls_label].model.get_params()
                            m.model.set_params(alpha=old_pars['alpha'], l1_ratio=old_pars['l1_ratio'])
                        except:
                            pass
                        m.train(sys_cls_data, hyper_parameters_search=hyper_parameters_search)
                        if not m.trained: 
                            print('    insufficient or uniform training data: using default value')
                        pop_models[cls_label] = m 
    return cls_models

def trainable_classification_models(data):
    sys_cls_labels = list(data['system_classification'].unique())
    if 'unidentified' in sys_cls_labels: sys_cls_labels.pop(sys_cls_labels.index('unidentified'))
    cls_models = dict.fromkeys(sys_cls_labels)
    if len(sys_cls_labels) > 0:
        cls_models['system_classification'] = None
    for sys_cls in sys_cls_labels:
        cls_models[sys_cls] = {}
        # get the slice of `data` that is relevant for this sys_cls
        sys_cls_data = data.loc[data['system_classification']==sys_cls].copy()
        # drop the columns where all values are None:
        sys_cls_data.dropna(axis=1,how='all',inplace=True)
        # every system class has a noise classifier
        cls_models[sys_cls]['noise'] = None
        # use the sys_cls to identify the populations and their structures
        pop_struct_specifiers = sys_cls.split('__')
        for pop_struct in pop_struct_specifiers:
            pop_id = pop_struct[:re.compile('pop._').match(pop_struct).end()-1]
            structure_id = pop_struct[re.compile('pop._').match(pop_struct).end():]
            cls_models[sys_cls][pop_id] = {}
            bas_cls_header = pop_id+'_basis_classification'
            bas_cls_labels = list(sys_cls_data[bas_cls_header].unique())
            # NOTE: will any of these len>0 checks ever evaluate to False? Is it practical to remove them?
            if len(bas_cls_labels) > 0: cls_models[sys_cls][pop_id]['basis_classification'] = None
            if structure_id == 'crystalline':
                lattice_cls_header = pop_id+'_lattice'
                lattice_cls_labels = list(sys_cls_data[lattice_cls_header].unique())
                if len(lattice_cls_labels) > 0: cls_models[sys_cls][pop_id]['lattice'] = None
            if structure_id == 'disordered':
                interxn_cls_header = pop_id+'_interaction'
                interxn_cls_labels = list(sys_cls_data[interxn_cls_header].unique())
                if len(interxn_cls_labels) > 0: cls_models[sys_cls][pop_id]['interaction'] = None
    return cls_models

def save_model_data(model,yml_path,txt_path):
    with open(yml_path,'w') as yml_file:
        model_data = dict(
            scaler=dict(),
            model=dict(hyper_parameters=dict(), trained_par=dict()),
            cross_valid_results=primitives(model.cross_valid_results),
            trained=model.trained,
            default_val = model.default_val
            )
        if model.trained:
            hyper_par = list(model.grid_search_hyperparameters.keys())
            for p in hyper_par:
                if p in model.model.__dict__:
                    model_data['model']['hyper_parameters'][p] = model.model.__dict__[p]
            # only "fitted" models can be used for prediction
            # a model is "fitted" when is has
            # "coef_", "intercept_", and "t_"(iteration count)
            tr_par_arrays = ['coef_', 'intercept_', 'classes_']
            for p in tr_par_arrays:
                if p in model.model.__dict__:
                    model_data['model']['trained_par'][p] = model.model.__dict__[p].tolist()
            model_data['model']['trained_par']['t_'] = model.model.__dict__['t_']
            model_data['scaler']['mean_'] = model.scaler.__dict__['mean_'].tolist()
            model_data['scaler']['scale_'] = model.scaler.__dict__['scale_'].tolist()
            if hasattr(model, 'scaler_y'): # only regression models have it
                model_data['scaler_y'] = dict(
                    mean_ = model.scaler_y.__dict__['mean_'].tolist(),
                    scale_ = model.scaler_y.__dict__['scale_'].tolist()
                    )
        yaml.dump(model_data,yml_file)
    with open(txt_path,'w') as txt_file:
        if model.trained:
            res_str = model.print_CV_report()
        else:
            res_str = 'The model was not trained'
        txt_file.write(res_str)

def save_regression_models(models=regression_models, test=False):
    """Serialize `models` to .yml files, and also save them as module attributes.

    The models and scalers are saved to .yml,
    and a report of the cross-validation is saved to .txt.
    xrsdkit.models.regression_models is updated with all `models`.

    Parameters
    ----------
    models : dict
        embedded dict of models, similar to output of train_regression_models().
    test : bool (optional)
        if True, the models will be saved in the testing dir.
    """
    rg_root_dir = regression_models_dir
    model_dict = regression_models
    if test: 
        rg_root_dir = test_regression_models_dir 
        model_dict = test_regression_models
    if not os.path.exists(rg_root_dir): os.mkdir(rg_root_dir)
    for sys_cls, sys_models in models.items():
        sys_dir_path = os.path.join(rg_root_dir,sys_cls)
        if not sys_cls in model_dict: model_dict[sys_cls] = {}
        if not os.path.exists(sys_dir_path): os.mkdir(sys_dir_path)
        for pop_id, pop_models in sys_models.items():
            pop_dir_path = os.path.join(sys_dir_path,pop_id)
            if not pop_id in model_dict[sys_cls]: model_dict[sys_cls][pop_id] = {}
            if not os.path.exists(pop_dir_path): os.mkdir(pop_dir_path)
            for k,v in pop_models.items():
                if k in _reg_params:
                    if v:
                        model_dict[sys_cls][pop_id][k] = v
                        yml_path = os.path.join(pop_dir_path,k+'.yml')
                        txt_path = os.path.join(pop_dir_path,k+'.txt')
                        save_model_data(v,yml_path,txt_path)
                else:
                    if not k in model_dict[sys_cls][pop_id]: model_dict[sys_cls][pop_id][k] = {}
                    pop_subdir_path = os.path.join(pop_dir_path,k)
                    if not os.path.exists(pop_subdir_path): os.mkdir(pop_subdir_path)
                    if k in xrsdefs.setting_selections['lattice']+xrsdefs.setting_selections['interaction']:
                        if k in xrsdefs.setting_selections['lattice']: param_nms = xrsdefs.setting_params['lattice'][k]
                        if k in xrsdefs.setting_selections['interaction']: param_nms = xrsdefs.setting_params['disordered'][k]
                        for param_nm in param_nms:
                            if v[param_nm]:
                                model_dict[sys_cls][pop_id][k][param_nm] = v[param_nm]
                                yml_path = os.path.join(pop_subdir_path,param_nm+'.yml')
                                txt_path = os.path.join(pop_subdir_path,param_nm+'.txt')
                                save_model_data(v[param_nm],yml_path,txt_path)
                    else:
                        # k is a basis class label,
                        # v is a dict of dicts of models for each specie
                        bas_dir = os.path.join(pop_dir_path,k)
                        if not os.path.exists(bas_dir): os.mkdir(bas_dir)

                        if k in xrsdefs.noise_model_names:
                            for param_nm, param_model in v.items():
                                if param_model:
                                    model_dict[sys_cls][pop_id][k][param_nm] = param_model
                                    yml_path = os.path.join(bas_dir,param_nm+'.yml')
                                    txt_path = os.path.join(bas_dir,param_nm+'.txt')
                                    save_model_data(param_model,yml_path,txt_path)
                        else:
                            for specie_id, specie_models in v.items():
                                if not specie_id in model_dict[sys_cls][pop_id][k]:
                                    model_dict[sys_cls][pop_id][k][specie_id] = {}
                                specie_dir = os.path.join(bas_dir,specie_id)
                                if not os.path.exists(specie_dir): os.mkdir(specie_dir)
                                for param_nm, param_model in specie_models.items():
                                    if param_model:
                                        model_dict[sys_cls][pop_id][k][specie_id][param_nm] = param_model
                                        yml_path = os.path.join(specie_dir,param_nm+'.yml')
                                        txt_path = os.path.join(specie_dir,param_nm+'.txt')
                                        save_model_data(param_model,yml_path,txt_path)

def save_classification_models(models=classification_models, test=False):
    """Serialize `models` to .yml files, and also save them as module attributes.

    The models and scalers are saved to .yml,
    and a report of the cross-validation is saved to .txt.
    xrsdkit.models.classification_models is updated with all `models`.

    Parameters
    ----------
    models : dict
        embedded dict of models, similar to output of train_regression_models().
    test : bool (optional)
        if True, the models will be saved in the testing dir.
    """
    cl_root_dir = classification_models_dir
    model_dict = classification_models 
    if test: 
        cl_root_dir = test_classification_models_dir
        model_dict = test_classification_models
    if not os.path.exists(cl_root_dir): os.mkdir(cl_root_dir)
    for sys_cls, sys_mod in models.items():
        if sys_cls == 'system_classification':
            if sys_mod:
                model_dict[sys_cls] = sys_mod
                yml_path = os.path.join(cl_root_dir,'system_classification.yml')
                txt_path = os.path.join(cl_root_dir,'system_classification.txt')
                save_model_data(sys_mod,yml_path,txt_path)
        else:
            sys_dir_path = os.path.join(cl_root_dir,sys_cls)
            if not sys_cls in model_dict: model_dict[sys_cls] = {}
            if not os.path.exists(sys_dir_path): os.mkdir(sys_dir_path)
            for pop_id, pop_mod in sys_mod.items():
                if pop_id == 'noise':
                    if pop_mod:
                        model_dict[sys_cls]['noise'] = pop_mod
                        yml_path = os.path.join(sys_dir_path,'noise_classification.yml')
                        txt_path = os.path.join(sys_dir_path,'noise_classification.txt')
                        save_model_data(pop_mod,yml_path,txt_path)
                else:
                    if not pop_id in model_dict[sys_cls]: model_dict[sys_cls][pop_id] = {}
                    pop_dir_path = os.path.join(sys_dir_path,pop_id)
                    if not os.path.exists(pop_dir_path): os.mkdir(pop_dir_path)
                    for cls_label, m in pop_mod.items():
                        if m:
                            model_dict[sys_cls][pop_id][cls_label] = m
                            yml_path = os.path.join(sys_dir_path,pop_id,cls_label+'.yml')
                            txt_path = os.path.join(sys_dir_path,pop_id,cls_label+'.txt')
                            save_model_data(m,yml_path,txt_path)

# TODO: generate all unique sets of labels and the corresponding dataframe groups 
def group_by_labels(df):
    grp_cols = ['experiment_id','system_classification']
    for col in df.columns:
        if re.compile('pop._basis_classification').match(col): grp_cols.append(col)
        if re.compile('pop._lattice').match(col): grp_cols.append(col)
        if re.compile('pop._interaction').match(col): grp_cols.append(col)
    all_groups = df.groupby(grp_cols)
    return grp_cols, all_groups


def predict(features,test=False):
    """Estimate physical parameters, given a feature vector.

    Evaluates classifiers and regression models to
    estimate physical parameters of a sample
    that produced the input `features`,
    from xrsdit.tools.profiler.profile_pattern().

    Parameters
    ----------
    features : OrderedDict
        OrderedDict of features with their values,
        similar to output of xrsdkit.tools.profiler.profile_pattern()

    Returns
    -------
    results : dict
        dictionary with predicted classifications and parameters
    """

    classifiers=classification_models
    regressors=regression_models
    if test:
        classifiers=test_classification_models
        regressors=test_regression_models

    results = {}

    results['system_classification'] = classifiers['system_classification'].classify(features)
    sys_cl = results['system_classification'][0]

    if sys_cl == 'unidentified':
        return results

    cl_models_to_use = classifiers[sys_cl]
    reg_models_to_use = regressors[sys_cl]
    pop_structures = {}
    
    # extract the structures of each population from the system class
    for pop_struct in sys_cl.split('__'):
        pop_id = re.search('^pop.*?_',pop_struct).group(0)[:-1]
        struct_id = re.sub('^pop.*?_','',pop_struct)
        pop_structures[pop_id] = struct_id

    results['noise'] = {}
    if 'noise' in cl_models_to_use and cl_models_to_use['noise'].trained:
        results['noise']['noise_classification'] = cl_models_to_use['noise'].classify(features)
    else:
        # the model was created but not trained: the default value for the model was saved
        if 'noise' in cl_models_to_use:
             results['noise']['noise_classification'] = (cl_models_to_use['noise'].default_val, 1.0)
        else:
            # we do not have a model: save the default value
            results['noise']['noise_classification'] = ('flat', 1.0) # TODO add to definitions?
    pop_structures['noise'] = results['noise']['noise_classification'][0]

    # evaluate noise parameters
    for param_nm in xrsdefs.noise_params[pop_structures['noise']]+['I0_fraction']:
        if param_nm in reg_models_to_use['noise'][pop_structures['noise']] \
                and reg_models_to_use['noise'][pop_structures['noise']][param_nm].trained:
            results['noise'][param_nm] = \
                reg_models_to_use['noise'][pop_structures['noise']][param_nm].predict(features)
        else:
            # the model was created but not trained:
            # the default value for the model was saved
            if param_nm in reg_models_to_use['noise'][pop_structures['noise']]:
                results['noise'][param_nm] = \
                    float(reg_models_to_use['noise'][pop_structures['noise']][param_nm].default_val)
            elif not param_nm == 'I0' and not param_nm == 'I0_fraction':
                # TODO do we need to add a default val for I0_fraction?
                # we do not have a model: save the default value unless the param is I0 or I0_fraction
                results['noise'][param_nm] = xrsdefs.noise_param_defaults[param_nm]['value']

    for pop_id, pop_mod in cl_models_to_use.items():
        if not pop_id == 'noise':
            results[pop_id] = {}
            # evaluate parameters of this population
            for param_nm in xrsdefs.structure_params[pop_structures[pop_id]]+['I0_fraction']:
                if param_nm in reg_models_to_use[pop_id] and reg_models_to_use[pop_id][param_nm].trained:
                    results[pop_id][param_nm] = reg_models_to_use[pop_id][param_nm].predict(features)
                else:
                    # the model was created but not trained:
                    # the default value for the model was saved
                    if param_nm in reg_models_to_use[pop_id]:
                        results[pop_id][param_nm] = float(reg_models_to_use[pop_id][param_nm].default_val)
                    elif not param_nm == 'I0':
                        # we do not have a model: save the default value unless the param is I0
                        results[pop_id][param_nm] = xrsdefs.param_defaults[param_nm]['value']

            # classify the basis of this population
            bas_clsfr = pop_mod['basis_classification'] 
            if bas_clsfr.trained:
                results[pop_id]['basis_classification'] = bas_clsfr.classify(features)
            else:
                results[pop_id]['basis_classification'] = (bas_clsfr.default_val,1.)
            bas_cl = results[pop_id]['basis_classification'][0]

            # extract the form factors of each specie from the basis class
            specie_forms = {}
            for spec_ff in bas_cl.split('__'):
                specie_id = re.search('^specie.*?_',spec_ff).group(0)[:-1]
                ff_id = re.sub('^specie.*?_','',spec_ff)
                specie_forms[specie_id] = ff_id

            # TODO (later): if the structure is crystalline or disordered,
            # evaluate the lattice classifier or interaction classifer,
            # respectively

            for specie_id, specie_ff in specie_forms.items():
                results[pop_id][specie_id] = {}
                for param_nm in xrsdefs.form_factor_params[specie_forms[specie_id]]:
                    if param_nm in reg_models_to_use[pop_id][bas_cl][specie_id] \
                    and reg_models_to_use[pop_id][bas_cl][specie_id][param_nm].trained:
                        results[pop_id][specie_id][param_nm] = \
                            reg_models_to_use[pop_id][bas_cl][specie_id][param_nm].predict(features)
                    else:
                        results[pop_id][specie_id][param_nm] = xrsdefs.param_defaults[param_nm]['value']
                
            # TODO (later): if the specie is atomic, classify its atom symbol

    return results

def system_from_prediction(prediction,q,I,source_wavelength):
    """Create a System object from output of predict() function.

    Parameters
    ----------
    prediction : dict
         dictionary with predicted system class and parameters
    q : array
        array of scattering vector magnitudes 
    I : array
        array of integrated scattering intensities corresponding to `q`

    Returns
    -------
    predicted_system : xrsdkit.system.System
        a System object built from the prediction dictionary
    """
    new_sys = dict()

    # find all pops and their structure:
    sys_class = prediction['system_classification']

    if sys_class[0] == 'unidentified':
        return System(), new_sys

    # else, create the noise model and build the populations
    new_sys['noise'] = {'model':prediction['noise']['noise_classification'][0],'parameters':{}}
    for param_nm in xrsdefs.noise_params[prediction['noise']['noise_classification'][0]]:
        if param_nm in prediction['noise']:
            new_sys['noise']['parameters'][param_nm] = dict(value = prediction['noise'][param_nm])
    new_sys['noise']['parameters']['I0'] = {}
    if prediction['noise']['I0_fraction'] > 0:
        new_sys['noise']['parameters']['I0']['value'] = prediction['noise']['I0_fraction']
    else:
        new_sys['noise']['parameters']['I0']['value'] = xrsdefs.param_defaults['I0']['bounds'][0]

    for p in sys_class[0].split("__"):
        pop_id = re.search('^pop.*?_',p).group(0)[:-1]
        struct_id = re.sub('^pop.*?_','',p)
        new_sys[pop_id]=dict(structure=struct_id)

        # add parameters of the population:
        new_sys[pop_id]['parameters'] = {}
        for par in xrsdefs.structure_params[struct_id]:
            # prediction is not expected to include 'I0'
            if par in prediction[pop_id].keys():
                new_sys[pop_id]['parameters'][par] = {'value':prediction[pop_id][par]}
        # substitute I0_fraction for I0
        if prediction[pop_id]['I0_fraction'] > 0:
            new_sys[pop_id]['parameters']['I0'] = {'value':prediction[pop_id]['I0_fraction']}
        else:
            new_sys[pop_id]['parameters']['I0'] = {'value': xrsdefs.param_defaults['I0']['bounds'][0]}

        # TODO (later - as for predict()): if the structure is crystalline or disordered,
        # evaluate the lattice classifier or interaction classifer, respectively
        # TODO: fill in any reasonable guesses for the other Population settings
        # (e.g. use min and max q-values for crystalline q_min and q_max).

        # add basis:
        new_sys[pop_id]['basis'] = {}
        for b in prediction[pop_id]['basis_classification'][0].split("__"): # we can have more than one specie
            specie_id = re.search('^specie.*?_',b).group(0)[:-1]
            ff_id = re.sub('^specie.*?_','',b)
            new_sys[pop_id]['basis'][specie_id] = {'form':ff_id, 'parameters': {}}
            # find all parameters for this specie:
            par_dict = prediction[pop_id][specie_id]
            for k,v in par_dict.items():
                new_sys[pop_id]['basis'][specie_id]['parameters'][k] = {'value':v}

            # TODO (later - as for predict()): if the specie is atomic, classify its atom symbol

    predicted_system = System(new_sys)
    Isum = np.sum(I)
    I_comp = predicted_system.compute_intensity(q,source_wavelength)
    Isum_comp = np.sum(I_comp)
    I_factor = Isum/Isum_comp
    predicted_system.noise_model.parameters['I0']['value'] *= I_factor
    for pop_nm,pop in predicted_system.populations.items():
        pop.parameters['I0']['value'] *= I_factor

    return predicted_system, predicted_system.to_dict() 

# TODO refactor the modeling dataset index: it can no longer be divided simply by system class
#def save_modeling_datasets(df,grp_cols,all_groups,all_samples,test=True):
#    dir_path = modeling_data_dir
#    if test:
#        dir_path = os.path.join(dir_path,'models','modeling_data','testing_data')
#    file_path = os.path.join(dir_path,'dataset_statistics.txt')
#
#    with open(file_path, 'w') as txt_file:
#        txt_file.write('Downsampling statistics:\n\n')
#        for grpk,samp in zip(all_groups.groups.keys(),all_samples):
#            txt_file.write(grp_cols+'\n')
#            txt_file.write(grpk+'\n')
#            txt_file.write(len(samp)+' / '+len(all_groups.groups[grpk])+'\n')
#
#    modeling_dsid_file = os.path.join(modeling_data_dir,'modeling_dataset_ids.yml')
#    all_dsids = yaml.load(open(modeling_dsid_file,'rb'))
#
#    ds_map_filepath = os.path.join(modeling_data_dir,'dsid_map.yml')
#    ds_map = yaml.load(open(ds_map_filepath,'rb'))
#
#    # TODO: Take all_dsids one at a time,
#    # and associate each one with a group.
#    # (NOTE: If the group already exists in the ds_map,
#    # should we re-use that dsid?)
#    # If we run out of available modeling datasets,
#    # we will add more to the list by hand.
#    # ds_map should be an embedded dict,
#    # keyed by all_groups.groups.keys.
#
#    # For each dataset that gets assigned to a group,
#    # set its title to 'xrsdkit modeling dataset',
#    # set its description to list the group labels,
#    # create a new version,
#    # and upload the group.
#
#    # Then, upload the entire sample for system_classifier,
#    # and upload each system_class into a dataset 
#    # for that system's basis_classifiers.
#    #            pif.dump(pp, open(jsf,'w'))
#    #            client.data.upload(ds_id, jsf)
#    #    with open(ds_map_filepath, 'w') as yaml_file:
#    #        yaml.dump(dataset_ids, yaml_file)

