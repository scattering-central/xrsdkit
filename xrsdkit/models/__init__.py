import os
from collections import OrderedDict

import yaml
import numpy as np
from citrination_client import CitrinationClient

from .regressor import Regressor
from .classifier import SystemClassifier
from .. import regression_params
from ..tools.profiler import profile_spectrum
from ..tools.citrination_tools import downsample_Citrination_datasets

file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(file_path))
root_dir = os.path.dirname(src_dir)
modeling_data_dir = os.path.join(src_dir,'models','modeling_data')
regression_models_dir = os.path.join(modeling_data_dir,'regressors')

api_key_file = os.path.join(root_dir, 'api_key.txt')
cl = None
if os.path.exists(api_key_file):
    a_key = open(api_key_file, 'r').readline().strip()
    cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)

src_dsid_file = os.path.join(src_dir,'models','modeling_data','source_dataset_ids.yml')
src_dsid_list = yaml.load(open(src_dsid_file,'r'))

system_classes = ['noise','pop0_unidentified']
regression_models = OrderedDict()
regression_models['noise'] = {}
regression_models['pop0_unidentified'] = {}
for fn in os.listdir(regression_models_dir):
    if fn.endswith(".yml"):
        sys_cls = fn.split(".")[0]
        system_classes.append(sys_cls)
        regression_models[sys_cls] = {}
        yml_path = os.path.join(regression_models_dir,fn)
        s_and_m_file = open(yml_path,'rb')
        content = yaml.load(s_and_m_file)
        labels = content.keys()
        for l in labels:
            regression_models[sys_cls][l] = Regressor(l, sys_cls)

def downsample_and_train(
    source_dataset_ids=src_dsid_list,
    citrination_client=cl,
    save_samples=False,
    save_models=False,
    train_hyperparameters=False):
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
        If true, downsampled datasets will be saved to their own datasets,
        according to xrsdkit/models/modeling_data/dataset_ids.yml
    save_models : bool
        If true, the models trained on the downsampled datasets
        will be saved to yml files in xrsdkit/models/modeling_data/
    train_hyperparameters : bool
        if True, the models will be optimized
        over a grid of hyperparameters during training
    """

    data = downsample_Citrination_datasets(citrination_client, source_dataset_ids, save_samples=save_samples)

    # system classifier:
    sys_cls = train_system_classifier(data, hyper_parameters_search=train_hyperparameters)

    # regression models:
    reg_models = train_regression_models(data, hyper_parameters_search=train_hyperparameters)
    # TODO: save cross-validation details as 
    # parameters of the XRSDModel objects during training,
    # and add a function for printing out a description of them
    if save_models:
        save_regression_models(reg_models, modeling_data_dir)
        # TODO: similar function for saving system classifier
        #save_system_classifier(sys_cls, modeling_data_dir)

def train_system_classifier(data, train_hyperparameters=False):
    cls = SystemClassifier()
    classifier.train(data, hyper_parameters_search=train_hyperparameters)
    return cls

def get_possible_regression_models(data):
    """Get dictionary of models that we can train using provided data.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels

    Returns
    -------
    model_labels : dict
        dictionary of possible regression models for each system_class
        (a possible model can be sufficiently trained using provided data)
    """

    sys_cls = list(data.system_class.unique())
    if 'noise' in sys_cls:
        sys_cls.remove('noise')
    if 'pop0_unidentified' in sys_cls:
        sys_cls.remove('pop0_unidentified')
    model_labels = OrderedDict.fromkeys(sys_cls)
    for cls in sys_cls:
        cls_data = data[(data['system_class']==cls)]

        print('determining regression models for system class {}'.format(cls))
        #drop the collumns where all values are None:
        cls_data.dropna(axis=1,how='all',inplace=True)
        cols = cls_data.columns
        possible_models = []
        for c in cols:
            if any([rp == c[-1*len(rp):] for rp in regression_params]):
                possible_models.append(c)
        model_labels[cls] = possible_models
    return model_labels 

def train_regression_models(data, hyper_parameters_search=False,
        system_class=['all'], testing_data=None, partial=False):
    """Train regression models, optionally searching for optimal hyperparameters.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels
    hyper_parameters_search : bool
        If true, grid-search model hyperparameters
        to seek high cross-validation R^2 score.
    system_class : array of str
        the names of system_class for which we want to train
        regression models.
    testing_data : pandas.DataFrame (optional)
        dataframe containing original training data plus new data
        for computing the cross validation errors of the updated models.
    partial : bool
        If true, the models will be updataed using new data
        (train_partial() instead of train() from sklearn is used).

    Returns
    -------
    models : dict
        the dict keys are system_class names, and the values are
        dictionaries of regression models for the system_class
    """
    possible_models = get_possible_regression_models(data)

    models = OrderedDict.fromkeys(possible_models.keys())
    for k in models.keys(): models[k] = {}

    for k, v in possible_models.items(): # v is the list of possible models for given system_class
        pop_data = data[(data['system_class']==k)]
        print('attempting to train regression models for {}'.format(k))
        for m in v:
            reg_model = Regressor(m, k)
            reg_model.train(pop_data, hyper_parameters_search)
            if reg_model.trained:
                models[k][m] = reg_model 
                print('- finished training model for {}'.format(m))
            else:
                print('- training failed for {}'.format(m))
    return models

def print_training_results(results):
    """Print parameters of models and cross validation accuracies.

    Parameters
    ----------
    results : dict
        the dict keys are system_class names, and the values are
        dictionaries of regression models for the system_class
    """
    for pop, models in results.items():
        print(pop)
        for k, m in models.items():
            print(k)
            try:
                print('accuracy : {}'.format(m.accuracy))
                print('parameters : {}'.format(m.parameters))
            except:
                print('failed to print training results for system {} class, model {}'.format(pop,k))

def save_regression_models(models, file_path=None):
    """Save model parameters and CV errors in YAML and .txt files.

    Parameters
    ----------
    models : dict
        the dict keys are system_class names, and the values are
        dictionaries of regression models for the system_class
    file_path : str (optional)
        Full path to the YAML file where the models will be saved.
        Scalers, models, parameters, and cross-validation errors
        will be saved at this path. 
    """
    p = os.path.abspath(__file__)
    d = os.path.dirname(p)
    for sys_cls, reg_mods in models.items():
        s_and_m = {}
        acc = {}
        file_path = os.path.join(d,'modeling_data','regressors',sys_cls+'.yml')
        cverr_txt_path = os.path.splitext(file_path)[0]+'.txt'
        for param_nm,m in reg_mods.items():
            if m.model is not None:
                acc[param_nm] = m.accuracy
                s_and_m[param_nm] = dict(
                    scaler = m.scaler.__dict__, 
                    model = m.model.__dict__,
                    parameters = m.parameters, 
                    accuracy = m.accuracy, 
                    system_class = m.system_class
                    )
        if any(s_and_m):
            with open(file_path, 'w') as yaml_file:
                yaml.dump(s_and_m, yaml_file)
            with open(cverr_txt_path, 'w') as txt_file:
                txt_file.write(str(acc))

def evaluate_params(q_I, system_class):
    """Evaluate regression models to estimate parameters for the sample

    Parameters
    ----------
    q_I : array
        n-by-2 array of scattering vector 
        (1/Angstrom) and intensities
    system_class : str
        label (string) for the system class,
        used for selecting regression models 

    Returns
    -------
    params_dict : dict
        dictionary with predicted parameters
    """
    f = profile_spectrum(q_I)
    #pop = system_class[0]
    params_dict = {}
    for param_nm,m in regression_models[system_class].items():
        params_dict[param_nm] = m.predict(f, q_I)
    return params_dict

# helper function - to set parameters for scalers and models
def set_param(m_s, param):
    for k, v in param.items():
        if isinstance(v, list):
            setattr(m_s, k, np.array(v))
        else:
            setattr(m_s, k, v)


