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
classification_models = OrderedDict()

# here we only recreate the models from yml files
# the models for new labels will be created in train_regression_models()
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
            regression_models[sys_cls][l] = Regressor(l, yml_path)

# recreate the classification models
yml_file_cl = os.path.join(modeling_data_dir,'classifiers','system_class.yml')
classification_models['system_class'] = SystemClassifier(yml_file_cl)

def downsample_and_train(
    source_dataset_ids=src_dsid_list,
    citrination_client=cl,
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
        If true, downsampled datasets will be saved to their own datasets,
        according to xrsdkit/models/modeling_data/dataset_ids.yml
    save_models : bool
        If true, the models trained on the downsampled datasets
        will be saved to yml files in xrsdkit/models/modeling_data/
    train_hyperparameters : bool
        if True, the models will be optimized
        over a grid of hyperparameters during training
    test : bool
        if True, the downsmapling statistics and models will be
        saved in modleling_data/tesding_data dir;
        if Fasle, in modleling_data dir.
    """

    data = downsample_Citrination_datasets(citrination_client, source_dataset_ids,
                                           save_samples=save_samples, test=test)

    # system classifier:
    sys_cls = train_system_classifier(data, hyper_parameters_search=train_hyperparameters)
    print(classifier_results_to_str(sys_cls))

    # regression models:
    reg_models = train_regression_models(data, hyper_parameters_search=train_hyperparameters)

    if save_models:
        save_regression_models(reg_models, test=test)
        save_classification_model(sys_cls, test=test)

def train_system_classifier(data, hyper_parameters_search=False):
    """Retrain the system_classifier using a given DataFrame `data`. 

    This is a developer tool for building models
    from a set of Citrination datasets.
    It is used by the package developers to deploy
    a standard set of models with xrsdkit.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels
    hyper_parameters_search : bool
        if True, the models will be optimized
        over a grid of hyperparameters during training

    Returns
    -------
    cls : sklearn SGDClassifier
        fitted sklearn SGDClassifier.
    """
    # TODO: update this when we have multiple classification models:
    # currently it only handles the system_class label.
    models = OrderedDict.fromkeys(['system_class'])
    for label,model in models.items():
        if label in classification_models:
            model = classification_models[label]
        else:
            print('training classifier for {}:'.format(label))
            #if label == 'system_class':
            model = SystemClassifier()
        model.train(data, hyper_parameters_search=hyper_parameters_search)
        classification_models['system_class'] = model
    return models

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
        testing_data=None, partial=False):
    """Train regression models, optionally searching for optimal hyperparameters.

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
            # check if we already recreated this model from a yml file:
            if (k in regression_models) and (m in regression_models[k]):
                reg_model = regression_models[k][m]
            else: # create a new model
                reg_model = Regressor(m, None)

            reg_model.train(pop_data, hyper_parameters_search)
            if reg_model.trained:
                models[k][m] = reg_model 
                regression_models[k][m] = reg_model
                print('- finished training model for {}'.format(m))
            else:
                print('- training failed for {}'.format(m))
    return models


def save_regression_models(models=regression_models, test=False):
    """Save models, scalers, and cross validation results in YAML;
     save training report in .txt files.

    Parameters
    ----------
    models : dict
        the dict keys are system_class names, and the values are
        dictionaries of regression models for the system_class
    test : bool (optional)
        if True, the models will be saved in the testing dir.
    """
    p = os.path.abspath(__file__)
    d = os.path.dirname(p)
    for sys_cls, reg_mods in models.items():
        s_and_m = {}
        cross_valid_results = {}
        if test:
            file_path = os.path.join(d,'modeling_data','testing_data','regressors',sys_cls+'.yml')
        else:
            file_path = os.path.join(d,'modeling_data','regressors',sys_cls+'.yml')
        cverr_txt_path = os.path.splitext(file_path)[0]+'.txt'
        for param_nm,m in reg_mods.items():
            if m.model is not None:
                cross_valid_results[param_nm] = m.cross_valid_results
                s_and_m[param_nm] = dict(
                    scaler = m.scaler.__dict__, 
                    model = m.model.__dict__,
                    cross_valid_results = m.cross_valid_results)
        if any(s_and_m):
            with open(file_path, 'w') as yaml_file:
                yaml.dump(s_and_m, yaml_file)
            with open(cverr_txt_path, 'w') as txt_file:
                res_str = regressors_results_to_str(cross_valid_results)
                txt_file.write(res_str)


def save_classification_model(models=classification_models, test=False):
    """Save models, scalers, and cross validation results in YAML;
     save training report in .txt files.

    Parameters
    ----------
    models : dict
        with scaler, model, parameters, and accuracy.
    test : bool (optional)
        if True, the models will be saved in the testing dir.
    """
    p = os.path.abspath(__file__)
    d = os.path.dirname(p)
    s_and_m = {}

    for label,model in models.items():
        if test:
            file_path = os.path.join(d,'modeling_data','testing_data','classifiers',label+'.yml')
        else:
            file_path = os.path.join(d,'modeling_data','classifiers', label+'.yml')
        cverr_txt_path = os.path.splitext(file_path)[0]+'.txt'
        if model is not None:
            s_and_m = dict(label = dict(
                    scaler = model.scaler.__dict__,
                    model = model.model.__dict__,
                    cross_valid_results = model.cross_valid_results))
        if any(s_and_m):
            with open(file_path, 'w') as yaml_file:
                yaml.dump(s_and_m, yaml_file)
            res_str = classifier_results_to_str(model_dict)
            with open(cverr_txt_path, 'w') as txt_file:
                txt_file.write(res_str)

def classifier_results_to_str(model_dict):
    """Convert the dict with cross validation results to str.

    Parameters
    ----------
    model_dict : dict
        with scaler, model, parameters, and cross validation results.

    Returns
    -------
    results_str : str
        string with formated results of cross validatin.
    """

    results_str = "Cross validation results for System Classifier \n"
    results_str +='confusion_matrix : \n'
    results_str += str(model_dict.cross_valid_results["confusion matrix :"])+ '\n'
    results_str += '\n System classes : \n'
    for s in model_dict.cross_valid_results["model was tested for :"]:
        results_str += str(s)+ '\n'
    results_str += '\n F1 score by classes : \n'
    results_str += str(model_dict.cross_valid_results["F1 score by sys_classes"])+ '\n'
    results_str += '\n F1 averaged not weighted : \n'
    results_str += str(model_dict.cross_valid_results["F1 score averaged not weighted :"])+ '\n'
    results_str += '\n mean not weighted acc by system classes : \n'
    for k, v in model_dict.cross_valid_results["mean not weighted accuracies by system classes :"].items():
        if isinstance(v, float):
            results_str += str(k)+ ": " + str(v) + '\n'
    results_str +='\n mean accuracy : \n'
    results_str += str(model_dict.cross_valid_results["mean not weighted accuracy :"])+ '\n'
    results_str += 'The accuracy was calculated for each system class for each split (one experiment out) \n ' \
                       'as percent of right predicted labels. Then accuracy was averaged for each system class, \n ' \
                       'and then averaged for all system class'

    return results_str

def regressors_results_to_str(cross_valid_results):
    """Convert the dict with cross validation results to str.

    Parameters
    ----------
    cross_valid_results : dict
        with cross validation relults.

    Returns
    -------
    results_str : str
        string with formated results of cross validatin.
    """

    results_str = "Cross validation results for Regressors \n"
    for a_k, a_v in cross_valid_results.items():
        results_str += (a_k + '\n')
        for k,v in a_v.items():
            results_str += (k + ' : ' + str(v) + '\n')
        results_str += '\n'
    return results_str

# helper function - to set parameters for scalers and models
def set_param(m_s, param):
    for k, v in param.items():
        if isinstance(v, list):
            setattr(m_s, k, np.array(v))
        else:
            setattr(m_s, k, v)


def predict(features):
    """Evaluate classifier and regression models to
    estimate parameters for the sample

    Parameters
    ----------
    features : OrderedDict
            OrderedDict of features with their values,
            similar to output of xrsdkit.tools.profiler.profile_spectrum()

    Returns
    -------
    results : dict
        dictionary with predicted system class and parameters
    """
    results = {}
    results['system_class'] = cls.classify(features)
    sys_cl = results['system_class'][0]

    for param_nm, regressor in regression_models[sys_cl].items():
        results[param_nm] = regressor.predict(features)

    return results


