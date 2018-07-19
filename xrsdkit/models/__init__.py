import os
import yaml
from collections import OrderedDict

from .regressor import Regressor
from .. import regression_params
from ..tools.profiler import profile_spectrum

#find all existing regression models:
p = os.path.abspath(__file__)
d = os.path.dirname(p)
regression_dir = os.path.join(d,'modeling_data','regressors')

system_classes = ['noise','pop0_unidentified']
regression_models = OrderedDict()
regression_models['noise'] = {}
regression_models['pop0_unidentified'] = {}
for fn in os.listdir(regression_dir):
    if fn.endswith(".yml"):
        cl = fn.split(".")[0]
        system_classes.append(cl)
        regression_models[cl] = {}
        yml_path = os.path.join(regression_dir,fn)
        s_and_m_file = open(yml_file,'rb')
        content = yaml.load(s_and_m_file)
        labels = content.keys()
        for l in labels:
            regression_models[s][l] = Regressor(l, s)

# helper function - to set parameters for scalers and models
def set_param(m_s, param):
    for k, v in param.items():
        if isinstance(v, list):
            setattr(m_s, k, np.array(v))
        else:
            setattr(m_s, k, v)

def get_possible_regression_models(data):
    """Get dictionary of models that we can train using provided data.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels
        returned by citrination_tools.get_data_from_Citrination

    Returns
    -------
    model_labels : dict
        dictionary of possible regression models for each system_class
        (can be trained using provided data)
    """

    sys_cls = list(data.system_class.unique())
    if 'noise' in sys_cls:
        sys_cls.remove('noise')
    if 'pop0_unidentified' in sys_cls:
        sys_cls.remove('pop0_unidentified')
    model_labels = OrderedDict.fromkeys(sys_cls)
    for cls in sys_cls:
        cls_data = data[(data['system_class']==cls)]

        #to find the list of possible models and train all possible regression models
        #drop the collumns where all values are None:
        cls_data.dropna(axis=1,how='all',inplace=True)
        cols = cls_data.columns
        possible_models = []
        for c in cols:
            if any([rp == c[-1*len(rp):] for rp in regression_params]):
            #end = c.split("_")[-1]
            #if end in regression_params:
                if data[c].shape[0] > 10: #TODO change to 100 when we will have more data
                    possible_models.append(c)
        model_labels[cls] = possible_models
    return model_labels 

def train_regression_models(data, hyper_parameters_search=False,
                                 system_class = ['all'], testing_data = None, partial = False):
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
        for m in v:
            reg_model = Regressor(m, k)
            reg_model.train(pop_data, hyper_parameters_search)
            models[k][m] = reg_model 
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
        for k, m in models.items():
            try:
                print('accuracy : {}'.format(m.accuracy))
                print('parameters : {}'.format(m.parameters))
            except:
                print('failed to print training results for system {} class, model {}'.format(pop,k)

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
        for param_nm,m in models.items():
            file_path = os.path.join(d,'modeling_data','regressors',sys_cls+'.yml')
            cverr_txt_path = os.path.splitext(file_path)[0]+'.txt'
            if m.model is not None:
                s_and_m[reg_label] = dict(
                    scaler = m.scaler.__dict__, 
                    model = m.model.__dict__,
                    parameters = m.parameters, 
                    accuracy = m.accuracy, 
                    system_class = m.system_class
                    )
        if any(s_and_m):
            with open(file_path, 'w') as yaml_file:
                yaml.dump(s_and_m, yaml_file)

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

