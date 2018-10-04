import os
import re
from collections import OrderedDict

import yaml
import pandas as pd
from citrination_client import CitrinationClient

from .. import * 
from .regressor import Regressor
from .classifier import Classifier
from ..tools import primitives
from ..tools.citrination_tools import downsample, get_data_from_Citrination

file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(file_path))
root_dir = os.path.dirname(src_dir)
modeling_data_dir = os.path.join(src_dir,'models','modeling_data')
testing_data_dir = os.path.join(src_dir,'models','modeling_data','test')
regression_models_dir = os.path.join(modeling_data_dir,'regressors')
classification_models_dir = os.path.join(modeling_data_dir,'classifiers')
test_regression_models_dir = os.path.join(testing_data_dir,'regressors')
test_classification_models_dir = os.path.join(testing_data_dir,'classifiers')

api_key_file = os.path.join(root_dir, 'api_key.txt')
citcl = None
if os.path.exists(api_key_file):
    a_key = open(api_key_file, 'r').readline().strip()
    citcl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)

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
regression_models = OrderedDict()
classification_models = OrderedDict()

# --- LOAD REGRESSION MODELS --- #
if not os.path.exists(regression_models_dir): os.mkdir(regression_models_dir)
for sys_cls in os.listdir(regression_models_dir):
    regression_models[sys_cls] = {}
    sys_cls_dir = os.path.join(regression_models_dir,sys_cls)
    # the directory for the system class has subdirectories
    # for each population in the system
    for pop_id in os.listdir(sys_cls_dir):
        regression_models[sys_cls][pop_id] = {}
        pop_dir = os.path.join(sys_cls_dir,pop_id)
        # the directory for the population has model parameter files,
        # subdirectories for parameters of each modellable structure,
        # and subdirectories for each modellable basis class
        for pop_itm in os.listdir(pop_dir):
            if pop_itm in crystalline_structures+disordered_structures:
                # subdirectory for lattice or disordered structure params
                regression_models[sys_cls][pop_id][pop_itm] = {}
                pop_subdir = os.path.join(pop_dir,pop_itm)
                for pop_subitm in os.listdir(pop_subdir):
                    if pop_subitm.endswith('.yml'):
                        param_nm = pop_subitm.split('.')[0]
                        yml_path = os.path.join(pop_subdir,pop_subitm)
                        regression_models[sys_cls][pop_id][pop_itm][param_nm] = Regressor(param_nm,yml_path) 
            elif pop_itm.endswith('.yml'):
                # model parameters
                param_nm = pop_itm.split('.')[0]
                yml_path = os.path.join(pop_dir,pop_itm)
                regression_models[sys_cls][pop_id][param_nm] = Regressor(param_nm,yml_path) 
            elif not pop_itm.endswith('.txt'):
                # subdirectory for a basis class
                bas_cls = pop_itm.split('.')[0]
                regression_models[sys_cls][pop_id][bas_cls] = {} 
                bas_cls_dir = os.path.join(pop_dir,bas_cls)
                # the directory for the basis class has subdirectories
                # for each specie in the basis
                for specie_id in os.listdir(bas_cls_dir):
                    regression_models[sys_cls][pop_id][bas_cls][specie_id] = {}
                    specie_dir = os.path.join(bas_cls_dir,specie_id)
                    # the directory for the specie should contain only model parameter files 
                    for specie_itm in os.listdir(specie_dir):
                        if specie_itm.endswith('.yml'):
                            param_nm = specie_itm.split('.')[0]
                            yml_path = os.path.join(specie_dir,specie_itm)
                            regression_models[sys_cls][pop_id][bas_cls][specie_id][param_nm] = \
                            Regressor(param_nm,yml_path)


# --- LOAD CLASSIFICATION MODELS --- #
if not os.path.exists(classification_models_dir): os.mkdir(classification_models_dir)
yml_path = os.path.join(classification_models_dir,'system_classification.yml')

if os.path.exists(yml_path):
    classification_models['system_classification'] = Classifier('system_classification',yml_path)
    
for sys_cls in os.listdir(classification_models_dir):
    if not sys_cls.endswith('.yml') and not sys_cls.endswith('.txt'):
        classification_models[sys_cls] = {}
        sys_cls_dir = os.path.join(classification_models_dir,sys_cls)
        # the directory for the system class has subdirectories
        # for each population in the system
        for pop_id in os.listdir(sys_cls_dir):
            classification_models[sys_cls][pop_id] = {}
            pop_dir = os.path.join(sys_cls_dir,pop_id)
            # the directory for the population class contains
            # yml files with model parameters
            # for the population's basis classifier
            # and lattice or interaction classifiers if applicable
            for pop_itm in os.listdir(pop_dir):
                if pop_itm.endswith('.yml'):
                    yml_path = os.path.join(pop_dir,pop_itm)
                    model_type = os.path.splitext(pop_itm)[0]
                    model_label = pop_id+'_'+model_type
                    classification_models[sys_cls][pop_id][model_type] = Classifier(model_label,yml_path)

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
    df, pifs_list = get_data_from_Citrination(citrination_client,source_dataset_ids)
    df_sample, all_lbls, all_grps, all_samples = downsample_by_group(df)
    train_from_dataframe(df_sample,train_hyperparameters,save_models,test)

def downsample_by_group(df):
    """Group and down-sample a DataFrame of xrsd records.
        
    Parameters
    ----------
    df : pandas.DataFrame
        dataframe containing xrsd samples 

    Returns
    -------
    unscaled_data : pandas.DataFrame
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

    all_samples = []
    #for exp_id in all_exp:
    # downsample each group independently
    for group_labels,grp in all_groups.groups.items():
        #lbl_df = _filter_by_labels(data,lbls)
        dsamp = downsample(df.iloc[grp].copy(), 1.0)
        data_sample = data_sample.append(dsamp)
        all_samples.append(dsamp)
    return data_sample, group_cols, all_groups, all_samples

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
        sys_cls_data = data[(data['system_classification']==sys_cls)]
        for pop_id,pop_models in sys_models.items():
            for k in pop_models.keys():
                if k in regression_params:
                    # train reg_models[sys_cls][pop_id][k]
                    target = pop_id+'_'+k
                    reg_model = Regressor(target, None)
                    reg_model.train(sys_cls_data, hyper_parameters_search)
                    if reg_model.trained:
                        pop_models[k] = reg_model 
                elif k in crystalline_structures:
                    for param_nm in crystalline_structure_params[k]:
                        # train reg_models[sys_cls][pop_id][k][param_nm]
                        lattice_label = pop_id+'_lattice'
                        sub_cls_data = sys_cls_data[(sys_cls_data[lattice_label]==k)]
                        target = pop_id+'_'+param_nm
                        reg_model = Regressor(target, None)
                        reg_model.train(sub_cls_data, hyper_parameters_search)
                        if reg_model.trained:
                            pop_models[k][param_nm] = reg_model 
                elif k in disordered_structures:
                    for param_nm in disordered_structure_params[k]:
                        # train reg_models[sys_cls][pop_id][k][param_nm]
                        interxn_label = pop_id+'_interaction'
                        sub_cls_data = sys_cls_data[(sys_cls_data[interxn_label]==k)]
                        target = pop_id+'_'+param_nm
                        reg_model = Regressor(target, None)
                        reg_model.train(sub_cls_data, hyper_parameters_search)
                        if reg_model.trained:
                            pop_models[k][param_nm] = reg_model 
                else:
                    # k is a basis classification
                    bas_cls = k
                    bas_models = pop_models[k] 
                    bas_cls_label = pop_id+'_basis_classification'
                    bas_cls_data = sys_cls_data[(sys_cls_data[bas_cls_label]==bas_cls)]
                    for specie_id,specie_models in bas_models.items():
                        for param_nm in specie_models.keys():
                            target = pop_id+'_'+specie_id+'_'+param_nm
                            reg_model = Regressor(target, None)
                            reg_model.train(bas_cls_data, hyper_parameters_search)
                            if reg_model.trained:
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
        # use the sys_cls to identify the populations and their structures
        pop_struct_specifiers = sys_cls.split('__')
        for pop_struct in pop_struct_specifiers:
            pop_id = pop_struct[:re.compile('pop._').match(pop_struct).end()-1]
            structure_id = pop_struct[re.compile('pop._').match(pop_struct).end():]
            reg_models[sys_cls][pop_id] = {}
            for param_nm in structure_params[structure_id]:
                reg_label = pop_id+'_'+param_nm
                if reg_label in sys_cls_data.columns and param_nm in regression_params:
                    reg_models[sys_cls][pop_id][param_nm] = None
            # if the structure is crystalline or disordered, add sub-dicts
            # for the parameters of the relevant lattices or interactions
            if structure_id == 'crystalline':
                lattice_header = pop_id+'_lattice'
                lattice_labels = sys_cls_data[lattice_header].unique()
                for ll in lattice_labels:
                    reg_models[sys_cls][pop_id][ll] = {} 
                    for param_nm in crystalline_structure_params[ll]:
                        reg_label = pop_id+'_'+param_nm
                        if reg_label in sys_cls_data.columns and param_nm in regression_params: 
                            reg_models[sys_cls][pop_id][ll][param_nm] = None
            elif structure_id == 'disordered':
                interxn_header = pop_id+'_interaction'
                interxn_labels = sys_cls_data[interxn_header].unique() 
                for il in interxn_labels:
                    reg_models[sys_cls][pop_id][il] = {} 
                    for param_nm in disordered_structure_params[il]:
                        reg_label = pop_id+'_'+param_nm
                        if reg_label in sys_cls_data.columns and param_nm in regression_params:
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
                    for param_nm in form_factor_params[form_id]:
                        reg_label = pop_id+'_'+specie_id+'_'+param_nm
                        if reg_label in bas_cls_data.columns and param_nm in regression_params:
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
    for sys_cls_lbl, sys_models in cls_models.items():
        if sys_cls_lbl == 'system_classification':
            model = Classifier('system_classification',None)
            model.train(data, hyper_parameters_search=hyper_parameters_search)
            cls_models['system_classification'] = model
        else:
            for pop_id,pop_models in sys_models.items():
                for cls_label in pop_models.keys():
                    model_label = pop_id+'_'+cls_label
                    data_subset = data[data['system_classification']==sys_cls_lbl]
                    m = Classifier(model_label,None)
                    m.train(data_subset, hyper_parameters_search=hyper_parameters_search)
                    pop_models[cls_label] = m 
    return cls_models

def trainable_classification_models(data):
    sys_cls_labels = list(data['system_classification'].unique())
    if 'unidentified' in sys_cls_labels: sys_cls_labels.pop(sys_cls_labels.index('unidentified'))
    cls_models = dict.fromkeys(sys_cls_labels)
    if len(sys_cls_labels) > 0: cls_models['system_classification'] = None
    for sys_cls in sys_cls_labels:
        cls_models[sys_cls] = {}
        # get the slice of `data` that is relevant for this sys_cls
        sys_cls_data = data.loc[data['system_classification']==sys_cls].copy()
        # drop the columns where all values are None:
        sys_cls_data.dropna(axis=1,how='all',inplace=True)
        # use the sys_cls to identify the populations and their structures
        pop_struct_specifiers = sys_cls.split('__')
        for pop_struct in pop_struct_specifiers:
            pop_id = pop_struct[:re.compile('pop._').match(pop_struct).end()-1]
            structure_id = pop_struct[re.compile('pop._').match(pop_struct).end():]
            cls_models[sys_cls][pop_id] = {}
            bas_cls_header = pop_id+'_basis_classification'
            bas_cls_labels = list(sys_cls_data[bas_cls_header].unique())
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
            hyper_par = ['loss', 'epsilon',  'penalty', 'alpha', 'l1_ratio']
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
        yaml.dump(model_data,yml_file)
    with open(txt_path,'w') as txt_file:
        res_str = model.print_CV_report()
        txt_file.write(res_str)

def save_regression_models(models=regression_models, test=False):
    """Serialize `models` to .yml files, and update local regression_models.

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
    if test: rg_root_dir = test_regression_models_dir 
    if not os.path.exists(rg_root_dir): os.mkdir(rg_root_dir)
    for sys_cls, sys_models in models.items():
        sys_dir_path = os.path.join(rg_root_dir,sys_cls)
        if not sys_cls in regression_models: regression_models[sys_cls] = {}
        if not os.path.exists(sys_dir_path): os.mkdir(sys_dir_path)
        for pop_id, pop_models in sys_models.items():
            pop_dir_path = os.path.join(sys_dir_path,pop_id)
            if not pop_id in regression_models[sys_cls]: regression_models[sys_cls][pop_id] = {}
            if not os.path.exists(pop_dir_path): os.mkdir(pop_dir_path)
            for k,v in pop_models.items():
                if k in regression_params:
                    if v:
                        regression_models[sys_cls][pop_id][k] = v
                        yml_path = os.path.join(pop_dir_path,k+'.yml')
                        txt_path = os.path.join(pop_dir_path,k+'.txt')
                        save_model_data(v,yml_path,txt_path)
                else:
                    if not k in regression_models[sys_cls][pop_id]: regression_models[sys_cls][pop_id][k] = {}
                    pop_subdir_path = os.path.join(pop_dir_path,k)
                    if not os.path.exists(pop_subdir_path): os.mkdir(pop_subdir_path)
                    if k in crystalline_structures+disordered_structures:
                        if k in crystalline_structures: param_nms = crystalline_structure_params[k]
                        if k in disordered_structures: param_nms = disordered_structure_params[k]
                        for param_nm in param_nms:
                            if v[param_nm]:
                                regression_models[sys_cls][pop_id][k][param_nm] = v[param_nm]
                                yml_path = os.path.join(pop_subdir_path,param_nm+'.yml')
                                txt_path = os.path.join(pop_subdir_path,param_nm+'.txt')
                                save_model_data(v[param_nm],yml_path,txt_path)
                    else:
                        # k is a basis class label,
                        # v is a dict of dicts of models for each specie
                        bas_dir = os.path.join(pop_dir_path,k)
                        if not os.path.exists(bas_dir): os.mkdir(bas_dir)
                        for specie_id, specie_models in v.items():
                            if not specie_id in regression_models[sys_cls][pop_id][k]:
                                regression_models[sys_cls][pop_id][k][specie_id] = {}
                            specie_dir = os.path.join(bas_dir,specie_id)
                            if not os.path.exists(specie_dir): os.mkdir(specie_dir)
                            for param_nm, param_model in specie_models.items():
                                if param_model:
                                    regression_models[sys_cls][pop_id][k][specie_id][param_nm] = param_model
                                    yml_path = os.path.join(specie_dir,param_nm+'.yml')
                                    txt_path = os.path.join(specie_dir,param_nm+'.txt')
                                    save_model_data(param_model,yml_path,txt_path)

def save_classification_models(models=classification_models, test=False):
    """Serialize `models` to .yml files, and update local classification_models.

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
    if test: cl_root_dir = test_classification_models_dir
    if not os.path.exists(cl_root_dir): os.mkdir(cl_root_dir)
    for sys_cls, sys_mod in models.items():
        if sys_cls == 'system_classification':
            if sys_mod:
                classification_models[sys_cls] = sys_mod
                yml_path = os.path.join(cl_root_dir,'system_classification.yml')
                txt_path = os.path.join(cl_root_dir,'system_classification.txt')
                save_model_data(sys_mod,yml_path,txt_path)
        else:
            sys_dir_path = os.path.join(cl_root_dir,sys_cls)
            if not sys_cls in classification_models: classification_models[sys_cls] = {}
            if not os.path.exists(sys_dir_path): os.mkdir(sys_dir_path)
            for pop_id, pop_models in sys_mod.items():
                if not pop_id in classification_models[sys_cls]: classification_models[sys_cls][pop_id] = {}
                pop_dir_path = os.path.join(sys_dir_path,pop_id)
                if not os.path.exists(pop_dir_path): os.mkdir(pop_dir_path)
                for cls_label, m in pop_models.items():
                    if m:
                        classification_models[sys_cls][pop_id][cls_label] = m
                        yml_path = os.path.join(sys_dir_path,pop_id,cls_label+'.yml')
                        txt_path = os.path.join(sys_dir_path,pop_id,cls_label+'.txt')
                        save_model_data(m,yml_path,txt_path)

# TODO refactor the modeling dataset index: it can no longer be divided simply by system class
def save_modeling_datasets(df,grp_cols,all_groups,all_samples,test=True):
    dir_path = modeling_data_dir
    if test:
        dir_path = os.path.join(dir_path,'models','modeling_data','testing_data')
    file_path = os.path.join(dir_path,'dataset_statistics.txt')

    with open(file_path, 'w') as txt_file:
        txt_file.write('Downsampling statistics:\n\n')
        for grpk,samp in zip(all_groups.groups.keys(),all_samples):
            txt_file.write(grp_cols+'\n')
            txt_file.write(grpk+'\n')
            txt_file.write(len(samp)+' / '+len(all_groups.groups[grpk])+'\n')

    modeling_dsid_file = os.path.join(modeling_data_dir,'modeling_dataset_ids.yml')
    all_dsids = yaml.load(open(modeling_dsid_file,'rb'))

    ds_map_filepath = os.path.join(modeling_data_dir,'dsid_map.yml')
    ds_map = yaml.load(open(ds_map_filepath,'rb'))

    # TODO: Take all_dsids one at a time,
    # and associate each one with a group.
    # (NOTE: If the group already exists in the ds_map,
    # should we re-use that dsid?)
    # If we run out of available modeling datasets,
    # we will add more to the list by hand.
    # ds_map should be an embedded dict,
    # keyed by all_groups.groups.keys.

    # For each dataset that gets assigned to a group,
    # set its title to 'xrsdkit modeling dataset',
    # set its description to list the group labels,
    # create a new version,
    # and upload the group.

    # Then, upload the entire sample for system_classifier,
    # and upload each system_class into a dataset 
    # for that system's basis_classifiers.
    #            pif.dump(pp, open(jsf,'w'))
    #            client.data.upload(ds_id, jsf)
    #    with open(ds_map_filepath, 'w') as yaml_file:
    #        yaml.dump(dataset_ids, yaml_file)


# TODO: generate all unique sets of labels and the corresponding dataframe groups 
def group_by_labels(df):
    grp_cols = ['experiment_id','system_classification']
    for col in df.columns:
        if re.compile('pop._basis_classification').match(col): grp_cols.append(col)
        if re.compile('pop._lattice').match(col): grp_cols.append(col)
        if re.compile('pop._interaction').match(col): grp_cols.append(col)
    all_groups = df.groupby(grp_cols)
    return grp_cols, all_groups

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

    results['system_classification'] = classification_models['system_classification'].classify(features)
    sys_cl = results['system_classification'][0]

    if sys_cl == 'unidentified':
        return results

    cl_models_to_use = classification_models[sys_cl]
    reg_models_to_use = regression_models[sys_cl]
    pop_structures = {}
    
    # extract the structures of each population from the system class
    for pop_struct in sys_cl.split('__'):
        pop_structures[pop_struct[:4]] = pop_struct[5:]

    for pop_id, pop_classifiers in cl_models_to_use.items():
        results[pop_id] = {'structure':pop_structures[pop_id]}

        # evaluate parameters of this population
        for param_nm in structure_params[pop_structures[pop_id]]:
            if param_nm in reg_models_to_use[pop_id] \
                    and reg_models_to_use[pop_id][param_nm].trained:
                results[pop_id][param_nm] = \
                    reg_models_to_use[pop_id][param_nm].predict(features)
            else:
                # the model was created but did not trained;
                # the default value for the model was saved
                if param_nm in reg_models_to_use[pop_id]:
                    results[pop_id][param_nm] = reg_models_to_use[pop_id].default_val
                else:
                    # we do not have a model
                    results[pop_id][param_nm] = param_defaults[param_nm]['value']

        # classify the basis of this population
        bas_clsfr = pop_classifiers['basis_classification'] 
        if bas_clsfr.trained:
            results[pop_id]['basis_classification'] = bas_clsfr.classify(features)

        else:
            results[pop_id]['basis_classification'] = (bas_clsfr.default_val,1.)
        bas_cl = results[pop_id]['basis_classification'][0]

        # extract the form factors of each specie from the basis class
        specie_forms = {}
        for spec_ff in bas_cl.split('__'):
            specie_forms[spec_ff[:7]] = spec_ff[8:]

        # TODO (later): if the structure is crystalline or disordered,
        # evaluate the lattice classifier or interaction classifer,
        # respectively

        for specie_id, specie_ff in specie_forms.items():
            results[pop_id][specie_id] = {}
            for param_nm in form_factor_params[specie_forms[specie_id]]:
                if param_nm in reg_models_to_use[pop_id][bas_cl][specie_id] \
                        and reg_models_to_use[pop_id][bas_cl][specie_id][param_nm].trained:
                    results[pop_id][specie_id][param_nm] = \
                        reg_models_to_use[pop_id][bas_cl][specie_id][param_nm].predict(features)
                else:
                    results[pop_id][specie_id][param_nm] = param_defaults[param_nm]['value']
                
            # TODO (later): if the specie is atomic, classify its atom symbol

    return results


