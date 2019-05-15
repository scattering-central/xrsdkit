from __future__ import print_function
import os
import itertools
from collections import OrderedDict

import yaml
import numpy as np

from . import get_regression_models, get_classification_models, get_reg_conf, get_cl_conf
from .. import definitions as xrsdefs
from ..tools import primitives
from .regressor import Regressor
from .classifier import Classifier

def train_from_dataframe(data, train_hyperparameters=False, select_features=False, 
                output_dir=None, model_config_path=None, old_summary_path=None, message_callback=print):
    """Train xrsdkit models from a pandas.DataFrame of labeled samples.

    This is the primary function for training xrsdkit models.
    All other training functions should collect a DataFrame,
    and then invoke this function on that DataFrame.
    """
    # if old_summary_path is provided, the new summary will attempt
    # to express the differences in performance from old to new
    old_summary = {}
    if old_summary_path: 
        with open(old_summary_path,'rb') as yml_file:
            old_summary = yaml.load(yml_file)
    # if the model_config_path is provided, the new models
    # will obey any relevant configurations specified in that file
    try:
        # see if user provided a model config file
        with open(model_config_path,'rb') as yml_file:
            model_configs = yaml.load(yml_file)
            model_configs_cl = model_configs['CLASSIFIERS']
            model_configs_reg = model_configs['REGRESSORS']
    except:
        # load the current model configs, if any
        model_configs_cl = get_cl_conf()
        model_configs_reg = get_reg_conf()
    cls_models, new_summary_cl, new_config_cl = train_classification_models(data, train_hyperparameters, select_features, model_configs_cl, message_callback)
    reg_models, new_summary_reg, new_config_reg = train_regression_models(data, train_hyperparameters, select_features, model_configs_reg, message_callback)
    if output_dir:
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        new_summary = collect_summary(new_summary_reg, new_summary_cl, old_summary)
        yml_f = os.path.join(output_dir,'training_summary.yml')
        with open(yml_f,'w') as yml_file:
            yaml.dump(new_summary,yml_file)
        if model_config_path:
            new_config = collect_config(new_config_reg, new_config_cl)
            with open(model_config_path,'w') as yml_file:
                yaml.dump(new_config,yml_file)
        cl_dir = os.path.join(output_dir,'classifiers')
        if not os.path.exists(cl_dir): os.mkdir(cl_dir)
        reg_dir = os.path.join(output_dir,'regressors')
        if not os.path.exists(reg_dir): os.mkdir(reg_dir)
        message_callback('SAVING CLASSIFICATION MODELS TO {}'.format(cl_dir))
        save_classification_models(cl_dir, cls_models)
        message_callback('SAVING REGRESSION MODELS TO {}'.format(reg_dir))
        save_regression_models(reg_dir, reg_models)
    return reg_models, cls_models

def train_classification_models(data, train_hyperparameters=False, select_features=False, model_configs={}, message_callback=print):
    """Train all classifiers that are trainable from `data`.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels
    train_hyperparameters : bool
        if True, cross-validation metrics are used to select model hyperparameters 
    select_features : bool
        if True, recursive feature elimination is used to select model input space
    model_configs : dict
        dict of dicts containing model types and training metrics-
        generally this should be read in from a model config file

    Returns
    -------
    new_cls_models : dict
        Dict containing all classification models trained on the given dataset
    summary : dict
        Dict of model performance metrics, collected during training
    config : dict
        Dict of model types and training targets, collected during training
    """

    # get a reference to the currently-loaded classification models dict
    classification_models = get_classification_models()
    
    new_cls_models = {}
    new_cls_models['main_classifiers'] = {}
    summary = {}
    summary['main_classifiers'] = {}
    config = {}
    config['main_classifiers'] = {}

    # find all system_class labels represented in `data`:
    all_sys_cls = data['system_class'].tolist()

    data_copy = data.copy()
    for struct_nm in xrsdefs.structure_names:
        message_callback('Training binary classifier for '+struct_nm+' structures')
        model_id = struct_nm+'_binary'
        try:
            new_model_type = model_configs['main_classifiers'][model_id]['model_type']
            metric = model_configs['main_classifiers'][model_id]['metric']
        except:
            new_model_type = 'logistic_regressor'
            metric = 'precision' # use 'precision' as the scoring function to avoid false positives

        model = Classifier(new_model_type, metric, model_id)
        labels = [struct_nm in sys_cls for sys_cls in all_sys_cls]
        data_copy.loc[:,model_id] = labels
        if ('main_classifiers' in classification_models) \
        and (model_id in classification_models['main_classifiers']) \
        and (classification_models['main_classifiers'][model_id].trained) \
        and (classification_models['main_classifiers'][model_id].model_type == new_model_type):
            old_pars = classification_models['main_classifiers'][model_id].model.get_params()
            for param_nm in model.models_and_params[new_model_type]:
                model.model.set_params(**{param_nm:old_pars[param_nm]})
        model.train(data_copy, train_hyperparameters, select_features)
        if model.trained:
            f1_score = model.cross_valid_results['f1']
            acc = model.cross_valid_results['accuracy']
            prec = model.cross_valid_results['precision']
            rec = model.cross_valid_results['recall']
            message_callback('--> f1: {}, accuracy: {}, precision: {}, recall: {}'.format(f1_score,acc,prec,rec))
        else:
            message_callback('--> {} untrainable- default value: {}'.format(model_id,model.default_val))
        new_cls_models['main_classifiers'][model_id] = model
        summary['main_classifiers'][model_id] = primitives(model.get_cv_summary())
        config['main_classifiers'][model_id] = dict(model_type=model.model_type, metric=model.metric) 
    # There are 2**n possible outcomes for n binary classifiers.
    # For the (2**n)-1 non-null outcomes, a second classifier is used,
    # to count the number of populations of each structural type.

    all_flag_combs = itertools.product([True,False],repeat=len(xrsdefs.structure_names))
    for flags in all_flag_combs:
        if sum(flags) > 0:
            flag_idx = np.ones(data.shape[0],dtype=bool)
            model_id = ''
            for struct_nm, flag in zip(xrsdefs.structure_names,flags):
                struct_flag_idx = np.array([(struct_nm in sys_cls) == flag for sys_cls in all_sys_cls])
                flag_idx = flag_idx & struct_flag_idx
                if flag:
                    if model_id: model_id += '__'
                    model_id += struct_nm
            message_callback('Training system classifier for '+model_id)
            # get all samples whose system_class matches the flags
            flag_data = data.loc[flag_idx,:].copy()
            if flag_data.shape[0] > 0: # we have data with these structure flags in the training set
                try:
                    new_model_type = model_configs['main_classifiers'][model_id]['model_type']
                    metric = model_configs['main_classifiers'][model_id]['metric']
                except:
                    new_model_type = 'logistic_regressor'
                    metric = 'accuracy'
                model = Classifier(new_model_type, metric, 'system_class')
                if ('main_classifiers' in classification_models) \
                and (model_id in classification_models['main_classifiers']) \
                and (classification_models['main_classifiers'][model_id].trained) \
                and (classification_models['main_classifiers'][model_id].model_type == new_model_type):
                    old_pars = classification_models['main_classifiers'][model_id].model.get_params()
                    for param_nm in model.models_and_params[new_model_type]:
                        model.model.set_params(**{param_nm:old_pars[param_nm]})
                model.train(flag_data, train_hyperparameters, select_features)
                if model.trained:
                    f1_score = model.cross_valid_results['f1']
                    acc = model.cross_valid_results['accuracy']
                    prec = model.cross_valid_results['precision']
                    rec = model.cross_valid_results['recall']
                    message_callback('--> f1: {}, accuracy: {}, precision: {}, recall: {}'.format(f1_score,acc,prec,rec))
                else:
                    message_callback('--> {} untrainable- default value: {}'.format(model_id,model.default_val))
                # save the classifier
                new_cls_models['main_classifiers'][model_id] = model
                summary['main_classifiers'][model_id] = primitives(model.get_cv_summary())
                config['main_classifiers'][model_id] = dict(model_type=model.model_type, metric=model.metric) 

    sys_cls_labels = list(data['system_class'].unique())
    # 'unidentified' systems will have no sub-classifiers; drop this label up front 
    if 'unidentified' in sys_cls_labels: sys_cls_labels.remove('unidentified')

    for sys_cls in sys_cls_labels:
        message_callback('Training classifiers for system class {}'.format(sys_cls))
        new_cls_models[sys_cls] = {}
        summary[sys_cls] = {}
        config[sys_cls] = {}
        sys_cls_data = data.loc[data['system_class']==sys_cls].copy()

        # every system class must have a noise classifier
        message_callback('    Training noise classifier for system class {}'.format(sys_cls))
        try:
            new_model_type = model_configs[sys_cls]['noise_model']['model_type']
            metric = model_configs[sys_cls]['noise_model']['metric']
        except:
            new_model_type = 'logistic_regressor'
            metric = 'accuracy'
        model = Classifier(new_model_type, metric, 'noise_model')
        if (sys_cls in classification_models) \
        and ('noise_model' in classification_models[sys_cls]) \
        and (classification_models[sys_cls]['noise_model'].trained) \
        and (classification_models[sys_cls]['noise_model'].model_type == new_model_type):
            old_pars = classification_models[sys_cls]['noise_model'].model.get_params()
            for param_nm in model.models_and_params[new_model_type]:
                model.model.set_params(**{param_nm:old_pars[param_nm]})
        model.train(sys_cls_data, train_hyperparameters, select_features)
        if model.trained:
            f1_score = model.cross_valid_results['f1']
            acc = model.cross_valid_results['accuracy']
            prec = model.cross_valid_results['precision']
            rec = model.cross_valid_results['recall']
            message_callback('    --> f1: {}, accuracy: {}, precision: {}, recall: {}'.format(f1_score,acc,prec,rec))
        else: 
            message_callback('    --> {} untrainable- default value: {}'.format('noise_model',model.default_val))
        new_cls_models[sys_cls]['noise_model'] = model
        summary[sys_cls]['noise_model'] = primitives(model.get_cv_summary())
        config[sys_cls]['noise_model'] = dict(model_type=model.model_type, metric=model.metric) 

        # each population has some classifiers for form factor and settings
        for ipop, struct in enumerate(sys_cls.split('__')):
            pop_id = 'pop{}'.format(ipop)
            new_cls_models[sys_cls][pop_id] = {}
            summary[sys_cls][pop_id] = {}
            config[sys_cls][pop_id] = {}
            message_callback('    Training classifiers for population {}'.format(pop_id))

            # every population must have a form classifier
            form_header = pop_id+'_form'
            message_callback('    Training: {}'.format(form_header))
            try:
                new_model_type = model_configs[sys_cls][pop_id]['form']['model_type']
                metric = model_configs[sys_cls][pop_id]['form']['metric']
            except:
                new_model_type = 'logistic_regressor'
                metric = 'accuracy'
            model = Classifier(new_model_type, metric, form_header)
            if (sys_cls in classification_models) \
            and (pop_id in classification_models[sys_cls]) \
            and ('form' in classification_models[sys_cls][pop_id]) \
            and (classification_models[sys_cls][pop_id]['form'].trained) \
            and (classification_models[sys_cls][pop_id]['form'].model_type == new_model_type):
                old_pars = classification_models[sys_cls][pop_id]['form'].model.get_params()
                for param_nm in model.models_and_params[new_model_type]:
                    model.model.set_params(**{param_nm:old_pars[param_nm]})
            model.train(sys_cls_data, train_hyperparameters, select_features)
            if model.trained:
                f1_score = model.cross_valid_results['f1']
                acc = model.cross_valid_results['accuracy']
                prec = model.cross_valid_results['precision']
                rec = model.cross_valid_results['recall']
                message_callback('    --> f1: {}, accuracy: {}, precision: {}, recall: {}'.format(f1_score,acc,prec,rec))
            else: 
                message_callback('    --> {} untrainable- default value: {}'.format(form_header,model.default_val))
            new_cls_models[sys_cls][pop_id]['form'] = model
            summary[sys_cls][pop_id]['form'] = primitives(model.get_cv_summary())
            config[sys_cls][pop_id]['form'] = dict(model_type=model.model_type, metric=model.metric) 

            # add classifiers for any model-able structure settings 
            for stg_nm in xrsdefs.modelable_structure_settings[struct]:
                stg_header = pop_id+'_'+stg_nm
                message_callback('    Training: {}'.format(stg_header))
                try:
                    new_model_type = model_configs[sys_cls][pop_id][stg_nm]['model_type']
                    metric = model_configs[sys_cls][pop_id][stg_nm]['metric']
                except:
                    new_model_type = 'logistic_regressor'
                    metric = 'accuracy'
                model = Classifier(new_model_type, metric, stg_header)
                if (sys_cls in classification_models) \
                and (pop_id in classification_models[sys_cls]) \
                and (stg_nm in classification_models[sys_cls][pop_id]) \
                and (classification_models[sys_cls][pop_id][stg_nm].trained) \
                and (classification_models[sys_cls][pop_id][stg_nm].model_type == new_model_type):
                    old_pars = classification_models[sys_cls][pop_id][stg_nm].model.get_params()
                    for param_nm in model.models_and_params[new_model_type]:
                        model.model.set_params(**{param_nm:old_pars[param_nm]})
                model.train(sys_cls_data, train_hyperparameters, select_features)
                if model.trained:
                    f1_score = model.cross_valid_results['f1']
                    acc = model.cross_valid_results['accuracy']
                    prec = model.cross_valid_results['precision']
                    rec = model.cross_valid_results['recall']
                    message_callback('    --> f1: {}, accuracy: {}, precision: {}, recall: {}'.format(f1_score,acc,prec,rec))
                else: 
                    message_callback('    --> {} untrainable- default value: {}'.format(stg_header,model.default_val))
                new_cls_models[sys_cls][pop_id][stg_nm] = model
                summary[sys_cls][pop_id][stg_nm] = primitives(model.get_cv_summary())
                config[sys_cls][pop_id][stg_nm] = dict(model_type=model.model_type, metric=model.metric) 

            # add classifiers for any model-able form factor settings
            all_ff_labels = list(sys_cls_data[form_header].unique())
            for ff in all_ff_labels:
                form_data = sys_cls_data.loc[sys_cls_data[form_header]==ff].copy()
                new_cls_models[sys_cls][pop_id][ff] = {}
                summary[sys_cls][pop_id][ff] = {}
                config[sys_cls][pop_id][ff] = {} 
                message_callback('    Training classifiers for {} with {} form factors'.format(pop_id,ff))
                for stg_nm in xrsdefs.modelable_form_factor_settings[ff]:
                    stg_header = pop_id+'_'+stg_nm
                    message_callback('        Training: {}'.format(stg_header))
                    try:
                        new_model_type = model_configs[sys_cls][pop_id][ff][stg_nm]['model_type']
                        metric = model_configs[sys_cls][pop_id][ff][stg_nm]['metric']
                    except:
                        new_model_type = 'logistic_regressor'
                        metric = 'accuracy'
                    model = Classifier(new_model_type, metric, stg_header)
                    if (sys_cls in classification_models) \
                    and (pop_id in classification_models[sys_cls]) \
                    and (ff in classification_models[sys_cls][pop_id]) \
                    and (stg_nm in classification_models[sys_cls][pop_id][ff]) \
                    and (classification_models[sys_cls][pop_id][ff][stg_nm].trained) \
                    and (classification_models[sys_cls][pop_id][ff][stg_nm].model_type == new_model_type):
                        old_pars = classification_models[sys_cls][pop_id][ff][stg_nm].model.get_params()
                        for param_nm in model.models_and_params[new_model_type]:
                            model.model.set_params(**{param_nm:old_pars[param_nm]})
                    model.train(form_data, train_hyperparameters, select_features)
                    if model.trained:
                        f1_score = model.cross_valid_results['f1']
                        acc = model.cross_valid_results['accuracy']
                        prec = model.cross_valid_results['precision']
                        rec = model.cross_valid_results['recall']
                        message_callback('        --> f1: {}, accuracy: {}, precision: {}, recall: {}'.format(f1_score,acc,prec,rec))
                    else: 
                        message_callback('        --> {} untrainable- default value: {}'.format(stg_header,model.default_val))
                    new_cls_models[sys_cls][pop_id][ff][stg_nm] = model
                    summary[sys_cls][pop_id][ff][stg_nm] = primitives(model.get_cv_summary())
                    config[sys_cls][pop_id][ff][stg_nm] = dict(model_type=model.model_type, metric=model.metric) 
    return new_cls_models, summary, config


def save_classification_models(output_dir, models):
    """Serialize `models` to a tree of .yml files.

    The models and scalers are saved to .yml,
    and a report of the cross-validation is saved to .txt.

    Parameters
    ----------
    output_dir : str
        path to directory where models should be saved
    models : dict
        embedded dict of models, similar to output of train_regression_models().
    """
    cl_root_dir = output_dir
    # get a reference to the currently-loaded classification models dict
    model_dict = get_classification_models()
    if not os.path.exists(cl_root_dir): os.mkdir(cl_root_dir)

    if 'main_classifiers' in models:
        model_dict['main_classifiers'] = models['main_classifiers']
        if not os.path.exists(os.path.join(cl_root_dir,'main_classifiers')):
            os.mkdir(os.path.join(cl_root_dir,'main_classifiers'))
        for model_name, mod in model_dict['main_classifiers'].items():
            yml_path = os.path.join(cl_root_dir,'main_classifiers', model_name + '.yml')
            txt_path = os.path.join(cl_root_dir,'main_classifiers', model_name + '.txt')
            pickle_path = os.path.join(cl_root_dir,'main_classifiers', model_name + '.pickle')
            mod.save_model_data(yml_path,txt_path, pickle_path)

    all_sys_cls = list(models.keys())
    all_sys_cls.remove('main_classifiers')
    for sys_cls in all_sys_cls: 
        sys_cls_dir = os.path.join(cl_root_dir,sys_cls)
        if not sys_cls in model_dict: model_dict[sys_cls] = {}
        if not os.path.exists(sys_cls_dir): os.mkdir(sys_cls_dir)
        if 'noise_model' in models[sys_cls]:
            model_dict[sys_cls]['noise_model'] = models[sys_cls]['noise_model']
            yml_path = os.path.join(sys_cls_dir,'noise_model.yml')
            txt_path = os.path.join(sys_cls_dir,'noise_model.txt')
            pickle_path = os.path.join(sys_cls_dir, 'noise_model.pickle')
            models[sys_cls]['noise_model'].save_model_data(yml_path,txt_path, pickle_path)

        for ipop,struct in enumerate(sys_cls.split('__')):
            pop_id = 'pop{}'.format(ipop)
            pop_dir = os.path.join(sys_cls_dir,pop_id)
            if not os.path.exists(pop_dir): os.mkdir(pop_dir)
            if not pop_id in model_dict[sys_cls]: model_dict[sys_cls][pop_id] = {}

            if 'form' in models[sys_cls][pop_id]:
                model_dict[sys_cls][pop_id]['form'] = models[sys_cls][pop_id]['form']
                yml_path = os.path.join(pop_dir,'form.yml')
                txt_path = os.path.join(pop_dir,'form.txt')
                pickle_path = os.path.join(pop_dir, 'form.pickle')
                models[sys_cls][pop_id]['form'].save_model_data(yml_path,txt_path, pickle_path)
               
            for stg_nm in xrsdefs.modelable_structure_settings[struct]:
                if stg_nm in models[sys_cls][pop_id]:
                    model_dict[sys_cls][pop_id][stg_nm] = models[sys_cls][pop_id][stg_nm]
                    yml_path = os.path.join(pop_dir,stg_nm+'.yml')
                    txt_path = os.path.join(pop_dir,stg_nm+'.txt')
                    pickle_path = os.path.join(pop_dir,stg_nm+ '.pickle')
                    models[sys_cls][pop_id][stg_nm].save_model_data(yml_path,txt_path, pickle_path)

            for ff_id in xrsdefs.form_factor_names:
                if ff_id in models[sys_cls][pop_id]:
                    form_dir = os.path.join(pop_dir,ff_id)
                    if not os.path.exists(form_dir): os.mkdir(form_dir)
                    model_dict[sys_cls][pop_id][ff_id] = {}
                    for stg_nm in xrsdefs.modelable_form_factor_settings[ff_id]:
                        model_dict[sys_cls][pop_id][ff_id][stg_nm] = models[sys_cls][pop_id][ff_id][stg_nm]
                        yml_path = os.path.join(form_dir,stg_nm+'.yml')
                        txt_path = os.path.join(form_dir,stg_nm+'.txt')
                        pickle_path = os.path.join(form_dir,stg_nm+'.pickle')
                        models[sys_cls][pop_id][ff_id][stg_nm].save_model_data(yml_path,txt_path, pickle_path)


def train_regression_models(data, train_hyperparameters=False, select_features=False, model_configs={}, message_callback=print):
    """Train all regression models trainable from `data`. 

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels
    train_hyperparameters : bool
        if True, the models will be optimized
        over a grid of hyperparameters during training
    select_features : bool
        if True, recursive feature elimination is used to select model input space
    model_configs : dict
        dict containing model types and training target metrics-
        generally this should be read in from a model config file

    Returns
    -------
    new_reg_models : dict
        Dict containing all regression models trained on the given dataset
    summary : dict
        Dict of model performance metrics, collected during training
    config : dict
        Dict of model types and training targets, collected during training
    """

    # get a reference to the currently-loaded regression models dict
    regression_models = get_regression_models()
    new_reg_models = {}
    summary = {}
    config = {}
    sys_cls_labels = list(data['system_class'].unique())
    # 'unidentified' systems will have no regression models:
    if 'unidentified' in sys_cls_labels: sys_cls_labels.pop(sys_cls_labels.index('unidentified'))
    for sys_cls in sys_cls_labels:
        message_callback('training regressors for system class {}'.format(sys_cls))
        new_reg_models[sys_cls] = {}
        summary[sys_cls] = {}
        config[sys_cls] = {}
        sys_cls_data = data.loc[data['system_class']==sys_cls].copy()

        # every system class has regressors for one or more noise models
        new_reg_models[sys_cls]['noise'] = {}
        summary[sys_cls]['noise'] = {}
        config[sys_cls]['noise'] = {}
        all_noise_models = list(sys_cls_data['noise_model'].unique())
        for modnm in all_noise_models:
            message_callback('    training regressors for noise model {}'.format(modnm))
            new_reg_models[sys_cls]['noise'][modnm] = {}
            summary[sys_cls]['noise'][modnm] = {}
            config[sys_cls]['noise'][modnm] = {}
            noise_model_data = sys_cls_data.loc[sys_cls_data['noise_model']==modnm].copy()
            for pnm in list(xrsdefs.noise_params[modnm].keys())+['I0_fraction']:
                if not pnm == 'I0':
                    param_header = 'noise_'+pnm
                    try:
                        new_model_type = model_configs[sys_cls]['noise'][modnm][pnm]['model_type']
                        metric = model_configs[sys_cls]['noise'][modnm][pnm]['metric']
                    except:
                        new_model_type = 'ridge_regressor'
                        metric = 'neg_mean_absolute_error'
                    model = Regressor(new_model_type, metric, param_header)
                    message_callback('        training {}'.format(param_header))
                    if (sys_cls in regression_models) \
                    and ('noise' in regression_models[sys_cls]) \
                    and (modnm in regression_models[sys_cls]['noise']) \
                    and (pnm in regression_models[sys_cls]['noise'][modnm]) \
                    and (regression_models[sys_cls]['noise'][modnm][pnm].trained) \
                    and (regression_models[sys_cls]['noise'][modnm][pnm].model_type == new_model_type): 
                        old_pars = regression_models[sys_cls]['noise'][modnm][pnm].model.get_params()
                        for param_nm in model.models_and_params[new_model_type]:
                            model.model.set_params(**{param_nm:old_pars[param_nm]})
                    model.train(noise_model_data, train_hyperparameters, select_features)
                    if model.trained:
                        grpsz_wtd_mean_MAE = model.cross_valid_results['groupsize_weighted_average_MAE']
                        message_callback('        --> weighted-average MAE: {}'.format(grpsz_wtd_mean_MAE))
                    else: 
                        message_callback('        --> {} untrainable- default result: {}'.format(param_header,model.default_val))
                    new_reg_models[sys_cls]['noise'][modnm][pnm] = model
                    summary[sys_cls]['noise'][modnm][pnm] = primitives(model.get_cv_summary())
                    config[sys_cls]['noise'][modnm][pnm] = dict(model_type=model.model_type, metric=model.metric)

        # use the sys_cls to identify the populations and their structures
        for ipop,struct in enumerate(sys_cls.split('__')):
            pop_id = 'pop{}'.format(ipop)
            new_reg_models[sys_cls][pop_id] = {}
            summary[sys_cls][pop_id] = {}
            config[sys_cls][pop_id] = {}
            # every population must have a model for I0_fraction
            param_header = pop_id+'_I0_fraction'
            try:
                new_model_type = model_configs[sys_cls][pop_id]['I0_fraction']['model_type']
                metric = model_configs[sys_cls][pop_id]['I0_fraction']['metric']
            except:
                new_model_type = 'ridge_regressor'
                metric = 'neg_mean_absolute_error'
            model = Regressor(new_model_type, metric, param_header)
            message_callback('    training regressors for population {}'.format(pop_id))
            message_callback('        training {}'.format(param_header))
            if (sys_cls in regression_models) \
            and (pop_id in regression_models[sys_cls]) \
            and ('I0_fraction' in regression_models[sys_cls][pop_id]) \
            and (regression_models[sys_cls][pop_id]['I0_fraction'].trained) \
            and (regression_models[sys_cls][pop_id]['I0_fraction'].model_type == new_model_type): 
                old_pars = regression_models[sys_cls][pop_id]['I0_fraction'].model.get_params()
                for param_nm in model.models_and_params[new_model_type]:
                    model.model.set_params(**{param_nm:old_pars[param_nm]})
            model.train(sys_cls_data, train_hyperparameters, select_features)
            if model.trained:
                grpsz_wtd_mean_MAE = model.cross_valid_results['groupsize_weighted_average_MAE']
                message_callback('        --> weighted-average MAE: {}'.format(grpsz_wtd_mean_MAE))
            else: 
                message_callback('        --> {} untrainable- default result: {}'.format(param_header,model.default_val))
            new_reg_models[sys_cls][pop_id]['I0_fraction'] = model
            summary[sys_cls][pop_id]['I0_fraction'] = primitives(model.get_cv_summary())
            config[sys_cls][pop_id]['I0_fraction'] = dict(model_type=model.model_type, metric=model.metric)

            # add regressors for any modelable structure params 
            for stg_nm in xrsdefs.modelable_structure_settings[struct]:
                stg_header = pop_id+'_'+stg_nm
                new_reg_models[sys_cls][pop_id][stg_nm] = {}
                summary[sys_cls][pop_id][stg_nm] = {}
                config[sys_cls][pop_id][stg_nm] = {}
                stg_labels = list(sys_cls_data[stg_header].unique())
                for stg_label in stg_labels:
                    new_reg_models[sys_cls][pop_id][stg_nm][stg_label] = {}
                    summary[sys_cls][pop_id][stg_nm][stg_label] = {}
                    config[sys_cls][pop_id][stg_nm][stg_label] = {}
                    stg_label_data = sys_cls_data.loc[sys_cls_data[stg_header]==stg_label].copy()
                    message_callback('    training regressors for {} with {}=={}'.format(pop_id,stg_nm,stg_label))
                    for pnm in xrsdefs.structure_params(struct,{stg_nm:stg_label}):
                        param_header = pop_id+'_'+pnm
                        try:
                            new_model_type = model_configs[sys_cls][pop_id][stg_nm][stg_label][pnm]['model_type']
                            metric = model_configs[sys_cls][pop_id][stg_nm][stg_label][pnm]['metric']
                        except:
                            new_model_type = 'ridge_regressor'
                            metric = 'neg_mean_absolute_error'
                        model = Regressor(new_model_type, metric, param_header)
                        message_callback('        training {}'.format(param_header))
                        if (sys_cls in regression_models) \
                        and (pop_id in regression_models[sys_cls]) \
                        and (stg_nm in regression_models[sys_cls][pop_id]) \
                        and (stg_label in regression_models[sys_cls][pop_id][stg_nm]) \
                        and (pnm in regression_models[sys_cls][pop_id][stg_nm][stg_label]) \
                        and (regression_models[sys_cls][pop_id][stg_nm][stg_label][pnm].trained) \
                        and (regression_models[sys_cls][pop_id][stg_nm][stg_label][pnm].model_type == new_model_type):
                            old_pars = regression_models[sys_cls][pop_id][stg_nm][stg_label][pnm].model.get_params()
                            for param_nm in model.models_and_params[new_model_type]:
                                model.model.set_params(**{param_nm:old_pars[param_nm]})
                        model.train(stg_label_data, train_hyperparameters, select_features)
                        if model.trained:
                            grpsz_wtd_mean_MAE = model.cross_valid_results['groupsize_weighted_average_MAE']
                            message_callback('        --> weighted-average MAE: {}'.format(grpsz_wtd_mean_MAE))
                        else: 
                            message_callback('        --> {} untrainable- default result: {}'.format(param_header,model.default_val))
                        new_reg_models[sys_cls][pop_id][stg_nm][stg_label][pnm] = model
                        summary[sys_cls][pop_id][stg_nm][stg_label][pnm] = primitives(model.get_cv_summary())
                        config[sys_cls][pop_id][stg_nm][stg_label][pnm] = dict(model_type=model.model_type, metric=model.metric)

            # get all unique form factors for this population
            form_header = pop_id+'_form'
            form_specifiers = list(sys_cls_data[form_header].unique())

            # for each form, make additional regression models
            for form_id in form_specifiers:
                form_data = sys_cls_data.loc[data[form_header]==form_id].copy()
                new_reg_models[sys_cls][pop_id][form_id] = {}
                summary[sys_cls][pop_id][form_id] = {}
                config[sys_cls][pop_id][form_id] = {}
                message_callback('    training regressors for {} with {} form factors'.format(pop_id,form_id))
                for pnm in xrsdefs.form_factor_params[form_id]:
                    param_header = pop_id+'_'+pnm
                    try:
                        new_model_type = model_configs[sys_cls][pop_id][form_id][pnm]['model_type']
                        metric = model_configs[sys_cls][pop_id][form_id][pnm]['metric']
                    except:
                        new_model_type = 'ridge_regressor'
                        metric = 'neg_mean_absolute_error'
                    model = Regressor(new_model_type, metric, param_header)
                    message_callback('        training {}'.format(param_header))
                    if (sys_cls in regression_models) \
                    and (pop_id in regression_models[sys_cls]) \
                    and (form_id in regression_models[sys_cls][pop_id]) \
                    and (pnm in regression_models[sys_cls][pop_id][form_id]) \
                    and (regression_models[sys_cls][pop_id][form_id][pnm].trained) \
                    and (regression_models[sys_cls][pop_id][form_id][pnm].model_type == new_model_type):
                        old_pars = regression_models[sys_cls][pop_id][form_id][pnm].model.get_params()
                        for param_nm in model.models_and_params[new_model_type]:
                            model.model.set_params(**{param_nm:old_pars[param_nm]})
                    model.train(form_data, train_hyperparameters, select_features)
                    if model.trained:
                        grpsz_wtd_mean_MAE = model.cross_valid_results['groupsize_weighted_average_MAE']
                        message_callback('        --> weighted-average MAE: {}'.format(grpsz_wtd_mean_MAE))
                    else: 
                        message_callback('        --> {} untrainable- default result: {}'.format(param_header,model.default_val))
                    new_reg_models[sys_cls][pop_id][form_id][pnm] = model
                    summary[sys_cls][pop_id][form_id][pnm] = primitives(model.get_cv_summary())
                    config[sys_cls][pop_id][form_id][pnm] = dict(model_type=model.model_type, metric=model.metric)

                # add regressors for any modelable form factor params 
                for stg_nm in xrsdefs.modelable_form_factor_settings[form_id]:
                    stg_header = pop_id+'_'+stg_nm
                    stg_labels = list(form_data[stg_header].unique())
                    new_reg_models[sys_cls][pop_id][form_id][stg_nm] = {}
                    summary[sys_cls][pop_id][form_id][stg_nm] = {}
                    config[sys_cls][pop_id][form_id][stg_nm] = {}
                    for stg_label in stg_labels:
                        new_reg_models[sys_cls][pop_id][form_id][stg_nm][stg_label] = {}
                        summary[sys_cls][pop_id][form_id][stg_nm][stg_label] = {}
                        config[sys_cls][pop_id][form_id][stg_nm][stg_label] = {}
                        stg_label_data = form_data.loc[form_data[stg_header]==stg_label].copy()
                        message_callback('    training regressors for {} with {} form factors with {}=={}'.format(pop_id,form_id,stg_nm,stg_label))
                        for pnm in xrsdefs.additional_form_factor_params(form_id,{stg_nm:stg_label}):
                            param_header = pop_id+'_'+pnm
                            try:
                                new_model_type = model_configs[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm]['model_type']
                                metric = model_configs[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm]['metric']
                            except:
                                new_model_type = 'ridge_regressor'
                                metric = 'neg_mean_absolute_error'
                            model = Regressor(new_model_type, metric, param_header)
                            message_callback('        training {}'.format(param_header))
                            if (sys_cls in regression_models) \
                            and (pop_id in regression_models[sys_cls]) \
                            and (form_id in regression_models[sys_cls][pop_id]) \
                            and (stg_nm in regression_models[sys_cls][pop_id][form_id]) \
                            and (stg_label in regression_models[sys_cls][pop_id][form_id][stg_nm]) \
                            and (pnm in regression_models[sys_cls][pop_id][form_id][stg_nm][stg_label]) \
                            and (regression_models[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm].trained) \
                            and (regression_models[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm].model_type == new_model_type):
                                old_pars = regression_models[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm].model.get_params()
                                for param_nm in model.models_and_params[new_model_type]:
                                    model.model.set_params(**{param_nm:old_pars[param_nm]})
                            model.train(stg_label_data, train_hyperparameters, select_features)
                            if model.trained:
                                grpsz_wtd_mean_MAE = model.cross_valid_results['groupsize_weighted_average_MAE']
                                message_callback('        --> weighted-average MAE: {}'.format(grpsz_wtd_mean_MAE))
                            else: 
                                message_callback('        --> {} untrainable- default result: {}'.format(param_header,model.default_val))
                            new_reg_models[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm] = model
                            summary[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm] = primitives(model.get_cv_summary())
                            config[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm] = dict(model_type=model.model_type, metric=model.metric)

    return new_reg_models, summary, config


def save_regression_models(output_dir, models):
    """Serialize `models` to .yml files.

    The models and scalers are saved to .yml,
    and a report of the cross-validation is saved to .txt.

    Parameters
    ----------
    models : dict
        embedded dict of models, similar to output of train_regression_models().
    """
    rg_root_dir = output_dir
    # get a reference to the currently-loaded regression models dict
    model_dict = get_regression_models()
    if not os.path.exists(rg_root_dir): os.mkdir(rg_root_dir)
    for sys_cls in models.keys():
        sys_cls_dir = os.path.join(rg_root_dir,sys_cls)
        if not os.path.exists(sys_cls_dir): os.mkdir(sys_cls_dir)
        if not sys_cls in model_dict: model_dict[sys_cls] = {}
        if 'noise' in models[sys_cls]:
            noise_dir = os.path.join(sys_cls_dir,'noise')
            if not os.path.exists(noise_dir): os.mkdir(noise_dir)
            if not 'noise' in model_dict[sys_cls]: model_dict[sys_cls]['noise'] = {}
            for modnm in models[sys_cls]['noise'].keys():
                noise_model_dir = os.path.join(noise_dir,modnm)
                if not os.path.exists(noise_model_dir): os.mkdir(noise_model_dir)
                if not modnm in model_dict[sys_cls]['noise']: model_dict[sys_cls]['noise'][modnm] = {}
                for pnm, model in models[sys_cls]['noise'][modnm].items():
                    model_dict[sys_cls]['noise'][modnm][pnm] = model 
                    yml_path = os.path.join(noise_model_dir,pnm+'.yml')
                    txt_path = os.path.join(noise_model_dir,pnm+'.txt')
                    pickle_path = os.path.join(noise_model_dir,pnm+'.pickle')
                    models[sys_cls]['noise'][modnm][pnm].save_model_data(yml_path,txt_path, pickle_path)

        for ipop,struct in enumerate(sys_cls.split('__')):
            pop_id = 'pop{}'.format(ipop)
            pop_dir = os.path.join(sys_cls_dir,pop_id)
            if not os.path.exists(pop_dir): os.mkdir(pop_dir)
            if not pop_id in model_dict[sys_cls]: model_dict[sys_cls][pop_id] = {}
            
            if 'I0_fraction' in models[sys_cls][pop_id]:
                model_dict[sys_cls][pop_id]['I0_fraction'] = models[sys_cls][pop_id]['I0_fraction']
                yml_path = os.path.join(pop_dir,'I0_fraction.yml')
                txt_path = os.path.join(pop_dir,'I0_fraction.txt')
                pickle_path = os.path.join(pop_dir,'I0_fraction.pickle')
                models[sys_cls][pop_id]['I0_fraction'].save_model_data(yml_path,txt_path, pickle_path)
               
            for stg_nm in xrsdefs.modelable_structure_settings[struct]:
                if stg_nm in models[sys_cls][pop_id]:
                    stg_dir = os.path.join(pop_dir,stg_nm)
                    if not os.path.exists(stg_dir): os.mkdir(stg_dir)
                    if not stg_nm in model_dict[sys_cls][pop_id]: model_dict[sys_cls][pop_id][stg_nm] = {}
                    for stg_label in models[sys_cls][pop_id][stg_nm].keys():
                        stg_label_dir = os.path.join(stg_dir,stg_label)
                        if not os.path.exists(stg_label_dir): os.mkdir(stg_label_dir)
                        if not stg_label in model_dict[sys_cls][pop_id][stg_nm]: 
                            model_dict[sys_cls][pop_id][stg_nm][stg_label] = {}
                        for pnm in xrsdefs.structure_params(struct,{stg_nm:stg_label}):
                            if pnm in models[sys_cls][pop_id][stg_nm][stg_label]:
                                model_dict[sys_cls][pop_id][stg_nm][stg_label][pnm] = \
                                models[sys_cls][pop_id][stg_nm][stg_label][pnm]
                                yml_path = os.path.join(stg_label_dir,pnm+'.yml')
                                txt_path = os.path.join(stg_label_dir,pnm+'.txt')
                                pickle_path = os.path.join(stg_label_dir,pnm+'.pickle')
                                models[sys_cls][pop_id][stg_nm][stg_label][pnm].save_model_data(yml_path,txt_path, pickle_path)
            
            for form_id in xrsdefs.form_factor_names:
                if form_id in models[sys_cls][pop_id]:
                    form_dir = os.path.join(pop_dir,form_id)
                    if not os.path.exists(form_dir): os.mkdir(form_dir)
                    if not form_id in model_dict[sys_cls][pop_id]: model_dict[sys_cls][pop_id][form_id] = {}
                    for pnm in xrsdefs.form_factor_params[form_id]:
                        if pnm in models[sys_cls][pop_id][form_id]:
                            model_dict[sys_cls][pop_id][form_id][pnm] = models[sys_cls][pop_id][form_id][pnm]
                            yml_path = os.path.join(form_dir,pnm+'.yml')
                            txt_path = os.path.join(form_dir,pnm+'.txt')
                            pickle_path = os.path.join(form_dir,pnm+'.pickle')
                            models[sys_cls][pop_id][form_id][pnm].save_model_data(yml_path,txt_path, pickle_path)

                    for stg_nm in xrsdefs.modelable_form_factor_settings[form_id]:
                        if stg_nm in models[sys_cls][pop_id][form_id]:
                            stg_dir = os.path.join(form_dir,stg_nm)
                            if not os.path.exists(stg_dir): os.mkdir(stg_dir)
                            model_dict[sys_cls][pop_id][form_id][stg_nm] = {}
                            for stg_label in models[sys_cls][pop_id][form_id][stg_nm].keys():
                                stg_label_dir = os.path.join(stg_dir,stg_label)
                                if not os.path.exists(stg_label_dir): os.mkdir(stg_label_dir)
                                if not stg_label in model_dict[sys_cls][pop_id][form_id][stg_nm]:
                                    model_dict[sys_cls][pop_id][form_id][stg_nm][stg_label] = {}
                                for pnm in xrsdefs.additional_form_factor_params(form_id,{stg_nm:stg_label}):
                                    model_dict[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm] = \
                                    models[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm]
                                    yml_path = os.path.join(stg_label_dir,pnm+'.yml')
                                    txt_path = os.path.join(stg_label_dir,pnm+'.txt')
                                    pickle_path = os.path.join(stg_label_dir,pnm+'.pickle')
                                    models[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm].save_model_data(yml_path,txt_path, pickle_path)

def collect_config(config_reg, config_cl):
    model_configs = OrderedDict.fromkeys(['DESCRIPTION','CLASSIFIERS','REGRESSORS'])
    model_configs['DESCRIPTION'] = { 
        'Model options for classifiers':
            ['logistic_regressor','sgd_classifier',
            'non_linear_svm','linear_svm','linear_svm_hinge',
            'random_forest','d_tree','knn'],
        'Training metric options for classifiers':
            ['f1','accuracy','precision','recall'],
        'Model options for regressors':
            ['ridge_regressor','elastic_net','sgd_regressor'],
        'Training metric options for regressors':
            ['MAE','coef_of_determination']
        }
    model_configs['REGRESSORS'] = config_reg 
    model_configs['CLASSIFIERS'] = config_cl 
    return model_configs

def collect_summary(summary_reg, summary_cl, old_summary={}):
    summary = OrderedDict.fromkeys(['DESCRIPTION','CLASSIFIERS','REGRESSORS'])
    summary['DESCRIPTION'] = 'Each metric is reported with two values: '\
        'The first value is the value of the metric, '\
        'and the second value is the difference in the metric '\
        'relative to a reference training summary, if provided'
    summary['CLASSIFIERS'] = summary_cl 
    summary['REGRESSORS'] = summary_reg 
    old_summary_reg = {}
    old_summary_cl = {}
    if old_summary:
        old_summary_reg = old_summary['REGRESSORS']
        old_summary_cl = old_summary['CLASSIFIERS']
    # TODO: properly unpack and compare these summaries
    #for k, v in summary_reg.items():
    #    if v:
    #        summary['REGRESSORS'][k] = {}
    #        summary['REGRESSORS'][k]['model_type'] = v['model_type']
    #        summary['REGRESSORS'][k]['scores'] = {}
    #        for metric, value in v['scores'].items():
    #            try:
    #                diff = value-old_summary['REGRESSORS'][k]['scores'][metric][0]
    #            except:
    #                diff = None
    #            summary['REGRESSORS'][k]['scores'][metric] = [value, diff]
    #for k, v in summary_cl.items():
    #    if v:
    #        summary['CLASSIFIERS'][k] = {}
    #        summary['CLASSIFIERS'][k]['model_type'] = v['model_type']
    #        summary['CLASSIFIERS'][k]['scores'] = {}
    #        for key, value in v['scores'].items():
    #            try:
    #                diff = value-old_summary['CLASSIFIERS'][k]['scores'][key][0]
    #            except:
    #                diff = None
    #            summary['CLASSIFIERS'][k]['scores'][key] = [value, diff]
    return summary
