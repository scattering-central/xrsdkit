import os

import pandas as pd
import numpy as np
import yaml

from . import regression_models_dir, classification_models_dir
from . import regression_models, classification_models
from . import test_regression_models_dir, test_classification_models_dir
from . import test_regression_models, test_classification_models
from .. import definitions as xrsdefs
from ..tools import primitives, profiler
from . import classification_models, regression_models
from .regressor import Regressor
from .classifier import Classifier


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


def train_classification_models(data,hyper_parameters_search=False):
    """Train all classifiers that are trainable from `data`.

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
        trained on the given dataset `data`.
    """
    
    cls_models = {} 
    print(os.linesep+'Training main system classifier')
    model = Classifier('system_class',None)
    if 'system_class' in classification_models.keys() \
    and classification_models['system_class'].trained: 
        old_pars = classification_models['system_class'].model.get_params()
        model.model.set_params(alpha=old_pars['alpha'], l1_ratio=old_pars['l1_ratio'])
    model.train(data, hyper_parameters_search=hyper_parameters_search)
    cls_models['system_class'] = model

    sys_cls_labels = list(data['system_class'].unique())
    # 'unidentified' systems will have no sub-classifiers:
    if 'unidentified' in sys_cls_labels: sys_cls_labels.pop(sys_cls_labels.index('unidentified'))
    for sys_cls in sys_cls_labels:
        print('Training classifiers for system: ')
        print(sys_cls)
        cls_models[sys_cls] = {}
        sys_cls_data = data.loc[data['system_class']==sys_cls].copy()
        # drop the columns where all values are None:
        #sys_cls_data.dropna(axis=1,how='all',inplace=True)

        # every system class must have a noise classifier
        print('    Training noise classifier for system class {}'.format(sys_cls))
        model = Classifier('noise_model',None)
        if (sys_cls in classification_models) \
        and ('noise_model' in classification_models[sys_cls]) \
        and (classification_models[sys_cls]['noise_model'].trained): 
            old_pars = classification_models[sys_cls]['noise_model'].model.get_params()
            model.model.set_params(alpha=old_pars['alpha'], l1_ratio=old_pars['l1_ratio'])
        model.train(sys_cls_data, hyper_parameters_search=hyper_parameters_search)
        cls_models[sys_cls]['noise_model'] = model

        # each population has some classifiers for form factor and settings
        for ipop, struct in enumerate(sys_cls.split('__')):
            pop_id = 'pop{}'.format(ipop)
            cls_models[sys_cls][pop_id] = {}
            print('    Training classifiers for population {}'.format(pop_id))

            # every population must have a form classifier
            form_header = pop_id+'_form'
            print('    Training: {}'.format(form_header))
            model = Classifier(form_header,None)
            if (sys_cls in classification_models) \
            and (pop_id in classification_models[sys_cls]) \
            and ('form' in classification_models[sys_cls][pop_id]) \
            and (classification_models[sys_cls][pop_id]['form'].trained): 
                old_pars = classification_models[sys_cls][pop_id]['form'].model.get_params()
                model.model.set_params(alpha=old_pars['alpha'], l1_ratio=old_pars['l1_ratio'])
            model.train(sys_cls_data, hyper_parameters_search=hyper_parameters_search)
            cls_models[sys_cls][pop_id]['form'] = model

            # add classifiers for any model-able structure settings 
            for stg_nm in xrsdefs.modelable_structure_settings[struct]:
                stg_header = pop_id+'_'+stg_nm
                print('    Training: {}'.format(stg_header))
                model = Classifier(stg_header,None)
                if (sys_cls in classification_models) \
                and (pop_id in classification_models[sys_cls]) \
                and (stg_nm in classification_models[sys_cls][pop_id]) \
                and (classification_models[sys_cls][pop_id][stg_nm].trained): 
                    old_pars = classification_models[sys_cls][pop_id][stg_nm].model.get_params()
                    model.model.set_params(alpha=old_pars['alpha'], l1_ratio=old_pars['l1_ratio'])
                model.train(sys_cls_data, hyper_parameters_search=hyper_parameters_search)
                cls_models[sys_cls][pop_id][stg_nm] = model

            # add classifiers for any model-able form factor settings
            all_ff_labels = list(sys_cls_data[form_header].unique())
            for ff in all_ff_labels:
                form_data = sys_cls_data.loc[sys_cls_data[form_header]==ff].copy()
                cls_models[sys_cls][pop_id][ff] = {}
                for stg_nm in xrsdefs.modelable_form_factor_settings[ff]:
                    stg_header = pop_id+'_'+stg_nm
                    print('    Training: {}'.format(stg_header))
                    model = Classifier(stg_header,None)
                    if (sys_cls in classification_models) \
                    and (pop_id in classification_models[sys_cls]) \
                    and (ff in classification_models[sys_cls][pop_id]) \
                    and (stg_nm in classification_models[sys_cls][pop_id][ff]) \
                    and (classification_models[sys_cls][pop_id][ff][stg_nm].trained): 
                        old_pars = classification_models[sys_cls][pop_id][ff][stg_nm].model.get_params()
                        model.model.set_params(alpha=old_pars['alpha'], l1_ratio=old_pars['l1_ratio'])
                    model.train(form_data, hyper_parameters_search=hyper_parameters_search)
                    cls_models[sys_cls][pop_id][ff][stg_nm] = model

    return cls_models


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

    if 'system_class' in models:
        model_dict['system_class'] = models['system_class']
        yml_path = os.path.join(cl_root_dir,'system_class.yml')
        txt_path = os.path.join(cl_root_dir,'system_class.txt')
        save_model_data(models['system_class'],yml_path,txt_path)

    all_sys_cls = list(models.keys())
    all_sys_cls.pop(all_sys_cls.index('system_class'))
    for sys_cls in all_sys_cls: 
        sys_cls_dir = os.path.join(cl_root_dir,sys_cls)
        if not sys_cls in model_dict: model_dict[sys_cls] = {}
        if not os.path.exists(sys_cls_dir): os.mkdir(sys_cls_dir)
        if 'noise_model' in models[sys_cls]:
            model_dict[sys_cls]['noise_model'] = models[sys_cls]['noise_model']
            yml_path = os.path.join(sys_cls_dir,'noise_model.yml')
            txt_path = os.path.join(sys_cls_dir,'noise_model.txt')
            save_model_data(models[sys_cls]['noise_model'],yml_path,txt_path)

        for ipop,struct in enumerate(sys_cls.split('__')):
            pop_id = 'pop{}'.format(ipop)
            pop_dir = os.path.join(sys_cls_dir,pop_id)
            if not os.path.exists(pop_dir): os.mkdir(pop_dir)
            if not pop_id in model_dict[sys_cls]: model_dict[sys_cls][pop_id] = {}

            if 'form' in models[sys_cls][pop_id]:
                model_dict[sys_cls][pop_id]['form'] = models[sys_cls][pop_id]['form']
                yml_path = os.path.join(pop_dir,'form.yml')
                txt_path = os.path.join(pop_dir,'form.txt')
                save_model_data(models[sys_cls][pop_id]['form'],yml_path,txt_path)
               
            for stg_nm in xrsdefs.modelable_structure_settings[struct]:
                if stg_nm in models[sys_cls][pop_id]:
                    model_dict[sys_cls][pop_id][stg_nm] = models[sys_cls][pop_id][stg_nm]
                    yml_path = os.path.join(pop_dir,stg_nm+'.yml')
                    txt_path = os.path.join(pop_dir,stg_nm+'.txt')
                    save_model_data(models[sys_cls][pop_id][stg_nm],yml_path,txt_path)

            for ff_id in xrsdefs.form_factor_names:
                if ff_id in models[sys_cls][pop_id]:
                    form_dir = os.path.join(pop_dir,ff_id)
                    if not os.path.exists(form_dir): os.mkdir(form_dir)
                    model_dict[sys_cls][pop_id][ff_id] = {}
                    for stg_nm in xrsdefs.modelable_form_factor_settings[ff_id]:
                        model_dict[sys_cls][pop_id][ff_id][stg_nm] = models[sys_cls][pop_id][ff_id][stg_nm]
                        yml_path = os.path.join(form_dir,stg_nm+'.yml')
                        txt_path = os.path.join(form_dir,stg_nm+'.txt')
                        save_model_data(models[sys_cls][pop_id][ff_id][stg_nm],yml_path,txt_path)


def train_regression_models(data,hyper_parameters_search=False):
    """Train all regression models trainable from `data`. 

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels
    hyper_parameters_search : bool
        if True, the models will be optimized
        over a grid of hyperparameters during training

    Returns
    -------
    reg_models : dict
        embedded dictionary with blank spaces for all trainable regression models 
    """
    reg_models = {} 
    sys_cls_labels = list(data['system_class'].unique())
    # 'unidentified' systems will have no regression models:
    if 'unidentified' in sys_cls_labels: sys_cls_labels.pop(sys_cls_labels.index('unidentified'))
    for sys_cls in sys_cls_labels:
        print('training regressors for system class {}'.format(sys_cls))
        reg_models[sys_cls] = {}
        sys_cls_data = data.loc[data['system_class']==sys_cls].copy()
        # drop the columns where all values are None:
        #sys_cls_data.dropna(axis=1,how='all',inplace=True)

        # every system class has regressors for one or more noise models 
        reg_models[sys_cls]['noise'] = {}
        all_noise_models = list(sys_cls_data['noise_model'].unique())
        for modnm in all_noise_models:
            print('    training regressors for noise model {}'.format(modnm))
            reg_models[sys_cls]['noise'][modnm] = {}
            noise_model_data = sys_cls_data.loc[sys_cls_data['noise_model']==modnm].copy()
            for pnm in list(xrsdefs.noise_params[modnm].keys())+['I0_fraction']:
                param_header = 'noise_'+pnm
                model = Regressor(param_header,None)
                print('        training {}'.format(param_header))
                if (sys_cls in regression_models) \
                and ('noise' in regression_models[sys_cls]) \
                and (modnm in regression_models[sys_cls]['noise']) \
                and (pnm in regression_models[sys_cls]['noise'][modnm]) \
                and (regression_models[sys_cls]['noise'][modnm][pnm].trained): 
                    old_pars = regression_models[sys_cls]['noise'][modnm][pnm].model.get_params()
                    model.model.set_params(alpha=old_pars['alpha'],
                    l1_ratio=old_pars['l1_ratio'],epsilon=old_pars['epsilon'])
                if not pnm == 'I0':
                    model.train(noise_model_data, hyper_parameters_search)
                    reg_models[sys_cls]['noise'][modnm][pnm] = model 

        # use the sys_cls to identify the populations and their structures
        for ipop,struct in enumerate(sys_cls.split('__')):
            pop_id = 'pop{}'.format(ipop)
            reg_models[sys_cls][pop_id] = {}
            # every population must have a model for I0_fraction
            param_header = pop_id+'_I0_fraction'
            model = Regressor(param_header,None)
            print('    training regressors for population {}'.format(pop_id))
            if (sys_cls in regression_models) \
            and (pop_id in regression_models[sys_cls]) \
            and ('I0_fraction' in regression_models[sys_cls][pop_id]) \
            and (regression_models[sys_cls][pop_id]['I0_fraction'].trained): 
                old_pars = regression_models[sys_cls][pop_id]['I0_fraction'].model.get_params()
                model.model.set_params(alpha=old_pars['alpha'],
                l1_ratio=old_pars['l1_ratio'],epsilon=old_pars['epsilon'])
            model.train(sys_cls_data, hyper_parameters_search)
            reg_models[sys_cls][pop_id]['I0_fraction'] = model 
                
            # add regressors for any modelable structure params 
            for stg_nm in xrsdefs.modelable_structure_settings[struct]:
                stg_header = pop_id+'_'+stg_nm
                reg_models[sys_cls][pop_id][stg_nm] = {}
                stg_labels = list(sys_cls_data[stg_header].unique())
                for stg_label in stg_labels:
                    reg_models[sys_cls][pop_id][stg_nm][stg_label] = {}
                    stg_label_data = sys_cls_data.loc[sys_cls_data[stg_header]==stg_label].copy()
                    for pnm in xrsdefs.structure_params(struct,{stg_nm:stg_label}):
                        param_header = pop_id+'_'+pnm
                        model = Regressor(param_header,None)
                        print('        training {}'.format(param_header))
                        if (sys_cls in regression_models) \
                        and (pop_id in regression_models[sys_cls]) \
                        and (stg_nm in regression_models[sys_cls][pop_id]) \
                        and (stg_label in regression_models[sys_cls][pop_id][stg_nm]) \
                        and (pnm in regression_models[sys_cls][pop_id][stg_nm][stg_label]) \
                        and (regression_models[sys_cls][pop_id][stg_nm][stg_label][pnm].trained):
                            old_pars = regression_models[sys_cls][pop_id][stg_nm][stg_label][pnm].model.get_params()
                            model.model.set_params(alpha=old_pars['alpha'],
                            l1_ratio=old_pars['l1_ratio'],epsilon=old_pars['epsilon'])
                        model.train(stg_label_data, hyper_parameters_search)
                        reg_models[sys_cls][pop_id][stg_nm][stg_label][pnm] = model 

            # get all unique form factors for this population
            form_header = pop_id+'_form'
            form_specifiers = list(sys_cls_data[form_header].unique())

            # for each form, make additional regression models
            for form_id in form_specifiers:
                form_data = sys_cls_data.loc[data[form_header]==form_id].copy()
                reg_models[sys_cls][pop_id][form_id] = {}
                for pnm in xrsdefs.form_factor_params[form_id]:
                    param_header = pop_id+'_'+pnm
                    model = Regressor(param_header,None)
                    print('        training {}'.format(param_header))
                    if (sys_cls in regression_models) \
                    and (pop_id in regression_models[sys_cls]) \
                    and (form_id in regression_models[sys_cls][pop_id]) \
                    and (pnm in regression_models[sys_cls][pop_id][form_id]) \
                    and (regression_models[sys_cls][pop_id][form_id][pnm].trained):
                        old_pars = regression_models[sys_cls][pop_id][form_id][pnm].model.get_params()
                        model.model.set_params(alpha=old_pars['alpha'],
                        l1_ratio=old_pars['l1_ratio'],epsilon=old_pars['epsilon'])
                    model.train(form_data, hyper_parameters_search)
                    reg_models[sys_cls][pop_id][form_id][pnm] = model 

                # add regressors for any modelable form factor params 
                for stg_nm in xrsdefs.modelable_form_factor_settings[form_id]:
                    stg_header = pop_id+'_'+stg_nm
                    stg_labels = list(form_data[stg_header].unique())
                    reg_models[sys_cls][pop_id][form_id][stg_nm] = {}
                    for stg_label in stg_labels:
                        reg_models[sys_cls][pop_id][form_id][stg_nm][stg_label] = {}
                        stg_label_data = form_data.loc[form_data[stg_header]==stg_label].copy()
                        for pnm in xrsdefs.additional_form_factor_params(form_id,{stg_nm:stg_label}):
                            param_header = pop_id+'_'+pnm
                            model = Regressor(param_header,None)
                            print('        training {}'.format(param_header))
                            if (sys_cls in regression_models) \
                            and (pop_id in regression_models[sys_cls]) \
                            and (form_id in regression_models[sys_cls][pop_id]) \
                            and (stg_nm in regression_models[sys_cls][pop_id][form_id]) \
                            and (stg_label in regression_models[sys_cls][pop_id][form_id][stg_nm]) \
                            and (pnm in regression_models[sys_cls][pop_id][form_id][stg_nm][stg_label]) \
                            and (regression_models[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm].trained):
                                old_pars = regression_models[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm].\
                                model.get_params()
                                model.model.set_params(alpha=old_pars['alpha'],
                                l1_ratio=old_pars['l1_ratio'],epsilon=old_pars['epsilon'])
                            model.train(stg_label_data, hyper_parameters_search)
                            reg_models[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm] = model 

    return reg_models


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
                    save_model_data(models[sys_cls]['noise'][modnm][pnm],yml_path,txt_path)

        for ipop,struct in enumerate(sys_cls.split('__')):
            pop_id = 'pop{}'.format(ipop)
            pop_dir = os.path.join(sys_cls_dir,pop_id)
            if not os.path.exists(pop_dir): os.mkdir(pop_dir)
            if not pop_id in model_dict[sys_cls]: model_dict[sys_cls][pop_id] = {}
            
            if 'I0_fraction' in models[sys_cls][pop_id]:
                model_dict[sys_cls][pop_id]['I0_fraction'] = models[sys_cls][pop_id]['I0_fraction']
                yml_path = os.path.join(pop_dir,'I0_fraction.yml')
                txt_path = os.path.join(pop_dir,'I0_fraction.txt')
                save_model_data(models[sys_cls][pop_id]['I0_fraction'],yml_path,txt_path)
               
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
                                save_model_data(models[sys_cls][pop_id][stg_nm][stg_label][pnm],yml_path,txt_path)
            
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
                            save_model_data(models[sys_cls][pop_id][form_id][pnm],yml_path,txt_path)

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
                                    save_model_data(models[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm],
                                    yml_path,txt_path)
