import os
from collections import OrderedDict

import yaml

from .. import definitions as xrsdefs 
from .regressor import Regressor
from .classifier import Classifier

file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(file_path)
package_dir = os.path.dirname(models_dir)
#root_dir = os.path.dirname(package_dir)

# find directory containing packaged modeling data
modeling_data_dir = os.path.join(package_dir,'models','modeling_data')
regression_models_dir = os.path.join(modeling_data_dir,'regressors')
classification_models_dir = os.path.join(modeling_data_dir,'classifiers')

# find directory containing training summary
training_summary_yml = os.path.join(models_dir,'training_summary.yml')

def load_classifier_from_yml(yml_file):
    ymlf = open(yml_file,'rb')
    content = yaml.load(ymlf)
    ymlf.close()
    cl = Classifier(content['model_type'],content['model_target'])
    cl.load_model_data(content)
    return cl

def load_regressor_from_yml(yml_file):
    ymlf = open(yml_file,'rb')
    content = yaml.load(ymlf)
    ymlf.close()
    reg = Regressor(content['model_type'],content['model_target'])
    reg.load_model_data(content)
    return reg

def load_classification_models(model_root_dir=classification_models_dir):  
    model_dict = OrderedDict()
    if not os.path.exists(model_root_dir):
        return model_dict
    all_sys_cls = os.listdir(model_root_dir)
    # this next line filters out hidden files
    all_sys_cls = [i for i in all_sys_cls if not i[0]=='.']

    # the top-level classifier is a collection of classifiers;
    # their cumulative effect is to find the number of distinct populations
    # for each structure
    main_cls_path =  os.path.join(model_root_dir, 'main_classifiers')
    model_dict['main_classifiers'] = {}
    if os.path.exists(main_cls_path):
        all_main_cls = os.listdir(main_cls_path)
        # this next line filters out hidden files
        all_main_cls = [i for i in all_main_cls if not i[0]=='.']
        all_main_cls = [cl for cl in all_main_cls if cl.endswith('.yml')]
        for cl in all_main_cls:
            cl_name = os.path.splitext(cl)[0]
            yml_path = os.path.join(main_cls_path, cl)
            model_dict['main_classifiers'][cl_name] = load_classifier_from_yml(yml_path)

    if 'main_classifiers' in all_sys_cls: all_sys_cls.remove('main_classifiers')
    for sys_cls in all_sys_cls:
        model_dict[sys_cls] = {}
        sys_cls_dir = os.path.join(model_root_dir,sys_cls)
        noise_yml_path = os.path.join(sys_cls_dir,'noise_model.yml')
        if os.path.exists(noise_yml_path):
            model_dict[sys_cls]['noise_model'] = load_classifier_from_yml(noise_yml_path)

        for ipop,struct in enumerate(sys_cls.split('__')):
            pop_id = 'pop{}'.format(ipop)
            pop_dir = os.path.join(sys_cls_dir,pop_id)
            model_dict[sys_cls][pop_id] = {}

            # each population must have a form classifier
            form_yml_path = os.path.join(pop_dir,'form.yml')
            if os.path.exists(form_yml_path):
                model_dict[sys_cls][pop_id]['form'] = load_classifier_from_yml(form_yml_path) 

            # other classifiers in this directory are for structure settings
            for stg_nm in xrsdefs.modelable_structure_settings[struct]:
                stg_yml_path = os.path.join(pop_dir,stg_nm+'.yml')
                if os.path.exists(stg_yml_path):
                    model_dict[sys_cls][pop_id][stg_nm] = load_classifier_from_yml(stg_yml_path) 

            # some additional directories may exist for form factor settings-
            # these would be named according to their form factors
            for ffnm in xrsdefs.form_factor_names:
                ff_dir = os.path.join(pop_dir,ffnm)
                if os.path.exists(ff_dir):
                    model_dict[sys_cls][pop_id][ffnm] = {}
                    for stg_nm in xrsdefs.modelable_form_factor_settings[ffnm]:
                        stg_yml_path = os.path.join(ff_dir,stg_nm+'.yml')
                        if os.path.exists(stg_yml_path):
                            model_dict[sys_cls][pop_id][ffnm][stg_nm] = load_classifier_from_yml(stg_yml_path) 
    return model_dict

def load_regression_models(model_root_dir=regression_models_dir):
    model_dict = OrderedDict()
    if not os.path.exists(model_root_dir):
        return model_dict

    all_sys_cls = os.listdir(model_root_dir)
    # this next line filters out hidden files
    all_sys_cls = [i for i in all_sys_cls if not i[0]=='.']
    for sys_cls in all_sys_cls:
        model_dict[sys_cls] = {}
        sys_cls_dir = os.path.join(model_root_dir,sys_cls)

        # every system class must have some noise parameters
        noise_dir = os.path.join(sys_cls_dir,'noise')
        model_dict[sys_cls]['noise'] = {}
        for modnm in xrsdefs.noise_model_names:
            noise_model_dir = os.path.join(noise_dir,modnm)
            if os.path.exists(noise_model_dir):
                model_dict[sys_cls]['noise'][modnm] = {}
                for pnm in list(xrsdefs.noise_params[modnm].keys())+['I0_fraction']:
                    param_yml_file = os.path.join(noise_model_dir,pnm+'.yml')
                    if os.path.exists(param_yml_file):
                        model_dict[sys_cls]['noise'][modnm][pnm] = load_regressor_from_yml(param_yml_file)

        for ipop,struct in enumerate(sys_cls.split('__')):
            pop_id = 'pop{}'.format(ipop)
            model_dict[sys_cls][pop_id] = {}
            pop_dir = os.path.join(sys_cls_dir,pop_id)

            # each population must have a model for its I0_fraction 
            I0_fraction_yml = os.path.join(pop_dir,'I0_fraction.yml')
            if os.path.exists(I0_fraction_yml): 
                model_dict[sys_cls][pop_id]['I0_fraction'] = load_regressor_from_yml(I0_fraction_yml)

            # each population may have additional parameters,
            # depending on settings
            for stg_nm in xrsdefs.modelable_structure_settings[struct]:
                stg_dir = os.path.join(pop_dir,stg_nm)
                if os.path.exists(stg_dir):
                    model_dict[sys_cls][pop_id][stg_nm] = {}
                    all_stg_labels = os.listdir(stg_dir)
                    # this next line filters out hidden files
                    all_stg_labels = [i for i in all_stg_labels if not i[0]=='.']
                    for stg_label in all_stg_labels:
                        stg_label_dir = os.path.join(stg_dir,stg_label)
                        if os.path.exists(stg_label_dir):
                            model_dict[sys_cls][pop_id][stg_nm][stg_label] = {}
                            for pnm in xrsdefs.structure_params(struct,{stg_nm:stg_label}):
                                param_yml = os.path.join(stg_label_dir,pnm+'.yml')
                                model_dict[sys_cls][pop_id][stg_nm][stg_label][pnm] = load_regressor_from_yml(param_yml)

            # each population may have still more parameters,
            # depending on the form factor selection
            for ff_nm in xrsdefs.form_factor_names:
                ff_dir = os.path.join(pop_dir,ff_nm)
                if os.path.exists(ff_dir):
                    model_dict[sys_cls][pop_id][ff_nm] = {}
                    for pnm in xrsdefs.form_factor_params[ff_nm]:
                        param_yml = os.path.join(ff_dir,pnm+'.yml')
                        model_dict[sys_cls][pop_id][ff_nm][pnm] = load_regressor_from_yml(param_yml)

                # the final layer of parameters depends on form factor settings
                for stg_nm in xrsdefs.modelable_form_factor_settings[ff_nm]:
                    stg_dir = os.path.join(ff_dir,stg_nm)
                    if os.path.exists(stg_dir): 
                        model_dict[sys_cls][pop_id][ff_nm][stg_nm] = {}
                        all_stg_labels = os.listdir(stg_dir)
                        # this next line filters out hidden files
                        all_stg_labels = [i for i in all_stg_labels if not i[0]=='.']
                        for stg_label in all_stg_labels:
                            stg_label_dir = os.path.join(stg_dir,stg_label)
                            if os.path.exists(stg_label_dir):
                                model_dict[sys_cls][pop_id][ff_nm][stg_nm][stg_label] = {}
                                for pnm in xrsdefs.additional_form_factor_params(ff_nm,{stg_nm:stg_label}):
                                    param_yml = os.path.join(stg_label_dir,pnm+'.yml')
                                    model_dict[sys_cls][pop_id][ff_nm][stg_nm][stg_label][pnm] = load_regressor_from_yml(param_yml)
    return model_dict

regression_models = load_regression_models(regression_models_dir)
classification_models = load_classification_models(classification_models_dir)
