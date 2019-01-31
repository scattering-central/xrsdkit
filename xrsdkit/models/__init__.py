import os
import re
from collections import OrderedDict

from .. import definitions as xrsdefs 
from .regressor import Regressor
from .classifier import Classifier
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

def load_classification_models(model_root_dir=classification_models_dir):  
    model_dict = OrderedDict()
    if not os.path.exists(model_root_dir):
        return model_dict
    all_sys_cls = os.listdir(model_root_dir)
 
    yml_path = os.path.join(model_root_dir,'system_class.yml')
    if os.path.exists(yml_path):
        model_dict['system_class'] = Classifier('system_class',yml_path)
        all_sys_cls.pop(all_sys_cls.index('system_class.yml'))
        all_sys_cls.pop(all_sys_cls.index('system_class.txt'))

    for sys_cls in all_sys_cls:
        model_dict[sys_cls] = {}
        sys_cls_dir = os.path.join(model_root_dir,sys_cls)
        noise_yml_path = os.path.join(sys_cls_dir,'noise_model.yml')
        if os.path.exists(noise_yml_path):
            model_dict[sys_cls]['noise_model'] = Classifier('noise_model',noise_yml_path)

        for ipop,struct in enumerate(sys_cls.split('__')):
            pop_id = 'pop{}'.format(ipop)
            pop_dir = os.path.join(sys_cls_dir,pop_id)
            model_dict[sys_cls][pop_id] = {}

            # each population must have a form classifier
            form_header = pop_id+'_form'
            form_yml_path = os.path.join(pop_dir,'form.yml')
            if os.path.exists(form_yml_path):
                model_dict[sys_cls][pop_id]['form'] = Classifier(form_header,form_yml_path) 

            # other classifiers in this directory are for structure settings
            for stg_nm in xrsdefs.modelable_structure_settings[struct]:
                stg_header = pop_id+'_'+stg_nm
                stg_yml_path = os.path.join(pop_dir,stg_nm+'.yml')
                if os.path.exists(stg_yml_path):
                    model_dict[sys_cls][pop_id][stg_nm] = Classifier(stg_header,stg_yml_path) 

            # some additional directories may exist for form factor settings-
            # these would be named according to their form factors
            for ffnm in xrsdefs.form_factor_names:
                ff_dir = os.path.join(pop_dir,ffnm)
                if os.path.exists(ff_dir):
                    model_dict[sys_cls][pop_id][ffnm] = {}
                    for stg_nm in xrsdefs.modelable_form_factor_settings[ffnm]:
                        stg_header = pop_id+'_'+stg_nm
                        stg_yml_path = os.path.join(ff_dir,stg_nm+'.yml')
                        if os.path.exists(stg_yml_path):
                            model_dict[sys_cls][pop_id][ffnm][stg_nm] = Classifier(stg_header,stg_yml_path) 
    return model_dict

def load_regression_models(model_root_dir=regression_models_dir):
    model_dict = OrderedDict()
    if not os.path.exists(model_root_dir):
        return model_dict
 
    for sys_cls in os.listdir(model_root_dir):
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
                        param_header = 'noise_'+pnm
                        model_dict[sys_cls]['noise'][modnm][pnm] = Regressor(param_header,param_yml_file)

        for ipop,struct in enumerate(sys_cls.split('__')):
            pop_id = 'pop{}'.format(ipop)
            model_dict[sys_cls][pop_id] = {}
            pop_dir = os.path.join(sys_cls_dir,pop_id)

            # each population must have a model for its I0_fraction 
            I0_fraction_yml = os.path.join(pop_dir,'I0_fraction.yml')
            if os.path.exists(I0_fraction_yml): 
                I0_fraction_header = pop_id+'_I0_fraction'
                model_dict[sys_cls][pop_id]['I0_fraction'] = Regressor(I0_fraction_header,I0_fraction_yml)

            # each population may have additional parameters,
            # depending on settings
            for stg_nm in xrsdefs.modelable_structure_settings[struct]:
                stg_dir = os.path.join(pop_dir,stg_nm)
                if os.path.exists(stg_dir):
                    model_dict[sys_cls][pop_id][stg_nm] = {}
                    for stg_label in os.listdir(stg_dir):
                        stg_label_dir = os.path.join(stg_dir,stg_label)
                        if os.path.exists(stg_label_dir):
                            model_dict[sys_cls][pop_id][stg_nm][stg_label] = {}
                            for pnm in xrsdefs.structure_params(struct,{stg_nm:stg_label}):
                                param_header = pop_id+'_'+pnm
                                param_yml = os.path.join(stg_label_dir,pnm+'.yml')
                                model_dict[sys_cls][pop_id][stg_nm][stg_label][pnm] = Regressor(param_header,param_yml)

            # each population may have still more parameters,
            # depending on the form factor selection
            for ff_nm in xrsdefs.form_factor_names:
                ff_dir = os.path.join(pop_dir,ff_nm)
                if os.path.exists(ff_dir):
                    model_dict[sys_cls][pop_id][ff_nm] = {}
                    for pnm in xrsdefs.form_factor_params[ff_nm]:
                        param_header = pop_id+'_'+pnm
                        param_yml = os.path.join(ff_dir,pnm+'.yml')
                        model_dict[sys_cls][pop_id][ff_nm][pnm] = Regressor(param_header,param_yml)

                # the final layer of parameters depends on form factor settings
                for stg_nm in xrsdefs.modelable_form_factor_settings[ff_nm]:
                    stg_dir = os.path.join(ff_dir,stg_nm)
                    if os.path.exists(stg_dir): 
                        model_dict[sys_cls][pop_id][ff_nm][stg_nm] = {}
                        for stg_label in os.listdir(stg_dir):
                            stg_label_dir = os.path.join(stg_dir,stg_label)
                            if os.path.exists(stg_label_dir):
                                model_dict[sys_cls][pop_id][ff_nm][stg_nm][stg_label] = {}
                                for pnm in xrsdefs.additional_form_factor_params(ff_nm,{stg_nm:stg_label}):
                                    param_header = pop_id+'_'+pnm
                                    param_yml = os.path.join(stg_label_dir,pnm+'.yml')
                                    model_dict[sys_cls][pop_id][ff_nm][stg_nm][stg_label][pnm] = Regressor(param_header,param_yml)
    return model_dict

regression_models = load_regression_models(regression_models_dir)
classification_models = load_classification_models(classification_models_dir) 
test_regression_models = load_regression_models(test_regression_models_dir) 
test_classification_models = load_classification_models(test_classification_models_dir)

