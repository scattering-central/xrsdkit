import re

import numpy as np

from . import regression_models, classification_models, test_regression_models, test_classification_models
from ..system import System
from .. import definitions as xrsdefs

def predict(features,test=False):
    """Estimate system identity and physical parameters, given a feature vector.

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

    # evaluate the system class
    sys_cls = classifiers['system_class'].classify(features)
    results['system_class'] = sys_cls 

    if sys_cls[0] == 'unidentified':
        return results

    cl_models_to_use = classifiers[sys_cls[0]]
    reg_models_to_use = regressors[sys_cls[0]]

    # evaluate the noise model
    if cl_models_to_use['noise_model'].trained:
        results['noise_model'] = cl_models_to_use['noise_model'].classify(features)
    else:
        results['noise_model'] = (cl_models_to_use['noise_model'].default_val, 0.0)

    # evaluate noise parameters
    nmodl = results['noise_model'][0]
    param_nms = list(xrsdefs.noise_params[nmodl].keys())
    # there is no model for I0, due to its arbitrary scale
    param_nms.pop(param_nms.index('I0'))
    for param_nm in param_nms+['I0_fraction']:
        if reg_models_to_use['noise'][nmodl][param_nm].trained:
            results['noise_'+param_nm] = reg_models_to_use['noise'][nmodl][param_nm].predict(features)
        else:
            results['noise_'+param_nm] = reg_models_to_use['noise'][nmodl][param_nm].default_val

    # evaluate population form factors and parameters
    for ipop, struct in enumerate(results['system_class'][0].split('__')):
        pop_id = 'pop{}'.format(ipop)
        if reg_models_to_use[pop_id]['I0_fraction'].trained:
            results[pop_id+'_I0_fraction'] = reg_models_to_use[pop_id]['I0_fraction'].predict(features)
        else:
            results[pop_id+'_I0_fraction'] = reg_models_to_use[pop_id]['I0_fraction'].predict(features)
        if cl_models_to_use[pop_id]['form'].trained:
            results[pop_id+'_form'] = cl_models_to_use[pop_id]['form'].classify(features)
        else:
            results[pop_id+'_form'] = (cl_models_to_use[pop_id]['form'].default_val, 0.0) 
        ff_nm = results[pop_id+'_form'][0]

        # evaluate any modelable settings for this structure 
        for stg_nm in xrsdefs.modelable_structure_settings[struct]:
            if cl_models_to_use[pop_id][stg_nm].trained:
                results[pop_id+'_'+stg_nm] = cl_models_to_use[pop_id][stg_nm].classify(features)
            else:
                results[pop_id+'_'+stg_nm] = (cl_models_to_use[pop_id][stg_nm].default_val, 0.0)
            stg_val = results[pop_id+'_'+stg_nm][0]
            
            # evaluate any additional parameters that depend on this setting
            for param_nm in xrsdefs.structure_params(struct,{stg_nm:stg_val}):
                if reg_models_to_use[pop_id][stg_nm][stg_val][param_nm].trained:
                    results[pop_id+'_'+param_nm] = \
                    reg_models_to_use[pop_id][stg_nm][stg_val][param_nm].predict(features)
                else:
                    results[pop_id+'_'+param_nm] = \
                    reg_models_to_use[pop_id][stg_nm][stg_val][param_nm].default_val

        # evaluate any modelable settings for this form factor 
        for stg_nm in xrsdefs.modelable_form_factor_settings[ff_nm]:
            if cl_models_to_use[pop_id][ff_nm][stg_nm].trained:
                results[pop_id+'_'+stg_nm] = cl_models_to_use[pop_id][ff_nm][stg_nm].classify(features)
            else:
                results[pop_id+'_'+stg_nm] = (cl_models_to_use[pop_id][ff_nm][stg_nm].default_val, 0.0)
            stg_val = results[pop_id+'_'+stg_nm][0]

            for param_nm in xrsdefs.additional_form_factor_params(ff_nm,{stg_nm:stg_val}):
                if reg_models_to_use[pop_id][ff_nm][stg_nm][stg_val][param_nm].trained:
                    results[pop_id+'_'+param_nm] = \
                    reg_models_to_use[pop_id][ff_nm][stg_nm][stg_val][param_nm].predict(features)
                else:
                    results[pop_id+'_'+param_nm] = \
                    reg_models_to_use[pop_id][ff_nm][stg_nm][stg_val][param_nm].default_val

    return results

def system_from_prediction(prediction,q,I,**kwargs):
    """Create a System object from output of predict() function.

    Keyword arguments are used to add metadata to the output System object.
    Supported keyword arguments: 'source_wavelength'

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
    new_sys : xrsdkit.system.System
        a System object built from the prediction dictionary
    """
    sys_cls = prediction['system_class'][0]
    if sys_cls == 'unidentified':
        return System()
    nmodl = prediction['noise_model'][0]
    noise_dict = {'model':nmodl,'parameters':{}}
    noise_dict['parameters']['I0'] = {'value':prediction['noise_I0_fraction']}
    if prediction['noise_I0_fraction'] < 0.:
        noise_dict['parameters']['I0']['value'] = 0. 
    for param_nm in xrsdefs.noise_params[nmodl]:
        if not param_nm == 'I0':
            param_header = 'noise_'+param_nm
            noise_dict['parameters'][param_nm] = {'value':prediction[param_header]}

    pops_dict = {}
    for ipop,struct in enumerate(sys_cls.split('__')):
        pop_id = 'pop{}'.format(ipop)
        form_header = pop_id+'_form'
        form = prediction[form_header][0]
        pop_dict = {'structure':struct,'form':form,'settings':{},'parameters':{}}
        pop_dict['parameters']['I0'] = {'value':prediction[pop_id+'_I0_fraction']}
        if prediction[pop_id+'_I0_fraction'] <= 0.:
            pop_dict['parameters']['I0']['value'] = 0
        else:
            for stg_nm in xrsdefs.modelable_structure_settings[struct]:
                stg_header = pop_id+'_'+stg_nm
                stg_val = prediction[stg_header][0]
                pop_dict['settings'][stg_nm] = stg_val 
                for param_nm,param_def in xrsdefs.structure_params(struct,{stg_nm:stg_val}).items():
                    param_header = pop_id+'_'+param_nm
                    param_val = prediction[param_header]
                    if param_def['bounds'][0] is not None:
                        if param_val < param_def['bounds'][0]: param_val = param_def['bounds'][0]
                    if param_def['bounds'][1] is not None:
                        if param_val > param_def['bounds'][1]: param_val = param_def['bounds'][1]
                    pop_dict['parameters'][param_nm] = {'value':param_val}
            for stg_nm in xrsdefs.modelable_form_factor_settings[form]:
                stg_header = pop_id+'_'+stg_nm
                stg_val = prediction[stg_header][0]
                pop_dict['settings'][stg_nm] = stg_val 
                for param_nm,param_def in xrsdefs.additional_form_factor_params(form,{stg_nm:stg_val}).items():
                    param_header = pop_id+'_'+param_nm
                    param_val = prediction[param_header]
                    if param_def['bounds'][0] is not None:
                        if param_val < param_def['bounds'][0]: param_val = param_def['bounds'][0]
                    if param_def['bounds'][1] is not None:
                        if param_val > param_def['bounds'][1]: param_val = param_def['bounds'][1]
                    pop_dict['parameters'][param_nm] = {'value':param_val}
            pops_dict[pop_id] = pop_dict

    kwargs.update(pops_dict)
    new_sys = System(noise=noise_dict,**pops_dict)

    # TODO: handle other kwargs for sample_metadata
    # TODO: System.features.update(feats)
    if 'source_wavelength' in kwargs:
        new_sys.update_from_dict({'sample_metadata':{'source_wavelength':kwargs['source_wavelength']}})

    Isum = np.sum(I)
    I_comp = new_sys.compute_intensity(q)
    Isum_comp = np.sum(I_comp)
    I_factor = Isum/Isum_comp
    new_sys.noise_model.parameters['I0']['value'] *= I_factor
    for pop_nm,pop in new_sys.populations.items():
        pop.parameters['I0']['value'] *= I_factor

    return new_sys 

