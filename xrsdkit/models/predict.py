import numpy as np

from . import get_regression_models, get_classification_models
from ..system import System
from .. import definitions as xrsdefs

def predict(features, system_class=None, noise_model=None):
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
    results = {}

    # TODO: unidentified systems should still have models
    # for predicting the noise model and parameters
    if system_class:
        system_class = (system_class,  None)
    else:
        system_class =  predict_system_class(features)
    results['system_class'] = system_class
    sys_cls = system_class[0]
    if sys_cls == 'unidentified':
        return results

    noise_model, noise_params = predict_noise(features, sys_cls, noise_model)

    # evaluate the noise model
    results['noise_model'] = noise_model
    results.update(noise_params)

    # evaluate population form factors
    form_factors = predict_form_factors(features, sys_cls)
    results.update(form_factors)

    # evaluate settings
    settings = predict_settigs(features, sys_cls,form_factors)
    results.update(settings)

    # evaluate parameters for all populations
    parameters = predict_parameters(features, sys_cls, form_factors, settings)
    results.update(parameters)

    return results

def predict_system_class(features):
    """Predict system class, given a feature vector.

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
    classifiers = get_classification_models()

    # use the main classifiers to evaluate the system class
    if 'main_classifiers' in classifiers \
    and all([s+'_binary' in classifiers['main_classifiers'] for s in xrsdefs.structure_names]):
        main_cls = classifiers['main_classifiers']
        sys_cls = ''
        flagged_structures = ''
        certainties = {}
        for struct_nm in xrsdefs.structure_names:
            model_id = struct_nm+'_binary'
            #if model_id in main_cls:
            struct_result = main_cls[model_id].classify(features)
            certainties[model_id] = struct_result[1]
            if struct_result[0]:
                if flagged_structures: flagged_structures += '__'
                flagged_structures += struct_nm
        if flagged_structures in main_cls:
            sys_cls_result = main_cls[flagged_structures].classify(features)
            sys_cls = sys_cls_result[0]
            certainties['system_class'] = sys_cls_result[1]
        else:
            sys_cls = 'unidentified'
    else:
        raise RuntimeError('attempted to predict() before creating main classifiers')

    return (sys_cls, certainties)

def predict_noise(features, sys_cls, noise_m):
    """Predict type of noise and parameters for it.

    Parameters
    ----------
    features : OrderedDict
        OrderedDict of features with their values,
        similar to output of xrsdkit.tools.profiler.profile_pattern()
    sys_cls : String
        system class
    noise_m : String
        type of noise

    Returns
    -------
    noise_model : (String, float)
        type of noise and predicted probability
    noise_params : dict
        dictionary with predicted parameters
    """

    classifiers = get_classification_models()
    regressors = get_regression_models()

    cl_models_to_use = classifiers[sys_cls]
    reg_models_to_use = regressors[sys_cls]

    if noise_m is None:
        noise_model = cl_models_to_use['noise_model'].classify(features)
    else:
        noise_model = (noise_m, None)

    # evaluate noise parameters
    nmodl = noise_model[0]
    param_nms = list(xrsdefs.noise_params[nmodl].keys())
    # there is no model for I0, due to its arbitrary scale
    param_nms.pop(param_nms.index('I0'))
    noise_params = {}
    for param_nm in param_nms+['I0_fraction']:
        noise_params['noise_'+param_nm] = reg_models_to_use['noise'][nmodl][param_nm].predict(features)

    return noise_model, noise_params

def predict_form_factors(features, sys_cl):
    """Predict form factor for each population.

    Parameters
    ----------
    features : OrderedDict
        OrderedDict of features with their values,
        similar to output of xrsdkit.tools.profiler.profile_pattern()
    sys_cls : String
        system class

    Returns
    -------
    form_factors : dict
        dictionary with predicted form factors
    """

    classifiers = get_classification_models()

    # evaluate population form factors
    cl_models_to_use = classifiers[sys_cl]
    form_factors = {}
    for ipop, struct in enumerate(sys_cl.split('__')):
        pop_id = 'pop{}'.format(ipop)
        form_factors[pop_id+'_form'] = cl_models_to_use[pop_id]['form'].classify(features)
    return form_factors

def predict_settigs(features, sys_cl, form_factors):
    """Predict form factor for each population.

    Parameters
    ----------
    features : OrderedDict
        OrderedDict of features with their values,
        similar to output of xrsdkit.tools.profiler.profile_pattern()
    sys_cls : String
        system class
    form_factors : dict
        dictionary with form factors for each population
    Returns
    -------
    settings : dict
        dictionary with settings
    """
    classifiers = get_classification_models()
    cl_models_to_use = classifiers[sys_cl]
    settings = {}
    for ipop, struct in enumerate(sys_cl.split('__')):
        pop_id = 'pop{}'.format(ipop)
        ff_nm = form_factors[pop_id+'_form'][0]

        # evaluate any modelable settings for this structure
        for stg_nm in xrsdefs.modelable_structure_settings[struct]:
            settings[pop_id+'_'+stg_nm] = cl_models_to_use[pop_id][stg_nm].classify(features)

        # evaluate any modelable settings for this form factor
        for stg_nm in xrsdefs.modelable_form_factor_settings[ff_nm]:
            settings[pop_id+'_'+stg_nm] = cl_models_to_use[pop_id][ff_nm][stg_nm].classify(features)
    return settings

def predict_parameters(features, sys_cls, form_factors, settings):
    """Predict form factor for each population.

    Parameters
    ----------
    features : OrderedDict
        OrderedDict of features with their values,
        similar to output of xrsdkit.tools.profiler.profile_pattern()
    sys_cls : String
        system class
    form_factors : dict
        dictionary with form factors for each population
    settings : dict
        dictionary with settings
    Returns
    -------
    parameters : dict
        dictionary with values for each parameter
    """
    regressors = get_regression_models()
    reg_models_to_use = regressors[sys_cls]
    parameters = {}
    for ipop, struct in enumerate(sys_cls.split('__')):
        pop_id = 'pop{}'.format(ipop)
        ff_nm = form_factors[pop_id+'_form'][0]

        # evaluate I0_fraction
        parameters[pop_id+'_I0_fraction'] = reg_models_to_use[pop_id]['I0_fraction'].predict(features)

        # evaluate form factor parameters
        for param_nm,param_default in xrsdefs.form_factor_params[ff_nm].items():
            parameters[pop_id+'_'+param_nm] = reg_models_to_use[pop_id][ff_nm][param_nm].predict(features)

        # take each setting
        for stg_nm in xrsdefs.modelable_structure_settings[struct]:
            stg_val = settings[pop_id+'_'+stg_nm][0]

            # evaluate any additional parameters that depend on this setting
            for param_nm in xrsdefs.structure_params(struct,{stg_nm:stg_val}):
                parameters[pop_id+'_'+param_nm] = \
                reg_models_to_use[pop_id][stg_nm][stg_val][param_nm].predict(features)

        # take each form factor setting
        for stg_nm in xrsdefs.modelable_form_factor_settings[ff_nm]:
            stg_val = settings[pop_id+'_'+stg_nm][0]

            # evaluate any additional parameters that depend on this setting
            for param_nm in xrsdefs.additional_form_factor_params(ff_nm,{stg_nm:stg_val}):
                parameters[pop_id+'_'+param_nm] = \
                reg_models_to_use[pop_id][ff_nm][stg_nm][stg_val][param_nm].predict(features)
    return parameters


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
    # create System object
    sys_cls = prediction['system_class'][0]
    if sys_cls == 'unidentified':
        return System(**kwargs)

    # create noise model
    nmodl = prediction['noise_model'][0]
    noise_dict = {'model':nmodl,'parameters':{}}
    noise_dict['parameters']['I0'] = {'value':prediction['noise_I0_fraction']}
    if prediction['noise_I0_fraction'] < 0.:
        noise_dict['parameters']['I0']['value'] = 0. 
    for param_nm in xrsdefs.noise_params[nmodl]:
        if not param_nm == 'I0':
            param_header = 'noise_'+param_nm
            noise_dict['parameters'][param_nm] = {'value':prediction[param_header]}

    # create populations
    pops_dict = {}
    for ipop,struct in enumerate(sys_cls.split('__')):
        pop_id = 'pop{}'.format(ipop)
        form_header = pop_id+'_form'
        form = prediction[form_header][0]
        pop_dict = {'structure':struct,'form':form,'settings':{},'parameters':{}}
        pop_dict['parameters']['I0'] = {'value':prediction[pop_id+'_I0_fraction']}
        if prediction[pop_id+'_I0_fraction'] <= 0.:
            pop_dict['parameters']['I0']['value'] = 0
        # set form factor parameters
        for param_nm,param_def in xrsdefs.form_factor_params[form].items():
            param_header = pop_id+'_'+param_nm
            param_val = prediction[param_header]
            if param_def['bounds'][0] is not None:
                if param_val < param_def['bounds'][0]: param_val = param_def['bounds'][0]
            if param_def['bounds'][1] is not None:
                if param_val > param_def['bounds'][1]: param_val = param_def['bounds'][1]
            pop_dict['parameters'][param_nm] = {'value':param_val}
        # set structure settings
        for stg_nm in xrsdefs.modelable_structure_settings[struct]:
            stg_header = pop_id+'_'+stg_nm
            stg_val = prediction[stg_header][0]
            pop_dict['settings'][stg_nm] = stg_val 
            # set any parameters that depend on the setting
            for param_nm,param_def in xrsdefs.structure_params(struct,{stg_nm:stg_val}).items():
                param_header = pop_id+'_'+param_nm
                param_val = prediction[param_header]
                if param_def['bounds'][0] is not None:
                    if param_val < param_def['bounds'][0]: param_val = param_def['bounds'][0]
                if param_def['bounds'][1] is not None:
                    if param_val > param_def['bounds'][1]: param_val = param_def['bounds'][1]
                pop_dict['parameters'][param_nm] = {'value':param_val}
        # set form factor settings
        for stg_nm in xrsdefs.modelable_form_factor_settings[form]:
            stg_header = pop_id+'_'+stg_nm
            stg_val = prediction[stg_header][0]
            pop_dict['settings'][stg_nm] = stg_val 
            # set any parameters that depend on the setting
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
    new_sys = System(noise=noise_dict,**kwargs)

    # TODO: handle other kwargs for sample_metadata
    # TODO: System.features.update(feats)
    #if 'source_wavelength' in kwargs:
    #    new_sys.update_from_dict({'sample_metadata':{'source_wavelength':kwargs['source_wavelength']}})

    Isum = np.sum(I)
    I_comp = new_sys.compute_intensity(q)
    Isum_comp = np.sum(I_comp)
    I_factor = Isum/Isum_comp
    new_sys.noise_model.parameters['I0']['value'] *= I_factor
    for pop_nm,pop in new_sys.populations.items():
        pop.parameters['I0']['value'] *= I_factor

    return new_sys 

