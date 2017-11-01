from collections import OrderedDict

import yaml
import sklearn
from sklearn import preprocessing,linear_model
from collections import OrderedDict
import os

from ... import pawstools

class SaxsClassifier(object):
    """A container for a set of classifiers to be used on SAXS spectra"""

    def __init__(self,yml_file=None):
        if yml_file is None:
            p = os.path.abspath(__file__)
            # p = (pawsroot)/paws/core/tools/saxs/SaxsClassifier.py
            d = os.path.dirname(p)
            # d = (pawsroot)/paws/core/tools/saxs/
            yml_file = os.path.join(d,'modeling_data','scalers_and_models.yml')

        s_and_m_file = open(yml_file,'rb')
        s_and_m = yaml.load(s_and_m_file)

        sk_version = s_and_m['version']
        cur_version = list(map(int,sklearn.__version__.split('.')))
        major = cur_version[0]
        minor = cur_version[1]

        if (major != sk_version[0] or minor != sk_version[1]):
            version_str = ".".join(map(str,sk_version))
            msg = 'found scikit-learn v{}.{}.x. '\
                'SAXS classification models were built '\
                'with scikit-learn v{}. '\
                'Please try again with a matching version of scikit-learn.'\
                .format(major,minor,version_str) 
            raise RuntimeError(msg)

        scalers_dict = s_and_m['scalers'] # dict of scalers parametrs
        classifier_dict = s_and_m['models'] # dict of models parametrs

        model_param_bad_data = classifier_dict['bad_data']
        model_param_form = classifier_dict['form_factor_scattering']
        model_param_precur = classifier_dict['precursor_scattering']
        model_param_diff_peaks = classifier_dict['diffraction_peaks']

        scaler_param_bad_data = scalers_dict['bad_data']
        scaler_param_form = scalers_dict['form_factor_scattering']
        scaler_param_precur = scalers_dict['precursor_scattering']
        scaler_param_diff_peaks = scalers_dict['diffraction_peaks']

        # recreate the scalers
        scaler_bad_data = preprocessing.StandardScaler()
        self.set_param( scaler_bad_data, scaler_param_bad_data)

        scaler_form = preprocessing.StandardScaler()
        self.set_param( scaler_form, scaler_param_form)

        scaler_precur = preprocessing.StandardScaler()
        self.set_param( scaler_precur, scaler_param_precur)

        scaler_diff_peaks = preprocessing.StandardScaler()
        self.set_param( scaler_diff_peaks, scaler_param_diff_peaks)

        # recreate the models
        model_bad_data = linear_model.SGDClassifier()
        self.set_param( model_bad_data, model_param_bad_data)

        model_form = linear_model.SGDClassifier()
        self.set_param( model_form, model_param_form)

        model_precur = linear_model.SGDClassifier()
        self.set_param( model_precur, model_param_precur)

        model_diff_peaks = linear_model.SGDClassifier()
        self.set_param( model_diff_peaks, model_param_diff_peaks)

        # save the dicts of classifiers and scalers as outputs
        self.models = {'bad_data': model_bad_data,
                       'form_factor_scattering': model_form,
                       'precursor_scattering': model_precur,
                       'diffraction_peaks': model_diff_peaks}
        self.scalers = {'bad_data': scaler_bad_data,
                       'form_factor_scattering': scaler_form,
                       'precursor_scattering': scaler_precur,
                       'diffraction_peaks': scaler_diff_peaks}

    # helper function - to set parametrs for scalers and models
    def set_param(self, m_s, param):
        for k, v in param.items():
            if isinstance(v, list):
                setattr(m_s, k, np.array(v))
            else:
                setattr(m_s, k, v)

    def classify(self, sample_params):
        """Apply self.models and self.scalers to sample_params.

        Parameters
        ----------
        sample_params : array
            array of floats representing features of test sample

        Returns
        -------
        flags : dict
            dictionary of boolean flags indicating sample populations
        """ 
        flags = OrderedDict()
        x_bd = self.scalers['bad_data'].transform(sample_params)
        f_bd = self.models['bad_data'].predict(x_bd)[0]
        p_bd = self.models['bad_data'].predict_proba(x_bd)[0,int(f_bd)]
        flags['bad_data'] = (f_bd,p_bd)
        # NOTE: this is temporary, until new models have been built
        #flags['bad_data'] = (True,1.)
        if not f_bd: 
            for k in self.models.keys():
                if not k == 'bad_data':
                    xk = self.scalers[k].transform(sample_params)
                    fk = self.models[k].predict(xk)
                    pk = self.models[k].predict_proba(xk)[0,int(fk)]
                    flags[k] = (fk,pk)
        return flags

        #model_bad_data = clsfrs['bad_data']
        #model_form = clsfrs['form_factor_scattering']
        #model_precur = clsfrs['precursor_scattering']
        #model_diff_peaks = clsfrs['diffraction_peaks']

        #scaler_bad_data = sclrs['bad_data']
        #scaler_form = sclrs['form_factor_scattering']
        #scaler_precur = sclrs['precursor_scattering']
        #scaler_diff_peaks = sclrs['diffraction_peaks']

        ### Classify the spectrum
        # (1) Apply model to input

        # transform the ordered dictionary into an array of features:
        #bin_strengths = x['q_bin_strengths']
        #del x['q_bin_strengths']

        #edges = x['q_bin_edges']
        #del x['q_bin_edges']

        # we can use edges instead to indexes, but it could cause a problem
        # when not all samples have a specific edge
        #for i in range(len(edges)):
        #    x[str(edges[i])] = bin_strengths[i]

        # should we do it in profile_spectrum()?
        #for i in range(100):
        #    x[str(i)] = bin_strengths[i]

        # features for bad_data, precursor, and structure labels
        #features_analytical_and_60 = ['q_Imax', 'Imax_over_Imean', 'Imax_over_Ilowq',
        #'Imax_over_Ihighq', 'Imax_sharpness', 'low_q_ratio', 'high_q_ratio',
        #'log_fluctuation','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
        #'14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
        #'26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
        #'38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
        #'50', '51', '52', '53', '54', '55', '56', '57', '58', '59' ]

        # features for form label
        #features60 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
        #'14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
        #'26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
        #'38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
        #'50', '51', '52', '53', '54', '55', '56', '57', '58', '59']

        #input_features1 = [] # analytical and 60 bins
        #for item in features_analytical_and_60:
        #    input_features1.append(x[item])

        #input_features2 = [] # 60 bins
        #for item in features60:
        #    input_features2.append(x[item])

        #transformed_input_features = scaler_bad_data.transform([input_features1])
        #bad_data = model_bad_data.predict(transformed_input_features)[0] # [0] since we have only one sample
        #if bad_data:
        #    pr_bad_data = model_bad_data.predict_proba(transformed_input_features)[0, 1]
        #else:
        #    pr_bad_data = model_bad_data.predict_proba(transformed_input_features)[0, 0]

        #flags['bad_data'] = (bad_data, pr_bad_data) # label and propability to have this label

        #if bad_data == True:
        #    flags['precursor_scattering'] = (False, None)
        #    flags['form_factor_scattering'] = (False, None)
        #    flags['diffraction_peaks'] = (False, None)
        #else:
        #    #form label
        #    transformed_input_features = scaler_form.transform([input_features2])
        #    form = model_form.predict(transformed_input_features)[0]
        #    if form:
        #        pr_form = model_form.predict_proba(transformed_input_features)[0, 1]
        #    else:
        #        pr_form = model_form.predict_proba(transformed_input_features)[0, 0]
        #    flags['form_factor_scattering'] = (form, pr_form) # label and propability to have this label

        #    # precursor label
        #    transformed_input_features = scaler_precur.transform([input_features1])
        #    prec = model_precur.predict(transformed_input_features)[0]
        #    if prec:
        #        pr_prec = model_precur.predict_proba(transformed_input_features)[0, 1]
        #    else:
        #        pr_prec = model_precur.predict_proba(transformed_input_features)[0, 0]
        #    flags['precursor_scattering'] = (prec, pr_prec)

        #    # difraction peaks label
        #    transformed_input_features = scaler_diff_peaks.transform([input_features1])
        #    picks = model_diff_peaks.predict(transformed_input_features)[0]
        #    if picks:
        #        pr_picks = model_diff_peaks.predict_proba(transformed_input_features)[0, 1]
        #    else:
        #        pr_picks = model_diff_peaks.predict_proba(transformed_input_features)[0, 0]
        #    flags['diffraction_peaks'] = (picks, pr_picks)

        # problems for later:
        # if flags['form_factor_scattering']:
        #     # classify the form factor based on inputs  
        #     flags['form_factor_id'] = ''     
        #
        # if flags['diffraction_peaks']:
        #     # classify the space group based on inputs
        #     flags['space_group_id'] = ''
        #

        #self.outputs['population_flags'] = flags 




