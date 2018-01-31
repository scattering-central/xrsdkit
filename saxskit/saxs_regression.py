import os
import copy
from collections import OrderedDict

import yaml
import numpy as np
from sklearn import preprocessing,linear_model

from . import saxs_math, saxs_fit
from . import parameter_keys, all_parameter_keys 
from . import peak_math, peak_finder

class SaxsRegressor(object):
    """A set of regression models to be used on SAXS spectra"""

    def __init__(self,yml_file=None):
        if yml_file is None:
            p = os.path.abspath(__file__)
            d = os.path.dirname(p)
            yml_file = os.path.join(d,'modeling_data','scalers_and_models_regression.yml')

        s_and_m_file = open(yml_file,'rb')
        s_and_m = yaml.load(s_and_m_file)

        reg_models_dict = s_and_m['models']
        scalers_dict = s_and_m['scalers']

        self.models = OrderedDict.fromkeys(all_parameter_keys)
        self.scalers = OrderedDict.fromkeys(all_parameter_keys)
        reg_models = reg_models_dict.keys()
        for model_name in reg_models:
            model_params = reg_models_dict[model_name]
            scaler_params = scalers_dict[model_name]
            if scaler_params is not None:
                s = preprocessing.StandardScaler()
                self.set_param(s,scaler_params)
                m = linear_model.SGDRegressor()
                self.set_param(m,model_params)
            self.models[model_name] = m
            self.scalers[model_name] = s

    # helper function - to set parameters for scalers and models
    def set_param(self, m_s, param):
        for k, v in param.items():
            if isinstance(v, list):
                setattr(m_s, k, np.array(v))
            else:
                setattr(m_s, k, v)

    def predict_params(self,populations,features,q_I):
        """Evaluate the scattering parameters of a sample.

        Parameters
        ----------
        populations : dict
            dictionary counting scatterer populations,
            similar to output of SaxsClassifier.classify()
        features : dict
            dictionary of sample numerical features,
            similar to output of saxs_math.profile_spectrum().
        q_I : array 
            n-by-2 array of scattering vector (1/Angstrom) and intensities. 

        Returns
        -------
        params : dict
            dictionary of with predicted parameters
        """
        feature_array = np.array(list(features.values())).reshape(1,-1)

        params = OrderedDict()    
        fixed_params = OrderedDict()    
        if bool(populations['unidentified']):
            return params 

        if bool(populations['spherical_normal']):
            x = self.scalers['r0_sphere'].transform(feature_array)
            r0sph = self.models['r0_sphere'].predict(x)
            params['r0_sphere'] = [float(r0sph[0])]
            additional_features = saxs_math.spherical_normal_profile(q_I)
            ss_features = np.append(feature_array, np.array(list(additional_features.values()))).reshape(1,-1)
            x = self.scalers['sigma_sphere'].transform(ss_features)
            sigsph = self.models['sigma_sphere'].predict(x)
            params['sigma_sphere'] = [float(sigsph[0])]

        if bool(populations['guinier_porod']):
            additional_features = saxs_math.guinier_porod_profile(q_I)
            rg_features = np.append(feature_array, np.array(list(additional_features.values()))).reshape(1,-1)
            x = self.scalers['rg_gp'].transform(rg_features)
            rg = self.models['rg_gp'].predict(x)
            params['rg_gp'] = [float(rg[0])]
            # TODO: add a model for the porod exponent.
            params['D_gp'] = [float(saxs_fit.param_defaults['D_gp'])]

        return params

