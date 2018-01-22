from collections import OrderedDict
import os

import numpy as np
import sklearn
from sklearn import preprocessing,linear_model
import yaml

from . import saxs_math

class SaxsClassifier(object):
    """A classifier to determine scatterer populations from SAXS spectra"""

    def __init__(self,yml_file=None):
        if yml_file is None:
            p = os.path.abspath(__file__)
            d = os.path.dirname(p)
            yml_file = os.path.join(d,'modeling_data','scalers_and_models.yml')

        s_and_m_file = open(yml_file,'rb')
        s_and_m = yaml.load(s_and_m_file)

        # dict of classification model parameters
        classifier_dict = s_and_m['models']
        # dict of scaler parameters
        scalers_dict = s_and_m['scalers'] 

        self.models = OrderedDict.fromkeys(saxs_math.population_keys)
        self.scalers = OrderedDict.fromkeys(saxs_math.population_keys)
        for model_name in saxs_math.population_keys:
            model_params = classifier_dict[model_name]
            scaler_params = scalers_dict[model_name] 
            if scaler_params is not None:
                s = preprocessing.StandardScaler()
                self.set_param(s,scaler_params)
                m = linear_model.SGDClassifier()
                self.set_param(m,model_params)
            self.models[model_name] = m
            self.scalers[model_name] = s

    # helper function - to set parametrs for scalers and models
    def set_param(self, m_s, param):
        for k, v in param.items():
            if isinstance(v, list):
                setattr(m_s, k, np.array(v))
            else:
                setattr(m_s, k, v)

    def classify(self, sample_features):
        """Classify a sample from its features dict.

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of saxs_math.profile_spectrum()

        Returns
        -------
        populations : dict
            dictionary of integers 
            counting predicted scatterer populations
            for all populations in saxs_math.population_keys.
        certainties : dict
            dictionary, similar to `populations`,
            but containing the certainty of the prediction
        """
        feature_array = np.array(list(sample_features.values())).reshape(1,-1)  

        populations = OrderedDict()
        certainties = OrderedDict()

        x = self.scalers['unidentified'].transform(feature_array)
        pop = self.models['unidentified'].predict(x)[0]
        cert = self.models['unidentified'].predict_proba(x)[0,int(pop)]
        populations['unidentified'] = pop 
        certainties['unidentified'] = cert 

        if not populations['unidentified']: 
            for k in saxs_math.population_keys:
                if not k == 'unidentified':
                    x = self.scalers[k].transform(feature_array)
                    pop = self.models[k].predict(x)[0]
                    cert = self.models[k].predict_proba(x)[0,int(pop)]
                    populations[k] = pop 
                    certainties[k] = cert 

        return populations, certainties

