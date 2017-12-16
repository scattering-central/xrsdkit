from collections import OrderedDict
import os

import numpy as np
import sklearn
from sklearn import preprocessing,linear_model
import yaml

from . import saxs_math

from citrination_client import CitrinationClient

class SaxsClassifier(object):
    """A set of classifiers to be used on SAXS spectra"""

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

    def classify(self, sample_params):
        """Apply self.models and self.scalers to sample_params.

        Parameters
        ----------
        sample_params : array
            array of floats representing features of test sample

        Returns
        -------
        flags : dict
            dictionary of (boolean,float) tuples,
            where the first item is the flag 
            and the second is the probability,
            for each of the potential scattering populations
        """ 
        flags = OrderedDict()
        if self.scalers['unidentified'] != None:
            x_bd = self.scalers['unidentified'].transform(sample_params)
            f_bd = self.models['unidentified'].predict(x_bd)[0]
            p_bd = self.models['unidentified'].predict_proba(x_bd)[0,int(f_bd)]
            flags['unidentified'] = (f_bd,p_bd)
        else:
            flags['unidentified'] = (None, None)
            # when we do not have a model for bad_data,
            # we can still try predictions for others labels
            f_bd = False 

        if not f_bd: 
            for k in self.models.keys():
                if not k == 'unidentified':
                    if self.scalers[k] != None:
                        xk = self.scalers[k].transform(sample_params)
                        fk = self.models[k].predict(xk)[0]
                        pk = self.models[k].predict_proba(xk)[0,int(fk)]
                        flags[k] = (fk,pk)
                    else:
                        flags[k] = (None, None)
        return flags

    def run_classifier(self, sample_params):

        flags = self.classify(np.array(list(sample_params.values())).reshape(1,-1))
        return flags


    def citrination_setup(self,address,api_key_file):
        """sets up a CitrinationClient
        Parameters
        ----------
        address : string
            address of Citrination page
        api_key_file : string
            path to the file with Citrination api_key
        """
        with open(api_key_file, "r") as g:
            api_key = g.readline()
        a_key = api_key.strip()

        self.client = CitrinationClient(site = address, api_key=a_key)


    def citrination_predict(self,sample_params):
        """Apply self.models and self.scalers to sample_params.

        Parameters
        ----------
        sample_params : ordered dictionary
            ordered dictionary of floats representing features of test sample

        Returns
        -------
        flags : dict
            dictionary of (boolean,float) tuples,
            where the first item is the flag
            and the second is the probability,
            for each of the potential scattering populations
        """
        if self.client == None:
            print("Client has not been set up")
            return None

        inputs = {}
        for k,v in sample_params.items():
            k = "Property " + k
            inputs[k] = v

        flags = OrderedDict()
        resp = self.client.predict("24", inputs) # "24" is ID of dataview on Citrination
        flags['unidentified'] = resp['candidates'][0]['Property unidentified']
        flags['guinier_porod'] = resp['candidates'][0]['Property guinier_porod']
        flags['spherical_normal'] = resp['candidates'][0]['Property spherical_normal']
        flags['diffraction_peaks'] = resp['candidates'][0]['Property diffraction_peaks']

        return flags
