from collections import OrderedDict
import os

import numpy as np
import sklearn
from sklearn import preprocessing,linear_model
import yaml

from .. import structure_names

class StructureClassifier(object):
    """Models for classifying structure from scattering/diffraction data"""

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
        # dict of accuracies
        acc_dict = s_and_m['accuracy']

        self.models = OrderedDict.fromkeys(structure_names)
        self.scalers = OrderedDict.fromkeys(structure_names)
        self.accuracy = OrderedDict.fromkeys(structure_names)
        for struct_name in structure_names:
            model_params = classifier_dict[struct_name]
            scaler_params = scalers_dict[struct_name]
            acc = acc_dict[struct_name]
            if scaler_params is not None:
                s = preprocessing.StandardScaler()
                self.set_param(s,scaler_params)
                m = linear_model.SGDClassifier()
                self.set_param(m,model_params)
            self.models[struct_name] = m
            self.scalers[struct_name] = s
            self.accuracy[struct_name] = acc

    # helper function - to set parametrs for scalers and models
    def set_param(self, m_s, param):
        for k, v in param.items():
            if isinstance(v, list):
                setattr(m_s, k, np.array(v))
            else:
                setattr(m_s, k, v)

    def classify(self, sample_features):
        """Determine the types of structures represented by the sample

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of saxs_math.profile_spectrum()

        Returns
        -------
        structure_flags : dict
            dictionary of booleans inidicating whether or not 
            the sample exhibits various structures 
        certainties : dict
            dictionary, similar to `structure_flags`,
            but containing the certainty of the prediction
        """
        feature_array = np.array(list(sample_features.values())).reshape(1,-1)  

        structs = OrderedDict.fromkeys(structure_names)
        certainties = OrderedDict.fromkeys(structure_names)

        for struct_name in structure_names:
            structs[struct_name] = False
            certainties[struct_name] = 0. 

        #x = self.scalers['unidentified'].transform(feature_array)
        #pop = int(self.models['unidentified'].predict(x)[0])
        #cert = self.models['unidentified'].predict_proba(x)[0,pop]
        #populations['unidentified'] = pop 
        #certainties['unidentified'] = cert 

        #if not populations['unidentified']: 
        #    for k in population_keys:
        #        if not k == 'unidentified':
        #            x = self.scalers[k].transform(feature_array)
        #            pop = int(self.models[k].predict(x)[0])
        #            cert = self.models[k].predict_proba(x)[0,pop]
        #            populations[k] = pop 
        #            certainties[k] = cert 

        return populations, certainties

    def get_accuracy(self):
        """Get accuracy for all classification models.

        Returns
        -------
        accuracy : dict
            Dictionary of models and their accuracies.
            to calculate the accuracy "Leave-N-Groups-Out" technique is used.
            Every cycle data from two experiments used for testing and the
            other data for training. The average accuracy is reported.
        """
        return self.accuracy
