from collections import OrderedDict
import os

import numpy as np
from sklearn import preprocessing,linear_model
import yaml

from .. import structure_names
from . import set_param
from ..tools.piftools import model_output_names

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

        self.models = OrderedDict.fromkeys(model_output_names)
        self.scalers = OrderedDict.fromkeys(model_output_names)
        self.cv_error = OrderedDict.fromkeys(model_output_names)

        for struct_name in model_output_names:
            model_params = classifier_dict[struct_name]
            scaler_params = scalers_dict[struct_name]
            acc = acc_dict[struct_name]
            if scaler_params is not None:
                s = preprocessing.StandardScaler()
                set_param(s,scaler_params)
                m = linear_model.SGDClassifier()
                set_param(m,model_params)
                self.models[struct_name] = m
                self.scalers[struct_name] = s
                self.cv_error[struct_name] = acc

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

        structs = OrderedDict.fromkeys(model_output_names)
        certainties = OrderedDict.fromkeys(model_output_names)

        for struct_name in model_output_names:
            structs[struct_name] = False
            certainties[struct_name] = 0.

        for k in model_output_names:
            if self.scalers[k] is not None:
                x = self.scalers[k].transform(feature_array)
                pop = int(self.models[k].predict(x)[0])
                cert = self.models[k].predict_proba(x)[0,pop]
                structs[k] = pop
                certainties[k] = cert


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

        return structs, certainties

    def training_cv_error(self):
        """Report cross-validation error for these classification models.

        "Leave-2-Groups-Out" cross-validation is used.
        For each train-test split, 
        two experiments are used for testing 
        and the rest are used for training. 
        The reported error is the average over all train-test splits. 
        TODO: what is the error metric for the classifier? 
        TODO: verify that this docstring is correct

        Returns
        -------
        cv_errors : dict
            Dictionary of models and their cross-validation errors. 
        """
        return self.cv_error
