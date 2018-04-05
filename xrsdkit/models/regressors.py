from collections import OrderedDict
import numpy as np
from .regressor import Regressor
from xrsdkit.tools.piftools import reg_model_output_names



class Regressors(object):
    """To create all possible or specified classifiers, train, and save them; make a prediction."""

    def __init__(self, reg_models = ['all']):

        if "all" in reg_models:
            my_reg_models = []
            my_reg_models.extend(reg_model_output_names)
            self.models = OrderedDict.fromkeys(my_reg_models)
        else:
            my_reg_models = reg_models
            self.models = OrderedDict.fromkeys(my_reg_models)

        for m in my_reg_models:
            self.models[m] = Regressor(m)


    def train_regression_models(self, data, hyper_parameters_search=False):
        #results = OrderedDict.fromkeys(self.models)
        results = OrderedDict.fromkeys(list(self.models.keys()))
        for k, v in self.models.items():
            results[k] = v.train(data, hyper_parameters_search)

        return results

    def save_regression_models(self, scalers_models, file_path=None):
        for k, v in self.models.items():
            v.save_models(scalers_models[k], file_path)


    def make_predictions(self, sample_features, populations, q_I):
        """Determine the types of structures represented by the sample

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of saxs_math.profile_spectrum()
        populations : dict
            dictionary counting scatterer populations,
            similar to output of SaxsClassifier.classify()
        q_I : array
            n-by-2 array of scattering vector (1/Angstrom) and intensities.

        Returns
        -------
        prediction : dict
            dictionary of with predicted parameters
        """
        predictions = {}

        for k,v in self.models.items():
            pr = v.predict(sample_features, populations, q_I)
            if pr:
                predictions[k] = pr

        return predictions
