import numpy as np
from collections import OrderedDict
from .structure_classifier import StructureClassifier
from .regressor import Regressor
from xrsdkit.tools.piftools import model_output_names


class Predictors(object):
    """Models for classifying structure from scattering/diffraction data"""

    def __init__(self, cl_models = ['all'], reg_models = ['all'], hyper_parameters_search=False):

        self.hyper_parameters_search = hyper_parameters_search
        self.models = {"classifiers" : None, "regressors" : None}

        if "all" in cl_models:
            my_cl_models = []
            my_cl_models.extend(model_output_names)
            self.models["classifiers"] = OrderedDict.fromkeys(my_cl_models)
        else:
            my_cl_models = cl_models
            self.models["classifiers"] = OrderedDict.fromkeys(my_cl_models)

        for m in my_cl_models:
            self.models["classifiers"][m] = StructureClassifier(m)

        if "all" in reg_models:
            my_reg_models = []
            my_reg_models.extend([])# TODO list of possible regressors
            self.models["regressors"] = OrderedDict.fromkeys(my_reg_models)
        else:
            my_reg_models = reg_models
            self.models["regressors"] = OrderedDict.fromkeys(my_reg_models)

        for m in my_cl_models:
            self.models["classifiers"][m] = StructureClassifier(m)

        for m in my_reg_models:
            self.models["regressors"][m] = Regressor(m)



    def train_predictiors(self, data):
        data_diffuse_only = data[(data['diffuse_structure_flag']=="1") & (data['crystalline_structure_flag']!= "1")]
        for k, v in self.models["classifiers"].items():
            if k in ['guinier_porod_population_count', 'spherical_normal_population_count']:
                v.train(data_diffuse_only, self.hyper_parameters_search)
            else:
                v.train(data, self.hyper_parameters_search)

        # about the same for regressors:
        # for k, v in self.models["regressors"].items():
        #        ....


    def make_predictions(self, sample_features):
        """Determine the types of structures represented by the sample

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of saxs_math.profile_spectrum()

        Returns
        -------
        prediction : dict
            dictionary of predictions
            from classifiers and regressors.
            Predictions from classifiers also
            have certainty of the prediction
        """

        predictions = {'classifications' : None, 'regressions' : None}

        for k,v in self.models['classifiers'].item():
            predictions['classifications'][k] = v.classify(sample_features)

        #for k,v in self.models['regressors'].item():
            #predictions['regressions'][k] = v.classify(sample_features)

        return predictions
