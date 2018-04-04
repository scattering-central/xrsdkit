from collections import OrderedDict
from .structure_classifier import StructureClassifier
from xrsdkit.tools.piftools import cl_model_output_names


class Classifiers(object):
    """To create all possible or specified classifiers, train, and save them; make a prediction."""

    def __init__(self, cl_models = ['all']):

        if "all" in cl_models:
            my_cl_models = []
            my_cl_models.extend(cl_model_output_names)
            self.models = OrderedDict.fromkeys(my_cl_models)
        else:
            my_cl_models = cl_models
            self.models = OrderedDict.fromkeys(my_cl_models)

        for m in my_cl_models:
            self.models[m] = StructureClassifier(m)


    def train_classification_models(self, data, hyper_parameters_search=False):
        results = OrderedDict.fromkeys(self.models)
        data_diffuse_only = data[(data['diffuse_structure_flag']=="1") & (data['crystalline_structure_flag']== "0")]
        for k, v in self.models.items():
            if k in ['guinier_porod_population_count', 'spherical_normal_population_count']:
                results[k] = v.train(data_diffuse_only, hyper_parameters_search)
            else:
                results[k] = v.train(data, hyper_parameters_search)

        return results

    def save_classification_models(self, scalers_models, file_path=None):
        for k, v in self.models.items():
            v.save_models(scalers_models[k], file_path)


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
        list_of_wanted_models = list(self.models.keys())
        list_of_models_to_start = ['crystalline_structure_flag', 'diffuse_structure_flag']
        predictions = {}

        for k,v in self.models.items():
            if k in list_of_wanted_models and k in list_of_models_to_start:
                struct, cert = v.classify(sample_features)
                predictions[k] = [struct, cert]

        if predictions['crystalline_structure_flag'][0] == 0 and \
                        predictions['diffuse_structure_flag'][0] == 0:
            cert = predictions['crystalline_structure_flag'][1] * predictions['diffuse_structure_flag'][1]
            predictions['unidentified_structure_flag'] = [1, cert]
            return predictions

        if predictions['crystalline_structure_flag'] == 1:
            return predictions

        # we have diffuse pops only and can predict
        # spherical_normal_population_count and guinier_porod_population_count
        predictions['spherical_normal_population_count'] = \
            self.models['spherical_normal_population_count'].classify(sample_features)

        predictions['guinier_porod_population_count'] = \
            self.models['guinier_porod_population_count'].classify(sample_features)

        return predictions
