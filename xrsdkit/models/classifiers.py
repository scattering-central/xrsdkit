from collections import OrderedDict
from .structure_classifier import StructureClassifier
from xrsdkit.tools.piftools import cl_model_output_names


class Classifiers(object):
    """To create all possible or specified classifiers, train, and save them; make a prediction."""

    def __init__(self):

        self.models = OrderedDict.fromkeys(cl_model_output_names)
        for k,v in self.models.items():
            self.models[k] = StructureClassifier(k)


    def train_classification_models(self, data, hyper_parameters_search=False,
                                     cl_models = ['all'], testing_data = None, partial = False):

        if "all" in cl_models:
            results = OrderedDict.fromkeys(cl_model_output_names)
        else:
            results = OrderedDict.fromkeys(cl_models)

        data_diffuse_only = data[(data['diffuse_structure_flag']=="1")
                                 & (data['crystalline_structure_flag']== "0")]

        test_diffuse_only = None
        if testing_data is not None:
            test_diffuse_only = testing_data[(testing_data['diffuse_structure_flag']=="1")
                                             &(testing_data['crystalline_structure_flag']== "0")]

        for k, v in results.items():
            if k == 'unidentified_structure_flag': # we do not build a model for this label
                continue
            if k in ['guinier_porod_population_count', 'spherical_normal_population_count']:
                if partial:
                    results[k] = self.models[k].train_partial(data_diffuse_only, test_diffuse_only)
                else:
                    results[k] = self.models[k].train(data_diffuse_only, hyper_parameters_search)
            else:
                if partial:
                    results[k] = self.models[k].train_partial(data, testing_data)
                else:
                    results[k] = self.models[k].train(data, hyper_parameters_search)

        return results

    def print_training_results(self, results):
        for k, v in results.items():
            print(k, ":")
            print("accuracy :  %10.3f" % (results[k]['accuracy']))
            print("parameters :", results[k]['parameters'])
            print()


    def save_classification_models(self, scalers_models, file_path=None):

        for k, v in scalers_models.items():
            if k == 'unidentified_structure_flag': # we do not build a model for this label
                continue
            self.models[k].save_models(scalers_models[k], file_path)


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
        #predictions = OrderedDict.fromkeys(list(self.models.keys()))
        predictions = {}

        for k,v in self.models.items():
            if k in list_of_wanted_models and k in list_of_models_to_start:
                struct, cert = v.classify(sample_features)
                predictions[k] = [struct, cert]

        cert = predictions['crystalline_structure_flag'][1] + predictions['diffuse_structure_flag'][1]  - \
               predictions['crystalline_structure_flag'][1] * predictions['diffuse_structure_flag'][1]
        if predictions['crystalline_structure_flag'][0] == 0 and \
                        predictions['diffuse_structure_flag'][0] == 0:
            predictions['unidentified_structure_flag'] = [1, cert]
            return predictions
        else:
            predictions['unidentified_structure_flag'] = [0, cert]

        if predictions['unidentified_structure_flag'][0] == 1:
            return predictions

        if predictions['crystalline_structure_flag'][0] == 1:
            return predictions

        # we have diffuse pops only and can predict
        # spherical_normal_population_count and guinier_porod_population_count
        predictions['spherical_normal_population_count'] = \
            self.models['spherical_normal_population_count'].classify(sample_features)

        predictions['guinier_porod_population_count'] = \
            self.models['guinier_porod_population_count'].classify(sample_features)

        return predictions

    def print_accuracies(self):
        """Report cross-validation error for the model.

        To calculate cv_error "Leave-2-Groups-Out" cross-validation is used.
        For each train-test split,
        two experiments are used for testing
        and the rest are used for training.
        The reported error is the average accuracy over all train-test splits
        """
        print("Averaged cross validation accuracies: ")
        for k, v in self.models.items():
            print(k," :  %10.3f" % (v.get_cv_error()))
