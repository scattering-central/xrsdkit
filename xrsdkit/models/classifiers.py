from collections import OrderedDict
from .structure_classifier import StructureClassifier
from xrsdkit.tools.piftools import cl_model_output_names


class Classifiers(object):
    """To create all possible classifiers, train, update, and save them; make a prediction."""

    def __init__(self):

        self.models = OrderedDict.fromkeys(cl_model_output_names)
        for k,v in self.models.items():
            self.models[k] = StructureClassifier(k)


    def train_classification_models(self, data, hyper_parameters_search=False,
                                     cl_models = ['all'], testing_data = None, partial = False):
        """Train classification models, optionally searching for optimal hyperparameters.
        Parameters
        ----------
        data : pandas.DataFrame
            dataframe containing features and labels
        hyper_parameters_search : bool
            If true, grid-search model hyperparameters
            to seek high cross-validation accuracy.
        cl_models : array of str
            the names of models to train ('crystalline_structure_flag',
            'diffuse_structure_flag',
            'guinier_porod_population_count',
            'spherical_normal_population_count', or "all" to train all models).
        testing_data : pandas.DataFrame (optional)
            dataframe containing original training data plus new data
            for computing the cross validation accuracies of the updated models.
        partial : bool
            If true, the models will be updataed using new data
            (train_partial() instead of train() from sklearn is used).
        Returns
        -------
        results : dict
            keys : models names;
            values : dictionaries with sklearn standard scalers,
            sklearn models, models paramenters,
            and cross validation accuracies.
        """
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
        """Print parameters of models and cross validation accuracies.
        Parameters
        ----------
        results : dict
            keys : models names;
            values : dictionaries with sklearn standard scalers,
            sklearn models, models paramenters,
            and cross validation accuracies.
        """
        for k, v in results.items():
            print(k, ":")
            if results[k]['accuracy']:
                print("accuracy :  %10.3f" % (results[k]['accuracy']))
                print("parameters :", results[k]['parameters'])
            else:
                print("training/updatin is not possible (not enough data)")
            print()


    def save_classification_models(self, scalers_models, file_path=None):
        """Save model parameters and CV accuracies in YAML and .txt files.
        Parameters
        ----------
        scalers_models : dict
            keys : models names;
            values : dictionaries with sklearn standard scalers,
            sklearn models, models paramenters,
            and cross validation accuracies.
        file_path : str (optional)
            Full path to the YAML file where the models will be saved.
            Scalers, models,parameters, and cross-validation accuracies
            will be saved at this path, and the cross-validation errors
            are also saved in a .txt file of the same name, in the same directory.
        """
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
            similar to output of xrsdkit.tools.profiler.profile_spectrum()
        Returns
        -------
        prediction : dict
            dictionary of predictions
            from classifiers and certainty of the predictions.
        """
        list_of_wanted_models = list(self.models.keys())
        list_of_models_to_start = ['crystalline_structure_flag', 'diffuse_structure_flag']
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
        The reported error is the average accuracy over all train-test splits.
        """
        print("Averaged cross validation accuracies: ")
        for k, v in self.models.items():
            print(k," :  %10.3f" % (v.get_cv_error()))
