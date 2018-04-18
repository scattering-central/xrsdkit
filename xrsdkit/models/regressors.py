from collections import OrderedDict
import numpy as np
from .regressor import Regressor
from xrsdkit.tools.piftools import reg_model_output_names



class Regressors(object):
    """To create all possible regressors, train, update, and save them; make a prediction."""

    def __init__(self):

        self.models = OrderedDict.fromkeys(reg_model_output_names)
        for k,v in self.models.items():
            self.models[k] = Regressor(k)


    def train_regression_models(self, data, hyper_parameters_search=False,
                                     reg_models = ['all'], testing_data = None, partial = False):
        """Train regression models, optionally searching for optimal hyperparameters.
        Parameters
        ----------
        data : pandas.DataFrame
            dataframe containing features and labels
        hyper_parameters_search : bool
            If true, grid-search model hyperparameters
            to seek high cross-validation R^2 score.
        model : array of str
            the names of models to train ('rg_0','sigma_0',
            'r0_0', or "all" to train all models).
        testing_data : pandas.DataFrame (optional)
            dataframe containing original training data plus new data
            for computing the cross validation errors of the updated models.
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
        if "all" in reg_models:
            results = OrderedDict.fromkeys(reg_model_output_names)
        else:
            results = OrderedDict.fromkeys(reg_models)

        g_p_data = data[(data['guinier_porod_population_count']=="1")&
                     (data['diffuse_structure_flag']=="1") &
                     (data['crystalline_structure_flag']=="0") ]
        spherical_data = data[(data['spherical_normal_population_count']=="1")
                                &(data['diffuse_structure_flag']=="1")
                                & (data['crystalline_structure_flag']=="0") ]

        test_g_p = None
        test_spherical = None
        if testing_data is not None:
            test_g_p = testing_data[(testing_data['guinier_porod_population_count']=="1")&
                     (testing_data['diffuse_structure_flag']=="1") &
                     (testing_data['crystalline_structure_flag']=="0") ]
            test_spherical = testing_data[(testing_data['spherical_normal_population_count']=="1")
                                &(testing_data['diffuse_structure_flag']=="1")
                                & (testing_data['crystalline_structure_flag']=="0") ]

        for k, v in results.items():
            if k in ['sigma_0', 'r0_0']:
                if partial:
                    results[k] = self.models[k].train_partial(spherical_data, test_spherical)
                else:
                    results[k] = self.models[k].train(spherical_data, hyper_parameters_search)
            else:
                if partial:
                    results[k] = self.models[k].train_partial(g_p_data, test_g_p)
                else:
                    results[k] = self.models[k].train(g_p_data, hyper_parameters_search)

        return results


    def print_training_results(self, results):
        """Print parameters of models and cross validation accuracies.
        Parameters
        ----------
        results : dict
            keys : models names;
            values : dictionaries with sklearn standard scalers,
            sklearn models, models paramenters,
            and cross validation errors.
        """
        for k, v in results.items():
            print(k, ":")
            if results[k]['accuracy']:
                print("accuracy :  %10.3f" % (results[k]['accuracy']))
                print("parameters :", results[k]['parameters'])
            else:
                print("training/updatin is not possible (not enough data)")
            print()


    def save_regression_models(self, scalers_models, file_path=None):
        """Save model parameters and CV errors in YAML and .txt files.
        Parameters
        ----------
        scalers_models : dict
            keys : models names;
            values : dictionaries with sklearn standard scalers,
            sklearn models, models paramenters,
            and cross validation accuracies.
        file_path : str (optional)
            Full path to the YAML file where the models will be saved.
            Scalers, models,parameters, and cross-validation errors
            will be saved at this path, and the cross-validation errors
            are also saved in a .txt file of the same name, in the same directory.
        """
        for k, v in self.models.items():
            v.save_models(scalers_models[k], file_path)


    def make_predictions(self, sample_features, populations, q_I):
        """Determine the types of structures represented by the sample
        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of xrsdkit.tools.profiler.profile_spectrum()
        populations : dict
            dictionary counting scatterer populations,
            similar to output of Classifiers.make_predictions()
        q_I : array
            n-by-2 array of scattering vector (1/Angstrom) and intensities.
        Returns
        -------
        prediction : dict
            dictionary with predicted parameters
        """
        predictions = {}

        for k,v in self.models.items():
            pr = v.predict(sample_features, populations, q_I)
            if pr:
                predictions[k] = pr

        return predictions

    def print_errors(self):
        """Report cross-validation error for the model.

        To calculate cv_error "Leave-2-Groups-Out" cross-validation is used.
        For each train-test split,
        two experiments are used for testing
        and the rest are used for training.
        The reported error is normalized mean absolute error over all train-test splits.
        """
        print("Normalized cross validation mean absolute error: ")
        for k, v in self.models.items():
            print(k," :  %10.3f" % (v.get_cv_error()))
