from collections import OrderedDict
import numpy as np
from .regressor import Regressor
from . import get_possible_regression_models
import os



class Regressors(object):
    """To create all possible regressors for every type of population,
     train, update, and save them; make a prediction."""

    def __init__(self):
        #find all existing regression models:
        p = os.path.abspath(__file__)
        d = os.path.dirname(p)
        models = []
        regression_dir = os.path.join(d,'modeling_data','regressors')

        for file in os.listdir(regression_dir):
            if file.endswith(".yml"):
                name = file.split(".")[0]
                models.append(name)

        self.models = OrderedDict.fromkeys(models)
        for k,v in self.models.items():
            self.models[k] = Regressor(k)


    def train_regression_models(self, data, hyper_parameters_search=False,
                                     populations = ['all'], testing_data = None, partial = False):
        """Train regression models, optionally searching for optimal hyperparameters.
        Parameters
        ----------
        data : pandas.DataFrame
            dataframe containing features and labels
        hyper_parameters_search : bool
            If true, grid-search model hyperparameters
            to seek high cross-validation R^2 score.
        populations : array of str
            the names of populations for which we want to train
            regression models.
        testing_data : pandas.DataFrame (optional)
            dataframe containing original training data plus new data
            for computing the cross validation errors of the updated models.
        partial : bool
            If true, the models will be updataed using new data
            (train_partial() instead of train() from sklearn is used).
        Returns
        -------
        results : dict
            keys : population names;
            values : dictionaries with regression models for given population.
        """

        possible_models = get_possible_regression_models(data)
        # TODO add support for selecton of models

        results = OrderedDict.fromkeys(possible_models.keys())
        for k,v in results.items():
            results[k] = {}

        for k, v in possible_models.items(): # v is the list of possible models for given population
            pop_data = data[(data['populations']==k)]
            for m in v:
                reg_model = Regressor(m)
                res =  reg_model.train(pop_data, hyper_parameters_search)
                res['population'] = k
                results[k][m] = res
        return results


    def print_training_results(self, results):
        """Print parameters of models and cross validation accuracies.
        Parameters
        ----------
        results : dict
            keys : population names;
            values : dictionaries with regression models for given population.
        """
        for pop, models in results.items():
            if results[pop] == {}:
                continue
            print(pop, ":") # populaton
            for k, v in models.items():
                print(k, ":")
                try:
                    print("accuracy :  %10.3f" % (models[k]['accuracy']))
                    print("parameters :", models[k]['parameters'])
                except:
                    print("training/updatin is not possible (not enough data)")
                print()


    def save_regression_models(self, results, file_path=None):
        """Save model parameters and CV errors in YAML and .txt files.
        Parameters
        ----------
        results : dict
            keys : population names;
            values : dictionaries with regression models for given population.
        file_path : str (optional)
            Full path to the YAML file where the models will be saved.
            Scalers, models,parameters, and cross-validation errors
            will be saved at this path, and the cross-validation errors
            are also saved in a .txt file of the same name, in the same directory.
        """
        for pops, models in results.items():
            for k,v in models.items():
                if self.models.get(k, 0) == 0:
                    self.models[k] = Regressor(k)
                self.models[k].save_models(v, file_path)


    def make_predictions(self, sample_features, population, q_I):
        """Determine the types of structures represented by the sample
        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of xrsdkit.tools.profiler.profile_spectrum()
        population : str
            Scatterer populations.
        q_I : array
            n-by-2 array of scattering vector (1/Angstrom) and intensities.
        Returns
        -------
        prediction : dict
            dictionary with predicted parameters
        """
        pop = population[0]
        predictions = {}
        if pop=='Noise' or pop=='pop0_unidentified':
            return predictions

        for k,v in self.models.items():
            if v.population == pop:
                pr = v.predict(sample_features, q_I)
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
