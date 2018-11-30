import numpy as np
import pandas as pd
from sklearn import linear_model, model_selection, preprocessing
from sklearn.metrics import mean_absolute_error

from .xrsd_model import XRSDModel
from ..tools import profiler

class Regressor(XRSDModel):
    """Class for generating models to predict real-valued parameters."""

    def __init__(self,label,yml_file):
        super(Regressor,self).__init__(label, yml_file)
        self.grid_search_hyperparameters = dict(
            epsilon = [10, 1, 0.1, 0.01, 0.001, 0],
            alpha = [0.00001, 0.0001, 0.001, 0.01], # regularisation coef, default 0.0001
            l1_ratio = [0, 0.15, 0.5, 0.85, 1.0] # default 0.15
            )
        self.scaler_y = None

    def load_model_data(self,model_data):
        super(Regressor,self).load_model_data(model_data)
        self.scaler_y = preprocessing.StandardScaler()
        if self.trained:
            setattr(self.scaler_y, 'mean_', np.array(model_data['scaler_y']['mean_']))
            setattr(self.scaler_y, 'scale_', np.array(model_data['scaler_y']['scale_']))

    def build_model(self, model_hyperparams={}):
        if all([p in model_hyperparams for p in ['alpha','l1_ratio','epsilon']]):
            new_model = linear_model.SGDRegressor(
                    alpha=model_hyperparams['alpha'],
                    l1_ratio=model_hyperparams['l1_ratio'],
                    epsilon=model_hyperparams['epsilon'],
                    loss= 'huber',
                    penalty='elasticnet',
                    max_iter=1000)
        else:
            # NOTE: max_iter is about 10^6 / number of tr samples 
            new_model = linear_model.SGDRegressor(loss= 'huber',
                    penalty='elasticnet',max_iter=1000)
        return new_model

    def standardize(self,data):
        """Standardize the columns of data that are used as model inputs

        Overriding of XRSDModel.standardize():
        For the regression models we also need to scale the target,
        since 'epsilon' and other hyperparameters will be affected
        by the scale of the training data outputs. 
        """
        data = super(Regressor,self).standardize(data)
        self.scaler_y = preprocessing.StandardScaler() 
        self.scaler_y.fit(data[self.target].values.reshape(-1, 1))
        data[self.target] = self.scaler_y.transform(data[self.target].values.reshape(-1, 1))
        return data

    def predict(self, sample_features):
        """Predict this model's scalar target for a given sample. 

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of xrsdkit.tools.profiler.profile_pattern()

        Returns
        -------
        prediction : float
            predicted parameter value
        """

        feature_array = np.array(list(sample_features.values())).reshape(1,-1)
        x = self.scaler.transform(feature_array)

        return float(self.scaler_y.inverse_transform(self.model.predict(x))[0])

    def cross_validate(self,model,df,features):
        """Test a model using scikit-learn 3-fold crossvalidation

        Parameters
        ----------
        model : sklearn.linear_model.SGDRegressor
            scikit-learn regression model to be cross-validated 
        df : pandas.DataFrame
            pandas dataframe of features and labels
        features : list of str
            list of features that were used for training

        Returns
        -------
        results : dict
            includes normalized mean abs error by splits,
            normalized average mean abs error (unweighted),
            weighted average mean abs error,
            number of experiments that were used for training/testing
            (can only be 1 or 2, since if we have data from 3 or more
            experiments, cross_validate_by_experiments() will be used),
            IDs of experiments, and a description 
            of the train-test split technique.
        """
        scores = np.absolute(model_selection.cross_val_score(
                model,df[features],df[self.target],
                cv=3,scoring='neg_mean_absolute_error'))

        results = dict(normalized_mean_abs_error_by_splits = str(scores),
                       normalized_mean_abs_error = sum(scores)/len(scores),
                       #weighted_av_mean_abs_error is the same us unweighted since the splits have the same sizes:
                       weighted_av_mean_abs_error = sum(scores)/len(scores),
                       number_of_experiments = len(df.experiment_id.unique()),
                       experiments = str(df.experiment_id.unique()),
                       test_training_split = 'random shuffle-split 3-fold cross-validation')

        return results

    def cross_validate_by_experiments(self, model, df, features):
        """Test a model by LeaveOneGroupOut cross-validation.

        Parameters
        ----------
        df : pandas.DataFrame
            pandas dataframe of features and labels
        model : sk-learn
            with specific parameters
        features : list of str
            list of features that were used for training.
            
        Returns
        -------
        results : dict
            includes normilezed mean abs error by splits,
            normilezed average mean abs error (unweighted),
            weighted average mean abs error,
            number of experiments that were used for training/testing
            (can be 3 or more  - if we have data from 1 or 2 experiments
            only, cross_validate() will be used),
            IDs of experiments,
            how the split was done.
        """
        experiments = df.experiment_id.unique()
        test_scores_by_ex = []
        test_scores_by_ex_weighted = []
        for i in range(len(experiments)):
            tr = df[(df['experiment_id']!= experiments[i])]
            test = df[(df['experiment_id']== experiments[i])]
            model.fit(tr[features], tr[self.target])
            pr = model.predict(test[features])
            test_score = mean_absolute_error(pr, test[self.target])
            test_scores_by_ex.append(test_score)
            test_scores_by_ex_weighted.append(test_score*(test.shape[0]/df.shape[0]))

        results = dict(normalized_mean_abs_error_by_splits = str(test_scores_by_ex),
                       normalized_mean_abs_error = sum(test_scores_by_ex)/len(test_scores_by_ex),
                       weighted_av_mean_abs_error = sum(test_scores_by_ex_weighted),
                       number_of_experiments = len(test_scores_by_ex),
                       experiments = str(df.experiment_id.unique()),
                       test_training_split = "by experiments")
        return results

    def print_mean_abs_errors(self):
        result = ''
        for r in self.cross_valid_results['normalized_mean_abs_error_by_splits'].split():
            result += (r + '\n')
        return result

    def check_label(self, dataframe):
        """Test whether or not `dataframe` has legal values for all labels.

        Returns "True" if the dataframe has enough rows,
        over which the labels exhibit at least two unique values.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            dataframe of sample features and corresponding labels

        Returns
        -------
        result : bool
            indicates whether or not training is possible
        n_groups_out : int or None
            using leaveGroupOut makes sense when we have at least 3 groups.
        dataframe : pandas.DataFrame
            same as the input dataframe
        """
        result, n_groups_out = super(Regressor,self).check_label(dataframe)
        return result, n_groups_out, dataframe

    # TODO
    def print_accuracies(self):
        return ''

    def average_mean_abs_error(self,weighted=False):
        if weighted:
            return str(self.cross_valid_results['weighted_av_mean_abs_error'])
        else:
            return str(self.cross_valid_results['normalized_mean_abs_error'])

    def print_CV_report(self):
        """Return a string describing the model's cross-validation metrics.

        Returns
        -------
        CV_report : str
            string with formated results of cross validatin.
        """
        CV_report = 'Cross validation results for {} Regressor\n'.format(self.target) + \
            'Data from {} experiments was used\n\n'.format(
            str(self.cross_valid_results['number_of_experiments'])) + \
            'Normalized mean abs error by train/test split:\n' + \
            self.print_mean_abs_errors() + '\n' + \
            'Weighted-average mean abs error by train/test split: {}\n'.format(self.average_mean_abs_error(True)) + \
            'Unweighted-average mean abs error: {}\n'.format(self.average_mean_abs_error(False)) + \
            '\n\nNOTE: Weighted metrics are weighted by test set size' 
        return CV_report






