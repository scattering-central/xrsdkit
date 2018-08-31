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
            loss = ['huber', 'squared_loss'], # huber with epsilon = 0 yields mean abs error (MAE)
            epsilon = [1, 0.1, 0.01, 0.001, 0],  
            penalty = ['none', 'l2', 'l1', 'elasticnet'], # default: l2  
            alpha = [0.0001, 0.001, 0.01], # regularisation coef, default 0.0001 
            l1_ratio = [0, 0.15, 0.5, 0.95] # default 0.15, only valid for elasticnet penalty 
            )

    def build_model(self, model_hyperparams={}):
        if all([p in model_hyperparams for p in ['alpha','loss','penalty','l1_ratio','epsilon']]):
            new_model = linear_model.SGDRegressor(
                    alpha=model_hyperparams['alpha'], 
                    loss=model_hyperparams['loss'],
                    penalty=model_hyperparams['penalty'],
                    l1_ratio=model_hyperparams['l1_ratio'],
                    epsilon=model_hyperparams['epsilon'],
                    max_iter=1000)
        else:
            # NOTE: max_iter is about 10^6 / number of tr samples 
            new_model = linear_model.SGDRegressor(max_iter=1000) 
        return new_model

    def predict(self, sample_features):
        """Predict this model's scalar target for a given sample. 

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of xrsdkit.tools.profiler.profile_spectrum()

        Returns
        -------
        prediction : float
            predicted parameter value
        """

        feature_array = np.array(list(sample_features.values())).reshape(1,-1)
        x = self.scaler.transform(feature_array)

        return float(self.model.predict(x)[0])


    def run_cross_validation(self,model,data,features, group_cv):
        label_std = pd.to_numeric(data[self.target]).std()
        if group_cv:
            new_accuracy = self.cross_validate_by_experiments(model,data,features,label_std)
        else:
            new_accuracy = self.cross_validate(model,data,features,label_std)
        return new_accuracy


    def cross_validate(self,model,df,features, label_std):
        """Test a model using scikit-learn 5-fold crossvalidation

        Parameters
        ----------
        model : sklearn model
            model to be cross-validated 
        df : pandas.DataFrame
            pandas dataframe of features and labels
        features : list of str
            list of features that were used for training
        label_std : float
            standard deviation of training data for the model label 

        Returns
        -------
        results : dict
            includes normilezed mean abs error,
            normilezed mean abs error by splits,
            labels std,
            number of experiments that were used for training/testing
            (can be 1 or 2 since if we have data from 3 or more
            experiments, cross_validate_by_experiments() will be used),
            IDs of experiments,
            how the split was done.


        """
        scaler = preprocessing.StandardScaler()
        scaler.fit(df[profiler.profile_keys])
        scores = np.absolute(model_selection.cross_val_score(
                model,scaler.transform(df[features]), df[self.target],
                cv=5, scoring='neg_mean_absolute_error')/ label_std)

        results = dict(normalized_mean_abs_error_by_splits = scores,
                       normalized_mean_abs_error = sum(scores)/len(scores),
                       labels_std = label_std,
                       number_of_experiments = len(df.experiment_id.unique()),
                       experiments = df.experiment_id.unique(),
                       test_training_split = "random 5 folders crossvalidation split")

        return results

    def cross_validate_by_experiments(self, model, df, features, label_std):
        """Test a model by LeaveOneGroupOut cross-validation.

        Parameters
        ----------
        df : pandas.DataFrame
            pandas dataframe of features and labels
        model : sk-learn
            with specific parameters
        features : list of str
            list of features that were used for training.
        label_std : float
            standard deviation of training data for the model label 
            
        Returns
        -------
        results : dict
            includes normilezed mean abs error,
            normilezed mean abs error by experiments,
            labels std,
            number of experiments that were used for training/testing
            (can be 3 or more  - if we have data from 1 or 2 experiments
            only, cross_validate() will be used),
            IDs of experiments,
            how the split was done.
        """
        experiments = df.experiment_id.unique()
        test_scores_by_ex = []
        for i in range(len(experiments)):
            tr = df[(df['experiment_id']!= experiments[i])]
            test = df[(df['experiment_id']== experiments[i])]

            scaler = preprocessing.StandardScaler()
            scaler.fit(tr[features])
            model.fit(scaler.transform(tr[features]), tr[self.target])
            transformed_data = scaler.transform(test[features])
            pr = model.predict(transformed_data)
            test_score = mean_absolute_error(pr, test[self.target])
            test_scores_by_ex.append(test_score/label_std)

        results = dict(normalized_mean_abs_error_by_experiments = test_scores_by_ex,
                       normalized_mean_abs_error = sum(test_scores_by_ex)/len(test_scores_by_ex),
                       labels_std = label_std,
                       number_of_experiments = len(test_scores_by_ex),
                       experiments = df.experiment_id.unique(),
                       test_training_split = "by experiments")
        return results

    # TODO
    def print_accuracies(self):
        return ''

    # TODO
    def average_accuracy(self,weighted=False):
        return -1.

    # TODO
    def print_CV_report(self):
    """Return a string describing the model's cross-validation metrics.

    Returns
    -------
    CV_report : str
        string with formated results of cross validatin.
    """
    CV_report = 'Cross validation results for {} Regressor\n'.format(self.target) \
        'Accuracies by train/test split:\n' \
        self.print_accuracies() \
        'Weighted-average accuracy: {}\n'.format(self.average_accuracy(False)) \
        'Unweighted-average accuracy: {}\n'.format(self.average_accuracy(True)) \
        '\n\nNOTE: Weighted metrics are weighted by test set size' 
    return CV_report






