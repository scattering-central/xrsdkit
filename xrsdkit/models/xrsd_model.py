import numpy as np
import yaml
from sklearn import model_selection, preprocessing
from dask_ml.model_selection import GridSearchCV

from ..tools import profiler

class XRSDModel(object):

    def __init__(self, label, yml_file=None):
        self.model = None
        self.scaler = preprocessing.StandardScaler()
        self.cross_valid_results = None
        self.target = label
        self.trained = False
        self.model_file = yml_file
        self.default_val = None

        if yml_file:
            content = yaml.load(open(yml_file,'rb'))
            self.load_model_data(content)
        else:
            self.set_model()

    def load_model_data(self,model_data):
        self.trained = model_data['trained']
        self.default_val = model_data['default_val']
        if self.trained:
            self.set_model(model_data['model']['hyper_parameters'])
            for k,v in model_data['model']['trained_par'].items():
                setattr(self.model, k, np.array(v))
            setattr(self.scaler, 'mean_', np.array(model_data['scaler']['mean_']))
            setattr(self.scaler, 'scale_', np.array(model_data['scaler']['scale_']))
            self.cross_valid_results = model_data['cross_valid_results']

    def set_model(self, model_hyperparams={}):
        self.model = self.build_model(model_hyperparams)

    def build_model(self,model_hyperparams):
        msg = 'subclasses of XRSDModel must implement build_model()'
        raise NotImplementedError(msg)

    def train(self, all_data, hyper_parameters_search=False):
        """Train the model, optionally searching for optimal hyperparameters.

        Parameters
        ----------
        all_data : pandas.DataFrame
            dataframe containing features and labels
        hyper_parameters_search : bool
            If true, grid-search model hyperparameters
            to seek high cross-validation accuracy.
        """
        training_possible, n_groups_out, d = self.check_label(all_data)
        if not training_possible:
            # all samples have identical labels or we have <5 samples
            self.default_val = d[self.target].unique()[0]
            self.trained = False
            return
        else:
            # NOTE: SGD models train more efficiently on shuffled data
            shuffled_rows = np.random.permutation(all_data.index)
            all_data = all_data.loc[shuffled_rows]
            data = all_data[all_data[self.target].isnull() == False]
            data = self.standardize(data)

            if hyper_parameters_search:
                new_parameters = self.hyperparameters_search(
                    data[profiler.profile_keys], data[self.target],
                    data['experiment_id'], n_groups_out)
                new_model = self.build_model(new_parameters)
            else:
                new_model = self.model

            # NOTE: after cross-validation for parameter selection,
            # the entire dataset is used for final training
            self.cross_valid_results = self.run_cross_validation(new_model,data,profiler.profile_keys,n_groups_out)
            new_model.fit(data[profiler.profile_keys], data[self.target])
            self.model = new_model
            self.trained = True

    def standardize(self,data):
        """Standardize the columns of data that are used as model inputs"""
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(data[profiler.profile_keys])
        data[profiler.profile_keys] = self.scaler.transform(data[profiler.profile_keys])
        return data

    def hyperparameters_search(self,transformed_data, data_labels, group_by=None, n_leave_out=None, scoring=None):
        """Grid search for optimal alpha, penalty, and l1 ratio hyperparameters.

        Parameters
        ----------
        transformed_data : array
            2D numpy array of scaled features, one row for each sample
        data_labels : array
            array of labels (as a DataFrame column), one label for each sample
        group_by: string
            DataFrame column header for LeavePGroupsOut(groups=group_by)
        n_leave_out: integer
            number of groups to leave out, if group_by is specified 

        Returns
        -------
        clf.best_params_ : dict
            Dictionary of the best found hyperparameters.
        """
        if n_leave_out:
            cv = model_selection.LeavePGroupsOut(n_groups=n_leave_out).split(
                transformed_data, np.ravel(data_labels), groups=group_by)
        else:
            cv = 3 # number of folds for cross validation
        test_model = self.build_model()

        # threaded scheduler with optimal number of threads
        # will be used by default for dask GridSearchCV
        clf = GridSearchCV(test_model,
                        self.grid_search_hyperparameters, cv=cv, scoring=scoring)
        clf.fit(transformed_data, np.ravel(data_labels))

        return clf.best_params_


    def check_label(self, dataframe):
        """Test whether or not `dataframe` has legal values for all labels.
 
        Returns "True" if the dataframe has enough rows, 
        over which the labels exhibit at least two unique values
        and there are at least three samples for each label.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            dataframe of sample features and corresponding labels

        Returns
        -------
        result : bool
            indicates whether or not training is possible.
        n_groups_out : int or None
            using leaveGroupOut makes sense when we have at least 3 groups.
        """
        if len(dataframe.experiment_id.unique()) > 2:
            n_groups_out = 1
        else:
            # use 3-fold cross validation
            n_groups_out = None

        if len(dataframe[self.target].unique()) > 1:
            if dataframe.shape[0] >= 5:
                return True, n_groups_out
            else:
                #print('model {}: insufficient training data ({} samples)'.format(
                #self.target,dataframe.shape[0]))
                return False, n_groups_out
        else:
            #print('model {}: all training data have identical outputs ({})'.format(
            #self.target,dataframe[self.target].iloc[0]))
            return False, n_groups_out

    def run_cross_validation(self,model,data,features,group_cv):
        """Run a cross-validation test and return a report of the results.

        Regression models are scored by the coefficient of determination (R^2 or 'r2'),
        in order to normalize by the variance of the dataset.
        Training reports also include the more intuitive normalized mean_abs_error.
        Scikit-learn does not currently provide API for scoring by mean_abs_error,
        so mean_abs_error is not currently supported for hyperparameter training.
        Classifiers are validated by the f1_macro scoring function;
        f1_macro is the average, unweighted f1 score across all labels.
        The reports also include mean unweighted accuracies for all labels.
        Scikit-learn does not expose the mean unweighted accuracy by labels
        as a scoring option, so it cannot currently be used
        for hyperparameter optimization.

        Parameters
        ----------
        model : sklearn.linear_model.SGDRegressor
            scikit-learn regression model to be cross-validated
        data : pandas.DataFrame
            pandas dataframe of features and labels
        features : list of str
            list of features that were used for training
        group_cv : bool
            indicate if cross validation by experiments can be performed.

        Returns
        -------
        cross_val_results : dict
            with cross validation results.
        """
        if group_cv:
            cross_val_results = self.cross_validate_by_experiments(model,data,features)
        else:
            cross_val_results = self.cross_validate(model,data,features)
        return cross_val_results
