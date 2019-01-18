import numpy as np
import yaml
from sklearn import model_selection, preprocessing, utils
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
        # TODO: add a docstring that describes the interface
        msg = 'subclasses of XRSDModel must implement build_model()'
        raise NotImplementedError(msg)

    def run_cross_validation(self,model,data,features,grouping):
        # TODO: add a docstring that describes the interface
        msg = 'subclasses of XRSDModel must implement run_cross_validation()'
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
        training_possible, grouping = self.check_label(all_data)
        if not training_possible:
            # not enough samples, or all have identical labels
            self.default_val = all_data[self.target].unique()[0]
            self.trained = False
            return
        else:
            # NOTE: SGD models train more efficiently on shuffled data
            d = utils.shuffle(all_data)
            data = d[d[self.target].isnull() == False]
            data = self.standardize(data)

            if hyper_parameters_search:
                new_parameters = self.hyperparameters_search(data, n_leave_out=1)
                new_model = self.build_model(new_parameters)
            else:
                new_model = self.model

            # NOTE: after cross-validation for parameter selection,
            # the entire dataset is used for final training
            self.cross_valid_results = self.run_cross_validation(new_model,data,profiler.profile_keys,grouping)
            new_model.fit(data[profiler.profile_keys], data[self.target])
            self.model = new_model
            self.trained = True

    def standardize(self,data):
        """Standardize the columns of data that are used as model inputs"""
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(data[profiler.profile_keys])
        data[profiler.profile_keys] = self.scaler.transform(data[profiler.profile_keys])
        return data

    def hyperparameters_search(self,transformed_data, group_by='experiment_id', n_leave_out=None, scoring=None):
        """Grid search for optimal alpha, penalty, and l1 ratio hyperparameters.

        Parameters
        ----------
        transformed_data : pandas.DataFrame
            dataframe containing features and labels;
            note that the features should be transformed/standardized beforehand
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
                transformed_data[profiler.profile_keys], 
                np.ravel(transformed_data[self.target]),
                groups=transformed_data[group_by]
                )
        else:
            cv = 3 # number of folds for cross validation
        test_model = self.build_model()
        # threaded scheduler with optimal number of threads
        # will be used by default for dask GridSearchCV
        clf = GridSearchCV(test_model,self.grid_search_hyperparameters,cv=cv,scoring=scoring)
        clf.fit(transformed_data[profiler.profile_keys], np.ravel(transformed_data[self.target]))
        return clf.best_params_

    def check_label(self, dataframe):
        """Check whether `dataframe` provides sufficient training data for self.target.
 
        Returns True if the dataframe has at least 10 rows, 
        over which the labels exhibit at least two unique values.
        All rows of `dataframe` are assumed to have valid labels for self.target.
        A group_id column is added to the dataframe for cross-validation grouping.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            dataframe of sample features and corresponding labels

        Returns
        -------
        result : bool
            indicates whether or not training is possible.
        grouping : str 
            using leaveGroupOut makes sense when we have at least 3 groups.
        """
        if len(dataframe.experiment_id.unique()) > 2:
            grouping = 'experiment_id' 
        else:
            grouping = None 
        if len(dataframe[self.target].unique()) > 1:
            if dataframe.shape[0] >= 10:
                # sufficient training data with at least 2 distinct labels
                return True, grouping 
            else:
                # insufficient training data
                return False, grouping 
        else:
            # all training data have identical outputs 
            return False, grouping 
