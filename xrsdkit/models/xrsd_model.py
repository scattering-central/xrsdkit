import os

import numpy as np
import yaml
from sklearn import model_selection, preprocessing

from ..tools import profiler

class XRSDModel(object):

    def __init__(self, label, yml_file=None):
        self.model = None
        self.scaler = preprocessing.StandardScaler()
        self.cross_valid_results = None
        self.target = label
        self.trained = False
        self.model_file = yml_file

        if yml_file:
            content = yaml.load(open(yml_file,'rb'))
            self.load_model_data(content[label])
        else:
            self.set_model()


    def load_model_data(self,model_data):
        self.set_model()
        # TODO: consider getting rid of the set_param method,
        # in favor of something more concrete
        set_param(self.model,model_data['model'])
        set_param(self.scaler,model_data['scaler'])
        self.cross_valid_results = model_data['cross_valid_results']

    def set_model(self, model_hyperparams={}):
        self.model = self.build_model(model_hyperparams)

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

        shuffled_rows = np.random.permutation(all_data.index)
        all_data = all_data.loc[shuffled_rows]
        d = all_data[all_data[self.target].isnull() == False]
        training_possible = self.check_label(d)
        if not training_possible:
            return 

        # drop the rows with Nans in profile_keys (features)
        # the scaler will crash if data includes rows with Nons
        data = d.dropna(subset=profiler.profile_keys)

        # using leaveGroupOut makes sense when we have at least 3 groups
        if len(data.experiment_id.unique()) > 2:
            n_groups_out = 1
        else:
            # use 5-fold cross validation
            n_groups_out = None

        new_scaler = preprocessing.StandardScaler()
        new_scaler.fit(data[profiler.profile_keys])
        data[profiler.profile_keys] = new_scaler.transform(data[profiler.profile_keys])

        if hyper_parameters_search:
            new_parameters = self.hyperparameters_search(
                        data[profiler.profile_keys], data[self.target],
                        data['experiment_id'], n_groups_out)
            new_model = self.build_model(new_parameters)
        else:
            new_model = self.model

        # NOTE: after cross-validation for parameter selection,
        # the entire dataset is used for final training
        new_model.fit(data[profiler.profile_keys], data[self.target])

        cross_valid_results = self.run_cross_validation(new_model,data,profiler.profile_keys,n_groups_out)

        self.scaler = new_scaler
        self.model = new_model
        self.cross_valid_results = cross_valid_results
        self.trained = True


    def hyperparameters_search(self,transformed_data, data_labels, group_by=None, n_leave_out=None):
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
        #print("all experiments: ", data['experiment_id'].unique())
        if n_leave_out:
            cv=model_selection.LeavePGroupsOut(n_groups=n_leave_out).split(
                transformed_data, np.ravel(data_labels), groups=group_by)

        else:
            cv = 5 # five-fold cross validation
        test_model = self.build_model()

        if self.target == 'system_class':
            # Calculate f1 for each label, and find their unweighted median
            scoring = "f1_macro"
        else:
            scoring = None
        clf = model_selection.GridSearchCV(test_model,
                        self.grid_search_hyperparameters, cv=cv, scoring=scoring)
        clf.fit(transformed_data, np.ravel(data_labels))

        return clf.best_params_


    def check_label(self, dataframe):
        """Test whether or not `dataframe` has legal values for all labels.
 
        Returns "True" if the dataframe has enough rows, 
        over which the labels exhibit at least two unique values 

        Parameters
        ----------
        dataframe : pandas.DataFrame
            dataframe of sample features and corresponding labels

        Returns
        -------
        bool
            indicates whether or not training is possible.
        """
        if len(dataframe[self.target].unique()) > 1:
            if dataframe.shape[0] >= 5:
                return True
            else:
                print('model {}: insufficient training data ({} samples)'.format(
                self.target,dataframe.shape[0]))
                return False
        else:
            print('model {}: all training data have identical outputs ({})'.format(
            self.target,float(dataframe[self.target].iloc[0])))
            return False


# helper function - to set parameters for scalers and models
def set_param(m_s, param):
    for k, v in param.items():
        if isinstance(v, list):
            setattr(m_s, k, np.array(v))
        else:
            setattr(m_s, k, v)

