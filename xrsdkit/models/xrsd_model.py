import os

import numpy as np
import pandas as pd
import yaml
from sklearn import model_selection, preprocessing, linear_model
from sklearn.metrics import mean_absolute_error

from ..tools import profiler

class XRSDModel(object):

    def __init__(self, label, yml_file):
        self.model = None
        # TODO: if the hyperparameters are accessible through self.model,
        # remove the self.parameters attribute
        self.parameters = None
        self.scaler = preprocessing.StandardScaler()
        self.accuracy = None
        self.target = label
        self.trained = False
        content = yaml.load(open(yml_file,'rb'))
        self.load_model_data(content[label])

    def load_model_data(model_data):
        self.set_model(model_data['parameters'])
        # TODO: consider getting rid of the set_param method,
        # in favor of something more concrete
        set_param(self.model,model_data['model'])
        set_param(self.scaler,scaler_data)
        self.scaler = self.build_scaler(model_data['scaler'])
        self.accuracy = model_data['accuracy']

    def set_model(self, model_hyperparams={}):
        self.model = self.build_model(model_hyperparams)
        self.parameters = model_hyperparams

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

        # TODO: add comment to explain what happens here
        data = d.dropna(subset=profiler.profile_keys)

        # using leaveGroupOut makes sense when we have at least 3 groups
        if len(data.experiment_id.unique()) > 2:
            group_cv = True
            n_groups_out = 1
        else:
            # use 5-fold cross validation
            group_cv = False
            n_groups_out = None

        new_scaler = preprocessing.StandardScaler()
        new_scaler.fit(data[profiler.profile_keys])
        transformed_data = new_scaler.transform(data[profiler.profile_keys])

        if hyper_parameters_search:
            new_parameters = self.hyperparameters_search(
                        transformed_data, data[self.target],
                        data['experiment_id'], group_cv, n_groups_out)
        else:
            new_parameters = self.parameters

        new_model = self.build_model(new_parameters)
        # NOTE: after cross-validation for parameter selection,
        # the entire dataset is used for final training
        # TODO: should we use cross-validation for the model.fit() as well?
        new_model.fit(transformed_data, data[self.target]) 

        new_accuracy = self.run_cross_validation(new_model,data,group_cv)
        # TODO: Run this CV the same way as hyperparameters_search().
        # Right now it appears that they are not the same.
        # TODO: save all cross-validation statistics as object attributes

        self.scaler = new_scaler
        self.model = new_model
        self.parameters = new_parameters
        self.accuracy = new_accuracy
        self.trained = True

    def hyperparameters_search(self, transformed_data, data_labels, group_by=None, n_leave_out=1):
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
        if group_by:
            cv=model_selection.LeavePGroupsOut(n_groups=n_leave_out).split(
                transformed_data, np.ravel(data_labels), groups=group_by)
        else:
            cv = 5 # five-fold cross validation
        test_model = self.build_model() 
        clf = model_selection.GridSearchCV(test_model, self.grid_search_hyperparameters, cv=cv)
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

    def print_accuracies(self):
        """Pretty-print a report of the cross-validation statistics."""
        msg = '{}: cross-validation summary'.format(self.__name__) 
        # TODO: extract the error objective that was used to cross-validate the model
        msg += 'model objective: ... '
        # TODO: save a representation of the cross-validation technique, and extract it here
        msg += os.linesep+'cross-validation technique: ... '
        # TODO: save cross-validation statistics during training,
        # then process the statistics and print them out here
        msg += os.linesep+'cross-validation statistics: '
        msg += os.linesep+' ... '
        return msg

# helper function - to set parameters for scalers and models
def set_param(m_s, param):
    for k, v in param.items():
        if isinstance(v, list):
            setattr(m_s, k, np.array(v))
        else:
            setattr(m_s, k, v)

