import os

import numpy as np
import pandas as pd
import yaml
from sklearn import model_selection, preprocessing, linear_model
from sklearn.metrics import mean_absolute_error

from ..tools import profiler


class XRSDModel(object):

    def __init__(self, label, system_class=None, yml_file=None, classifier=True): #system_class for reg only

        self.model = None
        self.parameters = None
        self.scaler = None
        self.accuracy = None
        self.system_class = None
        self.target = label
        self.classifier = classifier
        self.n_groups_out = 1
        self.features = profiler.profile_keys_1
        self.trained = False

        if yml_file is None:
            p = os.path.abspath(__file__)
            d = os.path.dirname(p)
            if classifier:
                file_name = label + '.yml'
                yml_file = os.path.join(d,'modeling_data','classifiers',file_name)
            else:
                file_name = system_class + '.yml'
                yml_file = os.path.join(d,'modeling_data','regressors',file_name)

        try:
            s_and_m_file = open(yml_file,'rb')
            content = yaml.load(s_and_m_file)
            s_and_m = content[label]
        except:
            s_and_m = None

        if classifier:
             self.parameters_to_try = \
             {'penalty':('none', 'l2', 'l1', 'elasticnet'), #default l2
               'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1], #regularisation coef, default 0.0001
              'l1_ratio': [0, 0.15, 0.5, 0.85, 1.0]} #using with elasticnet only; default 0.15
        else:
             self.parameters_to_try = \
             {'loss':('huber', 'squared_loss'), # huber with epsilon = 0 gives us abs error (MAE)
               'epsilon': [1, 0.1, 0.01, 0.001, 0],
               'penalty':['none', 'l2', 'l1', 'elasticnet'], #default l2
               'alpha':[0.0001, 0.001, 0.01], #default 0.0001
              'l1_ratio': [0, 0.15, 0.5, 0.95], #default 0.15
              }

        if s_and_m and s_and_m['scaler']: # we have a saved model
            self.scaler = preprocessing.StandardScaler()
            set_param(self.scaler,s_and_m['scaler'])
            if self.classifier:
                self.model = linear_model.SGDClassifier()
            else:
                self.model = linear_model.SGDRegressor()
                self.system_class = s_and_m['system_class']
            set_param(self.model,s_and_m['model'])
            self.accuracy = s_and_m['accuracy']
            self.parameters = s_and_m['parameters']


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
            self.new_scaler = None
            self.new_model = None
            self.new_accuracy = None
            self.new_parameters = None
            self.trained = False
            return 

        data = d.dropna(subset=self.features)# TODO update it when we will have new features for crystalline only

        # using leaveGroupOut makes sense when we have at least 3 groups
        if len(data.experiment_id.unique()) > 2:
            leaveGroupOut = True
        else:
            # use 5-fold cross validation
            leaveGroupOut = False

        new_scaler = preprocessing.StandardScaler()
        new_scaler.fit(data[self.features])
        transformed_data = new_scaler.transform(data[self.features])

        if hyper_parameters_search == True:
            new_parameters = self.hyperparameters_search(
                        transformed_data, data[self.target],
                        data['experiment_id'], leaveGroupOut, self.n_groups_out)
        else:
            new_parameters = self.parameters

        if self.classifier:
            if new_parameters:
                new_model = linear_model.SGDClassifier(
                    alpha=new_parameters['alpha'], loss='log',
                    penalty=new_parameters["penalty"], l1_ratio=new_parameters["l1_ratio"],
                         max_iter=10)
            else:
                new_model = linear_model.SGDClassifier(loss='log', max_iter=10)
        else:
            if new_parameters:
                new_model = linear_model.SGDRegressor(alpha=new_parameters['alpha'], loss= new_parameters['loss'],
                                        penalty = new_parameters["penalty"],l1_ratio = new_parameters["l1_ratio"],
                                        epsilon = new_parameters["epsilon"],
                                                      max_iter=1000)
            else:
                new_model = linear_model.SGDRegressor(max_iter=1000) # max_iter is about 10^6 / number of tr samples

        new_model.fit(transformed_data, data[self.target]) # using all data for final training

        if self.classifier:
            label_std = None
        else:
            label_std = pd.to_numeric(data[self.target]).std()# usefull for regressin only

        if leaveGroupOut:
            new_accuracy = self.cross_validate_by_experiments(data, new_model, label_std)
            if new_accuracy is None:
                new_accuracy = self.cross_validate(data, new_model,label_std)
        else:
            new_accuracy = self.cross_validate(data, new_model,label_std)

        self.scaler = new_scaler
        self.model = new_model
        self.parameters = new_parameters
        self.accuracy = new_accuracy
        self.trained = True


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

    def hyperparameters_search(self, transformed_data, data_labels, group_by, leaveNGroupOut, n_leave_out):
        """Grid search for optimal alpha, penalty, and l1 ratio hyperparameters.
        Parameters
        ----------
        transformed_data : array
            2D numpy array of features, one row for each sample
        data_labels : array
            array of labels (as a DataFrame column), one label for each sample
        group_by: string
            DataFrame column header for LeavePGroupsOut(groups=group_by)
        leaveNGroupOut: boolean
            Indicated whether or not we have enough experimental data
            to cross-validate by the leave-two-groups-out approach
        n: integer
            number of groups to leave out
        Returns
        -------
        clf.best_params_ : dict
            Dictionary of the best found parametrs.
        """

        if leaveNGroupOut == True:
            cv=model_selection.LeavePGroupsOut(n_groups=n_leave_out).split(
                transformed_data, np.ravel(data_labels), groups=group_by)
        else:
            cv = 5 # five folders cross validation

        if self.classifier:
            model = linear_model.SGDClassifier(loss='log',max_iter=10)
        else:
            model = linear_model.SGDRegressor(max_iter=1000)

        clf = model_selection.GridSearchCV(model, self.parameters_to_try, cv=cv)
        clf.fit(transformed_data, np.ravel(data_labels))

        return clf.best_params_


    def cross_validate(self, df, model, label_std):
        """Test a model using scikit-learn 5-fold crossvalidation

        Parameters
        ----------
        df : pandas.DataFrame
            pandas dataframe of features and labels
        model : sklearn model
            with specific parameters
        label_std : float
            is used for regression models only
        Returns
        -------
        float
            average crossvalidation score 
            (accuracy for classification, normalized MAE for regression)
        """
        scaler = preprocessing.StandardScaler()
        scaler.fit(df[self.features])
        if self.classifier:
            scores = model_selection.cross_val_score(
                model, scaler.transform(df[self.features]), df[self.target], cv=5)
            return scores.mean()
        else:
            scores = model_selection.cross_val_score(
                    model,scaler.transform(df[self.features]), df[self.target],
                    cv=5, scoring = 'neg_mean_absolute_error')
            return -1.0 * scores.mean()/label_std


    def cross_validate_by_experiments(self, df, model, label_std):
        """Test a model by leaveTwoGroupsOut cross-validation

        Parameters
        ----------
        df : pandas.DataFrame
            pandas dataframe of features and labels
        model : sk-learn
            with specific parameters
        label_std : float
            is used for regression models only
        Returns
        -------
        float
            average crossvalidation score by experiments 
            (accuracy for classification, normalized MAE for regression)
        """
        experiments = df.experiment_id.unique()# we have at least 5 experiments
        test_scores_by_ex = []
        count = 0
        for i in range(len(experiments)):
                tr = df[(df['experiment_id']!= experiments[i])]
                test = df[(df['experiment_id']== experiments[i])]
                if self.classifier:
                    # for testing, we want only the samples with labels that are
                    # included in training set:
                    tr_labels = tr[self.target].unique()
                    test = test[test[self.target].isin(tr_labels)]
                    if len(test)==0:
                        continue

                    # The number of class labels must be greater than one
                    if len(tr[self.target].unique()) < 2:
                        continue

                scaler = preprocessing.StandardScaler()
                scaler.fit(tr[self.features])
                model.fit(scaler.transform(tr[self.features]), tr[self.target])
                transformed_data = scaler.transform(test[self.features])
                if self.classifier:
                    test_score = model.score(
                        transformed_data, test[self.target])
                    test_scores_by_ex.append(test_score)
                else:
                    pr = model.predict(transformed_data)
                    test_score = mean_absolute_error(pr, test[self.target])
                    test_scores_by_ex.append(test_score/label_std)
                count +=1

        if count == 0:
            return None

        return sum(test_scores_by_ex)/count


    def get_cv_error(self):
        """Report cross-validation error for the model.

        Returns the average accuracy/error over all train-test splits
        (accuracy for classification,
        normalized mean absolute error for regression).

        Returns
        -------
        accuracy : float
            the cross-validation errors.
        """
        return self.accuracy

# helper function - to set parameters for scalers and models
def set_param(m_s, param):
    for k, v in param.items():
        if isinstance(v, list):
            setattr(m_s, k, np.array(v))
        else:
            setattr(m_s, k, v)
