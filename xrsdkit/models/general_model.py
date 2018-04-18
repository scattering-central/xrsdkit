import os
import numpy as np
import pandas as pd
import yaml
from sklearn import model_selection, preprocessing, linear_model
from sklearn.metrics import mean_absolute_error

from ..tools import profiler
from . import set_param

class XRSDModel(object):

    def __init__(self, label, yml_file=None, classifier=True):
        if yml_file is None:
            p = os.path.abspath(__file__)
            d = os.path.dirname(p)
            file_name = label + '.yml'
            yml_file = os.path.join(d,'modeling_data',file_name)

        try:
            s_and_m_file = open(yml_file,'rb')
            s_and_m = yaml.load(s_and_m_file)
        except:
            s_and_m = None

        self.model = None
        self.parameters = None
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
        self.scaler = None
        self.cv_error = None
        self.target = label
        self.classifier = classifier
        self.n_groups_out = 2
        self.features = profiler.profile_keys_1

        if s_and_m and s_and_m['scaler']: # we have a saved model
            self.scaler = preprocessing.StandardScaler()
            set_param(self.scaler,s_and_m['scaler'])
            if self.classifier:
                self.model = linear_model.SGDClassifier()
            else:
                self.model = linear_model.SGDRegressor()
            set_param(self.model,s_and_m['model'])
            self.cv_error = s_and_m['accuracy']
            self.parameters = s_and_m['parameters']


    def train(self, all_data, hyper_parameters_search=False):
        """Train SAXS classification models, optionally searching for optimal hyperparameters.

        Parameters
        ----------
        all_data : pandas.DataFrame
            dataframe containing features and labels
        hyper_parameters_search : bool
            If true, grid-search model hyperparameters
            to seek high cross-validation accuracy.

        Returns
        -------
        results : dict
            Dictionary with training results.

        The results include:

        - 'scaler': sklearn standard scaler
            used for transforming of new data

        - 'model':sklearn model
            trained on new data

        - 'parameters': dict
            Dictionary of parameters found by hyperparameters_search()

        - 'accuracy': float
            average crossvalidation score (accuracy for classification,
            normalized mean absolute error for regression)
        """
        d = all_data[all_data[self.target].isnull() == False]
        training_possible = self.check_label(d)

        new_scaler = None
        new_model = None
        new_accuracy = None
        new_parameters = None

        if not training_possible:
            print(self.target, "model was not trained.")# TODO decide what we should print (or not print) heare
            return new_scaler, new_model,new_parameters, new_accuracy

        data = d.dropna(subset=self.features)

        # using leaveTwoGroupOut makes sense when we have at least 5 groups
        if len(data.experiment_id.unique()) > 4:
            leaveTwoGroupOut = True
        else:
            # use 5-fold cross validation
            leaveTwoGroupOut = False

        new_scaler = preprocessing.StandardScaler()
        new_scaler.fit(data[self.features])
        transformed_data = new_scaler.transform(data[self.features])

        if hyper_parameters_search == True:
            new_parameters = self.hyperparameters_search(
                        transformed_data, data[self.target],
                        data['experiment_id'], leaveTwoGroupOut, self.n_groups_out)
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

        new_model.fit(transformed_data, data[self.target])

        if self.classifier:
            label_std = None
        else:
            label_std = pd.to_numeric(data[self.target]).std()# usefull for regressin only

        if leaveTwoGroupOut:
            new_accuracy = self.testing_by_experiments(data, new_model, label_std)
        else:
            new_accuracy = self.testing_using_crossvalidation(data, new_model,label_std)

        return {'scaler': new_scaler, 'model': new_model,
                'parameters' : new_parameters, 'accuracy': new_accuracy}


    def check_label(self, dataframe):
        """Test whether or not `dataframe` has legal values for the label

        For classification models:
        Because a model requires a distribution of training samples,
        this function checks `dataframe` to ensure that its
        labels are not all the same.
        For a model where the label is always the same,
        this function returns False,
        indicating that this `dataframe`
        cannot be used to train that model.

        For rergession models:
        The function return "True" if the dataframe has at least
        5 non-null values

        Parameters
        ----------
        dataframe : pandas.DataFrame
            dataframe of sample features and corresponding labels

        Returns
        -------
        boolean
            indicating whether or not training is possible.
        """

        if self.classifier:
            if len(dataframe[self.target].unique()) > 1:
                return True
            else:
                return False

        else:
            if dataframe.shape[0] > 4:
                return True
            else:
                return False


    def hyperparameters_search(self, transformed_data, data_labels, group_by, leaveNGroupOut, n):
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
            cv=model_selection.LeavePGroupsOut(n_groups=n).split(
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


    def testing_using_crossvalidation(self, df, model, label_std):
        """Fit a model, then test it using 5-fold crossvalidation

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
            average crossvalidation score (accuracy for classification,
            normalized mean absolute error for regression)
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


    def testing_by_experiments(self, df, model, label_std):
        """Fit a model, then test it by leaveTwoGroupsOut cross-validation

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
            average crossvalidation score by experiments (accuracy for classification,
            normalized mean absolute error for regression)
        """
        experiments = df.experiment_id.unique()# we have at least 5 experiments
        test_scores_by_ex = []
        count = 0
        for i in range(len(experiments)):
            for j in range(i+1, len(experiments)):
                tr = df[(df['experiment_id']!= experiments[i]) \
                    & (df['experiment_id']!= experiments[j])]
                test = df[(df['experiment_id']== experiments[i]) \
                    | (df['experiment_id']== experiments[j])]

                # The number of class labels must be greater than one
                if len(tr[self.target].unique()) < 2:
                    continue

                scaler = preprocessing.StandardScaler()
                scaler.fit(tr[self.features])
                model.fit(scaler.transform(tr[self.features]), tr[self.target])
                if self.classifier:
                    test_score = model.score(
                        scaler.transform(test[self.features]), test[self.target])
                    test_scores_by_ex.append(test_score)
                else:
                    pr = model.predict(scaler.transform(test[self.features]))
                    test_score = mean_absolute_error(pr, test[self.target])
                    test_scores_by_ex.append(test_score/label_std)
                count +=1

        return sum(test_scores_by_ex)/count

    def get_cv_error(self):
        """Report cross-validation error for the model.

        To calculate cv_error "Leave-2-Groups-Out" cross-validation is used.
        For each train-test split,
        two experiments are used for testing
        and the rest are used for training.
        The reported error is the average over all train-test splits
        (accuracy for classification,
        normalized mean absolute error for regression)

        Returns
        -------
        cv_errors : float
            the cross-validation errors.
        """
        return self.cv_error


    def save_models(self, scaler_model, file_path=None):
        """Save model parameters and CV errors in YAML and .txt files.

        Parameters
        ----------
        scaler_model : dict
            Dictionary with training results.

        The results include:
        - 'scaler': sklearn standard scaler
            used for transforming of new data

        - 'model':sklearn model
            trained on new data

        - 'parameters': dict
            Dictionary of parameters found by hyperparameters_search()

        - 'accuracy': float
            average crossvalidation score (accuracy for classification,
            normalized mean absolute error for regression)

        file_path : str
            Full path to the YAML file where the models will be saved.
            Scaler, model, and cross-validation error
            will be saved at this path, and the cross-validation error
            are also saved in a .txt file of the same name, in the same directory.
        """
        if scaler_model['model'] is None:
            return

        self.scaler = scaler_model['scaler']
        self.model = scaler_model['model']
        self.parameters = scaler_model['parameters']
        self.cv_error = scaler_model['accuracy']

        if file_path is None:
            p = os.path.abspath(__file__)
            d = os.path.dirname(p)
            suffix = 0
            file_path = os.path.join(d,'modeling_data',
                'custom_models_'+ self.target +str(suffix)+'.yml')
            while os.path.exists(file_path):
                suffix += 1
                file_path = os.path.join(d,'modeling_data',
                    'custom_models_'+ self.target + str(suffix)+'.yml')

        file_path = file_path + '/' + self.target + '.yml'
        cverr_txt_path = os.path.splitext(file_path)[0]+'.txt'

        s_and_m = {'scaler': self.scaler.__dict__, 'model': self.model.__dict__,
                   'parameters' : self.parameters, 'accuracy': self.cv_error}

        # save scalers and models
        with open(file_path, 'w') as yaml_file:
            yaml.dump(s_and_m, yaml_file)

        # save accuracy
        with open(cverr_txt_path, 'w') as txt_file:
            txt_file.write(str(s_and_m['accuracy']))


    def train_partial(self, new_data, testing_data = None):
        """
        Parameters
        ----------
        new_data : pandas.DataFrame
            dataframe with new data for updating models
            (containing features and labels)
        testing_data : pandas.DataFrame
            dataframe with data we wish use for testing the models

        Returns
        -------
        results : dict
            Dictionary with training results.

        The results include:

        - 'scaler': sklearn standard scaler
            used for transforming of new data

        - 'model':sklearn model
            trained on new data

        - 'parameters': dict
            dictionary of parameters that were used to train the model
            (were not changed)

        - 'accuracy': float
            average crossvalidation score (accuracy for classification,
            normalized mean absolute error for regression)
        """
        new_scaler = None
        new_model = None
        new_err = None

        scaler_par = self.scaler.__dict__
        model_par = self.model.__dict__
        if scaler_par == None: # the model was not trained yet
            return {'scaler': new_scaler, 'model': new_model,
                'parameters' : self.parameters, 'accuracy': new_err}

        d = new_data[new_data[self.target].isnull() == False]
        data2 = d.dropna(subset=self.features)
        training_possible = self.check_label(data2)
        if not training_possible:
            print(self.target, "model was not updated.")# TODO decide what we should print (or not print) heare
            return {'scaler': new_scaler, 'model': new_model,
                'parameters' : self.parameters, 'accuracy': new_err}

        new_scaler = preprocessing.StandardScaler()
        set_param(new_scaler,scaler_par)
        if self.classifier:
            new_model = linear_model.SGDClassifier()
        else:
            new_model = linear_model.SGDRegressor()

        set_param(new_model,model_par)

        data2.loc[ : , self.features] = new_scaler.transform(data2[self.features])
        new_model.partial_fit(data2[self.features], data2[self.target])

        if testing_data is not None:
            d = testing_data[testing_data[self.target].isnull() == False]
            data = d.dropna(subset=self.features)
            if self.classifier:
                label_std = None
            else:
                label_std = pd.to_numeric(data[self.target]).std()

            if len(data.experiment_id.unique()) > 4:
                new_err = self.testing_by_experiments(data, new_model, label_std)
            else:
                new_err = self.testing_using_crossvalidation(data, new_model, label_std)

        return {'scaler': new_scaler, 'model': new_model,
                'parameters' : self.parameters, 'accuracy': new_err}
