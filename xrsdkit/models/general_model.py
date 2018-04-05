import os
import numpy as np
import pandas as pd
import yaml
from sklearn import model_selection, preprocessing, linear_model
from sklearn.metrics import mean_absolute_error

from ..tools import profiler
from . import set_param

class XrsdModel(object):

    def __init__(self, label, yml_file=None, classifier = True):
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
        dict wtih : # TODO update discription
        new_scalers : sklearn standard scaler
            used for transforming of new data.
        new_models : sklearn model
            trained on new data.
        new_parameters : dict
            Dictionary of parameters found by hyperparameters_search().
        new_accuracy : float
            Accuracy of the model.
        """
        d = all_data[all_data[self.target].isnull() == False]
        training_possible = self.check_label(d)

        new_scaler = None
        new_model = None
        new_accuracy = None
        new_parameters = None

        if not training_possible:
            return new_scaler, new_model,new_parameters, new_accuracy # TODO we also need to return a warning "The model was not updated"

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
        #data.loc[ : , features] = scaler.transform(data[features])

        print(self.target, leaveTwoGroupOut)


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
                    penalty=new_parameters["penalty"], l1_ratio=new_parameters["l1_ratio"])
            else:
                new_model = linear_model.SGDClassifier(loss='log')
        else:
            if new_parameters:
                new_model = linear_model.SGDRegressor(alpha=new_parameters['alpha'], loss= new_parameters['loss'],
                                        penalty = new_parameters["penalty"],l1_ratio = new_parameters["l1_ratio"],
                                        epsilon = new_parameters["epsilon"], max_iter=1000)
            else:
                new_model = linear_model.SGDRegressor(max_iter=1000)

        new_model.fit(transformed_data, data[self.target])

        if self.classifier:
            label_std = None
        else:
            label_std = pd.to_numeric(data[self.target]).std()# usefull for regressin only
            print(self.target, label_std)

        if leaveTwoGroupOut:
            new_accuracy = self.testing_by_experiments(data, new_model, label_std)
        else:
            new_accuracy = self.testing_using_crossvalidation(data, new_model,label_std)


        #return {'scaler': new_scaler.__dict__, 'model': new_model.__dict__,
                #'parameters' : new_parameters, 'accuracy': new_accuracy}
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
            model = linear_model.SGDClassifier(loss='log')
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
                    model,scaler.transform(df[self.features]), df[self.target], cv=5, scoring = 'neg_mean_absolute_error')
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
            average crossvalidation score (accuracy for classification,
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

    def training_cv_error(self):
        """Report cross-validation error for the model.

        "Leave-2-Groups-Out" cross-validation is used.
        For each train-test split,
        two experiments are used for testing
        and the rest are used for training.
        The reported error is the average over all train-test splits.
        TODO: what is the error metric for the classifier?
        TODO: verify that this docstring is correct

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
        scaler_model : dict with  #TODO update
        new_scaler : sklearn scaler
        new_model : sklearn model
            with specific parameters
        new_parameters : dict
            Dictionary of parameters found by hyperparameters_search().
        cv_errors : float
            classifier: accuracy
            regression: normalized cross-validation errors each model.
        file_path : str
            Full path to the YAML file where the models will be saved.
            Scaler, model, and cross-validation error
            will be saved at this path, and the cross-validation error
            are also saved in a .txt file of the same name, in the same directory.
        """
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


        #if not os.path.splitext(file_path)[1] == '.yml':
            #file_path = file_path+'.yml'
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

'''


def train_classifiers_partial(new_data, file_path=None, all_training_data=None, model='all'):
    """Read SAXS classification models from a YAML file, then update them with new data.

    Parameters
    ----------
    new_data : pandas.DataFrame
        dataframe containing features and labels for updating models.
    file_path : str (optional)
        Full path to YAML file where scalers and models are saved.
        If None, the default saxskit models are used.
    all_training_data : pandas.DataFrame (optional)
        dataframe containing all of the original training data,
        for computing cross-validation errors of the updated models.
    model : str
        the name of model to train ("unidentified", "spherical_normal",
        "guinier_porod", "diffraction_peaks", or "all" to train all models).

    Returns
    -------
    scalers : dict
        Dictionary of sklearn standard scalers (one scaler per model).
    models : dict
        Dictionary of sklearn models.
    cv_errors : dict
        Dictionary of cross-validation errors for each model.
    """
    if file_path is None:
        p = os.path.abspath(__file__)
        d = os.path.dirname(p)
        file_path = os.path.join(d,'modeling_data','scalers_and_models.yml')
    s_and_m_file = open(file_path,'rb')
    s_and_m = yaml.load(s_and_m_file)

    models = s_and_m['models']
    scalers = s_and_m['scalers']
    cv_errors = s_and_m['accuracy']

    possible_models = check_labels(all_training_data)

    if model != 'all':
        for k in possible_models.keys():
            if k != model:
                # we do not want to train the other models
                possible_models[k] = False

    features = profile_keys['unidentified']

    # unidentified scatterer population model
    if possible_models['unidentified'] == True:
        scaler, model, cverr = train_partial(True, new_data, features, 'unidentified',
                                           models, scalers, all_training_data)
        if scaler:
            scalers['unidentified'] = scaler.__dict__
        if model:
            models['unidentified'] = model.__dict__
        if cverr:
            cv_errors['unidentified'] = cverr

    # For the rest of the models,
    # we will use only data with
    # identifiable scattering populations
    new_data = new_data[new_data['unidentified']==False]

    for k, v in possible_models.items():
        if v == True and k != 'unidentified':
            scaler, model, cverr = train_partial(True, new_data, features, k,
                                           models, scalers, all_training_data)
            if scaler:
                scalers[k] = scaler.__dict__
            if model:
                models[k] = model.__dict__
            if cverr:
                cv_errors[k] = cverr
    if all_training_data is None:
        cv_errors['NOTE'] = 'Cross-validation errors '\
        'were not re-computed after partial model training'
    return scalers, models, cv_errors

def train_regressors_partial(new_data, file_path=None, all_training_data=None, model='all'):
    """Read SAXS regression models from a YAML file, then update them with new data.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels for updating models
    file_path : str (optional)
        Full path to YAML file where scalers and models are saved.
        If None, the default saxskit models are used.
    all_training_data : pandas.DataFrame (optional)
        dataframe containing all of the original training data
        for computing the accuracy of the updated models.
    model : str
        the name of model to train ("r0_sphere", "sigma_sphere",
        "rg_gp", or "all" to train all models).

    Returns
    -------
    scalers : dict
        Dictionary of sklearn standard scalers (one scaler per model).
    models : dict
        Dictionary of sklearn models.
    accuracy : dict
        Dictionary of accuracies for each model.
    """
    if file_path is None:
        p = os.path.abspath(__file__)
        d = os.path.dirname(p)
        file_path = os.path.join(d,'modeling_data','scalers_and_models_regression.yml')

    s_and_m_file = open(file_path,'rb')
    s_and_m = yaml.load(s_and_m_file)
    models = s_and_m['models']
    scalers = s_and_m['scalers']
    cv_errors = s_and_m['accuracy']

    possible_models = check_labels_regression(new_data)

    if model != 'all':
        for k in possible_models.keys():
            if k != model:
                # we do not want to train the other models
                possible_models[k] = False

    # r0_sphere model
    if possible_models['r0_sphere'] == True:
        features = []
        features.extend(profile_keys['unidentified'])
        scaler, model, cverr = train_partial(False, new_data, features, 'r0_sphere',
                                           models, scalers, all_training_data)
        if scaler:
            scalers['r0_sphere'] = scaler.__dict__
        if model:
            models['r0_sphere'] = model.__dict__
        if cverr:
            cv_errors['r0_sphere'] = cverr

    # sigma_shpere model
    if possible_models['sigma_sphere'] == True:
        features = []
        features.extend(profile_keys['unidentified'])
        features.extend(profile_keys['spherical_normal'])
        scaler, model, cverr = train_partial(False, new_data, features, 'sigma_sphere',
                                           models, scalers, all_training_data)
        if scaler:
            scalers['sigma_sphere'] = scaler.__dict__
        if model:
            models['sigma_sphere'] = model.__dict__
        if cverr:
            cv_errors['sigma_sphere'] = cverr

    # rg_gp model
    if possible_models['rg_gp'] == True:
        features = []
        features.extend(profile_keys['unidentified'])
        features.extend(profile_keys['guinier_porod'])
        scaler, model, cverr = train_partial(False, new_data, features, 'rg_gp',
                                           models, scalers, all_training_data)
        if scaler:
            scalers['rg_gp'] = scaler.__dict__
        if model:
            models['rg_gp'] = model.__dict__
        if cverr:
            cv_errors['rg_gp'] = cverr
    if all_training_data is None:
        cv_errors['NOTE'] = 'Cross-validation errors '\
        'were not re-computed after partial model training'
    return scalers, models, cv_errors

# helper function - to set parametrs for scalers and models
def set_param(m_s, param):
        for k, v in param.items():
            if isinstance(v, list):
                setattr(m_s, k, np.array(v))
            else:
                setattr(m_s, k, v)

def train_partial(classifier, data, features, target, reg_models_dict, scalers_dict, testing_data):
    model_params = reg_models_dict[target]
    scaler_params = scalers_dict[target]
    if scaler_params is not None:
        scaler = preprocessing.StandardScaler()
        set_param(scaler,scaler_params)
        if classifier == True:
            model = linear_model.SGDClassifier()
        else:
            model = linear_model.SGDRegressor()
        set_param(model,model_params)
        d = data[data[target].isnull() == False]
        data2 = d.dropna(subset=features)
        scaler.fit(data2[features])
        data2.loc[ : , features] = scaler.transform(data2[features])
        model.partial_fit(data2[features], data2[target])
        if testing_data is None:
            accuracy = None
        else: # calculate training accuracy using all provided data
            d = testing_data[testing_data[target].isnull() == False]
            data = d.dropna(subset=features)
            if len(data.experiment_id.unique()) > 4:
                leaveNGroupOut = True
            else:
                leaveNGroupOut = False
            label_std = data[target].std()
            if leaveNGroupOut:
                if classifier == True:
                    accuracy = testing_by_experiments(
                        data, target, features, model_params['alpha'], model_params['l1_ratio'],
                        model_params['penalty'])
                else:
                    accuracy = testing_by_experiments_regression(
                        data, target, features, model_params['alpha'], model_params['l1_ratio'],
                        model_params['penalty'], model_params['loss'],
                        model_params['epsilon'], label_std)
            else:
                if classifier == True:
                    accuracy = testing_using_crossvalidation(
                        data, target, features,  model_params['alpha'], model_params['l1_ratio'],
                        model_params['penalty'])
                else:
                    accuracy = testing_using_crossvalidation_regression(
                        data, target, features,  model_params['alpha'], model_params['l1_ratio'],
                        model_params['penalty'], model_params['loss'],
                        model_params['epsilon'], label_std)
    else:
        scaler = None
        model = None
        accuracy = None
    return scaler, model, accuracy
'''



