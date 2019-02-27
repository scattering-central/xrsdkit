import random

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
        self.features = []

        if yml_file:
            content = yaml.load(open(yml_file,'rb'))
            self.load_model_data(content)
        else:
            self.set_model()

    def load_model_data(self,model_data):
        self.trained = model_data['trained']
        self.default_val = model_data['default_val']
        if self.trained:
            self.features = model_data['features']
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

    def run_cross_validation(self,model,data,feature_names):
        # TODO: add a docstring that describes the interface
        msg = 'subclasses of XRSDModel must implement run_cross_validation()'
        raise NotImplementedError(msg)

    def validate_feature_set(self, model, df, feature_names):
        msg = 'subclasses of XRSDModel must implement validate_feature_set()'
        raise NotImplementedError(msg)

    def train(self, model_data, hyper_parameters_search=False):
        """Train the model, optionally searching for optimal hyperparameters.

        Parameters
        ----------
        model_data : pandas.DataFrame
            dataframe containing features and labels for this model.
        hyper_parameters_search : bool
            If true, grid-search model hyperparameters
            to seek high cross-validation accuracy.
        """
        # copy the model dataframe: this avoids pandas SettingWithCopyWarning
        # TODO: find a more elegant solution to the SettingWithCopyWarning
        model_data = model_data.copy()
        model_data = self.standardize(model_data)
        training_possible = self.assign_groups(model_data)
        if not training_possible:
            # not enough samples, or all have identical labels
            self.default_val = model_data[self.target].unique()[0]
            self.trained = False
            return
        else:
            # NOTE: SGD models train more efficiently on shuffled data
            d = utils.shuffle(model_data)
            data = d[d[self.target].isnull() == False]
            # exclude samples with group_id==0
            valid_data = data[data.group_id>0]
            '''
            if hyper_parameters_search:
                new_parameters = self.hyperparameters_search(valid_data, n_leave_out=1)
                new_model = self.build_model(new_parameters)
            else:
                new_model = self.model
            '''

            '''

            cv = model_selection.LeavePGroupsOut(n_groups=1).split(
                valid_data[profiler.profile_keys], np.ravel(valid_data[self.target]),
                                                                  groups=valid_data['group_id'])

            rfecv = RFECV(estimator=self.model, step=1, cv=cv, scoring=self.score, min_features_to_select=3, n_jobs=-1)
            rfecv.fit(valid_data[profiler.profile_keys], valid_data[self.target])

            print("Optimal number of features : %d" % rfecv.n_features_)
            print(rfecv.score(valid_data[profiler.profile_keys], valid_data[self.target]))
            print(rfecv.support_ )
            print(rfecv.grid_scores_)

            best_features = [profiler.profile_keys[i] for i in range(len(profiler.profile_keys)) if rfecv.support_[i]]
            '''
            # features selection
            features = []
            features.extend(profiler.profile_keys)
            score_features = []
            while len(features) > 2:
                if hyper_parameters_search:
                    new_parameters = self.hyperparameters_search(valid_data, features, n_leave_out=1)
                    new_model = self.build_model(new_parameters)
                else:
                    new_model = self.model
                optimization_obj, ind = self.validate_feature_set(new_model,valid_data,features)
                score_features.append((optimization_obj, features.copy()))
                del features[ind]
            score_features.sort()
            best_features = score_features[0][1]
            # NOTE: after cross-validation for parameter selection,
            # the entire dataset is used for final training,
            #self.cross_valid_results = self.run_cross_validation(new_model,valid_data,profiler.profile_keys)
            #new_model.fit(valid_data[profiler.profile_keys], valid_data[self.target])
            self.cross_valid_results = self.run_cross_validation(new_model,valid_data,best_features)
            new_model.fit(valid_data[best_features], valid_data[self.target])
            self.model = new_model
            self.features = best_features
            self.trained = True

    def standardize(self,data):
        """Standardize the columns of data that are used as model inputs"""
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(data[profiler.profile_keys])
        data[profiler.profile_keys] = self.scaler.transform(data[profiler.profile_keys])
        return data

    def hyperparameters_search(self,transformed_data,features,group_by='group_id',n_leave_out=1,scoring=None):
        """Grid search for optimal alpha, penalty, and l1 ratio hyperparameters.

        Parameters
        ----------
        transformed_data : pandas.DataFrame
            dataframe containing features and labels;
            note that the features should be transformed/standardized beforehand
        features : list of str
            list of features to use
        group_by: string
            DataFrame column header for LeavePGroupsOut(groups=group_by)
        n_leave_out: integer
            number of groups to leave out, if group_by is specified 
        scoring : str
            Selection of scoring function.
            if None, the default scoring function of the model will be used

        Returns
        -------
        clf.best_params_ : dict
            Dictionary of the best found hyperparameters.
        """
        if n_leave_out:
            cv = model_selection.LeavePGroupsOut(n_groups=n_leave_out).split(
                transformed_data[features],
                np.ravel(transformed_data[self.target]),
                groups=transformed_data[group_by]
                )
        else:
            cv = 3 # number of folds for cross validation
        test_model = self.build_model()
        # threaded scheduler with optimal number of threads
        # will be used by default for dask GridSearchCV
        clf = GridSearchCV(test_model,self.grid_search_hyperparameters,cv=cv,scoring=scoring,n_jobs=-1)
        clf.fit(transformed_data[features], np.ravel(transformed_data[self.target]))
        return clf.best_params_

    def assign_groups(self, dataframe):
        """Assign cross-validation groups to all samples in input `dataframe`.
 
        A `group_id` column is added to `dataframe` for cross-validation grouping.
        All rows of `dataframe` are assumed to have valid target values-
        unlabeled samples should be filtered out before calling this.
        The base class implementation simply assigns the groups 
        based on the `experiment_id` labels.
        A sample missing an `experiment_id` label gets a `group_id` of 0. 
        Returns False if the data are insufficient for model training,
        i.e., if the target values are all identical.
        This does NOT return False if there is only one group-
        the splitting of a monolithic group should occur in subclasses, 
        based on the grouping requirements of the subclass.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            dataframe of sample features and corresponding labels

        Returns
        -------
        trainable : bool
            indicates whether or not training is possible
        """
        all_exp_ids = dataframe['experiment_id'].unique()
        all_labels = dataframe[self.target].unique()
        trainable = len(all_labels)>1
        group_ids = np.zeros(dataframe.shape[0],dtype=int)
        gid = 1
        for exp_id in all_exp_ids:
            # all samples without an experiment_id get stuck with group_id==0
            if exp_id:
                group_ids[dataframe['experiment_id']==exp_id] = gid
                gid += 1 
        dataframe.loc[:,'group_id'] = group_ids
        return trainable

    @staticmethod
    def shuffle_split_3fold(nsamp):
        group_ids = np.zeros(nsamp,dtype=int)
        nsamp1 = nsamp//3
        nsamp2 = (nsamp-nsamp1)//2
        all_idx = range(nsamp)
        idx_group1 = random.sample(all_idx,nsamp1)
        all_idx = [idx for idx in all_idx if not idx in idx_group1]
        idx_group2 = random.sample(all_idx,nsamp2)
        idx_group3 = [idx for idx in all_idx if not idx in idx_group2]
        group_ids[idx_group1] = 1
        group_ids[idx_group2] = 2
        group_ids[idx_group3] = 3
        return group_ids

