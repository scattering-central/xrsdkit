import pickle
import copy

import numpy as np
import yaml
from sklearn import preprocessing, utils
from sklearn.model_selection import LeavePGroupsOut
from dask_ml.model_selection import GridSearchCV 

from ..tools import primitives, profiler

class XRSDModel(object):

    def __init__(self, model_type, metric, label):
        self.model_type = model_type 
        self.metric = metric
        self.models_and_params = {}
        self.model = None
        self.scaler = preprocessing.StandardScaler()
        self.cross_valid_results = {} 
        self.target = label
        self.trained = False
        self.default_val = None
        self.features = []
        self.model = self.build_model()

    def load_model_data(self, model_data, pickle_file):
        if self.model_type == model_data['model_type'] \
        and self.target == model_data['model_target']:
            self.default_val = model_data['default_val']
            self.features = model_data['features']
            if model_data['trained']:
                self.trained = True
                setattr(self.scaler, 'mean_', np.array(model_data['scaler']['mean_']))
                setattr(self.scaler, 'scale_', np.array(model_data['scaler']['scale_']))
                self.model = pickle.load(open(pickle_file, 'rb'))
                self.cross_valid_results = model_data['cross_valid_results']
        else:
            raise ValueError('Tried to load modeling data with non-matching target or model type')

    def save_model_data(self, yml_path, txt_path, pickle_path):
        with open(yml_path,'w') as yml_file:
            yaml.dump(self.collect_model_data(),yml_file)
        with open(pickle_path,'wb') as pickle_file:
            pickle.dump(self.model, pickle_file, protocol=2)
        with open(txt_path,'w') as txt_file:
            if self.trained:
                res_str = self.print_CV_report()
            else:
                res_str = 'The model was not trained'
            txt_file.write(res_str)

    def collect_model_data(self):
        model_data = dict(
            model_type = self.model_type,
            metric = self.metric,
            model_target = self.target,
            scaler = dict(),
            model = dict(hyper_parameters=dict(), trained_par=dict()),
            cross_valid_results = primitives(self.cross_valid_results),
            trained = self.trained,
            default_val = primitives(self.default_val),
            features = self.features
            )
        if self.trained:
            hyper_par = list(self.models_and_params[self.model_type].keys())
            for p in hyper_par:
                if p in self.model.__dict__:
                    try:
                        model_data['model']['hyper_parameters'][p] = self.model.__dict__[p].tolist()
                    except:
                        model_data['model']['hyper_parameters'][p] = self.model.__dict__[p]
            # models are checked for several attributes before being used for predictions.
            # those attributes are listed here- if the model has any of them,
            # they must be saved so that they can be re-set when the model is loaded. 
            tr_par_arrays = ['coef_', 'intercept_', 'classes_']
            for p in tr_par_arrays:
                if p in self.model.__dict__:
                    model_data['model']['trained_par'][p] = self.model.__dict__[p].tolist()
            tr_par_ints = ['n_iter_','t_']
            for p in tr_par_ints:
                if p in self.model.__dict__:
                    try:
                        model_data['model']['trained_par'][p] = int(self.model.__dict__[p])
                    except TypeError:
                        model_data['model']['trained_par'][p] = self.model.__dict__[p]
            model_data['scaler']['mean_'] = self.scaler.__dict__['mean_'].tolist()
            model_data['scaler']['scale_'] = self.scaler.__dict__['scale_'].tolist()
        return model_data

    def build_model(self,model_hyperparams):
        # TODO: add a docstring that describes the interface
        msg = 'subclasses of XRSDModel must implement build_model()'
        raise NotImplementedError(msg)

    def train(self, model_data, train_hyperparameters=False, select_features=False):
        """Train the model, optionally searching for optimal hyperparameters.

        Parameters
        ----------
        model_data : pandas.DataFrame
            DataFrame containing features and labels for this model
        train_hyperparameters : bool
            If true, cross-validation metrics are used to select model hyperparameters 
        select_features : bool
            If true, before cross-validation, the model's default hyperparameters
            are used to recursively eliminate features
            based on best cross-validation metrics
        """
        training_possible = self.group_by_pc1(model_data,profiler.profile_keys)
        #training_possible = self.assign_groups(model_data)
        if not training_possible:
            # not enough samples, or all have identical labels-
            # take a non-standardized default value
            self.default_val = model_data[self.target].unique()[0]
            self.model = None
            self.features = []
            self.trained = False
            return
        else:
            s_model_data = self.standardize(model_data)
            # remove unlabeled samples
            s_model_data = s_model_data[s_model_data[self.target].isnull() == False]
            # maybe shuffle: SGD models train more efficiently on shuffled data
            if self.model_type in ['sgd_regressor','sgd_classifier']:
                s_model_data = utils.shuffle(s_model_data)
            # exclude samples with group_id==0
            valid_data = s_model_data[s_model_data.group_id>0]

            # begin by recursively eliminating features on a simple model (default parameters)
            model_feats = copy.deepcopy(profiler.profile_keys)
            if select_features:
                model_feats = self.cross_validation_rfe(valid_data,model_feats)
                test_model = self.build_model()
                test_model.fit(valid_data[model_feats], valid_data[self.target])
                cv = self.run_cross_validation(test_model,valid_data,model_feats)

            # use model_feats to grid-search hyperparameters
            model_hyperparams = {}
            if train_hyperparameters:
                test_model = self.build_model()
                param_grid = self.models_and_params[self.model_type]
                model_hyperparams = self.grid_search_hyperparams(test_model,valid_data,model_feats,param_grid)

            # after parameter and feature selection,
            # the entire dataset is used for final training,
            new_model = self.build_model(model_hyperparams)
            self.cross_valid_results = self.run_cross_validation(new_model,valid_data,model_feats)
            new_model.fit(valid_data[model_feats], valid_data[self.target])
            self.model = new_model
            self.features = model_feats 
            self.trained = True

    def standardize(self,data):
        """Standardize the columns of data that are used as model inputs"""
        data = data.copy()
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(data[profiler.profile_keys])
        data[profiler.profile_keys] = self.scaler.transform(data[profiler.profile_keys])
        return data

    def cross_validation_rfe(self,data,model_feats):
        model_outputs = np.ravel(data[self.target])
        cv_metrics = []
        rfe_feats = []
        test_model = self.build_model()
        cv = self.run_cross_validation(test_model,data,model_feats)
        cv_metrics.append(cv['minimization_score'])
        rfe_feats.append(copy.deepcopy(model_feats))
        nfeats = len(model_feats)
        for ifeat in range(1,nfeats):
            # removal based on coef_
            #test_model.fit(valid_data[model_feats], model_outputs)
            #lowest_coef_idx = np.argmin(np.abs(test_model.coef_))
            #worst_feat = model_feats[lowest_coef_idx]
            #model_feats.remove(worst_feat)
            feat_cv_metrics = []
            for feat in model_feats:
                trial_feats = copy.deepcopy(model_feats)
                trial_feats.remove(feat)
                cv = self.run_cross_validation(test_model,data,trial_feats)
                feat_cv_metrics.append(cv['minimization_score'])
            best_cv_idx = np.argmin(feat_cv_metrics)
            cv_metrics.append(feat_cv_metrics[best_cv_idx])
            worst_feat = model_feats.pop(best_cv_idx)
            rfe_feats.append(copy.deepcopy(model_feats))
        best_feats_idx = np.argmin(cv_metrics)
        best_feats = rfe_feats[best_feats_idx]
        #print(best_feats)
        #from matplotlib import pyplot as plt
        #plt.plot(range(1,nfeats+1),cv_metrics[::-1])
        #plt.xlabel('remaining features')
        #plt.ylabel('minimization score')
        #plt.show()
        return best_feats

    def group_by_pc1(self,dataframe,feature_names,n_groups=5):
        """Group samples by dividing them along the first principal component.

        For regressors, this grouping should be applied once to the whole dataset.        
        For classifiers, this grouping should be applied once for each possible label.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame containing modeling dataset
        feature_names : list
            Column headers that are used as input features
        n_groups : int
            Number of groups to create
        """
        raise NotImplementedError('XRSDModel subclasses must implement group_by_pc1()')

    def cv_report(self,data,y_true,y_pred):
        """Yield key cross-validation metrics.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing modeling dataset
        y_true : dict of np.array 
            Keys are group_ids, values are arrays of y-values for the group 
        y_pred : dict of np.array 
            Keys are group_ids, values are predictions corresponding to `y_true` 
        """
        raise NotImplementedError('XRSDModel subclasses must implement cv_report()')

    def _diverse_groups_possible(self,dataframe,n_groups,n_distinct):
        # check for at least n_groups sets of n_distinct distinct values:
        val_counts = dataframe.loc[:,self.target].value_counts()
        if len(val_counts)<n_distinct: 
            return False
        distinct_value_counts = np.zeros(n_groups)
        for val,ct in val_counts.items(): 
            if ct >= n_groups: 
                distinct_value_counts += 1
            else:
                # add to the emptiest buckets first
                gp_order = np.argsort(distinct_value_counts)
                for ict in range(ct):
                    distinct_value_counts[gp_order[ict]] += 1
        if any([ct<n_distinct for ct in distinct_value_counts]): return False
        return True

    def run_cross_validation(self,model,data,feature_names):
        """Cross-validate a model by LeaveOneGroupOut. 

        The train/test groupings are defined by the 'group_id' labels,
        which are added to the dataframe during self.assign_groups().

        Parameters
        ----------
        model : object 
            scikit-learn model to be cross-validated
        data : pandas.DataFrame
            pandas dataframe of features and labels
        feature_names : list of str
            list of feature names (column headers) used for training

        Returns
        -------
        result : dict
            dictionary of cross validation metrics.
        """
        y_true = {} 
        y_pred = {} 
        group_ids = data.group_id.unique()
        for gid in group_ids:
            tr = data[(data['group_id'] != gid)]
            test = data[(data['group_id'] == gid)]
            model.fit(tr[feature_names], tr[self.target])
            yp = model.predict(test[feature_names])
            yt = test[self.target] 
            y_pred[gid] = yp
            y_true[gid] = yt
        return self.cv_report(data,y_true,y_pred)

    def grid_search_hyperparams(self,model,data,feature_names,hyperparam_grid,n_leave_out=1):
        cv_splits = LeavePGroupsOut(n_groups=n_leave_out).split(
            data[feature_names],
            np.ravel(data[self.target]),
            groups=data['group_id']
            )
        gs_models = GridSearchCV(model,hyperparam_grid,cv=cv_splits,scoring=self.metric,n_jobs=-1)
        gs_models.fit(data[feature_names], np.ravel(data[self.target]))
        return gs_models.best_params_

