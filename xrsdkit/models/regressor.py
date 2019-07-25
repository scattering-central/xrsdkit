from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans

from .xrsd_model import XRSDModel
from ..tools import profiler, Rsquared


class Regressor(XRSDModel):
    """Class for generating models to predict real-valued parameters."""

    def __init__(self, model_type, metric, label):
        super(Regressor,self).__init__(model_type, metric, label)
        self.scaler_y = preprocessing.StandardScaler() 
        self.models_and_params = dict(
            ridge_regressor = dict(
                alpha = np.logspace(0,2,num=5,endpoint=True,base=10.)
                ),
            elastic_net = dict(
                alpha = np.logspace(0,2,num=5,endpoint=True,base=10.),
                l1_ratio = np.linspace(0,1,num=5,endpoint=True)
                ),
            sgd_regressor = dict(
                epsilon = [1, 0.1, 0.01],
                alpha = [0.0001, 0.001, 0.01],
                l1_ratio = [0., 0.15, 0.5, 0.85, 1.0]
                )
            )

    def build_model(self, model_hyperparams={}):
        if self.model_type == 'ridge_regressor':
            alpha = 1.
            if 'alpha' in model_hyperparams: alpha = model_hyperparams['alpha']
            new_model = linear_model.Ridge(alpha=alpha, max_iter=10000)
        elif self.model_type == 'elastic_net':
            alpha = 1.
            if 'alpha' in model_hyperparams: alpha = model_hyperparams['alpha']
            l1_ratio = 0.5 
            if 'l1_ratio' in model_hyperparams: l1_ratio = model_hyperparams['l1_ratio']
            new_model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        elif self.model_type == 'sgd_regressor':
            new_model = self.build_sgd_model(model_hyperparams)
        else:
            raise ValueError('Unrecognized model type: {}'.format(self.model_type))
        return new_model

    def build_sgd_model(self, model_hyperparams={}):
        alpha = 1.E-4
        if 'alpha' in model_hyperparams: alpha = model_hyperparams['alpha']
        l1_ratio = 0.15 
        if 'l1_ratio' in model_hyperparams: l1_ratio = model_hyperparams['l1_ratio']
        epsilon = 0.15 
        if 'epsilon' in model_hyperparams: epsilon = model_hyperparams['epsilon']
        new_model = linear_model.SGDRegressor(alpha=alpha, l1_ratio=l1_ratio, epsilon=epsilon,
            loss='huber', penalty='elasticnet', max_iter=10000, tol=1.E-3)
        return new_model

    def load_model_data(self,model_data, pickle_file):
        super(Regressor,self).load_model_data(model_data, pickle_file)
        self.scaler_y = preprocessing.StandardScaler()
        if self.trained:
            setattr(self.scaler_y, 'mean_', np.array(model_data['scaler_y']['mean_']))
            setattr(self.scaler_y, 'scale_', np.array(model_data['scaler_y']['scale_']))

    def collect_model_data(self):
        model_data = super(Regressor,self).collect_model_data()
        if self.trained:
            model_data['scaler_y'] = dict(
                mean_ = self.scaler_y.__dict__['mean_'].tolist(),
                scale_ = self.scaler_y.__dict__['scale_'].tolist()
                )
        return model_data

    def standardize(self,data,features):
        """Standardize the columns that are used as inputs and outputs.

        Reimplementation of XRSDModel.standardize():
        For the regression models the target must also be standardized,
        since the effects of model hyperparameters 
        are relative to the scale of the outputs. 

        Parameters
        ----------
        data : pandas.DataFrame
            modeling dataset
        features : list
            features to be standardized

        Returns
        -------
        s_data : pandas.DataFrame
        """
        s_data = super(Regressor,self).standardize(data,features)
        self.scaler_y = preprocessing.StandardScaler() 
        self.scaler_y.fit(data[self.target].values.reshape(-1, 1))
        s_data[self.target] = self.scaler_y.transform(data[self.target].values.reshape(-1, 1))
        return s_data

    def predict(self,data):
        """Run predictions for each row of input `data`.

        Each row of `data` represents one sample.
        The `data` columns are assumed to match self.features.

        Parameters
        ----------
        data : array-like
        
        Returns
        -------
        preds : array
        """
        if self.trained and data.shape[0]>0:
            X = self.scaler.transform(data)
            preds = self.scaler_y.inverse_transform(self.model.predict(X))
        else:
            preds = self.default_val*np.ones(data.shape[0])
        return preds 

    def get_cv_summary(self):
        return dict(model_type=self.model_type,
                    scores={k:self.cross_valid_results.get(k,None) for k in ['MAE','coef_of_determination']})

    def print_CV_report(self):
        """Return a string describing the model's cross-validation metrics.

        Returns
        -------
        CV_report : str
            string with formated results of cross validatin.
        """
        # TODO: add sample_ids, groupings, and group sizes 
        # TODO: add feature names 
        # TODO: document the computation of these metrics, 
        # then refer to documentation in this report
        CV_report = 'Cross validation results for {} Regressor\n\n'.format(self.target) + \
            'Mean absolute error: {}\n\n'.format(
            self.cross_valid_results['MAE']) + \
            'Coefficient of determination (R^2): {}\n\n'.format(
            self.cross_valid_results['coef_of_determination']) 
        return CV_report

    def group_by_pc1(self,dataframe,feature_names,n_groups=5):
        groups_possible = self._diverse_groups_possible(dataframe,n_groups,2)
        group_ids = pd.Series(np.zeros(dataframe.shape[0]),index=dataframe.index,dtype=int)
        if not groups_possible: 
            return group_ids, False

        gids = range(1,n_groups+1)
        pc1 = PCA(n_components=1)
        data_pc = pc1.fit_transform(dataframe[feature_names]).ravel()
        pc_rank = np.argsort(data_pc)
        gp_size = int(round(dataframe.shape[0]/n_groups))
        #groups = np.zeros(dataframe.shape[0])
        for igid,gid in enumerate(gids):
            group_ids.iloc[pc_rank[igid*gp_size:(igid+1)*gp_size]] = int(gid)

        # check all groups for at least two distinct target values-
        # for any deficient groups, swap samples to balance
        val_cts_by_group = OrderedDict()
        deficient_gids = []
        for gid in gids:
            val_cts_by_group[gid] = dataframe.loc[(group_ids==gid),self.target].value_counts()
            if len(val_cts_by_group[gid]) < 2: deficient_gids.append(gid)

        temp_gid = max(gids)+1
        while len(deficient_gids) > 0:
            gid = deficient_gids.pop(0)
            val = val_cts_by_group[gid].keys()[0]
            ct = val_cts_by_group[gid][val]
            # find a group to swap with:
            # the swap group should have plenty of distinct values
            candidate_nvals = [len(vcts) for ggiidd,vcts in val_cts_by_group.items()]
            swap_gid = list(val_cts_by_group.keys())[np.argmax(candidate_nvals)]
            # swap the groups half-half:
            # 1. temporarily assign half of swap_gid to temp_gid 
            group_ids.loc[group_ids==swap_gid][:int(round(ct/2))] = temp_gid
            # 2. assign half of gid to swap_gid 
            group_ids.loc[group_ids==gid][:int(round(ct/2))] = swap_gid
            # 3. assign temp_gid to gid
            group_ids.loc[group_ids==temp_gid] = gid
        return group_ids, True 

    def cv_report(self,data,y_true,y_pred):
        group_MAE = {}
        groupsize_weighted_MAE = {}
        all_gids = data['group_id'].unique()
        gids = np.array(data['group_id'])
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        for gid in all_gids:
            gid_idx = gids==gid
            group_MAE[gid] = mean_absolute_error(y_true[gid_idx],y_pred[gid_idx])
            groupsize_weighted_MAE[gid] = group_MAE[gid]*y_true[gid_idx].shape[0]/data.shape[0]
        result = dict(
            MAE = mean_absolute_error(y_true,y_pred),
            group_average_MAE = np.mean(list(group_MAE.values())),
            groupsize_weighted_average_MAE = np.sum(list(groupsize_weighted_MAE.values())),
            coef_of_determination = Rsquared(np.array(y_true),np.array(y_pred))
            )
        #result['minimization_score'] = result['MAE']
        result['minimization_score'] = -1*result['coef_of_determination']
        return result
