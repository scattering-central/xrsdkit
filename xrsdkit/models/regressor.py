from collections import OrderedDict

import numpy as np
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

    def standardize(self,data):
        """Standardize the columns that are used as inputs and outputs.

        Reimplementation of XRSDModel.standardize():
        For the regression models the target must also be standardized,
        since the effects of model hyperparameters 
        are relative to the scale of the outputs. 
        """
        data = super(Regressor,self).standardize(data)
        self.scaler_y = preprocessing.StandardScaler() 
        self.scaler_y.fit(data[self.target].values.reshape(-1, 1))
        data[self.target] = self.scaler_y.transform(data[self.target].values.reshape(-1, 1))
        return data

    def predict(self, sample_features):
        """Predict this model's scalar target for a given sample. 

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of xrsdkit.tools.profiler.profile_pattern()

        Returns
        -------
        prediction : float
            predicted parameter value
        """
        if self.trained:
            feature_array = np.array(list(sample_features.values())).reshape(1,-1)
            feature_idx = [k in self.features for k in sample_features.keys()]
            x = self.scaler.transform(feature_array)[:, feature_idx]
            return float(self.scaler_y.inverse_transform(self.model.predict(x))[0])
        else:
            return self.default_val

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
        if not groups_possible: return False

        group_ids = range(1,n_groups+1)
        pc1 = PCA(n_components=1)
        data_pc = pc1.fit_transform(dataframe[feature_names]).ravel()
        pc_rank = np.argsort(data_pc)
        gp_size = int(round(dataframe.shape[0]/n_groups))
        groups = np.zeros(dataframe.shape[0])
        for igid,gid in enumerate(group_ids):
            groups[pc_rank[igid*gp_size:(igid+1)*gp_size]] = int(gid)
        dataframe.loc[:,'group_id'] = groups

        # check all groups for at least two distinct target values-
        # for any deficient groups, swap samples to balance
        val_cts_by_group = OrderedDict()
        deficient_gids = []
        for gid in group_ids:
            val_cts_by_group[gid] = dataframe.loc[dataframe.loc[:,'group_id']==gid,self.target].value_counts()
            if len(val_cts_by_group[gid]) < 2: deficient_gids.append(gid)

        temp_gid = max(group_ids)+1
        while len(deficient_gids) > 0:
            gid = deficient_gids.pop(0)
            val = val_cts_by_group[gid].keys()[0]
            ct = val_cts_by_group[gid][val]
            # find a group to swap with:
            # the swap group should have plenty of distinct values
            candidate_nvals = [len(vcts) for ggiidd,vcts in val_cts_by_group.items()]
            swap_gid = val_cts_by_group.keys()[np.argmax(candidate_nvals)]
            # swap the groups half-half:
            # 1. temporarily assign half of swap_gid to temp_gid 
            dataframe.loc[(dataframe.loc[:,'group_id']==swap_gid)[:round(ct/2)],'group_id'] = temp_gid
            # 2. assign half of gid to swap_gid 
            dataframe.loc[(dataframe.loc[:,'group_id']==gid)[:round(ct/2)],'group_id'] = swap_gid
            # 3. assign temp_gid to gid
            dataframe.loc[dataframe.loc[:,'group_id']==temp_gid,'group_id'] = gid
        return True 

    def cv_report(self,data,y_true,y_pred):
        y_true_all = []
        y_pred_all = []
        group_MAE = {}
        groupsize_weighted_MAE = {}
        for gid,yt in y_true.items():
            y_true_all.extend(yt)
        for gid,yp in y_pred.items():
            y_pred_all.extend(yp)
        for gid in y_true.keys():
            group_MAE[gid] = mean_absolute_error(y_true[gid],y_pred[gid])
            groupsize_weighted_MAE[gid] = group_MAE[gid]*y_true[gid].shape[0]/data.shape[0]
        result = dict(
            MAE = mean_absolute_error(y_true_all,y_pred_all),
            group_average_MAE = np.mean(list(group_MAE.values())),
            groupsize_weighted_average_MAE = np.sum(list(groupsize_weighted_MAE.values())),
            coef_of_determination = Rsquared(np.array(y_true_all),np.array(y_pred_all))
            )
        #result['minimization_score'] = result['MAE']
        result['minimization_score'] = -1*result['coef_of_determination']
        return result
