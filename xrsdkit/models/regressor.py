import copy

import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans

from .xrsd_model import XRSDModel
from ..tools import profiler


class Regressor(XRSDModel):
    """Class for generating models to predict real-valued parameters."""

    def __init__(self,label,yml_file):
        self.scaler_y = None
        super(Regressor,self).__init__(label, yml_file)
        self.grid_search_hyperparameters = dict(
            epsilon = [10, 1, 0.1, 0.01, 0.001, 0],
            alpha = [0.00001, 0.0001, 0.001, 0.01], # regularisation coef, default 0.0001
            l1_ratio = [0, 0.15, 0.5, 0.85, 1.0] # default 0.15
            )

    def load_model_data(self,model_data):
        super(Regressor,self).load_model_data(model_data)
        self.scaler_y = preprocessing.StandardScaler()
        if self.trained:
            setattr(self.scaler_y, 'mean_', np.array(model_data['scaler_y']['mean_']))
            setattr(self.scaler_y, 'scale_', np.array(model_data['scaler_y']['scale_']))

    def build_model(self, model_hyperparams={}):
        if all([p in model_hyperparams for p in ['alpha','l1_ratio','epsilon']]):
            new_model = linear_model.SGDRegressor(
                    alpha=model_hyperparams['alpha'],
                    l1_ratio=model_hyperparams['l1_ratio'],
                    epsilon=model_hyperparams['epsilon'],
                    loss= 'huber',
                    penalty='elasticnet',
                    max_iter=10000, tol=1e-5)
        else:
            # NOTE: max_iter is about 10^6 / number of tr samples 
            new_model = linear_model.SGDRegressor(loss= 'huber',
                    penalty='elasticnet',max_iter=10000, tol=1e-5)
        return new_model

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

        feature_array = np.array(list(sample_features.values())).reshape(1,-1)
        x = self.scaler.transform(feature_array)

        return float(self.scaler_y.inverse_transform(self.model.predict(x))[0])

    def print_mean_abs_errors(self):
        result = ''
        for r in self.cross_valid_results['normalized_mean_abs_error_by_splits'].split():
            result += (r + '\n')
        return result

    def average_mean_abs_error(self,weighted=False):
        if weighted:
            return str(self.cross_valid_results['weighted_av_mean_abs_error'])
        else:
            return str(self.cross_valid_results['normalized_mean_abs_error'])

    def print_CV_report(self):
        """Return a string describing the model's cross-validation metrics.

        Returns
        -------
        CV_report : str
            string with formated results of cross validatin.
        """
        CV_report = 'Cross validation results for {} Regressor\n'.format(self.target) + \
            'Data from {} experiments was used\n\n'.format(
            str(self.cross_valid_results['number_of_experiments'])) + \
            'Normalized mean abs error by train/test split:\n' + \
            self.print_mean_abs_errors() + '\n' + \
            'Weighted-average mean abs error by train/test split: {}\n'.format(self.average_mean_abs_error(True)) + \
            'Unweighted-average mean abs error: {}\n'.format(self.average_mean_abs_error(False)) + \
            '\n\nNOTE: Weighted metrics are weighted by test set size' 
        return CV_report

    def assign_groups(self, dataframe, min_groups=5):
        """Assign train/test groups to `dataframe`.

        This reimplementation invokes the base class method, 
        and then updates the groups if necessary
        to ensure there are at least min_groups groups,
        where each group has nonzero variance for for the target.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            dataframe of sample features and corresponding labels

        Returns
        -------
        trainable : bool
            indicates whether or not training is possible
        """
        trainable = super(Regressor,self).assign_groups(dataframe)

        # check for at least min_groups pairs of distinct values:
        count_0 = 0
        count_1 = 0
        val_counts = dataframe.loc[dataframe['group_id']>0,self.target].value_counts()
        sorted_counts = np.sort(list(val_counts.values))[::-1]
        for ct in sorted_counts: 
            if count_0 < min_groups: 
                count_0 += ct
            else:
                count_1 += ct
        if count_0 < min_groups or count_1 < min_groups: return False

        if trainable:
            group_ids = dataframe.loc[dataframe['group_id']>0,'group_id'].unique()
            gp_counts = dataframe.loc[dataframe['group_id']>0,'group_id'].value_counts()

            # if we have too few groups, use KMeans to split them until we have enough 
            while len(group_ids) < min_groups: 
                gid_to_split = gp_counts.keys()[gp_counts.values.argmax()]
                new_gid = 1
                while new_gid in group_ids: new_gid += 1
                # TODO: come up with a more balanced clustering mechanism
                km = KMeans(2,n_init=10)
                new_gids = km.fit_predict(dataframe.loc[dataframe['group_id']==gid_to_split,profiler.profile_keys])
                idx_gp0 = new_gids==0
                idx_gp1 = new_gids==1
                new_gids[idx_gp0] = gid_to_split
                new_gids[idx_gp1] = new_gid
                dataframe.loc[dataframe['group_id']==gid_to_split,'group_id'] = new_gids
                group_ids = dataframe.loc[dataframe['group_id']>0,'group_id'].unique()
                gp_counts = dataframe.loc[dataframe['group_id']>0,'group_id'].value_counts()

            # if we have any groups with zero variance,
            # balance with samples from other groups
            groups_to_check = list(group_ids.copy())
            while len(groups_to_check) > 0:
                gid = groups_to_check.pop(0) 
                vals = dataframe.loc[dataframe['group_id']==gid,self.target]
                if np.std(vals) == 0:
                    val = vals.iloc[0]
                    # find the group with the greatest population of values not equal to val
                    gp_otherval_counts = gp_counts.copy()
                    for ggiidd in gp_otherval_counts.keys():
                        val_cts = dataframe.loc[dataframe['group_id']==ggiidd,self.target].value_counts()
                        if val in val_cts.keys():
                            gp_otherval_counts[ggiidd] = gp_otherval_counts[ggiidd]-val_cts[val]
                    gid_to_split = gp_otherval_counts.keys()[gp_otherval_counts.values.argmax()]

                    # split the gid_to_split, reassign
                    # TODO: come up with a more balanced clustering mechanism
                    km = KMeans(2,n_init=10)
                    new_gids = km.fit_predict(dataframe.loc[
                        (dataframe['group_id']==gid_to_split)&(dataframe[self.target]!=val),
                        profiler.profile_keys])
                    idx_gp0 = new_gids==0
                    idx_gp1 = new_gids==1
                    new_gids[idx_gp0] = gid_to_split
                    new_gids[idx_gp1] = gid
                    dataframe.loc[
                        (dataframe['group_id']==gid_to_split)&(dataframe[self.target]!=val),
                        'group_id'] = new_gids
                    gp_counts = dataframe.loc[dataframe['group_id']>0,'group_id'].value_counts()
                    # if this leaves gid_to_split with zero variance, add it to groups_to_check
                    new_vals = dataframe.loc[dataframe['group_id']==gid_to_split,self.target]
                    if np.std(new_vals) == 0: groups_to_check.append(gid_to_split)

                    
            # check that we have only nonzero variances remaining 
            for gid in group_ids:
                vals = dataframe.loc[dataframe['group_id']==gid,self.target]
                if np.std(vals) == 0:
                    raise ValueError('zero variance in target labels for a regression group')

        return trainable


    def run_cross_validation(self,model,data,feature_names):
        """Cross-validate a model by LeaveOneGroupOut. 

        Regression models are scored by the coefficient of determination (R^2 or 'r2'),
        in order to normalize by the variance of the dataset.
        Validation reports also include the more intuitive normalized mean_abs_error.
        Scikit-learn does not currently provide API for scoring by mean_abs_error,
        so mean_abs_error is not currently supported for hyperparameter training.

        Parameters
        ----------
        model : sklearn.linear_model.SGDRegressor
            scikit-learn regression model to be cross-validated
        data : pandas.DataFrame
            pandas dataframe of features and labels
        feature_names : list of str
            list of feature names (column headers) used for training

        Returns
        -------
        result : dict
            with cross validation results.
        """
        grp_ids = data.group_id.unique()
        test_scores_by_grp = []
        test_scores_by_grp_weighted = []
        for igrp,grp in enumerate(grp_ids):
            tr = data[(data['group_id']!=grp)]
            test = data[(data['group_id']==grp)]
            model.fit(tr[feature_names], tr[self.target])
            pr = model.predict(test[feature_names])
            test_score = mean_absolute_error(pr, test[self.target])
            test_scores_by_grp.append(test_score)
            test_scores_by_grp_weighted.append(test_score*(test.shape[0]/data.shape[0]))

        result = dict(normalized_mean_abs_error_by_splits = str(test_scores_by_grp),
                       normalized_mean_abs_error = sum(test_scores_by_grp)/len(test_scores_by_grp),
                       weighted_av_mean_abs_error = sum(test_scores_by_grp_weighted),
                       number_of_experiments = len(test_scores_by_grp),
                       experiments = str(data.experiment_id.unique()),
                       test_training_split = "by experiments")
        return result
