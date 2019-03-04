from collections import OrderedDict

import numpy as np
from sklearn import linear_model, model_selection
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from dask_ml.model_selection import GridSearchCV

from .xrsd_model import XRSDModel
from ..tools import profiler


class Classifier(XRSDModel):
    """Class for models that classify attributes of material systems."""

    def __init__(self,label,yml_file):
        super(Classifier,self).__init__(label, yml_file)
        self.grid_search_hyperparameters = dict(
            alpha = [0.0001, 0.001, 0.01, 0.1], # regularisation coef, default 0.0001
            l1_ratio = [0.15, 0.5, 0.85, 1.0] # default 0.15
            #C = [ 0.1, 1.0]
            )

    def minimization_score(self,true_labels,pred_labels):
        return -1*f1_score(true_labels, pred_labels, average='macro')

    def build_model(self,model_hyperparams={}):
        if all([p in model_hyperparams for p in ['alpha','l1_ratio']]):
            new_model = linear_model.SGDClassifier(
                    alpha=model_hyperparams['alpha'], 
                    loss='log',
                    penalty='elasticnet',
                    l1_ratio=model_hyperparams['l1_ratio'],
                    max_iter=1000000, class_weight='balanced', tol=1e-10#, eta0 = 0.001, learning_rate='adaptive'
                    )
        else:
            new_model = linear_model.SGDClassifier(
                loss='log', penalty='elasticnet',
                max_iter=1000000, class_weight='balanced', tol=1e-10#, eta0 = 0.001, learning_rate='adaptive'
                )
        return new_model

    def classify(self, sample_features):
        """Classify the model target for a sample.

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of xrsdkit.tools.profiler.profile_pattern().

        Returns
        -------
        sys_cls : object 
            Predicted classification value for self.target, given `sample_features`
        cert : float or None
            the certainty of the prediction
            (For models that are not trained, cert=None)
        """
        feature_array = np.array(list(sample_features.values())).reshape(1,-1)
        feature_idx = [k in self.features for k in sample_features.keys()]
        x = self.scaler.transform(feature_array)[:, feature_idx]
        sys_cls = self.model.predict(x)[0]
        cert = max(self.model.predict_proba(x)[0])
        return sys_cls, cert

    def run_cross_validation(self, model, df, feature_names):
        """Cross-validate a model by LeaveOneGroupOut. 

        The train/test groupings are defined by the 'group_id' labels,
        which are added to the dataframe during self.assign_groups().

        Parameters
        ----------
        df : pandas.DataFrame
            pandas dataframe of features and labels,
            including at least three distinct experiment_id labels 
        model : sklearn.linear_model.SGDClassifier
            an sklearn classifier instance trained on some dataset
            with some choice of hyperparameters
        feature_names : list of str
            list of feature names (column headers) used for training.

        Returns
        -------
        result : dict
            list of all labels from df,
            number of experiments,
            list of all experiments,
            F1_macro, accuracy, confusion matrix,
            and testing-training splits.
        """
        group_ids = df.group_id.unique()
        all_classes = df[self.target].unique().tolist()
        true_labels = []
        pred_labels = []
        for gid in group_ids:
            tr = df[(df['group_id'] != gid)]
            test = df[(df['group_id'] == gid)]
            model.fit(tr[feature_names], tr[self.target])
            y_pred = model.predict(test[feature_names])
            pred_labels.extend(y_pred)
            true_labels.extend(test[self.target])
        cm = confusion_matrix(true_labels, pred_labels, all_classes)
        experiments = df.experiment_id.unique() 
        result = dict(all_classes = all_classes,
                        number_of_experiments = len(experiments),
                        experiments = str(experiments),
                        confusion_matrix = str(cm),
                        F1_score_by_classes = f1_score(true_labels, pred_labels,
                                    labels=all_classes, average=None).tolist(),
                        F1_score_averaged_not_weighted = f1_score(true_labels,
                                    pred_labels, labels=all_classes, average='macro'),
                        accuracy = accuracy_score(true_labels, pred_labels, sample_weight=None),
                        test_training_split = 'for classes with samples from 3 or more experiment_ids, \n'\
                                            'the data are split according to experiment_id; \n'\
                                            'for classes with samples from 2 or fewer experiment_ids, \n'\
                                            'the data are randomly shuffled and split into three groups'
                        )
        return result

    def hyperparameters_search(self,transformed_data,features,group_by='group_id',n_leave_out=1,scoring='f1_macro'):
        """Grid search for optimal alpha and l1 ratio hyperparameters.

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
        params : dict
            dictionary of the parameters to get the best f1 score.
        """

        cv = model_selection.LeavePGroupsOut(n_groups=n_leave_out).split(
                transformed_data[features], np.ravel(transformed_data[self.target]),
                                                                  groups=transformed_data[group_by])
        test_model = self.build_model()

        # threaded scheduler with optimal number of threads
        # will be used by default for dask GridSearchCV
        clf = GridSearchCV(test_model,
                        self.grid_search_hyperparameters, cv=cv, scoring=scoring, n_jobs=-1)
        clf.fit(transformed_data[features], np.ravel(transformed_data[self.target]))
        params = clf.best_params_
        return params

    def print_labels(self, all=True):
        if all:
            labels = self.cross_valid_results['all_classes']
        else:
            labels = self.cross_valid_results['model_was_NOT_tested_for']
        if labels:
            result = ''
            for l in labels:
                result += l
                result += '\n'
            return result
        else:
            return "The model was tested for all labels"

    def assign_groups(self, dataframe, min_groups=5):
        """Assign train/test groups to `dataframe`.

        This reimplementation invokes the base class method, 
        and then updates the groups if necessary
        to balance the representation of classes among the groups. 

        Parameters
        ----------
        dataframe : pandas.DataFrame
            dataframe of sample features and corresponding labels
        min_groups : int
            minimum number of groups to create

        Returns
        -------
        trainable : bool
            indicates whether or not training is possible
        """
        trainable = super(Classifier,self).assign_groups(dataframe)
        if trainable:
            cl_counts = dataframe.loc[dataframe['group_id']>0,self.target].value_counts()

            # if any labels do not have at least `min_groups` representative samples,
            # the label should not be trained-
            # ignore these samples by setting their group_id to 0
            for cl,count in cl_counts.items():
                if count<min_groups:
                    dataframe.loc[dataframe[self.target]==cl,'group_id']=0

            # if this leaves us with only one distinct label, 
            # return False (not trainable).
            cl_counts = dataframe.loc[dataframe['group_id']>0,self.target].value_counts()
            n_cl_total = len(cl_counts.keys())
            if n_cl_total < 2: return False

            # count the number of samples for each label in each group,
            # and take note of any underrepresented groups
            group_ids = dataframe.loc[dataframe['group_id']>0,'group_id'].unique()
            #group_ids.sort()
            cl_counts_by_group = OrderedDict.fromkeys(group_ids)
            underrep_gids = [] 
            for gid in group_ids:
                cl_counts_by_group[gid] = dataframe.loc[dataframe['group_id']==gid,self.target].value_counts()
                if len(cl_counts_by_group[gid].keys()) < n_cl_total:
                    underrep_gids.append(gid)

            # merge groups to eliminate underrepresentation
            while len(underrep_gids) > 0:
                gid = underrep_gids.pop(0)
                cl_in_group = set(cl_counts_by_group[gid].keys())
                n_cl_in_pair = OrderedDict.fromkeys(cl_counts_by_group.keys())
                pairing_scores = OrderedDict.fromkeys(cl_counts_by_group.keys())
                #pairing_scores[0] = float('inf')

                # evaluate best pairing
                for pair_gid in cl_counts_by_group.keys():
                    if pair_gid == gid: 
                        pairing_scores[pair_gid] = float('inf')
                        n_cl_in_pair[pair_gid] = cl_counts_by_group[gid] 
                    else:
                        cl_in_pair = cl_in_group.union(set(cl_counts_by_group[pair_gid].keys()))
                        n_cl_in_pair[pair_gid] = len(cl_in_pair)
                        if n_cl_in_pair[pair_gid] == n_cl_total:
                            pair_cl_counts = dict([(cl_label,0) for cl_label in cl_counts.keys()])
                            for cl_label, cl_count in cl_counts_by_group[gid].items():
                                pair_cl_counts[cl_label] += cl_count
                            for cl_label, cl_count in cl_counts_by_group[pair_gid].items():
                                pair_cl_counts[cl_label] += cl_count
                            #pairing_scores[pair_gid] = np.min(pair_cl_counts.values())
                            pairing_scores[pair_gid] = np.std(list(pair_cl_counts.values()))
                        else:
                            pairing_scores[pair_gid] = float('inf')

                # assign the best pairing
                underrep_pair_scores = [pairing_scores[ugid] for ugid in underrep_gids] 
                underrep_pair_n_cl = [n_cl_in_pair[ugid] for ugid in underrep_gids] 
                if not underrep_pair_scores and not underrep_pair_n_cl:
                    # there are no underrepresented groups that are valid pairing candidates- 
                    # pair with the fully represented group that gives the best pairing score
                    best_pairing_gid = list(pairing_scores.keys())[np.argmin(list(pairing_scores.values()))]
                    # if `gid` is the only remaining underrepresented group, 
                    # we have the possibility of `best_pairing_gid` == `gid`-
                    # use np.argmax to select a fully represented group
                    if best_pairing_gid == gid:
                        best_pairing_gid = list(n_cl_in_pair.keys())[np.argmax(list(n_cl_in_pair.values()))]
                else:
                    # if possible, get full representation by pairing 
                    # with another underrepresented group
                    if any([ps<float('inf') for ps in underrep_pair_scores]):
                        best_pairing_gid = underrep_gids[np.argmin(underrep_pair_scores)] 
                        underrep_gids.pop(underrep_gids.index(best_pairing_gid))
                    else:
                        # else, take underrepresented pair with the best partial representation 
                        best_pairing_gid = underrep_gids[np.argmax(underrep_pair_n_cl)]
                dataframe.loc[dataframe['group_id']==gid,'group_id'] = best_pairing_gid 

                # update cl_counts_by_group
                cl_counts_by_group.pop(gid)
                cl_counts_by_group[best_pairing_gid] = dataframe.loc[dataframe['group_id']==best_pairing_gid,self.target].value_counts()

            # if we are left with fewer than min_groups,
            # split the groups with k-means,
            # until we are up to min_groups
            while len(cl_counts_by_group) < min_groups:
                group_ids = cl_counts_by_group.keys()
                new_gid = 1
                while new_gid in group_ids: new_gid += 1
                # perform split for each class to preserve full representation
                for cl in cl_counts.keys():
                    # get the fully-represented group with the greatest sample count-
                    # NOTE that this condition should enforce full representation in all groups
                    sample_counts = [cl_counts_by_group[gid][cl] for gid in cl_counts_by_group.keys()]
                    gid_to_split = list(cl_counts_by_group.keys())[np.argmax(sample_counts)]

                    # TODO: come up with a more balanced clustering mechanism
                    km = KMeans(2,n_init=10)
                    new_grps = km.fit_predict(
                    dataframe.loc[(dataframe['group_id']==gid_to_split)&(dataframe[self.target]==cl),
                    profiler.profile_keys]
                    )
                    idx_gp0 = new_grps==0
                    idx_gp1 = new_grps==1
                    new_grps[idx_gp0] = gid_to_split
                    new_grps[idx_gp1] = new_gid
                    dataframe.loc[(dataframe['group_id']==gid_to_split)&(dataframe[self.target]==cl),'group_id'] = new_grps 

                # re-count all groups
                for gid in group_ids:
                    cl_counts_by_group[gid] = dataframe.loc[dataframe['group_id']==gid,self.target].value_counts()
                cl_counts_by_group[new_gid] = dataframe.loc[dataframe['group_id']==new_gid,self.target].value_counts()

        return trainable 

    def print_confusion_matrix(self):
        if self.cross_valid_results['confusion_matrix']:
            result = ''
            matrix = self.cross_valid_results['confusion_matrix'].split('\n')
            for i in range(len(self.cross_valid_results['all_classes'])):
                result += (matrix[i] + "  " +
                        self.cross_valid_results['all_classes'][i] + '\n')
            return result
        else:
            return "Confusion matrix was not created"

    def print_F1_scores(self):
        result = ''
        for i in range(len(self.cross_valid_results['F1_score_by_classes'])):
            result += (self.cross_valid_results['all_classes'][i] +
                       " : " + str(self.cross_valid_results['F1_score_by_classes'][i]) + '\n')
        return result

    def print_accuracy(self):
        if self.cross_valid_results['accuracy']:
            return str(self.cross_valid_results['accuracy'])
        else:
            return "Mean accuracies by classes were not calculated"

    def average_F1(self):
        return str(self.cross_valid_results['F1_score_averaged_not_weighted'])

    def print_CV_report(self):
        """Return a string describing the model's cross-validation metrics.

        Returns
        -------
        CV_report : str
            string with formated results of cross validatin.
        """
        CV_report = 'Cross validation results for {} Classifier\n\n'.format(self.target) + \
            'Data from {} experiments was used\n\n'.format(
            str(self.cross_valid_results['number_of_experiments'])) + \
            'Confusion matrix:\n' + \
            self.print_confusion_matrix()+'\n\n' + \
            'F1 scores by label:\n' + \
            self.print_F1_scores() + '\n'\
            'Label-averaged unweighted F1 score: {}\n\n'.format(self.average_F1()) + \
            'Accuracy:\n' + \
            self.print_accuracy() + '\n'+\
            "Test/training split: " + self.cross_valid_results['test_training_split']
        return CV_report
