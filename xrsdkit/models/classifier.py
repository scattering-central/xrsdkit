import numpy as np

from sklearn import linear_model, model_selection
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from .xrsd_model import XRSDModel
from dask_ml.model_selection import GridSearchCV
from ..tools import profiler


class Classifier(XRSDModel):
    """Class for models that classify attributes of material systems."""

    def __init__(self,label,yml_file):
        super(Classifier,self).__init__(label, yml_file)
        self.grid_search_hyperparameters = dict(
            alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1], # regularisation coef, default 0.0001
            l1_ratio = [0, 0.15, 0.5, 0.85, 1.0] # default 0.15
            )

    def build_model(self,model_hyperparams={}):
        if all([p in model_hyperparams for p in ['alpha','l1_ratio']]):
            new_model = linear_model.SGDClassifier(
                    alpha=model_hyperparams['alpha'], 
                    loss='log',
                    penalty='elasticnet',
                    l1_ratio=model_hyperparams['l1_ratio'],
                    max_iter=1000, class_weight='balanced', tol=1e-5
                    )
        else:
            new_model = linear_model.SGDClassifier(
                loss='log', penalty='elasticnet',
                max_iter=1000, class_weight='balanced', tol=1e-5
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
        x = self.scaler.transform(feature_array)
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
        groups = df.group_id.unique()
        all_classes = df[self.target].unique().tolist()
        true_labels = []
        pred_labels = []
        for i in range(len(groups)):
            tr = df[(df['group_id'] != groups[i])]
            test = df[(df['group_id'] == groups[i])]
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

    def hyperparameters_search(self,transformed_data, group_by='group_id', n_leave_out=1, scoring='f1_macro'):
        """Grid search for optimal alpha and l1 ratio hyperparameters.

        Returns
        -------
        params : dict
            dictionary of the parameters to get the best f1 score.
        """

        cv = model_selection.LeavePGroupsOut(n_groups=n_leave_out).split(
                transformed_data[profiler.profile_keys], np.ravel(transformed_data[self.target]),
                                                                  groups=transformed_data[group_by])
        test_model = self.build_model()

        # threaded scheduler with optimal number of threads
        # will be used by default for dask GridSearchCV
        clf = GridSearchCV(test_model,
                        self.grid_search_hyperparameters, cv=cv, scoring=scoring)
        clf.fit(transformed_data[profiler.profile_keys], np.ravel(transformed_data[self.target]))
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

    def assign_groups(self, dataframe):
        """Assign train/test groups to `dataframe`.

        This reimplementation invokes the base class method, 
        and then updates the groups if necessary
        to balance the representation of classes among the groups. 

        Parameters
        ----------
        dataframe : pandas.DataFrame
            dataframe of sample features and corresponding labels

        Returns
        -------
        trainable : bool
            indicates whether or not training is possible
        """
        trainable = super(Classifier,self).assign_groups(dataframe)
        if trainable:
            cl_counts = dataframe[self.target].value_counts()
            for cl,nsamp in cl_counts.items():
                cl_idx = dataframe[self.target] == cl
                cl_data = dataframe.loc[cl_idx]
                if nsamp < 10:
                    # insufficient samples for the class: 
                    # set group_id to zero to leave this class out of the model
                    dataframe.loc[cl_idx,'group_id'] = 0
                else:
                    # sufficient samples exist for training this class
                    cl_grp_ids = cl_data['group_id'].unique()
                    if len(cl_grp_ids) < 3:
                        # less than three groups are represented in this class:
                        # the group_id assignments should be rebalanced.
                        # shuffle-split samples into 3 groups.
                        cl_group_ids = self.shuffle_split_3fold(nsamp) 
                        dataframe.loc[cl_idx,'group_id'] = cl_group_ids 
            #check if we still have at least two fully represented classes:
            if len(dataframe.loc[dataframe.group_id>0,self.target].unique()) < 2:
                return False
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
