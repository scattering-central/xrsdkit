import numpy as np
import pandas as pd
import random

from sklearn import linear_model, model_selection
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from .xrsd_model import XRSDModel
from dask_ml.model_selection import GridSearchCV
from ..tools import profiler


class Classifier(XRSDModel):
    """Class for generating models to classifying material systems."""

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
                    max_iter=1000, class_weight='balanced')
        else:
            new_model = linear_model.SGDClassifier(loss='log', penalty='elasticnet',
                                                   max_iter=1000, class_weight='balanced')
        return new_model

    def classify(self, sample_features):
        """Classify the model target for a sample.

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of xrsdkit.tools.profiler.profile_pattern()

        Returns
        -------
        structure_flags : bool or None
            a boolean inidicating whether or not
            the sample exhibits the structure
            None is reterned for models that was not trained yet
        cert : float or None
            the certainty of the prediction
            None is reterned for models that was not trained yet
        """
        feature_array = np.array(list(sample_features.values())).reshape(1,-1)
        x = self.scaler.transform(feature_array)
        sys_cls = self.model.predict(x)[0]
        cert = max(self.model.predict_proba(x)[0])
        return sys_cls, cert


    def run_cross_validation(self, model, df, features, _):
        """Test a model by LeaveOneGroupOut cross-validation.
        In this case, the groupings are defined by the group_id.

        Parameters
        ----------
        df : pandas.DataFrame
            pandas dataframe of features and labels,
            including at least three distinct experiment_id labels 
        model : sklearn.linear_model.SGDClassifier
            an sklearn classifier instance trained on some dataset
            with some choice of hyperparameters
        features : list of str
            list of features that were used for training.

        Returns
        -------
        test_scores_by_ex : dict
            list of all labels from df,
            number of experiments,
            list of all experiments,
            F1_macro, accuracy,confusion matrix,
            and testing-training splits.
        """
        experiments = df.experiment_id.unique()# we have at least 3 experiments
        groups = df.group_id.unique()
        all_classes = df[self.target].unique().tolist()
        true_labels = []
        pred_labels = []
        for i in range(len(groups)):
            tr = df[(df['group_id'] != groups[i])]
            if len(tr[self.target].unique()) < 2:
                continue
            test = df[(df['group_id'] == groups[i])]

            model.fit(tr[features], tr[self.target])
            y_pred = model.predict(test[features])

            pred_labels.extend(y_pred)
            true_labels.extend(test[self.target])

        cm = confusion_matrix(true_labels, pred_labels, all_classes)

        result = dict(all_classes = all_classes,
                      number_of_experiments = len(experiments),
                      experiments = str(df.experiment_id.unique()),
                      confusion_matrix = str(cm),
                      F1_score_by_classes = f1_score(true_labels, pred_labels,
                                    labels=all_classes, average=None).tolist(),
                      F1_score_averaged_not_weighted = f1_score(true_labels,
                                        pred_labels, labels=all_classes, average='macro'),
                      accuracy = accuracy_score(true_labels, pred_labels, sample_weight=None),
                      test_training_split = "by group: for classes that includes data from 3 or more experiments, \n "
                                            "the data was splited such that all data from each experiment was\n "
                                            "placed in one group; for the other classes - the data was randomply\n "
                                            "splited into three groups")
        return result


    def hyperparameters_search(self,transformed_data, group_by='group_id', n_leave_out=None, scoring='f1_macro'):
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


    def check_label(self, dataframe):
        """Test whether or not `dataframe` has legal values for all labels.
        This invokes the method from the base class and then add a new
        column "group_by" to the dataframe. It also removes data from
        the classes that have less than 10 samples.

        Returns "True" if the dataframe has enough rows,
        over which the labels exhibit at least two unique values

        Parameters
        ----------
        dataframe : pandas.DataFrame
            dataframe of sample features and corresponding labels

        Returns
        -------
        result : bool
            indicates whether or not training is possible
            (the dataframe has enough rows,
            over which the labels exhibit at least two unique values)
        _ : int or None
            for classification models LeaveOneGroupOut by "group_id" always is used
        dataframe : pandas.DataFrame
            updated dataframe: classes with less than 10 samples are removed;
            a new column "group_by" is added.
        """
        result, _ = super(Classifier,self).check_label(dataframe)
        if result:
            if min(dataframe[self.target].value_counts().tolist()) < 10:
                # remove these samples
                all_classes = dataframe[self.target].value_counts().keys()
                number_of_samles_by_cl = dataframe[self.target].value_counts().tolist()
                for i in range(len(number_of_samles_by_cl)):
                    if number_of_samles_by_cl[i] < 10:
                        dataframe = dataframe.loc[~(dataframe[self.target] == all_classes[i]) ]
                #check if we still have at least two different classes:
                if len(dataframe[self.target].unique()) < 2:
                    result = False
                    return result, _, dataframe

            gr_list = []
            all_classes = dataframe[self.target].value_counts().keys()
            for cl in all_classes:
                d = dataframe.loc[dataframe[self.target]==cl]
                if len(d['experiment_id'].unique()) > 2:
                # for classes that have data from 3 or more experiments:
                # the data from one experiment will be in the same group
                    all_exp = d['experiment_id'].value_counts().keys().tolist()
                    all_exp = random.sample(all_exp, len(all_exp)) # to shuffle the list
                    exp_per_group = len(all_exp)//3
                    if len(all_exp)%3 == 2: # if we have 5 exeriments: 2 - 2 - 1; 4: 1 - 1 - 2
                        exp_per_group +=1
                    gr_1 = all_exp[ : exp_per_group] # these experiments will be at the group 1
                    gr_2 = all_exp[exp_per_group : exp_per_group *2]
                    gr_3 = all_exp[exp_per_group * 2 : ]
                    d_1 = d.loc[d['experiment_id'].isin(gr_1)].copy()
                    d_2 = d.loc[d['experiment_id'].isin(gr_2)].copy()
                    d_3 = d.loc[d['experiment_id'].isin(gr_3)].copy()
                else:
                    # split the samples of this class in 3 groups:
                    samp_per_group = d.shape[0]//3
                    if d.shape[0]%3 == 2:
                        samp_per_group +=1
                    d_1 = d.iloc[ :samp_per_group]
                    d_2 = d.iloc[samp_per_group : 2 * samp_per_group]
                    d_3 = d.iloc[2 * samp_per_group : ]
                d_1.loc[ :, 'group_id'] = 1
                gr_list.append(d_1)
                d_2.loc[ :, 'group_id'] = 2
                gr_list.append(d_2)
                d_3.loc[ :, 'group_id'] = 3
                gr_list.append(d_3)

            dataframe = pd.concat(gr_list)

        return result, _, dataframe


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

