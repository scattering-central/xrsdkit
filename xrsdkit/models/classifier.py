import numpy as np
import pandas as pd
from collections import OrderedDict

from sklearn import linear_model, model_selection
from sklearn.metrics import f1_score, confusion_matrix

from .xrsd_model import XRSDModel

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
                    max_iter=10)
        else:
            new_model = linear_model.SGDClassifier(loss='log', penalty='elasticnet', max_iter=10)
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

    def cross_validate_by_experiments(self, model, df, features):
        """Test a model by LeaveOneGroupOut cross-validation.

        In this case, the groupings are defined by the experiment_id labels.

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
            includes list of all labels,
            list of all experiments, confusion matrix,
            list of labels for which the models were and were not tested,
            F1 score by classes, averaged F1 score weighted and unweighted,
            mean accuracies by classes,
            mean unweighted and class-size-weighted accuracy,
            and testing-training splits.
        """
        experiments = df.experiment_id.unique()# we have at least 3 experiments
        all_classes = df[self.target].unique().tolist()
        test_scores_by_ex = {}
        test_scores_by_classes = dict.fromkeys(all_classes)
        for k,v in test_scores_by_classes.items():
            test_scores_by_classes[k] = []
        scores = []
        true_labels = []
        pred_labels = []
        acc_weighted_by_classes = []
        for i in range(len(experiments)):
            tr = df[(df['experiment_id'] != experiments[i])]
            if len(tr[self.target].unique()) < 2:
                continue
            test = df[(df['experiment_id'] == experiments[i])]
            # remove from the test set the samples of the classes
            # that do not exists in the training set:
            cl_in_training_set = tr[self.target].unique()
            test = test[(test[self.target].isin(cl_in_training_set))]

            model.fit(tr[features], tr[self.target])
            y_pred = model.predict(test[features])

            pred_labels.extend(y_pred)
            true_labels.extend(test[self.target])

            labels_from_test = test[self.target].value_counts().keys().tolist()

            cmat = confusion_matrix(test[self.target], y_pred, labels_from_test)

            # for each class we devided the number of right predictions
            # by the total number of samples for this class at test set:
            accuracies_by_classes = cmat.diagonal()/test[self.target].value_counts().tolist()
            acc_weighted_by_classes.append(sum(cmat.diagonal())/test.shape[0])

            average_acc_for_this_exp = sum(accuracies_by_classes)/len(accuracies_by_classes)
            scores.append(average_acc_for_this_exp)

            for k in range(len(labels_from_test)):
                test_scores_by_classes[labels_from_test[k]].append(accuracies_by_classes[k])

        # we may not be able to test for all classes
        # (if samples of a class are included into only one experiment)
        not_tested_classes = []
        tested_classes = []
        score_by_cl = []
        for k, v in test_scores_by_classes.items():
            if test_scores_by_classes[k] == []:
                not_tested_classes.append(k)
            else:
                tested_classes.append(k)
                av = sum(test_scores_by_classes[k])/len(test_scores_by_classes[k])
                test_scores_by_classes[k] = av
                score_by_cl.append(av)

        result = dict(all_classes = all_classes,
                      number_of_experiments = len(experiments),
                      experiments = str(df.experiment_id.unique()),
                      confusion_matrix = str(confusion_matrix(true_labels, pred_labels, all_classes)),
                      model_was_NOT_tested_for = not_tested_classes,
                      model_was_tested_for = tested_classes,
                      F1_score_by_classes = f1_score(true_labels, pred_labels,
                                    labels=tested_classes, average=None).tolist(),
                      F1_score_averaged_not_weighted = f1_score(true_labels,
                                        pred_labels, labels=tested_classes, average='macro'),
                      F1_score_averaged_weighted = f1_score(true_labels,
                                        pred_labels, labels=tested_classes, average='weighted'),
                      mean_accuracies_by_classes = test_scores_by_classes,
                      mean_not_weighted_accuracy = sum(score_by_cl)/len(score_by_cl),
                      mean_weighted_by_classes_accuracy = sum(acc_weighted_by_classes)/len(acc_weighted_by_classes),
                      test_training_split = "by experiments")
        return result


    def cross_validate(self,model,df,features):
        # some of metrics have None since they cannot be calculated using cross_val_score from sklearn
        results = dict(all_classes = df[self.target].unique().tolist(),
                       number_of_experiments = len(df.experiment_id.unique()),
                       experiments = str(df.experiment_id.unique()),
                       confusion_matrix = None,
                       model_was_NOT_tested_for = None,
                       model_was_tested_for = df[self.target].unique().tolist(), #same as all_classes
                       F1_score_by_classes = [],
                       mean_accuracies_by_classes = None,
                       mean_not_weighted_accuracy= None,
                       test_training_split = "random 3 folders split")

        if min(df[self.target].value_counts()) > 2:
            results['mean_weighted_by_classes_accuracy'] = model_selection.cross_val_score(model,df[features],
                                                        df[self.target],cv=3, scoring='accuracy').tolist()
            results['F1_score_averaged_not_weighted'] = model_selection.cross_val_score(model,df[features],
                                                        df[self.target],cv=3, scoring='f1_macro').tolist()
            results['F1_score_averaged_weighted'] = model_selection.cross_val_score(model,df[features],
                                                        df[self.target],cv=3, scoring='f1_weighted').tolist()
            results['test_training_split'] = 'random shuffle-split 3-fold cross-validation'
        else:
            results['mean_weighted_by_classes_accuracy'] = None
            results['F1_score_averaged_not_weighted'] = None
            results['F1_score_averaged_weighted'] = None
            results['test_training_split'] = 'The model was not cross-validated, '\
                'because some labels in the training set included less than 3 samples.'
        return results

    def hyperparameters_search(self,transformed_data, data_labels,
                               group_by=None, n_leave_out=None, scoring='accuracy'):
        """Grid search for optimal alpha, penalty, and l1 ratio hyperparameters.

        This invokes the method from the base class with a different scoring argument.

        Returns
        -------
        params : dict
            dictionary of the parameters to get the best f1 score.
        """
        # TODO (later): try scoring "f1_macro"
        # or implement customized grid search.
        # problem with any f1: for each split,
        # f1 is calculated for EACH class that is present
        # in the testing or training set,
        # which currently produces a lot of zeros.
        # For each split, we want to calculate f1 only for the classes
        # that are present in training set
        params = super(Classifier,self).hyperparameters_search(
                transformed_data,data_labels,group_by,n_leave_out,scoring)
        return  params

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

        This checks whether or not each cross-validation split will include
        at least two classes for training
        (as required for hyperparameters_search).
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
        n_groups_out : int or None
            using leaveGroupOut makes sense when we have at least 3 groups
            and each split by groups has legal values for all labels
        dataframe : pandas.DataFrame
            updated dataframe, if required 
            (if the input dataframe includes 
            only 1 or 2 samples for some classes, 
            these samples are cloned two times,
            so that simple non-validated models
            are still produced for those classes).
        """
        result, n_groups_out = super(Classifier,self).check_label(dataframe)
        if result:
            # check if there are skewed classes and fix them:
            if min(dataframe[self.target].value_counts().tolist()) < 3:
                # find samples of skewed classes and add two clones of them
                all_classes = dataframe[self.target].value_counts().keys()
                number_of_samles_by_cl = dataframe[self.target].value_counts().tolist()
                for i in range(len(number_of_samles_by_cl)):
                    if number_of_samles_by_cl[i] < 3:
                        sampls = dataframe.loc[dataframe[self.target] == all_classes[i]].copy()
                        dataframe = pd.concat([dataframe,sampls,sampls])

            # check if threre are splits when all testing data have identical labels
            experiments = dataframe.experiment_id.unique()
            for i in range(len(experiments)):
                tr = dataframe[(dataframe['experiment_id'] != experiments[i])]
                if len(tr[self.target].unique()) < 2:
                    n_groups_out = None # 3-fold cross validation will be used

        return result, n_groups_out, dataframe

    def print_confusion_matrix(self):
        if self.cross_valid_results['confusion_matrix']:
            result = ''
            matrix = self.cross_valid_results['confusion_matrix'].split('\n')
            #for i in range(len(self.cross_valid_results['model_was_tested_for'])):
            for i in range(len(self.cross_valid_results['all_classes'])):
                result += (matrix[i] + "  " +
                        #self.cross_valid_results['model_was_tested_for'][i] + '\n')
                        self.cross_valid_results['all_classes'][i] + '\n')
            return result
        else:
            return "Confusion matrix was not created"

    def print_F1_scores(self):
        result = ''
        for i in range(len(self.cross_valid_results['F1_score_by_classes'])):
            result += (self.cross_valid_results['model_was_tested_for'][i] +
                       " : " + str(self.cross_valid_results['F1_score_by_classes'][i]) + '\n')
        return result

    def print_accuracies(self):
        if self.cross_valid_results['mean_accuracies_by_classes']:
            result = ''
            for k,v in self.cross_valid_results['mean_accuracies_by_classes'].items():
                result += (k + " : " + str(v) + '\n')
            return result
        else:
            return "Mean accuracies by classes were not calculated"

    def average_F1(self,weighted=False):
        if weighted:
            return str(self.cross_valid_results['F1_score_averaged_weighted'])
        else:
            return str(self.cross_valid_results['F1_score_averaged_not_weighted'])

    def average_accuracy(self,weighted=False):
        if weighted:
            return str(self.cross_valid_results['mean_weighted_by_classes_accuracy'])
        else:
            if self.cross_valid_results['mean_not_weighted_accuracy']:
                return str(self.cross_valid_results['mean_not_weighted_accuracy'])
            else:
                return "Mean not weighted accuracy was not calculated"

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
            'All labels : \n' + self.print_labels()+'\n\n'+ \
            'model was NOT tested for :' '\n'+ self.print_labels(False) +'\n\n'+ \
            'Confusion matrix:\n' + \
            'if the model was not tested for some labels, the corresponding rows will have all zeros\n' + \
            self.print_confusion_matrix()+'\n\n' + \
            'F1 scores by label:\n' + \
            self.print_F1_scores() + '\n'\
            'Label-averaged unweighted F1 score: {}\n\n'.format(self.average_F1(False)) + \
            'Label-averaged weighted F1 score: {}\n\n'.format(self.average_F1(True)) + \
            'Accuracies by label:\n' + \
            self.print_accuracies() + '\n'+\
            'Label-averaged unweighted accuracy: {}\n\n'.format(self.average_accuracy(False)) + \
            'Label-averaged weighted accuracy: {}\n'.format(self.average_accuracy(True))+ '\n\n'+\
            'NOTE: Weighted metrics are weighted by class size' + '\n' + \
            "Test/training split: " + self.cross_valid_results['test_training_split']
        return CV_report

