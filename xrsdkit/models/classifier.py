import numpy as np
from collections import OrderedDict

from sklearn import linear_model
from sklearn.metrics import f1_score, confusion_matrix

from .xrsd_model import XRSDModel

class Classifier(XRSDModel):
    """Class for generating models to classifying material systems."""

    def __init__(self,label,yml_file):
        super(Classifier,self).__init__(label, yml_file)
        self.grid_search_hyperparameters = dict(
            penalty = ['none', 'l2', 'l1', 'elasticnet'], # default: l2
            alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1], # regularisation coef, default 0.0001
            l1_ratio = [0, 0.15, 0.5, 0.85, 1.0] # default 0.15, only valid for elasticnet penalty
            )

    def build_model(self,model_hyperparams={}):
        if all([p in model_hyperparams for p in ['alpha','penalty','l1_ratio']]):
            new_model = linear_model.SGDClassifier(
                    alpha=model_hyperparams['alpha'], 
                    loss='log',
                    penalty=model_hyperparams['penalty'], 
                    l1_ratio=model_hyperparams['l1_ratio'],
                    max_iter=10)
        else:
            new_model = linear_model.SGDClassifier(loss='log', max_iter=10)
        return new_model

    def classify(self, sample_features):
        """Classify the model target for a sample.

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of xrsdkit.tools.profiler.profile_spectrum()

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

    def run_cross_validation(self,model,data,features,n_groups_out):
        if n_groups_out:
            new_accuracy = self.cross_validate_by_experiments(model,data,features)
        else:
            new_accuracy = self.cross_validate(model,data,features)
        return new_accuracy


    def cross_validate_by_experiments(self, model, df, features):
        """Test a model by LeaveOneGroupOut cross-validation.

        Parameters
        ----------
        df : pandas.DataFrame
            pandas dataframe of features and labels
            must include the data from at least 3 experiments
        model : sk-learn
            with specific parameters
        features : list of str
            list of features that were used for training.

        Returns
        -------
        test_scores_by_ex : dict
            includes list of all system classes, confusion matrix,
            F1 score by sys_classes, averaged F1 score,
            mean not weighted accuracies by system classes and experiments,
            mean not weighted accuracy.
        """
        experiments = df.experiment_id.unique()# we have at least 3 experiments
        sys_classes = df.system_class.unique().tolist()
        test_scores_by_ex = {}
        test_scores_by_sys_classes = dict.fromkeys(sys_classes) # do not use (sys_classes, [])
        for k,v in test_scores_by_sys_classes.items():
            test_scores_by_sys_classes[k] = []
        scores = []
        true_labels = []
        pred_labels = []
        for i in range(len(experiments)):
            tr = df[(df['experiment_id'] != experiments[i])]
            test = df[(df['experiment_id'] == experiments[i])]
            # remove from the test set the samples of the classes
            # that do not exists in the training set:
            cl_in_training_set = tr.system_class.unique()
            test = test[(test['system_class'].isin(cl_in_training_set))]

            model.fit(tr[features], tr[self.target])
            y_pred = model.predict(test[features])

            pred_labels.extend(y_pred)
            true_labels.extend(test[self.target])

            labels_from_test = test.system_class.value_counts().keys().tolist()
            cmat = confusion_matrix(test[self.target], y_pred, labels_from_test)

            # for each class we devided the number of right predictions
            # by the total number of samples for this class at test set:
            accuracies_by_classes = cmat.diagonal()/test.system_class.value_counts().tolist()

            average_acc_for_this_exp = sum(accuracies_by_classes)/len(accuracies_by_classes)
            scores.append(average_acc_for_this_exp)

            test_scores_by_ex[experiments[i]] = {}
            test_scores_by_ex[experiments[i]]['average for exp'] = average_acc_for_this_exp
            test_scores_by_ex[experiments[i]]['by classes'] = {}
            for k in range(len(labels_from_test)):
                test_scores_by_ex[experiments[i]]['by classes'][labels_from_test[k]] = accuracies_by_classes[k]
                test_scores_by_sys_classes[labels_from_test[k]].append(accuracies_by_classes[k])


        result = OrderedDict()
        result["all system classes :"] = sys_classes

        # we may not be able to test for all sys_classes
        # (if samples of a sys_class are included into only one experiment)
        not_tested_sys_classes = []
        tested_sys_classes = []
        score_by_cl = []
        for k, v in test_scores_by_sys_classes.items():
            if test_scores_by_sys_classes[k] == []:
                not_tested_sys_classes.append(k)
            else:
                tested_sys_classes.append(k)
                av = sum(test_scores_by_sys_classes[k])/len(test_scores_by_sys_classes[k])
                test_scores_by_sys_classes[k] = av
                score_by_cl.append(av)

        result['confusion matrix :'] = confusion_matrix(true_labels, pred_labels, tested_sys_classes)

        result["model was NOT tested for :"] = not_tested_sys_classes
        result["model was tested for :"] = tested_sys_classes
        result["F1 score by sys_classes"] = f1_score(true_labels, pred_labels,
                                                     labels=tested_sys_classes, average=None)
        result["F1 score averaged not weighted :"] = f1_score(true_labels,
                                        pred_labels, labels=tested_sys_classes, average='macro')
        result["mean not weighted accuracies by system classes :"] = test_scores_by_sys_classes
        result["mean not weighted accuracies by exp :"] = test_scores_by_ex
        result["mean not weighted accuracy :"]= sum(score_by_cl)/len(score_by_cl)

        return result

    def hyperparameters_search(self,transformed_data, data_labels, group_by=None, n_leave_out=None, scoring='f1_macro'):
        """Grid search for optimal alpha, penalty, and l1 ratio hyperparameters.

        This invokess the method from the base class with a different scoring argument.

        Returns
        -------
        params : dict
            dictionary of the parameters to get the best f1 score.
        """
        params = super(Classifier,self).hyperparameters_search(
        transformed_data,data_labels,group_by,n_leave_out,scoring)

        return  params


class SystemClassifier(Classifier):
    """Classifier for determining the material system (structures and form factors).

    See the main xrsdkit package documentation 
    for all supported structures and parameters
    for defining the material system.
    """

    def __init__(self,yml_file):
        super(SystemClassifier,self).__init__('system_class', yml_file)


