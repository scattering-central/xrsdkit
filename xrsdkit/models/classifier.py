import numpy as np
from sklearn import linear_model, model_selection

from .xrsd_model import XRSDModel

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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



    def cross_validate(self, model, df,features):
        """Test a model using scikit-learn 5-fold crossvalidation

        Parameters
        ----------
        model : sklearn model
            model to be cross-validated 
        df : pandas.DataFrame
            pandas dataframe of features and labels

        Returns
        -------
        scores : object
            TODO: describe scores output
        """
        scores = model_selection.cross_val_score(
            model, df[features],
            df[self.target], cv=5)
        score = {}
        score["5 folders cv"]= scores
        return scores


    def cross_validate_by_experiments(self, model, df, features):
        """Test a model by LeaveOneGroupOut cross-validation.

        Parameters
        ----------
        df : pandas.DataFrame
            pandas dataframe of features and labels
        model : sk-learn
            with specific parameters

        Returns
        -------
        test_scores_by_ex : object
            TODO: describe scores output
        """
        experiments = df.experiment_id.unique()# we have at least 3 experiments
        test_scores_by_ex = {}
        scores = []
        for i in range(len(experiments)):
            tr = df[(df['experiment_id'] != experiments[i])]
            test = df[(df['experiment_id'] == experiments[i])]
            #TODO  decide if we want to remove from the test data
            # the samples with the the labels that were not included in
            # training data
            model.fit(tr[features], tr[self.target])

            y_pred = model.predict(test[features])
            cmat = confusion_matrix(test[self.target], y_pred)

            print(cmat)

            #TOD0 : remove thit ref:
            # https://www.quora.com/How-do-you-measure-the-accuracy-score-for-each-class-when-testing-classifier-in-sklearn
            # the correct number of classifications for each label are given
            # by the diagonal entries. The totals can be found by summing
            # the rows. The fraction of correctly classified labels for
            # each case is then given by:
            accuracies_by_classes = cmat.diagonal()/cmat.sum(axis=1)
            average_acc_for_this_exp = sum(accuracies_by_classes)/len(accuracies_by_classes)
            scores.append(average_acc_for_this_exp)
            test_scores_by_ex[experiments[i]] = {}

            test_scores_by_ex[experiments[i]]['average for exp'] = average_acc_for_this_exp
            test_scores_by_ex[experiments[i]]['by classes'] = accuracies_by_classes

        mean_score = sum(scores)/len(scores)

        result = {}
        result["mean_score"]= mean_score
        result["score_by_exp"] = test_scores_by_ex

        return result

    '''
    # There is a very simple variant of saving classifier,
    # but it looks different from save_regression_models()
    def save_classification_model(self, test=False):
        """Save model parameters and CV errors in YAML and .txt files.
        """
        p = os.path.abspath(__file__)
        d = os.path.dirname(p)
        if test:
            file_path = os.path.join(d,'modeling_data','testing_data','regressors','system_class.yml')
        else:
            file_path = self.model_file

        cverr_txt_path = os.path.splitext(file_path)[0]+'.txt'

        s_and_m = {self.target : {'scaler': self.scaler.__dict__, 'model': self.model.__dict__,
                   'parameters' : self.parameters, 'accuracy': self.accuracy}}

        # save scalers and models
        with open(self.model_file, 'w') as yaml_file:
            yaml.dump(s_and_m, yaml_file)

        # save accuracy
        with open(cverr_txt_path, 'w') as txt_file:
            txt_file.write(str(s_and_m[self.target]['accuracy']))
    '''

class SystemClassifier(Classifier):
    """Classifier for determining the material system (structures and form factors).

    See the main xrsdkit package documentation 
    for all supported structures and parameters
    for defining the material system.
    """

    def __init__(self,yml_file):
        super(SystemClassifier,self).__init__('system_class', yml_file)


