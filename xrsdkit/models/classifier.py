import numpy as np
from sklearn import linear_model, model_selection, preprocessing

from .xrsd_model import XRSDModel
from ..tools import profiler


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

    def run_cross_validation(self,model,data,group_cv):
        if group_cv:
            new_accuracy = self.cross_validate_by_experiments(model,data)
        else:
            new_accuracy = self.cross_validate(model,data)
        return new_accuracy

    def cross_validate(self, model, df):
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
        scaler = preprocessing.StandardScaler()
        scaler.fit(df[profiler.profile_keys_1])
        scores = model_selection.cross_val_score(
            model, scaler.transform(df[profiler.profile_keys_1]),
            df[self.target], cv=5)
        return scores

    def cross_validate_by_experiments(self, model, df):
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
        experiments = df.experiment_id.unique()# we have at least 5 experiments
        test_scores_by_ex = []
        for i in range(len(experiments)):
            tr = df[(df['experiment_id'] != experiments[i])]
            test = df[(df['experiment_id'] == experiments[i])]
            # for testing, we want only the samples with labels that are
            # included in training set:

            # TODO: not clear why this is needed- consider removing?
            # (all samples in the dataset should have some label for the target)
            tr_labels = tr[self.target].unique()
            test = test[test[self.target].isin(tr_labels)]

            scaler = preprocessing.StandardScaler()
            scaler.fit(tr[profiler.profile_keys_1])
            model.fit(scaler.transform(tr[profiler.profile_keys_1]), tr[self.target])
            transformed_data = scaler.transform(test[profiler.profile_keys_1])
            test_score = model.score(
                transformed_data, test[self.target])
            test_scores_by_ex.append(test_score)

        return test_scores_by_ex

class SystemClassifier(Classifier):
    """Classifier for determining the material system (structures and form factors).

    See the main xrsdkit package documentation 
    for all supported structures and parameters
    for defining the material system.
    """

    def __init__(self,yml_file):
        super(SystemClassifier,self).__init__('system_class', yml_file)


