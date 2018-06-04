import numpy as np
from .general_model import XRSDModel


class StructureClassifier(XRSDModel):
    """To create classifier for classifying structure from scattering/diffraction data;
    train, update, and save it; make a prediction."""

    def __init__(self,label,yml_file_cl=None):
        # 'populations' is the column name for target
        super(StructureClassifier,self).__init__('populations', yml_file_cl)

    def classify(self, sample_features):
        """Determine the types of structures represented by the sample
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
        struct = None
        cert = None

        feature_array = np.array(list(sample_features.values())).reshape(1,-1)

        if self.scaler: # we have a saved model
            x = self.scaler.transform(feature_array)
            struct = self.model.predict(x)[0]
            cert = max(self.model.predict_proba(x)[0])

        return struct, cert


    def print_training_results(self, results):
        """Print parameters of model and cross validation accuracies.
        Parameters
        ----------
        results : dict
            Dictionary with training results.
        The results include:
        - 'scaler': sklearn standard scaler
            used for transforming of new data
        - 'model':sklearn model
            trained on new data
        - 'parameters': dict
            Dictionary of parameters found by hyperparameters_search()
        - 'accuracy': float
            average crossvalidation score (accuracy for classification,
            normalized mean absolute error for regression)
        """
        print('population', ":")
        if results['accuracy']:
            print("accuracy :  %10.3f" % (results['accuracy']))
            print("parameters :", results['parameters'])
        else:
            print("training/updatin is not possible (not enough data)")
        print()


    def print_accuracies(self):
        """Report cross-validation error for the model.
        To calculate cv_error "Leave-2-Groups-Out" cross-validation is used.
        For each train-test split,
        two experiments are used for testing
        and the rest are used for training.
        The reported error is the average accuracy over all train-test splits.
        """
        print("Averaged cross validation accuracies: ")
        print("populations :  %10.3f" % (self.get_cv_error()))
