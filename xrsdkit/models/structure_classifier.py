import numpy as np
import os
import yaml
from .general_model import XRSDModel


class StructureClassifier(XRSDModel):
    """To create classifier for classifying structure from scattering/diffraction data;
    train, update, and save it; make a prediction."""

    def __init__(self,label,yml_file_cl=None):
        # 'system_class' is the column name for target
        super(StructureClassifier,self).__init__('system_class', yml_file_cl)

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
        print('system_class', ":")
        if results['accuracy']:
            print("accuracy :  %10.3f" % (results['accuracy']))
            print("parameters :", results['parameters'])
        else:
            print("training/updatin is not possible (not enough data)")
        print()


    def save_models(self, file_path=None):
        """Save model parameters and CV errors in YAML and .txt files.
        Parameters
        ----------
        scaler_model : dict
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
        file_path : str
            Full path to the YAML file where the models will be saved.
            Scaler, model, and cross-validation error
            will be saved at this path, and the cross-validation error
            are also saved in a .txt file of the same name, in the same directory.
        """

        if file_path is None:
            p = os.path.abspath(__file__)
            d = os.path.dirname(p)
            suffix = 0
            file_path = os.path.join(d,'modeling_data', 'classifiers',
                'custom_models_'+ self.target +str(suffix)+'.yml')
            while os.path.exists(file_path):
                suffix += 1
                file_path = os.path.join(d,'modeling_data', 'classifiers',
                    'custom_models_'+ self.target + str(suffix)+'.yml')

        file_path = file_path + '/classifiers/' + self.target + '.yml'

        cverr_txt_path = os.path.splitext(file_path)[0]+'.txt'

        s_and_m = {self.target : {'scaler': self.scaler.__dict__, 'model': self.model.__dict__,
                   'parameters' : self.parameters, 'accuracy': self.cv_error}}

        # save scalers and models
        with open(file_path, 'w') as yaml_file:
            yaml.dump(s_and_m, yaml_file)

        # save accuracy
        with open(cverr_txt_path, 'w') as txt_file:
            txt_file.write(str(s_and_m[self.target]['accuracy']))



    def print_accuracies(self):
        """Report cross-validation error for the model.
        To calculate cv_error "Leave-2-Groups-Out" cross-validation is used.
        For each train-test split,
        two experiments are used for testing
        and the rest are used for training.
        The reported error is the average accuracy over all train-test splits.
        """
        if self.get_cv_error():
            print("Averaged cross validation accuracies: ")
            print("system_class :  %10.3f" % (self.get_cv_error()))
        else:
            print("The model was not trained yet.")
