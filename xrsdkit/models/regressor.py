import numpy as np
from .general_model import XrsdModel


class Regressor(XrsdModel):
    """Models for prediction continuous values from scattering/diffraction data"""

    def __init__(self,label,yml_file_cl=None):

        XrsdModel.__init__(self, label, yml_file=yml_file_cl, classifier = False)

        self.parameters_to_try = \
            {'loss':('huber', 'squared_loss'), # huber with epsilon = 0 gives us abs error (MAE)
              'epsilon': [1, 0.1, 0.01, 0.001, 0],
              'penalty':['none', 'l2', 'l1', 'elasticnet'], #default l2
              'alpha':[0.0001, 0.001, 0.01], #default 0.0001
             'l1_ratio': [0, 0.15, 0.5, 0.95], #default 0.15
             }

        # TODO set features

    def predict(self, sample_features, populations):
        """Determine the types of structures represented by the sample

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of saxs_math.profile_spectrum()
        populations : dict
            dictionary counting scatterer populations,
            similar to output of SaxsClassifier.classify()

        Returns
        -------
        prediction : float or None
            predicted parameter
            None is reterned for models that was not trained yet
        """
        feature_array = np.array(list(sample_features.values())).reshape(1,-1)

        if self.scaler: # we have a saved model
            x = self.scaler.transform(feature_array)
            prediction = int(self.model.predict(x)[0])

            return prediction
        else:
            return None
