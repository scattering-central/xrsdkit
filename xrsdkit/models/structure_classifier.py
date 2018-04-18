import numpy as np
from .general_model import XRSDModel


class StructureClassifier(XRSDModel):
    """To create classifier for classifying structure from scattering/diffraction data;
    train, update, and save it; make a prediction."""

    def __init__(self,label,yml_file_cl=None):
        super(StructureClassifier,self).__init__(label, yml_file_cl)
        # use all default settings of xrsdModel

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
            struct = int(self.model.predict(x)[0])
            cert = self.model.predict_proba(x)[0,struct]

        return struct, cert
