from collections import OrderedDict
import os

import numpy as np
import sklearn
from sklearn import preprocessing,linear_model
import yaml

from .. import structure_names
from . import set_param

class SphericalClassifier(object):
    """Models for identifying spherical populations from scattering data"""

    def __init__(self,yml_file=None):
        if yml_file is None:
            p = os.path.abspath(__file__)
            d = os.path.dirname(p)
            yml_file = os.path.join(d,'modeling_data','spherical_classifier.yml')

        s_and_m_file = open(yml_file,'rb')
        s_and_m = yaml.load(s_and_m_file)

    def classify(self, sample_features):
        """Count the (normal-distributed) spherical scatterers in the sample

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of profiler.profile_spectrum()

        Returns
        -------
        pop_count : integer 
            predicted number of spherical scatterer populations 
        certainty : float
            certainty of the prediction
        """
        feature_array = np.array(list(sample_features.values())).reshape(1,-1)  
        return

    def training_cv_error(self):
        """Return training cross-validation error for this model.

        TODO: describe exactly what value is being reported.

        Returns
        -------
        cv_error : dict
            dict of metrics of training error for this model.
        """
        return
