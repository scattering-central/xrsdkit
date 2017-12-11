from collections import OrderedDict

import yaml
import os

class SaxsRegressor(object):
    """A set of regression models to be used on SAXS spectra"""

    def __init__(self,yml_file=None):
        pass

    # helper function - to set parametrs for scalers and models
    def set_param(self, m_s, param):
        for k, v in param.items():
            if isinstance(v, list):
                setattr(m_s, k, np.array(v))
            else:
                setattr(m_s, k, v)

    def predict_params(self,populations,features):
        pass

