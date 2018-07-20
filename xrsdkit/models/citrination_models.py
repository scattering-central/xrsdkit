from collections import OrderedDict

import pandas as pd
import numpy as np
from citrination_client import CitrinationClient

from .. import structure_names
        
class CitrinationSystemClassifier(object):
    """Citrination-backed model for system classification"""

    def __init__(self,api_key_file,address='https://slac.citrination.com'):
        with open(api_key_file, "r") as g:
            api_key = g.readline()
        a_key = api_key.strip()
        self.client = CitrinationClient(site = address, api_key=a_key)

    def classify(self,sample_params):
        """Determine the types of structures represented by the sample

        Parameters
        ----------
        sample_params : ordered dictionary
            ordered dictionary of floats representing features of test sample

        Returns
        -------
        population : str
            predicted system class.
        uncertainty : float
            uncertainty of the prediction.
        """

        inputs = self.append_str_property(sample_params)
        # TODO: add a .yml index of dataview_ids like dataset_ids
        resp = self.client.predict("64", inputs) # "64" is ID of dataview on Citrination

        population = resp[0].get_value("system_classification").value
        uncertainty = resp[0].get_value("system_classification").loss

        return population, uncertainty

    # helper function
    def append_str_property(self, sample_params):
        inputs = {}
        for k,v in sample_params.items():
            k = "Property " + k
            inputs[k] = v
        return inputs

