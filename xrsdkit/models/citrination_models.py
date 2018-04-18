from collections import OrderedDict

import pandas as pd
import numpy as np
from citrination_client import CitrinationClient

from .. import structure_names

class CitrinationStructureClassifier(object):
    """Citrination models for classifying structure from scattering/diffraction data"""

    def __init__(self,api_key_file,address='https://slac.citrination.com'):
        with open(api_key_file, "r") as g:
            api_key = g.readline()
        a_key = api_key.strip()
        self.client = CitrinationClient(site = address, api_key=a_key)

    def classify(self,sample_features):
        """Determine the types of structures represented by the sample"""
        struct_flags = OrderedDict.fromkeys(structure_names)
        for struct_name in structure_names:
            struct_flags[struct_name] = False
        return struct_flags

#
#class CitrinationSaxsModels(object):
#    """A set of models that uses Citrination to evaluate SAXS spectra.
#
#    Use of this class requires a Citrination api key.
#    You can get one by making an account on Citrination.
#    The api key should be copy/pasted into a file,
#    and the path to the file should be provided
#    as an instantiation argument.
#    """
#
#    def __init__(self, api_key_file, address='https://slac.citrination.com'):
#        with open(api_key_file, "r") as g:
#            api_key = g.readline()
#        a_key = api_key.strip()
#
#        self.client = CitrinationClient(site = address, api_key=a_key)
#
#
#    def classify(self,sample_params):
#        """
#        Parameters
#        ----------
#        sample_params : ordered dictionary
#            ordered dictionary of floats representing features of test sample
#
#        Returns
#        -------
#        Returns
#        -------
#        populations : dict
#            dictionary of integers
#            counting predicted scatterer populations
#            for all populations in population_keys.
#        uncertainties : dict
#            dictionary, similar to `populations`,
#            but containing the uncertainty of the prediction
#        """
#
#        inputs = self.append_str_property(sample_params)
#
#        populations = OrderedDict()
#        uncertainties = OrderedDict()
#        resp = self.client.predict("33", inputs) # "33" is ID of dataview on Citrination
#        for popname in population_keys:
#            populations[popname] = int(resp['candidates'][0]['Property '+popname][0])
#            uncertainties[popname] = float(resp['candidates'][0]['Property '+popname][1])
#
#        return populations, uncertainties
#
#
#    # helper function
#    def append_str_property(self, sample_params):
#        inputs = {}
#        for k,v in sample_params.items():
#            k = "Property " + k
#            inputs[k] = v
#        return inputs
#
#    def predict_params(self,populations,features,q_I):
#        """Use Citrination to predict the scattering parameters.
#
#        Parameters
#        ----------
#        populations : dict
#            dictionary counting scatterer populations,
#            similar to output of self.classify()
#        features : dict
#            dictionary of sample numerical features,
#            similar to output of profiler.profile_spectrum().
#        q_I : array
#            n-by-2 array of scattering vector (1/Angstrom) and intensities.
#
#        Returns
#        -------
#        params : dict
#            dictionary of predicted and calculated scattering parameters:
#            r0_sphere, sigma_sphere, and rg_gp 
#            are predicted using Citrination models.
#        """
#        # TODO: The predictions need to handle 
#        # multiple populations of the same type:
#        # include this once we have training data
#
#        features = self.append_str_property(features)
#
#        params = OrderedDict()
#        uncertainties = OrderedDict()
#        if bool(populations['unidentified']):
#            return params, uncertainties
#
#        if bool(populations['spherical_normal']):
#            resp = self.client.predict("34", features) # "34" is ID of dataview on Citrination
#            params['r0_sphere'] = [float(resp['candidates'][0]['Property r0_sphere'][0])]
#            uncertainties['r0_sphere'] = float(resp['candidates'][0]['Property r0_sphere'][1])
#            additional_features = profiler.spherical_normal_profile(q_I)
#            additional_features = self.append_str_property(additional_features)
#            ss_features = OrderedDict(features)
#            ss_features.update(additional_features)
#            resp = self.client.predict("31", ss_features)
#            params['sigma_sphere'] = [float(resp['candidates'][0]['Property sigma_sphere'][0])]
#            uncertainties['sigma_sphere'] = float(resp['candidates'][0]['Property sigma_sphere'][1])
#
#        if bool(populations['guinier_porod']):
#            additional_features = profiler.guinier_porod_profile(q_I)
#            additional_features = self.append_str_property(additional_features)
#            rg_features = dict(features)
#            rg_features.update(additional_features)
#            resp =self.client.predict("35", rg_features)
#            params['rg_gp'] = [float(resp['candidates'][0]['Property rg_gp'][0])]
#            uncertainties['rg_gp'] = float(resp['candidates'][0]['Property rg_gp'][1])
#
#        return params,uncertainties
