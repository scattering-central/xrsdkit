from collections import OrderedDict

from . import saxs_math

from citrination_client import CitrinationClient

class SaxsCitrination(object):
    """A set of classifiers to be used on SAXS spectra"""

    def __init__(self, address, api_key_file):
        with open(api_key_file, "r") as g:
            api_key = g.readline()
        a_key = api_key.strip()

        self.client = CitrinationClient(site = address, api_key=a_key)


    def citrination_classify(self,sample_params):
        """
        Parameters
        ----------
        sample_params : ordered dictionary
            ordered dictionary of floats representing features of test sample

        Returns
        -------
        flags : dict
            dictionary of (boolean,float) tuples,
            where the first item is the flag
            and the second is the probability,
            for each of the potential scattering populations
        """
        if self.client == None:
            print("Client has not been set up")
            return None

        inputs = self.append_str_property(sample_params)

        flags = OrderedDict()
        resp = self.client.predict("24", inputs) # "24" is ID of dataview on Citrination
        flags['unidentified'] = resp['candidates'][0]['Property unidentified']
        flags['guinier_porod'] = resp['candidates'][0]['Property guinier_porod']
        flags['spherical_normal'] = resp['candidates'][0]['Property spherical_normal']
        flags['diffraction_peaks'] = resp['candidates'][0]['Property diffraction_peaks']

        return flags


    # helper function
    def append_str_property(self, sample_params):
        inputs = {}
        for k,v in sample_params.items():
            k = "Property " + k
            inputs[k] = v
        return inputs


    def citrination_predict(self, populations, sample_params, q_I):
        """Apply self.models and self.scalers to sample_params.

        Parameters
        ----------
        sample_params : ordered dictionary
            ordered dictionary of floats representing features of test sample

        Returns
        -------
        flags : dict
            dictionary of (boolean,float) tuples,
            where the first item is the flag
            and the second is the probability,
            for each of the potential scattering populations
        """
        if self.client == None:
            print("Client has not been set up")
            return None

        features = self.append_str_property(sample_params)

        params = OrderedDict.fromkeys(saxs_math.all_parameter_keys)

        if populations['unidentified'][0] == '1':
            # TODO: we could still use a fit to 'predict' I0_floor...
            return params # all params are "None"

        if populations['spherical_normal'][0] == '1' and populations['diffraction_peaks'][0] == '0':
            resp = self.client.predict("27", features) # "27" is ID of dataview on Citrination
            params['r0_sphere'] = resp['candidates'][0]['Property r0_sphere']

            additional_features = saxs_math.spherical_normal_profile(q_I)
            additional_features = self.append_str_property(additional_features)
            ss_features = dict(features)
            ss_features.update(additional_features)
            resp = self.client.predict("28", ss_features)
            params['sigma_sphere'] = resp['candidates'][0]['Property sigma_sphere']

        if populations['guinier_porod'][0] == '1':
            additional_features = saxs_math.guinier_porod_profile(q_I)
            additional_features = self.append_str_property(additional_features)
            rg_features = dict(features)
            rg_features.update(additional_features)
            resp =self.client.predict("29", rg_features)
            params['rg_gp'] = resp['candidates'][0]['Property rg_gp']

        return params
