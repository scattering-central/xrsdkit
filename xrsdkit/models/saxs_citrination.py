from collections import OrderedDict

from ..tools import profiler 

from citrination_client import CitrinationClient
from citrination_client import PifSystemReturningQuery, DatasetQuery, DataQuery, Filter


# TODO: refactor to new data model
population_keys = ['unidentified','guinier_porod','spherical_normal','diffraction_peaks']

def get_data_from_Citrination(client, dataset_id_list):
    """Get data from Citrination and create a dataframe.

    Parameters
    ----------
    client : citrination_client.CitrinationClient
        A python Citrination client for fetching data
    dataset_id_list : list of int
        List of dataset ids (integers) for fetching SAXS records

    Returns
    -------
    df_work : pandas.DataFrame
        dataframe containing features and labels
        obtained through `client` from the Citrination datasets
        listed in `dataset_id_list`
    """
    data = []

    pifs = get_pifs_from_Citrination(client,dataset_id_list)

    for pp in pifs:
        feats = OrderedDict.fromkeys(all_profile_keys)
        pops = OrderedDict.fromkeys(population_keys)
        par = OrderedDict.fromkeys(all_parameter_keys)
        expt_id,t_utc,q_I,temp,pif_feats,pif_pops,pif_par,rpt = saxs_piftools.unpack_pif(pp)
        feats.update(saxs_math.profile_spectrum(q_I))
        feats.update(saxs_math.detailed_profile(q_I,pif_pops))
        pops.update(pif_pops)
        par.update(pif_par)
        param_list = []
        for k in par.keys():
            if par[k] is not None:
                val = par[k][0]
            else:
                val = None
            param_list.append(val)

        data_row = [expt_id]+list(feats.values())+list(pops.values())+param_list
        data.append(data_row)

    colnames = ['experiment_id']
    colnames.extend(all_profile_keys)
    colnames.extend(population_keys)
    colnames.extend(all_parameter_keys)

    d = pd.DataFrame(data=data, columns=colnames)
    d = d.where((pd.notnull(d)), None) # replace all NaN by None
    shuffled_rows = np.random.permutation(d.index)
    df_work = d.loc[shuffled_rows]

def get_pifs_from_Citrination(client, dataset_id_list):
    all_hits = []
    for dataset in dataset_id_list:
        query = PifSystemReturningQuery(
            from_index=0,
            size=100,
            query=DataQuery(
                dataset=DatasetQuery(
                    id=Filter(
                    equal=dataset))))

        current_result = client.search(query)
        while current_result.hits is not None:
            all_hits.extend(current_result.hits)
            n_current_hits = len(current_result.hits)
            #n_hits += n_current_hits
            query.from_index += n_current_hits 
            current_result = client.search(query)

    pifs = [x.system for x in all_hits]
    return pifs

class CitrinationSaxsModels(object):
    """A set of models that uses Citrination to evaluate SAXS spectra.

    Use of this class requires a Citrination api key.
    You can get one by making an account on Citrination.
    The api key should be copy/pasted into a file,
    and the path to the file should be provided
    as an instantiation argument.
    """

    def __init__(self, api_key_file, address='https://slac.citrination.com'):
        with open(api_key_file, "r") as g:
            api_key = g.readline()
        a_key = api_key.strip()

        self.client = CitrinationClient(site = address, api_key=a_key)


    def classify(self,sample_params):
        """
        Parameters
        ----------
        sample_params : ordered dictionary
            ordered dictionary of floats representing features of test sample

        Returns
        -------
        Returns
        -------
        populations : dict
            dictionary of integers
            counting predicted scatterer populations
            for all populations in population_keys.
        uncertainties : dict
            dictionary, similar to `populations`,
            but containing the uncertainty of the prediction
        """

        inputs = self.append_str_property(sample_params)

        populations = OrderedDict()
        uncertainties = OrderedDict()
        resp = self.client.predict("33", inputs) # "33" is ID of dataview on Citrination
        for popname in population_keys:
            populations[popname] = int(resp['candidates'][0]['Property '+popname][0])
            uncertainties[popname] = float(resp['candidates'][0]['Property '+popname][1])

        return populations, uncertainties


    # helper function
    def append_str_property(self, sample_params):
        inputs = {}
        for k,v in sample_params.items():
            k = "Property " + k
            inputs[k] = v
        return inputs

    def predict_params(self,populations,features,q_I):
        """Use Citrination to predict the scattering parameters.

        Parameters
        ----------
        populations : dict
            dictionary counting scatterer populations,
            similar to output of self.classify()
        features : dict
            dictionary of sample numerical features,
            similar to output of profiler.profile_spectrum().
        q_I : array
            n-by-2 array of scattering vector (1/Angstrom) and intensities.

        Returns
        -------
        params : dict
            dictionary of predicted and calculated scattering parameters:
            r0_sphere, sigma_sphere, and rg_gp 
            are predicted using Citrination models.
        """
        # TODO: The predictions need to handle 
        # multiple populations of the same type:
        # include this once we have training data

        features = self.append_str_property(features)

        params = OrderedDict()
        uncertainties = OrderedDict()
        if bool(populations['unidentified']):
            return params, uncertainties

        if bool(populations['spherical_normal']):
            resp = self.client.predict("34", features) # "34" is ID of dataview on Citrination
            params['r0_sphere'] = [float(resp['candidates'][0]['Property r0_sphere'][0])]
            uncertainties['r0_sphere'] = float(resp['candidates'][0]['Property r0_sphere'][1])
            additional_features = profiler.spherical_normal_profile(q_I)
            additional_features = self.append_str_property(additional_features)
            ss_features = OrderedDict(features)
            ss_features.update(additional_features)
            resp = self.client.predict("31", ss_features)
            params['sigma_sphere'] = [float(resp['candidates'][0]['Property sigma_sphere'][0])]
            uncertainties['sigma_sphere'] = float(resp['candidates'][0]['Property sigma_sphere'][1])

        if bool(populations['guinier_porod']):
            additional_features = profiler.guinier_porod_profile(q_I)
            additional_features = self.append_str_property(additional_features)
            rg_features = dict(features)
            rg_features.update(additional_features)
            resp =self.client.predict("35", rg_features)
            params['rg_gp'] = [float(resp['candidates'][0]['Property rg_gp'][0])]
            uncertainties['rg_gp'] = float(resp['candidates'][0]['Property rg_gp'][1])

        return params,uncertainties
