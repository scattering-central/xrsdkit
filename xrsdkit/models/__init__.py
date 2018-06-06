from collections import OrderedDict
from .. import regression_params

"""Data processing modules based on scikit-learn and Citrination"""

# helper function - to set parameters for scalers and models
def set_param(m_s, param):
    for k, v in param.items():
        if isinstance(v, list):
            setattr(m_s, k, np.array(v))
        else:
            setattr(m_s, k, v)

def get_possible_regression_models(data):
    """Get dictionary of models that we can train
    using provided data.
    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels
        returned by citrination_tools.get_data_from_Citrination
    Returns
    -------
    result : dict
        dictionary of possible regression models for each population
        (can be trained using provided data)
    """

    pops = list(data.populations.unique())
    if "Noise" in pops:
        pops.remove("Noise")
    if "pop0_unidentified"in pops:
        pops.remove("pop0_unidentified")
    result = OrderedDict.fromkeys(pops)
    for p in pops:
        pop_data = data[(data['populations']==p)]

        #to find the list of possible models and train all possible regression models
        #drop the collumns where all values are None:
        pop_data.dropna(axis=1, how='all',inplace=True)
        cols = pop_data.columns
        possible_models = []
        for c in cols:
            end = c.split("_")[-1]
            if end in regression_params:
                if data[c].shape[0] > 10: #TODO change to 100 when we will have more data
                    possible_models.append(c)
        result[p] = possible_models
    return result



