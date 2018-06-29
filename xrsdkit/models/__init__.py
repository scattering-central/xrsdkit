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
    """Get dictionary of models that we can train using provided data.

    Parameters
    ----------
    data : pandas.DataFrame
        dataframe containing features and labels
        returned by citrination_tools.get_data_from_Citrination

    Returns
    -------
    model_labels : dict
        dictionary of possible regression models for each system_class
        (can be trained using provided data)
    """

    sys_cls = list(data.system_class.unique())
    if 'noise' in sys_cls:
        sys_cls.remove('noise')
    if 'pop0_unidentified' in sys_cls:
        sys_cls.remove('pop0_unidentified')
    model_labels = OrderedDict.fromkeys(sys_cls)
    for cls in sys_cls:
        cls_data = data[(data['system_class']==cls)]

        #to find the list of possible models and train all possible regression models
        #drop the collumns where all values are None:
        cls_data.dropna(axis=1,how='all',inplace=True)
        cols = cls_data.columns
        possible_models = []
        for c in cols:
            if any([rp == c[-1*len(rp):] for rp in regression_params]):
            #end = c.split("_")[-1]
            #if end in regression_params:
                if data[c].shape[0] > 10: #TODO change to 100 when we will have more data
                    possible_models.append(c)
        model_labels[cls] = possible_models
    return model_labels 



