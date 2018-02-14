import os
import warnings
warnings.filterwarnings("ignore")

from citrination_client import CitrinationClient
from saxskit.saxs_models import get_data_from_Citrination
from saxskit.saxs_models import train_classifiers, train_regressors

path = os.getcwd()
api_key_file = path + '/citrination_api_key_ssrl.txt'

with open(api_key_file, "r") as g:
    a_key = g.readline().strip()
cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)

data = get_data_from_Citrination(client = cl, dataset_id_list= [1,15]) # [1,15] is a list of datasets ids

train_classifiers(data,  hyper_parameters_search = True)

with open(path + "/saxskit/saxskit/modeling_data/accuracy.txt", "r") as g:
    accuracy = g.readline()
print(accuracy)

train_regressors(data,  hyper_parameters_search = True)

with open(path + "/saxskit/saxskit/modeling_data/accuracy_regression.txt", "r") as g:
    accuracy = g.readline()
print(accuracy)