import os
import warnings
warnings.filterwarnings("ignore")
import time

from citrination_client import CitrinationClient
from citrination_client.data.client import DataClient
from xrsdkit.models.structure_classifier import StructureClassifier
from xrsdkit.models.regressors import Regressors

from xrsdkit.tools.citrination_tools import get_data_from_Citrination, sampl_data_on_Citrination

p = os.path.abspath(__file__)
d = os.path.dirname(os.path.dirname(os.path.dirname(p)))

api_key_file = os.path.join(d, 'api_key.txt')
if not os.path.exists(api_key_file):
    print("Citrination api key file did not find")

with open(api_key_file, "r") as g:
    a_key = g.readline().strip()
cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)
data_cl = DataClient(host='https://slac.citrination.com',api_key=a_key)

id, count = sampl_data_on_Citrination(cl,data_cl, [21,22,23,28,29,30,31,32,33,34,35,36])
#print(id)
time.sleep(300) # wait 5 minutes

data = get_data_from_Citrination(client = cl, dataset_id_list= [id])

while data.shape[0] != count:
    print("we got only ", data.shape[0])
    time.sleep(300)
    data = get_data_from_Citrination(client = cl, dataset_id_list= [id])

#data = get_data_from_Citrination(client = cl, dataset_id_list= [21,22,23,28,29,30,31,32,33,34,35,36])
models_path = os.path.join(d,'xrsdkit','models','modeling_data')

my_classifier = StructureClassifier("system_class")
print("Old accuracies for classifiers:")
my_classifier.print_accuracies()

results = my_classifier.train(data, hyper_parameters_search = True)
print("New accuracies and parameters for classifiers:")
my_classifier.print_training_results(results)
my_classifier.save_models(results, models_path)

# regression models:
rg_models = Regressors()
#print("Old accuracies for regressors:")
#rg_models.print_errors()
results = rg_models.train_regression_models(data, hyper_parameters_search = True)
print("New accuracies and parameters for regressors:")
rg_models.print_training_results(results)
rg_models.save_regression_models(results, models_path)

