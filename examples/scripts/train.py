import os
import warnings
warnings.filterwarnings("ignore")
import time

from citrination_client import CitrinationClient
from xrsdkit.models.structure_classifier import StructureClassifier
from xrsdkit.models import train_regression_models, print_training_results, \
    save_regression_models

from xrsdkit.tools.citrination_tools import get_data_from_Citrination, downsample_Citrination_datasets

p = os.path.abspath(__file__)
d = os.path.dirname(os.path.dirname(os.path.dirname(p)))

api_key_file = os.path.join(d, 'api_key.txt')
if not os.path.exists(api_key_file):
    print("Citrination api key file did not find")

with open(api_key_file, "r") as g:
    a_key = g.readline().strip()
cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)

data = downsample_Citrination_datasets(cl, [22,23,28,29,30,31,32,33,34,35,36],save_sample=False)

#data = get_data_from_Citrination(client = cl, dataset_id_list= [21,22,23,28,29,30,31,32,33,34,35,36])

models_path = os.path.join(d,'xrsdkit','models','modeling_data')
'''
my_classifier = StructureClassifier("system_class")
print("Old accuracies for classifiers:")
my_classifier.print_accuracies()

my_classifier.train(data, hyper_parameters_search = True)
print("New accuracies and parameters for classifiers:")
my_classifier.print_accuracies()
my_classifier.save_models(models_path)
'''
# regression models:
reg_models = train_regression_models(data, hyper_parameters_search=True)
print(reg_models)
print("New accuracies and parameters for regressors:")
print_training_results(reg_models)
#save_regression_models(reg_models, models_path)

