import os
import warnings
warnings.filterwarnings("ignore")

from citrination_client import CitrinationClient
from saxskit.saxs_models import get_data_from_Citrination
from saxskit.saxs_models import train_classifiers, train_regressors, save_models

p = os.path.abspath(__file__)
d = os.path.dirname(os.path.dirname(p))
classifiers_path = os.path.join(d,'saxskit','modeling_data','scalers_and_models.yml')
regressors_path = os.path.join(d,'saxskit','modeling_data','scalers_and_models_regression.yml')

api_key_file = os.path.join(d, 'api_key.txt')
if not os.path.exists(api_key_file):
    print("Citrination api key file did not find")

with open(api_key_file, "r") as g:
    a_key = g.readline().strip()
cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)

data = get_data_from_Citrination(client = cl, dataset_id_list= [1,15])

scalers, models, accuracy = train_classifiers(data, hyper_parameters_search = True, model='all')
save_models(scalers, models, accuracy, classifiers_path)

scalers, models, accuracy = train_regressors(data, hyper_parameters_search = True, model= 'all')

# if we want to train only "r0_sphere" model:
#scalers, models, accuracy = train_regressors(data, hyper_parameters_search = False, model= 'r0_sphere')
save_models(scalers, models, accuracy, regressors_path)