import os
import warnings
warnings.filterwarnings("ignore")

from citrination_client import CitrinationClient
from xrsdkit.models.classifiers import Classifiers
from xrsdkit.models.regressor import Regressor

from xrsdkit.tools.citrination_tools import get_data_from_Citrination

p = os.path.abspath(__file__)
d = os.path.dirname(os.path.dirname(os.path.dirname(p)))

api_key_file = os.path.join(d, 'api_key.txt')
if not os.path.exists(api_key_file):
    print("Citrination api key file did not find")

with open(api_key_file, "r") as g:
    a_key = g.readline().strip()
cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)

#data = get_data_from_Citrination(client = cl, dataset_id_list= [21,22,23,24])
data = get_data_from_Citrination(client = cl, dataset_id_list= [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
print(data.shape)

classifiers_path = os.path.join(d,'xrsdkit','models','modeling_data')

my_classifiers = Classifiers() # we can specify the list of classifiers to train
results = my_classifiers.train_classification_models(data, hyper_parameters_search = True)
my_classifiers.save_classification_models(results, classifiers_path)



'''
# regression models:
rg_model = Regressor('r_g_0')
scaler, model, parameters,  accuracy = rg_model.train(data_diffuse_only, hyper_parameters_search = True)
print(accuracy)
classifiers_path = os.path.join(d,'xrsdkit','models','modeling_data','scalers_and_models_rg_0.yml')
rg_model.save_models(scaler, model, parameters,  accuracy, classifiers_path)
'''
#scalers, models, accuracy = train_regressors(data, hyper_parameters_search = True, model= 'all')

# if we want to train only "r0_sphere" model:
#scalers, models, accuracy = train_regressors(data, hyper_parameters_search = False, model= 'r0_sphere')
#save_models(scalers, models, accuracy, regressors_path)