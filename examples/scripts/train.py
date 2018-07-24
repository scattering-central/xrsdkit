import os

from citrination_client import CitrinationClient
from xrsdkit.models import \
    train_regression_models, \
    print_training_results, \
    save_regression_models

from xrsdkit.tools.citrination_tools import downsample_Citrination_datasets

file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(os.path.dirname(file_path))
root_dir = os.path.dirname(src_dir)
modeling_data_dir = os.path.join(src_dir,'models','modeling_data')

api_key_file = os.path.join(root_dir, 'api_key.txt')

src_dsid_file = os.path.join(src_dir,'models','modeling_data','source_dataset_ids.yml')
src_dsid_list = yaml.load(open(src_dsid_file,'r'))
data = downsample_Citrination_datasets(cl, src_dsid_list, save_sample=False)

def train_models(save_models=False):
    if not os.path.exists(api_key_file):
        msg = 'No api_key.txt file found in {}'.format(root_dir)
        raise FileNotFoundError(msg) 
    a_key = open(api_key_file, 'r').readline().strip()
    cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)

    # system classifier:
    # TODO: add training for system classifier

    # regression models:
    reg_models = train_regression_models(data, hyper_parameters_search=True)
    print_training_results(reg_models)
    if save_models:
        save_regression_models(reg_models, modeling_data_dir)

