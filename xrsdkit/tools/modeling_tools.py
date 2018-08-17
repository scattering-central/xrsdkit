import os

from citrination_client import CitrinationClient
from xrsdkit.models import root_dir,downsample_and_train

def train_models(hyperparameters_search=False):
    api_key_file = os.path.join(root_dir, 'api_key.txt')
    dataset_id_index = os.path.join(root_dir,'xrsdkit','models','modeling_data','source_dataset_ids.yml')
    dataset_id_list = yaml.load(open(dataset_id_index,'r'))
    if os.path.exists(api_key_file):
        a_key = open(api_key_file, 'r').readline().strip()
        cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)
        downsample_and_train(
            dataset_id_list,cl,
            save_samples=True,
            save_models=True,
            train_hyperparameters=hyperparameters_search)

