import os

from citrination_client import CitrinationClient
from xrsdkit.models import root_dir,downsample_and_train

def test_training():
    api_key_file = os.path.join(root_dir, 'api_key.txt')
    if os.path.exists(api_key_file):
        a_key = open(api_key_file, 'r').readline().strip()
        cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)
        downsample_and_train(
            [22,23,28,29,30],cl,
            save_samples=False,
            save_models=False,
            train_hyperparameters=False)

