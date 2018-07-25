import os
import glob

import numpy as np

from xrsdkit.tools import profiler
from xrsdkit.models.structure_classifier import StructureClassifier

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

    #my_classifier = StructureClassifier("system_class")
    #my_classifier.train(train, hyper_parameters_search = False)
    #my_classifier.save_models(test_path)

    #reg_models = train_regression_models(data, hyper_parameters_search=False)
    #save_regression_models(reg_models, test_path)

