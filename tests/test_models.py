import os
import glob

import numpy as np

from xrsdkit.tools import profiler
from xrsdkit.models.classifiers import Classifiers
from xrsdkit.models.regressors import Regressors

from citrination_client import CitrinationClient
from xrsdkit.tools.citrination_tools import get_data_from_Citrination

def test_classifiers_and_regressors():
    cl_models = Classifiers()
    reg_models = Regressors()
    for data_type in ['precursors','spheres','peaks']:
        data_path = os.path.join(os.getcwd(),'tests','test_data','solution_saxs',data_type)
        data_files = glob.glob(os.path.join(data_path,'*.csv'))
        for fpath in data_files:
            print('testing classifiers and regressors on {}'.format(fpath))
            q_I = np.loadtxt(fpath,delimiter=',')
            prof = profiler.profile_spectrum(q_I)
            cl_result = cl_models.make_predictions(prof)
            for k, v in cl_result.items():
                print(k, ' :', v[0], "  with probability: %1.3f" % (v[1]))
            reg_result = reg_models.make_predictions(prof, cl_result, q_I)
            for k, v in reg_result.items():
                print(k, " :   %10.3f" % (v))


def test_training():
    p = os.path.dirname(os.path.abspath(__file__))
    d = os.path.dirname(p)
    api_key_file = os.path.join(d,'api_key.txt')
    if not os.path.exists(api_key_file):
        return
    with open(api_key_file, "r") as g:
        a_key = g.readline().strip()
    cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)

    data = get_data_from_Citrination(client=cl, dataset_id_list=[21,22,24,25,26,27])
    data_len = data.shape[0]
    train = data.iloc[:int(data_len*0.9),:]
    train_part = data.iloc[int(data_len*0.9):,:]

    test_path = os.path.join(d,'xrsdkit','models','modeling_data', 'test_classifiers')

    # train from scratch
    my_classifiers = Classifiers() # we can specify the list of classifiers to train
    results = my_classifiers.train_classification_models(train, hyper_parameters_search = False)
    my_classifiers.print_training_results(results)
    my_classifiers.save_classification_models(results, test_path)

    rg_models = Regressors()
    results = rg_models.train_regression_models(train, hyper_parameters_search = False)
    rg_models.print_training_results(results)
    rg_models.save_regression_models(results, test_path)

    # update models
    my_classifiers = Classifiers() # we can specify the list of classifiers to train
    results = my_classifiers.train_classification_models(train_part, testing_data = data, partial = True)
    my_classifiers.print_training_results(results)
    my_classifiers.save_classification_models(results, test_path)

    rg_models = Regressors()
    results = rg_models.train_regression_models(train_part, testing_data = data, partial = True)
    rg_models.print_training_results(results)
    rg_models.save_regression_models(results, test_path)
