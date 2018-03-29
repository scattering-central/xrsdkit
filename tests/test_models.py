from collections import OrderedDict
import os
import glob

import numpy as np

from xrsdkit.tools import profiler
from xrsdkit.models.structure_classifier import StructureClassifier
#from xrsdkit.models.saxs_regression import SaxsRegressor
from collections import OrderedDict

from citrination_client import CitrinationClient
from xrsdkit.tools.citrination_tools import get_data_from_Citrination
from xrsdkit.tools.piftools import model_output_names


def test_classifier():
    p = os.path.dirname(os.path.abspath(__file__))
    d = os.path.dirname(p)

    classifiers = OrderedDict.fromkeys(model_output_names)
    for model in model_output_names:
        classifiers[model] = StructureClassifier(model)

    for data_type in ['precursors','spheres','peaks']:
        data_path = os.path.join(os.getcwd(),'tests','test_data','solution_saxs',data_type)
        data_files = glob.glob(os.path.join(data_path,'*.csv'))
        for fpath in data_files:
            print('testing classifier on {}'.format(fpath))
            q_I = np.loadtxt(fpath,delimiter=',')
            prof = profiler.profile_spectrum(q_I)

            for k, v in classifiers.items():
                pop,cert = v.classify(prof)  # was tmp_prof
                print('\t{} populations: {} ({} certainty)'.format(k,pop,cert))
#
#def test_regression():
#    p = os.path.dirname(os.path.abspath(__file__))
#    d = os.path.dirname(p)
#    p_clsmod = os.path.join(d,'xrsdkit','models','modeling_data','scalers_and_models.yml')
#    p_regmod = os.path.join(d,'xrsdkit','models','modeling_data','scalers_and_models_regression.yml')
#    sxc = SaxsClassifier(p_clsmod)
#    sxr = SaxsRegressor(p_regmod)
#    for data_type in ['precursors','spheres','peaks']:
#        data_path = os.path.join(p,'tests','test_data','solution_saxs',data_type)
#        data_files = glob.glob(os.path.join(data_path,'*.csv'))
#        for fpath in data_files:
#            print('testing regression on {}'.format(fpath))
#            q_I = np.loadtxt(fpath,delimiter=',')
#            prof = profiler.profile_spectrum(q_I)
#
#            # TODO: make all models work with profile_spectrum() output
#            tmp_prof = OrderedDict()
#            for k in prof.keys():
#                if prof[k] is not None:
#                    tmp_prof[k] = prof[k]
#
#            pops,certs = sxc.classify(tmp_prof)
#            params = sxr.predict_params(pops,prof,q_I)
#            for k, v in params.items():
#                print('\t{} parameter: {} '.format(k,v))
#
def test_training():
    p = os.path.dirname(os.path.abspath(__file__))
    d = os.path.dirname(p)
    api_key_file = os.path.join(d,'api_key.txt')
    if not os.path.exists(api_key_file):
        return
    with open(api_key_file, "r") as g:
        a_key = g.readline().strip()
    cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)

    data = get_data_from_Citrination(client=cl, dataset_id_list=[21,22,23,24,25,26,27])
    data_len = data.shape[0]
    train = data.iloc[:int(data_len*0.9),:]
#    train_part = data.iloc[int(data_len*0.9):,:]

    data_diffuse_only = data[(data['diffuse_structure_flag']=="1") & (data['crystalline_structure_flag']!= "1")]
    train_d = data_diffuse_only.iloc[:int(data_diffuse_only.shape[0] * 0.9), :]

    for model in model_output_names:
        cl = StructureClassifier(model)
        fl_name = 'test_classifiers_' + model + '.yml'
        test_classifiers_path = os.path.join(d,'xrsdkit','models','modeling_data',fl_name)

        if model in ['guinier_porod_population_count', 'spherical_normal_population_count']:
            scaler, model, par, accuracy = cl.train(train_d, hyper_parameters_search=False)
        else:
            scaler, model, par, accuracy = cl.train(train, hyper_parameters_search=False)
        cl.save_models(scaler, model, par, accuracy, test_classifiers_path)
#
#    scalers, models, accuracy = train_regressors(train, hyper_parameters_search=False, model='all')
#    save_models(scalers, models, accuracy, test_regressors_path)
#
#    scalers, models, accuracy = train_classifiers_partial(
#        train_part, test_classifiers_path, all_training_data=data, model='all')
#    save_models(scalers, models, accuracy, test_classifiers_path)
#
#    scalers, models, accuracy = train_regressors_partial(
#        train_part, test_regressors_path, all_training_data=data, model='all')
#    save_models(scalers, models, accuracy, test_regressors_path)
#
#
#
#
#
