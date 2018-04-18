import os

import numpy as np
from citrination_client import CitrinationClient

from xrsdkit.tools import profiler
#from xrsdkit.models.saxs_citrination import CitrinationSaxsModels 
#from xrsdkit.models.saxs_citrination import get_data_from_Citrination
#from saxskit.saxs_models import train_classifiers, train_regressors
#from saxskit.saxs_models import train_classifiers_partial, train_regressors_partial
#from saxskit.saxs_models import save_models

def test_citrination_models():
    p = os.path.dirname(os.path.abspath(__file__))
    api_key_file = os.path.join(p,'api_key.txt')
    if not os.path.exists(api_key_file):
        print('api_key.txt not found: skipping Citrination test')
        return
    test_path = os.path.join(head,'test_data','solution_saxs','spheres','spheres_1.csv')
    q_I = np.genfromtxt (test_path, delimiter=",")
    features = profiler.profile_spectrum(q_I)
    saxs_models = CitrinationSaxsModels(api_key_file,'https://slac.citrination.com')
    print('testing Citrination models on {}'.format(test_path))
    flags, uncertainties = saxs_models.classify(features)
    print("scatterer populations: ")
    for popk in flags.keys():
        print('\t{} populations: {} ({} certainty)'.format(popk,flags[popk],uncertainties[popk]))
    params,uncertainties = saxs_models.predict_params(flags, features, q_I)
    print("scattering and intensity parameters: ")
    for popk in params.keys():
        print('\t{} populations: {} ({} certainty)'.format(popk,params[popk],uncertainties[popk]))


#def test_citrination_classifier(address,api_key_file):
#    model_file_path = os.path.join(os.getcwd(),'saxskit','modeling_data','scalers_and_models.yml')
#    sxc = saxs_classify.SaxsClassifier(model_file_path)
#    sxc.citrination_setup(address,api_key_file)
#    for data_type in ['precursors','spheres','peaks']:
#        data_path = os.path.join(os.getcwd(),'tests','test_data','solution_saxs',data_type)
#        data_files = glob.glob(os.path.join(data_path,'*.csv'))
#        for fpath in data_files:
#            print('testing classifier on {}'.format(fpath))
#            q_I = np.loadtxt(fpath,delimiter=',')
#            prof = saxs_math.profile_spectrum(q_I)
#            pops = sxc.citrination_predict(prof)
#            for popk,pop in pops.items():
#                print('\t{} populations: {} ({} certainty)'.format(popk,pop[0],pop[1]))
#

def test_getdata():
    p = os.path.dirname(os.path.abspath(__file__))
    api_key_file = os.path.join(p,'api_key.txt')
    if not os.path.exists(api_key_file):
        return
    with open(api_key_file, "r") as g:
        a_key = g.readline().strip()
    cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)
    data = get_data_from_Citrination(client=cl, dataset_id_list=[16])
    #data_len = data.shape[0]

    #train = data.iloc[:int(data_len*0.9),:]
    #train_part = data.iloc[int(data_len*0.9):,:]

    #p = os.path.abspath(__file__)
    #d = os.path.dirname(os.path.dirname(p))
    #test_classifiers_path = os.path.join(d,'saxskit','modeling_data','test_classifiers.yml')
    #test_regressors_path = os.path.join(d,'saxskit','modeling_data','test_regressors.yml')
    
    #scalers, models, accuracy = train_classifiers(train, hyper_parameters_search=False, model='all')
    #save_models(scalers, models, accuracy, test_classifiers_path)

    #scalers, models, accuracy = train_regressors(train, hyper_parameters_search=False, model='all')
    #save_models(scalers, models, accuracy, test_regressors_path)

    #scalers, models, accuracy = train_classifiers_partial(
    #    train_part, test_classifiers_path, all_training_data=data, model='all')
    #save_models(scalers, models, accuracy, test_classifiers_path)

    #scalers, models, accuracy = train_regressors_partial(
    #    train_part, test_regressors_path, all_training_data=data, model='all')
    #save_models(scalers, models, accuracy, test_regressors_path)


