import os
import numpy as np

from xrsdkit.tools import profiler
from xrsdkit.models.citrination_models import CitrinationSystemClassifier

def test_citrination_models():
    p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    api_key_file = os.path.join(p,'example','api_key.txt')
    if not os.path.exists(api_key_file):
        print('api_key.txt not found: skipping Citrination test')
        return
    test_path = os.path.join(p,'test_data','solution_saxs','spheres','spheres_1.csv')
    q_I = np.genfromtxt (test_path, delimiter=",")
    features = profiler.profile_spectrum(q_I)
    saxs_model = CitrinationSystemClassifier(api_key_file,'https://slac.citrination.com')
    print('testing Citrination models on {}'.format(test_path))
    population, uncertaintie = saxs_model.classify(features)
    print("scatterer populations: ")
    print(population, "  with uncertainties: %1.3f" % (uncertaintie))
'''
    params,uncertainties = saxs_models.predict_params(flags, features, q_I)
    print("scattering and intensity parameters: ")
    for popk in params.keys():
        print('\t{} populations: {} ({} certainty)'.format(popk,params[popk],uncertainties[popk]))
'''



