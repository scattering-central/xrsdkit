# output for predict.py:
# https://github.com/scattering-central/saxskit/blob/examples/examples/output.png

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from xrsdkit.tools.profiler import profile_spectrum

# for using saxskit models:
from xrsdkit.models.structure_classifier import StructureClassifier
from xrsdkit.models.saxs_regression import SaxsRegressor

# for using Citrination models:
from xrsdkit.models.citrination_models import CitrinationStructureClassifier

# for computing parameters related with intensity:
#from xrsdkit import saxs_fit

p = os.path.abspath(__file__)
d = os.path.dirname(os.path.dirname(os.path.dirname(p)))
#path = os.path.join(d,'tests','test_data','solution_saxs','peaks','peaks_0.csv')
path = os.path.join(d,'tests','test_data','solution_saxs','spheres','spheres_1.csv')

q_i = np.genfromtxt (path, delimiter=",")

#Profile a saxs spectrum:
features = profile_spectrum(q_i)

#Using saxskit models:
print("\033[1m" + "Prediction from saxskit models: " + "\033[0;0m", "\n")
print("scatterer populations: ")

crystalline_model = StructureClassifier('crystalline_structure_flag')
crystalline_pop, probability = crystalline_model.classify(features)
print('crystalline_structure_flag: ', crystalline_pop, "  with probability: %1.3f" % (probability))

diffuse_model = StructureClassifier('diffuse_structure_flag')
diffuse_pop, probability = diffuse_model.classify(features)
print('diffuse_structure_flag: ', diffuse_pop, "  with probability: %1.3f" % (probability))

print()
'''
r = SaxsRegressor()
params = r.predict_params(populations,features, q_i)

sxf = saxs_fit.SaxsFitter(q_i,populations)
params, report = sxf.fit_intensity_params(params)
print("scattering and intensity parameters: ")
for k,v in params.items():
    print(k, ":", end="")
    for n in v:
        print(" %10.3f" % (n))
print()

#Using Citrination models:
print("\033[1m" + "Prediction from Citrination models: " + "\033[0;0m", "\n")

api_key_file = os.path.join(d, 'api_key.txt')
if not os.path.exists(api_key_file):
    print("Citrination api key file did not find")

saxs_models = CitrinationStructureClassifier(api_key_file,'https://slac.citrination.com')

populations, uncertainties = saxs_models.classify(features)
print("scatterer populations: ")
for k,v in populations.items():
    print(k, ":", v, "  with uncertainties: %1.3f" % (uncertainties[k]))
print()

params,uncertainties = saxs_models.predict_params(populations, features, q_i)
print("scatterer parameters: ")
for k,v in params.items():
    print(k, ":", v, " +/- %1.3f" % (uncertainties[k]))
print()

sxf = saxs_fit.SaxsFitter(q_i,populations)
params, report = sxf.fit_intensity_params(params)
print("scattering and intensity parameters: ")
for k,v in params.items():
    print(k, ":", end="")
    for n in range(len(v)):
        print(" %10.3f" % (v[n]) )
print()
'''