# output for predict.py:
# https://github.com/scattering-central/saxskit/blob/examples/examples/output.png

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from saxskit.saxs_math import profile_spectrum

# for using saxskit models:
from saxskit.saxs_classify import SaxsClassifier
from saxskit.saxs_regression import SaxsRegressor

# for using Citrination models:
from saxskit.saxs_citrination import CitrinationSaxsModels

# for computing parameters related with intensity:
from saxskit import saxs_fit

path = os.getcwd()
q_i = np.genfromtxt (path + '/saxskit/examples/sample_0.csv', delimiter=",")

#Profile a saxs spectrum:
features = profile_spectrum(q_i)

#Using SAXSKIT models:
print("\033[1m" + "Prediction from SAXSKIT models: " + "\033[0;0m", "\n")

m = SaxsClassifier()
populations, propability = m.classify(features)
print("scatterer populations: ")
for k,v in populations.items():
    print(k, ":", v, "  with propability: %1.3f" % (propability[k]))
print()

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

api_key_file = path + '/citrination_api_key_ssrl.txt'
saxs_models = CitrinationSaxsModels(api_key_file,'https://slac.citrination.com')

populations, uncertainties = saxs_models.classify(features)
print("scatterer populations: ")
for k,v in populations.items():
    print(k, ":", v, "  with uncertainties: %1.3f" % (uncertainties[k]))
print()

params,uncertainties = saxs_models.predict_params(populations, features, q_i)

sxf = saxs_fit.SaxsFitter(q_i,populations)
params, report = sxf.fit_intensity_params(params)
print("scattering and intensity parameters: ")
for k,v in params.items():
    print(k, ":", end="")
    for n in range(len(v)):
        print(" %10.3f" % (v[n]) )
print()

print("uncertainties: ")
for k,v in uncertainties.items():
    print(k, ": %1.3f" % (v))