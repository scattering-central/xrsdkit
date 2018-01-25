# output for predict.py is in output_predict.png

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

path = os.getcwd()
q_i = np.genfromtxt (path + '/saxskit/examples/sample_0.csv', delimiter=",")

#Profile a saxs spectrum:
features = profile_spectrum(q_i)

#Using SAXSKIT models:
print("\033[1m" + "Prediction form SAXSKIT models: " + "\033[0;0m")

m = SaxsClassifier()
flags, propability = m.classify(features)
print("scatterer populations: ", flags, '\n')

print("propability: ", propability, '\n')

r = SaxsRegressor()
params = r.predict_params(flags,features, q_i)
print("scattering parameters: ", params, '\n')

#Using Citrination models:
print("\033[1m" + "Prediction form Citrination models: " + "\033[0;0m")

api_key_file = path + '/citrination_api_key_ssrl.txt'
saxs_models = CitrinationSaxsModels(api_key_file,'https://slac.citrination.com')

flags, uncertainties = saxs_models.classify(features)
print("scatterer populations: ", flags, '\n')

print("uncertainties: ", uncertainties, '\n')

params,uncertainties = saxs_models.predict_params(flags, features, q_i)
print("scattering parameters: ", params, '\n')

print("uncertainties: ", uncertainties)