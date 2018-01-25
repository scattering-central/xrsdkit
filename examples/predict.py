import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from saxskit.saxs_math import profile_spectrum

# for using saxskit models:
from saxskit.saxs_classify import SaxsClassifier
from saxskit.saxs_regression import SaxsRegressor

path = os.getcwd()
q_i = np.genfromtxt (path + '/saxskit/examples/sample_0.csv', delimiter=",")

#Profile a saxs spectrum:
features = profile_spectrum(q_i)

#Using SAXSKIT models:
m = SaxsClassifier()
flags, propability = m.classify(features)
print("scatterer populations: ", flags, '\n')

print("propability: ", propability, '\n')

r = SaxsRegressor()
params = r.predict_params(flags,features, q_i)
print("scattering parameters: ", params, '\n')

#Using Citrination models:
from saxskit.saxs_citrination import CitrinationSaxsModels

api_key_file = path + '/citrination_api_key_ssrl.txt'
saxs_models = CitrinationSaxsModels(api_key_file,'https://slac.citrination.com')

flags, uncertainties = saxs_models.classify(features)
print("scatterer populations: ", flags, '\n')

print("uncertainties: ", uncertainties, '\n')

params,uncertainties = saxs_models.predict_params(flags, features, q_i)
print("scattering parameters: ", params, '\n')

print("uncertainties: ", uncertainties)