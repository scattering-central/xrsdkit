from __future__ import print_function
import os
import glob
from collections import OrderedDict

import numpy as np

from xrsdkit.scattering import form_factors as xrff


#from saxskit.saxs_math import profile_spectrum
#from saxskit.saxs_citrination import CitrinationSaxsModels

def test_guinier_porod():
    qvals = np.arange(0.01,1.,0.01)
    ff = xrff.guinier_porod(qvals,20,4,120)

def test_spherical_normal():
    qvals = np.arange(0.01,1.,0.01)
    ff = xrff.spherical_normal_ff(qvals,20,0.2)
    ff_mono = xrff.spherical_normal_ff(qvals,20,0.)

