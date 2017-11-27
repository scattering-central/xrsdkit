from __future__ import print_function
import os

import numpy as np

from saxskit import saxs_math, saxs_fit, saxs_classify, saxs_regression

def test_guinier_porod():
    qvals = np.arange(0.01,1.,0.01)
    Ivals = saxs_math.guinier_porod(qvals,20,4,120)
    assert isinstance(Ivals,np.ndarray)

def test_spherical_normal_saxs():
    qvals = np.arange(0.01,1.,0.01)
    Ivals = saxs_math.spherical_normal_saxs(qvals,20,0.2)
    assert isinstance(Ivals,np.ndarray)
    Ivals = saxs_math.spherical_normal_saxs(qvals,20,0.)
    assert isinstance(Ivals,np.ndarray)

def test_profile_spectrum():
    datapath = os.path.join(os.path.dirname(__file__),
        'test_data','solution_saxs','precursors','precursors_0.csv')
    test_data = open(datapath,'r')
    q_I = np.loadtxt(test_data,dtype=float,delimiter=',')
    assert isinstance(q_I,np.ndarray) 
    prof = saxs_math.profile_spectrum(q_I)
    assert isinstance(prof,dict)

#if __name__ == '__main__':
#    test_guinier_porod()
#    test_spherical_normal_saxs()

