from __future__ import print_function
import os
import glob

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

def test_classifier():
    model_file_path = os.path.join(os.getcwd(),'saxskit','modeling_data','models_test.yml')
    sxc = saxs_classify.SaxsClassifier(model_file_path)
    for data_type in ['precursors','spheres']:
        data_path = os.path.join(os.getcwd(),'tests','test_data','solution_saxs',data_type)
        data_files = glob.glob(os.path.join(data_path,'*.csv'))
        for fpath in data_files:
            print('testing classifier on {}'.format(fpath))
            q_I = np.loadtxt(fpath,delimiter=',')
            prof = saxs_math.profile_spectrum(q_I)
            pops = sxc.run_classifier(prof)
            for pk,pop in pops.items():
                print('\t{} populations: {} ({} certainty)'.format(pk,pop[0],pop[1]))

# TODO: next, test_regressions()
# TODO: then, test_population_profiles()

