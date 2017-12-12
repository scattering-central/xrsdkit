from __future__ import print_function
import os
import glob
from collections import OrderedDict

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
    q_I_gp = np.loadtxt(test_data,dtype=float,delimiter=',')
    prof = saxs_math.profile_spectrum(q_I_gp)
    gp_prof = saxs_math.guinier_porod_profile(q_I_gp)
    gp_pops = OrderedDict.fromkeys(saxs_math.population_keys)
    gp_pops.update({'guinier_porod':1})
    #sxf_gp = saxs_fit.SaxsFitter(q_I_gp,gp_pops)
    #gp_params,rpt = sxf_gp.fit()

    datapath = os.path.join(os.path.dirname(__file__),
        'test_data','solution_saxs','spheres','spheres_0.csv')
    test_data = open(datapath,'r')
    q_I_sph = np.loadtxt(test_data,dtype=float,delimiter=',')
    prof = saxs_math.profile_spectrum(q_I_sph)
    sph_prof = saxs_math.spherical_normal_profile(q_I_sph)
    #assert isinstance(prof,dict)
    sph_pops = OrderedDict.fromkeys(saxs_math.population_keys)
    sph_pops.update({'spherical_normal':1})
    #sxf_sph = saxs_fit.SaxsFitter(q_I_sph,sph_pops)
    #sph_params,rpt = sxf_sph.fit()

    pops = OrderedDict.fromkeys(saxs_math.population_keys)
    pops.update(gp_pops)
    pops.update(sph_pops)

    params = OrderedDict.fromkeys(saxs_math.parameter_keys)
    params.update(dict(
        I0_floor=[13.854987111947105],
        G_gp=[0.15633292606854013],
        rg_gp=[2.1217098827006495],
        D_gp=[4.0],
        I0_sphere=[69.738299480524972],
        r0_sphere=[9.9309300490573076],
        sigma_sphere=[0.0]))

    #params.update(gp_params)
    #params.update(sph_params)
    #params.update({'I0_floor':[1E-4]})
    I_tot = q_I_gp[:,1] + q_I_sph[:,1]
    q_I_tot = np.vstack([q_I_gp[:,1],I_tot]).T
    #sxf = saxs_fit.SaxsFitter(q_I_tot,pops)
    #fit_params,rpt = sxf.fit(params)
    #print('MC anneal ...')
    #better_params,last_params,rpt = sxf.MC_anneal_fit(params,0.01,1000,0.2)
    #print('Final fit ...')
    #final_params,rpt = sxf.fit(better_params)
    #print(fit_params)
    pop_profs = saxs_math.population_profiles(q_I_tot,pops,params)

def test_classifier():
    model_file_path = os.path.join(os.getcwd(),'saxskit','modeling_data','scalers_and_models.yml')
    sxc = saxs_classify.SaxsClassifier(model_file_path)
    for data_type in ['precursors','spheres']:
        data_path = os.path.join(os.getcwd(),'tests','test_data','solution_saxs',data_type)
        data_files = glob.glob(os.path.join(data_path,'*.csv'))
        for fpath in data_files:
            print('testing classifier on {}'.format(fpath))
            q_I = np.loadtxt(fpath,delimiter=',')
            prof = saxs_math.profile_spectrum(q_I)
            pops = sxc.run_classifier(prof)
            if data_type == 'spheres':
                sph_prof = saxs_math.spherical_normal_profile(q_I)
            if data_type == 'precursors':
                gp_prof = saxs_math.guinier_porod_profile(q_I)
            for pk,pop in pops.items():
                print('\t{} populations: {} ({} certainty)'.format(pk,pop[0],pop[1]))

def test_regressions():
    model_file_path = os.path.join(os.getcwd(),'saxskit','modeling_data','scalers_and_models.yml')
    model_file_path_reg = os.path.join(os.getcwd(),'saxskit','modeling_data','scalers_and_models_regression.yml')
    sxc = saxs_classify.SaxsClassifier(model_file_path)
    sxr = saxs_regression.SaxsRegressor(model_file_path_reg)
    for data_type in ['precursors','spheres']:
        data_path = os.path.join(os.getcwd(),'tests','test_data','solution_saxs',data_type)
        data_files = glob.glob(os.path.join(data_path,'*.csv'))
        for fpath in data_files:
            print('testing regression on {}'.format(fpath))
            q_I = np.loadtxt(fpath,delimiter=',')
            prof = saxs_math.profile_spectrum(q_I)
            pops = sxc.run_classifier(prof)
            reg_prediction = sxr.predict_params(pops,prof,q_I)
            for k, v in reg_prediction.items():
                print('\t{} parameter: {} '.format(k,v))


