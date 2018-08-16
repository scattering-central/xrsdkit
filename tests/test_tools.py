import os

import numpy as np

from xrsdkit.tools import profiler

def test_profile_spectrum():
    datapath = os.path.join(os.path.dirname(__file__),
        'test_data','solution_saxs','precursors','precursors_0.csv')
    test_data = open(datapath,'r')
    q_I_gp = np.loadtxt(test_data,dtype=float,delimiter=',')
    gp_prof = profiler.profile_spectrum(q_I_gp)

    datapath = os.path.join(os.path.dirname(__file__),
        'test_data','solution_saxs','spheres','spheres_0.csv')
    test_data = open(datapath,'r')
    q_I_sph = np.loadtxt(test_data,dtype=float,delimiter=',')
    sph_prof = profiler.profile_spectrum(q_I_sph)

    datapath = os.path.join(os.path.dirname(__file__),
        'test_data','solution_saxs','peaks','peaks_0.csv')
    test_data = open(datapath,'r')
    q_I_pks = np.loadtxt(test_data,dtype=float,delimiter=',')
    prof_pks = profiler.profile_spectrum(q_I_pks)

    #pops = OrderedDict.fromkeys(saxs_fit.population_keys)
    #pops.update(gp_pops)
    #pops.update(sph_pops)
    #pops.update(pks_pops)

    #I_tot = q_I_gp[:,1] + q_I_sph[:,1] + q_I_pks[:,1]
    #q_I_tot = np.vstack([q_I_gp[:,1],I_tot]).T
    #pop_profs = saxs_math.detailed_profile(q_I_tot,pops)


