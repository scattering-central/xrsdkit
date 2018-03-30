from collections import OrderedDict
import os

import numpy as np

from xrsdkit import compute_intensity
from xrsdkit.tools import profiler 
from xrsdkit.fitting.xrsd_fitter import XRSDFitter 
    
src_wl = 0.8265616

def test_fit():
    datapath = os.path.join(os.path.dirname(__file__),
        'test_data','solution_saxs','spheres','spheres_0.csv')
    #datapath = os.path.join(os.path.dirname(__file__),
    #    'test_data','solution_saxs','peaks','peaks_0.csv')
    print('testing XRSDFitter on {}'.format(datapath))
    f = open(datapath,'r')
    q_I = np.loadtxt(f,dtype=float,delimiter=',')
    populations = OrderedDict() 
    populations['noise'] = dict(
        structure='diffuse',
        parameters={'I0':0.1},
        basis={'flat_noise':{'flat':{'amplitude':1}}}
        )
    populations['nanoparticles'] = dict(
        structure='diffuse',
        parameters={'I0':1000},
        basis={'spherical_nanoparticles':{'spherical_normal':{'r0':20,'sigma':0.05}}}
        )
    I_guess = compute_intensity(q_I[:,0],populations,src_wl)
    #from matplotlib import pyplot as plt
    #plt.figure(3)
    #plt.semilogy(q_I[:,0],q_I[:,1],'k')
    #plt.semilogy(q_I[:,0],I_guess,'g')
    #plt.show()

    ftr = XRSDFitter(q_I,populations,src_wl)
    fit_pops,rpt = ftr.fit()
    print('optimization objective: {} --> {}'.format(rpt['initial_objective'],rpt['final_objective']))
    init_flat_params = ftr.flatten_params(populations)
    fit_flat_params = ftr.flatten_params(fit_pops)
    for k, v in fit_flat_params.items():
        print('\t{}: {} --> {}'.format(k,init_flat_params[k],v))


