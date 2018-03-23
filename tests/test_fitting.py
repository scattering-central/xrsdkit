from collections import OrderedDict
import os

import numpy as np

#from xrsdkit.models.saxs_classify import SaxsClassifier
#from xrsdkit.models.saxs_regression import SaxsRegressor
from xrsdkit import compute_intensity
from xrsdkit.tools import profiler 
from xrsdkit.fitting.xrsd_fitter import XRSDFitter 

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
        basis={(0,0,0):{'flat':{'amplitude':1}}}
        )
    populations['nanoparticles'] = dict(
        structure='diffuse',
        parameters={'I0':1000},
        basis={(0,0,0):{'spherical_normal':{'r0':20,'sigma':0.05}}}
        )
    I_guess = compute_intensity(q_I[:,0],populations,0.8265616)
    #from matplotlib import pyplot as plt
    #plt.figure(3)
    #plt.semilogy(q_I[:,0],q_I[:,1],'k')
    #plt.semilogy(q_I[:,0],I_guess,'g')
    #plt.show()

    ftr = XRSDFitter(q_I,populations)
    #fit_pops = ftr.fit()
    #params,rpt = sxf.fit_intensity_params(params)
    #p_opt,rpt_opt = sxf.fit(params)
    #obj_init = sxf.evaluate(params)
    #obj_opt = sxf.evaluate(p_opt)
    #print('optimization objective: {} --> {}'.format(obj_init,obj_opt))
    #for k, v in params.items():
    #    print('\t{}: {} --> {}'.format(k,v,p_opt[k]))


