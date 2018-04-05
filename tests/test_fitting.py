from collections import OrderedDict
import os

import numpy as np

from xrsdkit import compute_intensity
from xrsdkit.tools import profiler 
from xrsdkit.fitting.xrsd_fitter import XRSDFitter 
    
src_wl = 0.8265616

def test_fit_spheres():
    datapath = os.path.join(os.path.dirname(__file__),
        'test_data','solution_saxs','spheres','spheres_0.csv')
    print('testing XRSDFitter on {}'.format(datapath))
    f = open(datapath,'r')
    q_I = np.loadtxt(f,dtype=float,delimiter=',')
    populations = OrderedDict() 
    populations['noise'] = dict(
        structure='diffuse',
        parameters={'I0':0.1},
        basis={'flat_noise':{'flat':{}}}
        )
    populations['nanoparticles'] = dict(
        structure='diffuse',
        parameters={'I0':1000},
        basis={'spherical_nanoparticles':{'spherical_normal':{'r0':20,'sigma':0.05}}}
        )
    ftr = XRSDFitter(q_I,populations,src_wl)
    fit_pops,rpt = ftr.fit()

    print('optimization objective: {} --> {}'.format(rpt['initial_objective'],rpt['final_objective']))
    init_flat_params = ftr.flatten_params(populations)
    fit_flat_params = ftr.flatten_params(fit_pops)
    for k, v in init_flat_params.items():
        print('\t{}: {} --> {}'.format(k,v,fit_flat_params[k]))

    I_guess = compute_intensity(q_I[:,0],populations,src_wl)
    I_fit = compute_intensity(q_I[:,0],fit_pops,src_wl)
    #from matplotlib import pyplot as plt
    #plt.figure(2)
    #plt.semilogy(q_I[:,0],q_I[:,1],'k')
    #plt.semilogy(q_I[:,0],I_guess,'r')
    #plt.semilogy(q_I[:,0],I_fit,'g')
    #plt.show()



def test_fit_sphere_diffraction():
    datapath = os.path.join(os.path.dirname(__file__),
        'test_data','solution_saxs','peaks','peaks_0.csv')
    f = open(datapath,'r')
    q_I = np.loadtxt(f,dtype=float,delimiter=',')
    populations = OrderedDict() 
    populations['noise'] = dict(
        structure='diffuse',
        parameters={'I0':0.01},
        basis={'flat_noise':{'flat':{}}}
        )
    populations['nanoparticles'] = dict(
        structure='diffuse',
        parameters={'I0':100},
        basis={'spherical_nanoparticles':{'spherical_normal':{'r0':40,'sigma':0.03}}}
        )
    populations['superlattice'] = dict(
        structure='fcc',
        settings={'profile':'voigt','q_min':0.001,'q_max':0.4},
        parameters=dict(
            a=40.*4./np.sqrt(2),
            hwhm_g=0.001,
            hwhm_l=0.001,
            I0=1.E-3),
        basis={'spherical_nanoparticles':dict(
            spherical={'r':40},
            coordinates=[0.,0.,0.]
            )}
        )

    print('optimization objective: {} --> {}'.format(rpt['initial_objective'],rpt['final_objective']))
    init_flat_params = ftr.flatten_params(populations)
    fit_flat_params = ftr.flatten_params(fit_pops)
    for k, v in init_flat_params.items():
        print('\t{}: {} --> {}'.format(k,v,fit_flat_params[k]))
    
    I_guess = compute_intensity(q_I[:,0],populations,src_wl)
    I_fit = compute_intensity(q_I[:,0],fit_pops,src_wl)
    #from matplotlib import pyplot as plt
    #plt.figure(3)
    #plt.semilogy(q_I[:,0],q_I[:,1],'k')
    #plt.semilogy(q_I[:,0],I_guess,'r')
    #plt.semilogy(q_I[:,0],I_fit,'g')
    #plt.show()


