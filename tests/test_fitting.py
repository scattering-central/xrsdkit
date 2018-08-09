from collections import OrderedDict
import os

import numpy as np

from xrsdkit.system import System, fit 
    
src_wl = 0.8265616

noise_dict = dict(
    structure='diffuse',
    parameters={'I0':{'value':0.1}},
    basis={'flat_noise':{'form':'flat'}}
    )
np_dict =  dict(
    structure='diffuse',
    parameters={'I0':{'value':1000}},
    basis=dict(
        spherical_nanoparticles=dict(
            form='spherical',
            parameters={'r':{'value':40.}},
            )   
        )
    )
np_sl_dict = dict(
    structure='crystalline',
    parameters=dict(
        I0={'value':1.E-4},
        hwhm_g={'value':0.001},
        hwhm_l={'value':0.001},
        a={'value':40.*4./np.sqrt(2.)}
        ),
    settings={'lattice':'fcc','q_min':0.,'q_max':0.2,'profile':'voigt'},
    basis=dict(
        spherical_nanoparticles=dict(
            form='spherical',
            coordinates=[{'value':0.},{'value':0.},{'value':0.}],
            parameters={'r':{'value':40.}},
            )   
        )
    )

np_pops_dict = dict(
    noise = noise_dict,
    nanoparticles = np_dict
    )
np_sl_pops_dict = dict(
    noise = noise_dict,
    nanoparticles = np_dict,
    np_superlattice = np_sl_dict
    )
np_sys = System(np_pops_dict)
np_sl_sys = System(np_sl_pops_dict)


def test_fit_spheres():
    datapath = os.path.join(os.path.dirname(__file__),
        'test_data','solution_saxs','spheres','spheres_0.csv')
    f = open(datapath,'r')
    q_I = np.loadtxt(f,dtype=float,delimiter=',')
    q = q_I[:,0]
    I = q_I[:,1]
    fit_sys = fit(np_sys,q,I,src_wl)

    I_guess = np_sys.compute_intensity(q,src_wl)
    I_fit = fit_sys.compute_intensity(q,src_wl)

    #from matplotlib import pyplot as plt
    #plt.figure(2)
    #plt.semilogy(q,I,'k')
    #plt.semilogy(q,I_guess,'r')
    #plt.semilogy(q,I_fit,'g')
    #plt.show()


#def test_fit_superlattice():
#    datapath = os.path.join(os.path.dirname(__file__),
#        'test_data','solution_saxs','peaks','peaks_0.csv')
#    f = open(datapath,'r')
#    q_I = np.loadtxt(f,dtype=float,delimiter=',')
#    q = q_I[:,0]
#    I = q_I[:,1]
#    fit_sys = fit(np_sl_sys,q,I,src_wl)
#
#    I_guess = np_sl_sys.compute_intensity(q_I[:,0],src_wl)
#    I_fit = fit_sys.compute_intensity(q_I[:,0],src_wl)
#
#    from matplotlib import pyplot as plt
#    plt.figure(3)
#    plt.semilogy(q,I,'k')
#    plt.semilogy(q,I_guess,'r')
#    plt.semilogy(q,I_fit,'g')
#    plt.show()




