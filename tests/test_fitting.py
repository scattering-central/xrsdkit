import os

import numpy as np

from xrsdkit.system import System, fit 
from xrsdkit.visualization.gui import run_fit_gui

src_wl = 0.8265616

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
np_sys = System({'nanoparticles':np_dict,'noise':{'model':'flat','parameters':{'I0':{'value':0.1}}}})

datapath = os.path.join(os.path.dirname(__file__),
    'test_data','solution_saxs','spheres','spheres_0.csv')
f = open(datapath,'r')
q_I = np.loadtxt(f,dtype=float,delimiter=',')
q = q_I[:,0]
I = q_I[:,1]

def test_fit():
    fit_sys = fit(np_sys,q,I,src_wl)
    #I_guess = np_sys.compute_intensity(q,src_wl)
    #I_fit = fit_sys.compute_intensity(q,src_wl)

def test_fit_gui():
    if 'DISPLAY' in os.environ:
        fit_sys, good_fit_flag = run_fit_gui(np_sys,q_I,src_wl)


