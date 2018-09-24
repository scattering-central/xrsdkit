import os

import numpy as np

from xrsdkit.system import System, fit 
from xrsdkit.visualization.gui import run_fit_gui

src_wl = 0.8265616

spheres=dict(
    form='spherical_normal',
    parameters={'r0':{'value':40.},'sigma':{'value':0.1}},
    )   
flat_noise=dict(model='flat',parameters={'I0':{'value':0.1}})
nps =  dict(
    structure='diffuse',
    parameters={'I0':{'value':1000}},
    basis={'spheres':spheres}
    )
np_sys = System({'nanoparticles':nps,'noise':flat_noise})

datapath = os.path.join(os.path.dirname(__file__),
    'test_data','solution_saxs','spheres','spheres_0.csv')
f = open(datapath,'r')
q_I = np.loadtxt(f,dtype=float,delimiter=',')
q = q_I[:,0]
I = q_I[:,1]

def test_fit_gui():
    if 'DISPLAY' in os.environ:
        fit_sys, good_fit_flag = run_fit_gui(np_sys,src_wl,q,I)



