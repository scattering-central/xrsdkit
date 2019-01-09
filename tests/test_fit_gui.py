import os

import numpy as np

from xrsdkit.system import System, fit 
from xrsdkit.visualization.gui import run_fit_gui

src_wl = 0.8265616

spheres = dict(
    form='spherical_normal',
    parameters={'r0':{'value':35.},'sigma':{'value':0.1}},
    )
mono_spheres = dict(
    form='spherical',
    parameters={'r':{'value':35.}},
    )
flat_noise=dict(model='flat',parameters={'I0':{'value':0.1}})
nps = dict(
    structure='diffuse',
    parameters={'I0':{'value':1000}},
    basis={'spheres':spheres}
    )
np_sl = dict(
    structure='crystalline',
    settings={'q_min':0.,'q_max':0.2,'lattice':'cubic','centering':'F','space_group':'Fm-3m'},
    parameters={'I0':{'value':1.E-5},'a':{'value':130.}},
    basis={'spheres':mono_spheres}
    )
np_sys = System(
    nanoparticles=nps,
    noise=flat_noise
    )
np_sl_sys = System(
    superlattice=np_sl,
    nanoparticles=nps,
    noise=flat_noise
    )

datapath = os.path.join(os.path.dirname(__file__),
    'test_data','solution_saxs','spheres','spheres_0.csv')
f = open(datapath,'r')
q_I = np.loadtxt(f,dtype=float,delimiter=',')
q = q_I[:,0]
I = q_I[:,1]

datapath = os.path.join(os.path.dirname(__file__),
    'test_data','solution_saxs','peaks','peaks_0.csv')
f = open(datapath,'r')
q_I_sl = np.loadtxt(f,dtype=float,delimiter=',')
q_sl = q_I_sl[:,0]
I_sl = q_I_sl[:,1]

def test_fit_gui():
    if 'DISPLAY' in os.environ:
        #fit_sys = run_fit_gui(np_sys,q,I,src_wl)
        fit_sys = run_fit_gui(np_sl_sys,q_sl,I_sl,src_wl)



