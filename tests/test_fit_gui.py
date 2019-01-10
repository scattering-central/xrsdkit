import os

import numpy as np

from xrsdkit.system import System, Population
from xrsdkit.system.noise import NoiseModel
from xrsdkit.visualization.gui import run_fit_gui

src_wl = 0.8265616

flat_noise=NoiseModel(model='flat',parameters={'I0':{'value':0.1}})
nps = Population(
    structure='diffuse',
    form='spherical_normal',
    parameters={'I0':{'value':1000},'r':{'value':35.},'sigma':{'value':0.1}},
    basis={'spheres':spheres}
    )
np_sl = Population(
    structure='crystalline',
    form='spherical',
    settings={'q_min':0.,'q_max':0.2,'lattice':'cubic','centering':'F','space_group':'Fm-3m'},
    parameters={'I0':{'value':1.E-5},'a':{'value':130.},'r':{'value':35.}},
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
    'test_data','solution_saxs','peaks','peaks_0.csv')
q_I_sl = np.loadtxt(open(datapath,'r'),dtype=float,delimiter=',')

def test_fit_gui():
    if 'DISPLAY' in os.environ:
        fit_sys = run_fit_gui(np_sl_sys,q_I_sl[:,0],q_I_sl[:,1])

