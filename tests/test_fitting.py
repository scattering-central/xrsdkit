import os

import numpy as np

from xrsdkit.system import System, Population, fit 
from xrsdkit.system.noise import NoiseModel
from xrsdkit.visualization.gui import run_fit_gui

src_wl = 0.8265616

nps = Population(
    structure='diffuse',
    form='spherical',
    parameters={'I0':{'value':1000},'r':{'value':40.}}
    )
np_sys = System(
    nanoparticles=nps,
    noise=NoiseModel(
        model='flat',
        parameters={'I0':{'value':0.1}}
        )
    )
np_sys = System(
    nanoparticles=np_dict,
    noise={'model':'flat','parameters':{'I0':{'value':0.1}}}
    )

datapath = os.path.join(os.path.dirname(__file__),
    'test_data','solution_saxs','spheres','spheres_0.csv')
q_I = np.loadtxt(open(datapath,'r'),dtype=float,delimiter=',')

def test_fit():
    fit_sys = fit(np_sys,q_I[:,0],q_I[:,1],src_wl)
    #I_guess = np_sys.compute_intensity(q,src_wl)
    #I_fit = fit_sys.compute_intensity(q,src_wl)

