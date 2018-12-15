import os

import numpy as np

from xrsdkit.system import System, fit 
from xrsdkit.visualization import plot_xrsd_fit, draw_xrsd_fit

src_wl = 0.8265616

nps = dict(
    structure='diffuse',
    form='spherical',
    parameters={'I0':{'value':1000},'r':{'value':40.}}
    )
np_sys = System(
    nanoparticles=nps,
    noise={'model':'flat','parameters':{'I0':{'value':0.1}}}
    )
np_sys = System(
    nanoparticles=np_dict,
    noise={'model':'flat','parameters':{'I0':{'value':0.1}}}
    )

datapath = os.path.join(os.path.dirname(__file__),
    'test_data','solution_saxs','spheres','spheres_0.csv')
f = open(datapath,'r')
q_I = np.loadtxt(open(datapath,'r'),dtype=float,delimiter=',')

def test_plot():
    if 'DISPLAY' in os.environ:
        mpl_fig,I_comp = plot_xrsd_fit(np_sys,q_I[:,0],q_I[:,1],src_wl,True) 
        opt_np_sys = fit(np_sys,q_I[:,0],q_I[:,1],src_wl)
        draw_xrsd_fit(mpl_fig,opt_np_sys,q_I[:,0],q_I[:,1],src_wl,True)

