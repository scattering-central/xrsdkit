import os

import numpy as np

from xrsdkit.system import System, fit 
from xrsdkit.visualization import plot_xrsd_fit, draw_xrsd_fit
    
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

def test_plot():
    # NOTE: if this runs on python 2.7 with no 'DISPLAY',
    # it crashes even if show=False,
    # when matplotlib tries to create its figure().
    if 'DISPLAY' in os.environ:
        mpl_fig = plot_xrsd_fit(np_sys,src_wl,q,I,True) 
        opt_np_sys = fit(np_sys,src_wl,q,I)
        draw_xrsd_fit(mpl_fig,opt_np_sys,src_wl,q,I,True)

