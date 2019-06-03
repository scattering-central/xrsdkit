import os
import copy

import numpy as np
from matplotlib import pyplot as plt

from xrsdkit.system import System, fit 
from xrsdkit.visualization import plot_xrsd_fit, draw_xrsd_fit

src_wl = 0.8265616

nps = dict(
    structure='diffuse',
    form='spherical',
    parameters={'I0':{'value':1000},'r':{'value':40.}}
    )
np_sys = System(
    nanoparticles=copy.deepcopy(nps),
    noise={'model':'flat','parameters':{'I0':{'value':0.1}}},
    sample_metadata={'source_wavelength':src_wl}
    )
np_sys = System(
    nanoparticles=copy.deepcopy(nps),
    noise={'model':'flat','parameters':{'I0':{'value':0.1}}},
    sample_metadata={'source_wavelength':src_wl}
    )

datapath = os.path.join(os.path.dirname(__file__),
    'test_data','solution_saxs','spheres','spheres_0.dat')
f = open(datapath,'r')
q_I = np.loadtxt(open(datapath,'r'),dtype=float)

def test_plot():
    if 'DISPLAY' in os.environ:
        mpl_fig,I_comp = plot_xrsd_fit(np_sys,q_I[:,0],q_I[:,1]) 
        opt_np_sys = fit(np_sys,q_I[:,0],q_I[:,1])
        draw_xrsd_fit(mpl_fig.gca(),opt_np_sys,q_I[:,0],q_I[:,1])
        mpl_fig.show()

def test_multipanel_plot():
    if 'DISPLAY' in os.environ:
        fig, ax = plt.subplots(2,2)
        draw_xrsd_fit(ax[0,0],np_sys,q_I[:,0],q_I[:,1])
        draw_xrsd_fit(ax[0,1],np_sys,q_I[:,0],q_I[:,1])
        draw_xrsd_fit(ax[1,0],np_sys,q_I[:,0],q_I[:,1])
        draw_xrsd_fit(ax[1,1],np_sys,q_I[:,0],q_I[:,1])
        ax[0,0].set_xlabel('')
        ax[0,1].set_xlabel('')
        ax[0,1].set_ylabel('')
        ax[1,1].set_ylabel('')
        plt.show()

