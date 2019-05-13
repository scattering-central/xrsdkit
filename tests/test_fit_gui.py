import copy
import os

import numpy as np

from xrsdkit.system import System, Population
from xrsdkit.system.noise import NoiseModel
from xrsdkit.visualization.gui import run_gui
from xrsdkit.tools import ymltools as xrsdyml

src_wl = 0.8265616

flat_noise=NoiseModel('flat',parameters={'I0':{'value':0.1}})
nps = Population('diffuse','spherical',
    settings={'distribution':'r_normal'},
    parameters={'I0':{'value':1000},'r':{'value':35.},'sigma':{'value':0.1}}
    )
np_sl = Population('crystalline','spherical',
    settings={'q_min':0.,'q_max':0.2,'lattice':'F_cubic','space_group':'Fm-3m','distribution':'single'},
    parameters={'I0':{'value':1.},'a':{'value':130.},'r':{'value':35.}}
    )
np_sys = System(
    nanoparticles=copy.deepcopy(nps),
    noise=flat_noise,
    sample_metadata={'source_wavelength':src_wl}
    )
np_sl_sys = System(
    superlattice=copy.deepcopy(np_sl),
    nanoparticles=copy.deepcopy(nps),
    noise=flat_noise,
    sample_metadata={'source_wavelength':src_wl}
    )

datapath = os.path.join(os.path.dirname(__file__),
    'test_data','solution_saxs','peaks','peaks_0.dat')
sysfpath = os.path.splitext(datapath)[0]+'.yml'
xrsdyml.save_sys_to_yaml(sysfpath,np_sl_sys)

def test_fit_gui():
    if 'DISPLAY' in os.environ:
        fit_sys = run_gui([datapath],[sysfpath])
    os.remove(sysfpath)

