from __future__ import print_function

import numpy as np

from xrsdkit import scattering as xrs
from xrsdkit.tools import peak_math
from xrsdkit.system import System, Population
from xrsdkit.tools import ymltools as xrsdyml

fcc_Al = Population(
    structure='crystalline',
    form='atomic',
    settings={'lattice':'F_cubic','space_group':'Fm-3m',
        'q_max':5.,'structure_factor_mode':'local','symbol':'Al'},
    parameters=dict(
        a={'value':4.046},
        hwhm_g={'value':0.002},
        hwhm_l={'value':0.0018}
        )
    )
hcp_spheres = Population(
    structure='crystalline',
    form='spherical',
    settings={'lattice':'hcp',
        'space_group':'P6(3)/mmc','q_max':0.6,
        'structure_factor_mode':'radial'},
    parameters=dict(
        a={'value':120.},
        hwhm_g={'value':0.002},
        hwhm_l={'value':0.002},
        r={'value':40.}
        )
    )

fcc_Al_system = System(
    fcc_Al=fcc_Al.to_dict(),
    sample_metadata={'source_wavelength':0.8265617}
    )
hcp_sphere_system = System(
    hcp_spheres=hcp_spheres.to_dict(),
    sample_metadata={'source_wavelength':0.8265617}
    )

glassy_Al = Population(
    structure='disordered',
    form='atomic',
    settings={'interaction':'hard_spheres','symbol':'Al'},
    parameters=dict(
        r_hard={'value':4.046*np.sqrt(2)/4},
        v_fraction={'value':0.6},
        I0={'value':1.E5}
        )
    )

glassy_Al_system = System(glassy_Al=glassy_Al.to_dict())

mixed_Al_system = System(
    glassy_Al=glassy_Al.to_dict(),
    fcc_Al=fcc_Al.to_dict(),
    sample_metadata={'source_wavelength':0.8265617}
    )

def test_Al_scattering():
    qvals = np.arange(1.,5.,0.001)
    I_fcc = fcc_Al_system.compute_intensity(qvals)
    I_gls = glassy_Al_system.compute_intensity(qvals)
    I_mxd = mixed_Al_system.compute_intensity(qvals)

def test_sphere_scattering():
    qvals = np.arange(0.02,0.6,0.001)
    I_sl = hcp_sphere_system.compute_intensity(qvals) 
    
def test_gaussian():
    qvals = np.arange(0.01,4.,0.01)
    for hwhm in [0.01,0.03,0.05,0.1]:
        g = peak_math.gaussian(qvals-2.,hwhm)

def test_lorentzian():
    qvals = np.arange(0.01,4.,0.01)
    for hwhm in [0.01,0.03,0.05,0.1]:
        l = peak_math.lorentzian(qvals-2.,hwhm)

def test_voigt():
    qvals = np.arange(0.01,4.,0.01)
    for hwhm_g in [0.01,0.03,0.05,0.1]:
        for hwhm_l in [0.01,0.03,0.05,0.1]:
            v = peak_math.voigt(qvals-2.,hwhm_g,hwhm_l)

