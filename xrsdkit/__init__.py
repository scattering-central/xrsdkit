"""xrsdkit: a package for handling scattering and diffraction data"""
import os
from sys import platform as sys_pf

import numpy as np
import pprint
import matplotlib
if 'DISPLAY' in os.environ or sys_pf == 'darwin':
    matplotlib.use("TkAgg")

from xrsdkit.tools.profiler import profile_spectrum
from xrsdkit.models import predict as xrsdpred
from xrsdkit import system as xrsdsys 
from xrsdkit.visualization import gui as xrsdgui

def read_q_I(path):
    """Read q, I array from a text file.

    Parameters
    ----------
    path : str
        path to text file that contains q, I data.

    Returns
    -------
    q : array
        array of scattering vectors 
    I : array
        array of intensities corresponding to `q` 
    """
    q_I = np.loadtxt(path)
    return q_I[:,0],q_I[:,1]


def profile(q,I):
    """Numerically profile a scattering/diffraction pattern.

    Parameters
    ----------
    q : array
        array of scattering vectors 
    I : array
        array of intensities corresponding to `q`

    Returns
    -------
    profile : dict
        Dictionary of featuress computed from `q` and `I` 
    """
    return xrsdprof.profile_pattern(q,I) 


def predict_system(q,I,source_wavelength): 
    """Evaluate statistical models to classify and parameterize the pattern. 

    Parameters
    ----------
    q : array
        array of scattering vectors 
    I : array
        array of intensities corresponding to `q`
    source_wavelength : float
        wavelength of light source used for measuring intensity pattern

    Returns
    -------
    xrsdkit.system.System
        System object with predicted populations and parameters
    """
    return xrsdpred.system_from_prediction(
        xrsdpred.predict(profile(q,I)),
        source_wavelength=source_wavelength
        )


def print_system(sys):
    """Pretty-print xrsdkit.system.System object content.

    Parameters
    ----------
    sys : xrsdkit.system.System
        System object to be printed
    """
    pprint.pprint(sys.to_dict())


def fit(sys,q,I):
    """Fit the I(q) pattern and return a dict of optimized parameters.

    Parameters
    ----------
    sys : xrsdkit.system.System
        System object to fit parameters to `q` and `I`
    q : array
        array of scattering vectors 
    I : array
        array of intensities corresponding to `q`

    Returns
    -------
    xrsdkit.system.System
        Similar to input `sys`, but with fit-optimized parameters.
    """
    return xrsdsys.fit(sys,q,I)


def start_gui(sys,q,I):
    """Fit the I(q) pattern and return a dict of optimized parameters.

    Parameters
    ----------
    sys : xrsdkit.system.System
        System object to fit parameters to `q` and `I`
    q : array
        array of scattering vectors 
    I : array
        array of intensities corresponding to `q`
    """
    fit_sys = xrsdgui.run_fit_gui(sys,q,I)
    return fit_sys
