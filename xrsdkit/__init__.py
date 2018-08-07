"""This package provides tools for analysis of scattering and diffraction patterns."""

from collections import OrderedDict
import copy

import numpy as np

# TODO: update convenience constructors to return System objects. 

def fcc_crystal(atom_symbol,a_lat=10.,pk_profile='voigt',I0=1.E-3,q_min=0.,q_max=1.,hwhm_g=0.001,hwhm_l=0.001):
    return dict(
        structure='fcc',
        settings={'q_min':q_min,'q_max':q_max,'profile':pk_profile},
        parameters={'I0':I0,'a':a_lat,'hwhm_g':hwhm_g,'hwhm_l':hwhm_l},
        basis={atom_symbol+'_atom':dict(
            coordinates=[0,0,0],
            form='atomic',
            settings={'symbol':atom_symbol}
            )}
        )

def unidentified_population():
    return dict(
        structure='unidentified',
        settings={}, 
        parameters={},
        basis={}
        )

def empty_site():
    return dict(
        form='diffuse',
        settings={},
        parameters={}
        )
        
def flat_noise(I0=1.E-3):
    return dict(
        structure='diffuse',
        settings={},
        parameters={'I0':I0},
        basis={'flat_noise':{'form':'flat'}}
        )

