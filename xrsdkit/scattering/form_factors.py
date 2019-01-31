import os

import yaml
import numpy as np

from .. import definitions as xrsdefs

def atomic_ff(q,atom_symbol):
    pars = xrsdefs.atomic_params[atom_symbol]
    return compute_atomic_ff(q,pars['Z'],pars['a'],pars['b'])

def compute_atomic_ff(q,Z,a,b):
    g = q*1./(2*np.pi)
    s = g*1./2
    s2 = s**2
    ff = Z - 41.78214 * s2 * np.sum([aa*np.exp(-1*bb*s2) for aa,bb in zip(a,b)],axis=0)
    return ff

def spherical_ff(q,r):
    ff = np.zeros(q.shape)
    x = q*r
    if q[0] == 0:
        ff[0] = 1.
        ff[1:] = 3.*(np.sin(x[1:])-x[1:]*np.cos(x[1:]))*x[1:]**-3
        return ff
    else:
        return 3.*(np.sin(x)-x*np.cos(x))*x**-3

def atomic_ff_func(atom_symbol):
    return lambda q,atom_symbol=atom_symbol: atomic_ff(q,atom_symbol)

def spherical_ff_func(r):
    return lambda q,r=r: spherical_ff(q,r)


