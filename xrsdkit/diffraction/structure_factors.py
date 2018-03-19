from functools import partial

import numpy as np
import quadpy

from xrsdkit import scattering
from xrsdkit.scattering import form_factors as xrff

fcc_coords = np.array([
    (0.,0.,0.),
    (0.5,0.5,0.),
    (0.5,0.,0.5),
    (0.,0.5,0.5)])

def fcc_sf(q_hkl,hkl,popd):
    F_hkl = np.zeros(hkl.shape[1],dtype=complex) 
    basis = popd['basis']
    for fcc_coord in fcc_coords:
        for coord, species in basis.items():
            for specie_name, specie_params in species.items(): 
                g_dot_r = np.dot(fcc_coord+coord,hkl)
                if isinstance(specie_params,list):
                    ff = np.sum([specie_params[i]['occupancy'] *
                        xrff.compute_ff(np.array([q_hkl]),specie_name,specie_params[i]) 
                        for i in range(len(specie_params))]) 
                else: 
                    ff = xrff.compute_ff(np.array([q_hkl]),specie_name,specie_params)[0]*np.exp(2j*np.pi*g_dot_r)
                F_hkl += ff
    return F_hkl

def fcc_sf_spherical_average(q,popd):
    n_q = len(q)
    sf_func = lambda qi,ph,th: fcc_sf(qi,
            np.array([
            qi*np.sin(th)*np.cos(ph),
            qi*np.sin(th)*np.sin(ph),
            qi*np.cos(th)]),
            popd)
    sf_integral_func = lambda qi: quadpy.sphere.integrate_spherical(
        partial(sf_func,qi),rule=quadpy.sphere.Lebedev(35))
    sf = 1./(2*np.pi**2)*np.array(
        [sf_integral_func(qq) for qq in q],
        dtype=complex)
    return sf

