from functools import partial

import numpy as np
#import quadpy

from xrsdkit import scattering
from xrsdkit.scattering import form_factors as xrff

fcc_coords = np.array([
    [0.,0.,0.],
    [0.5,0.5,0.],
    [0.5,0.,0.5],
    [0.,0.5,0.5]])

def fcc_sf(q_hkl,hkl,basis):
    """Compute the FCC structure factor

    Parameters
    ----------
    q_hkl : float
        single q-value at which the structure factor is computed
    hkl : array
        array of plane indices to include in structure factor computation 
    basis : dict
        dict specifying scatterer coordinates and parameters 

    Returns
    -------
    F_hkl : complex 
        complex structure factor at `q_hkl`
    """
    # TODO: make this work for q_hkl as an array (hkl as a matrix)
    F_hkl = np.zeros(hkl.shape[1],dtype=complex) 
    for fcc_coord in fcc_coords:
        for site_name, site_items in basis.items():
            coord = site_items['coordinates']
            g_dot_r = np.dot(fcc_coord+coord,hkl)
            for site_item_name, site_item in site_items.items():
                if not site_item_name == 'coordinates':
                    if isinstance(site_item,list):
                        # TODO: defend against weird or nonphysical occupancy choices?
                        ff = np.sum([site_item[i]['occupancy'] \
                        * xrff.compute_ff(np.array([q_hkl]),site_item_name,site_item[i]) \
                        * np.exp(2j*np.pi*g_dot_r) \
                        for i in range(len(site_item))]) 
                    else: 
                        occ = 1.
                        if 'occupancy' in site_item: occ = site_item['occupancy']
                        ff = occ * xrff.compute_ff(np.array([q_hkl]),site_item_name,site_item)[0] \
                        * np.exp(2j*np.pi*g_dot_r)
                    F_hkl += ff
    return F_hkl

#def fcc_sf_spherical_average(q,popd):
#    n_q = len(q)
#    sf_func = lambda qi,ph,th: fcc_sf(qi,
#            np.array([
#            qi*np.sin(th)*np.cos(ph),
#            qi*np.sin(th)*np.sin(ph),
#            qi*np.cos(th)]),
#            popd)
#    sf_integral_func = lambda qi: quadpy.sphere.integrate_spherical(
#        partial(sf_func,qi),rule=quadpy.sphere.Lebedev(35))
#    sf = 1./(2*np.pi**2)*np.array(
#        [sf_integral_func(qq) for qq in q],
#        dtype=complex)
#    return sf

