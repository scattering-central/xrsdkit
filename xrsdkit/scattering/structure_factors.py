from functools import partial

import numpy as np

from . import form_factors as xrff

fcc_coords = np.array([
    [0.,0.,0.],
    [0.5,0.5,0.],
    [0.5,0.,0.5],
    [0.,0.5,0.5]])

def fcc_sf(q_hkl,hkl,basis):
    """Computes the FCC structure factor.

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
                    if not isinstance(site_item,list): site_item = [site_item]
                    # TODO: defend against weird or nonphysical occupancy choices?
                    for itm in site_item:
                        occ = 1.
                        if 'occupancy' in itm: occ = itm['occupancy']
                        F_hkl += occ * \
                        xrff.specie_ff(np.array([q_hkl]),site_item_name,itm) \
                        * np.exp(2j*np.pi*g_dot_r) 
    return F_hkl

def hard_sphere_sf(q,r_sphere,volume_fraction):
    """Computes the Percus-Yevick hard-sphere structure factor. 

    Parameters
    ----------
    q : array
        array of q values 

    Returns
    -------
    F : float 
        Structure factor at `q`
    """
    p = volume_fraction
    d = 2*r_sphere
    qd = q*d
    qd2 = qd**2
    qd3 = qd**3
    qd4 = qd**4
    qd6 = qd**6
    sinqd = np.sin(qd)
    cosqd = np.cos(qd)
    l1 = (1+2*p)**2/(1-p)**4
    l2 = -1*(1+p/2)**2/(1-p)**4
    nc = -24*p*(
        l1*( (sinqd - qd*cosqd) / qd3 )
        -6*p*l2*( (qd2*cosqd - 2*qd*sinqd - 2*cosqd + 2) / qd4 )
        -p*l1/2*( (qd4*cosqd - 4*qd3*sinqd - 12*qd2*cosqd + 24*qd*sinqd + 24*cosqd - 24) / qd6 )
        )
    F = 1/(1-nc)
    return F

#def fcc_sf_spherical_average(q,popd):
#    n_q = len(q)
#    sf_func = lambda qi,ph,th: fcc_sf(qi,
#            np.array([
#            qi*np.sin(th)*np.cos(ph),
#            qi*np.sin(th)*np.sin(ph),
#            qi*np.cos(th)]),
#            popd)
#    import quadpy
#    sf_integral_func = lambda qi: quadpy.sphere.integrate_spherical(
#        partial(sf_func,qi),rule=quadpy.sphere.Lebedev(35))
#    sf = 1./(2*np.pi**2)*np.array(
#        [sf_integral_func(qq) for qq in q],
#        dtype=complex)
#    return sf

