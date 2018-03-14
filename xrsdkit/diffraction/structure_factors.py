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

def fcc_sf(q_hkl,hkl,populations):
    if not isinstance(populations,list):
        populations = [populations]
    F_hkl = np.zeros(hkl.shape[1],dtype=complex) 
    for popd in populations:
        basis = popd['basis']
        for fcc_coord in fcc_coords:
            for coord,ffspec in basis.items():
                g_dot_r = np.dot(fcc_coord+coord,hkl)
                ff = xrff.compute_ff(np.array([q_hkl]),ffspec)[0]*np.exp(2j*np.pi*g_dot_r)
                F_hkl += ff
    return F_hkl

def fcc_sf_spherical_average(q,populations):
    if not isinstance(populations,list):
        populations = [populations]
    n_q = len(q)
    sf_func = lambda qi,ph,th: fcc_sf(qi,
        np.array([
            qi*np.sin(th)*np.cos(ph),
            qi*np.sin(th)*np.sin(ph),
            qi*np.cos(th)]),
        populations)
    
    sf_integral_func = lambda qi: quadpy.sphere.integrate_spherical(
        partial(sf_func,qi),rule=quadpy.sphere.Lebedev(35))
    sf = 1./(2*np.pi**2)*np.array(
        [sf_integral_func(qq) for qq in q],
        dtype=complex)
    return sf
    #for popd in populations:
    #    # compute the average intensity,
    #    # over the reciprocal space sphere,
    #    # for each q
    #    basis = popd['basis']
    #    for fcc_coord in fcc_coords:
    #        for coord,ffspec in basis.items():
    #            ff = compute_ff(q,ffspec)
    #            site_coord = fcc_coord+coord
    #            site_sf = np.zeros(n_q,dtype=complex)
    #            for idx in range(n_q):
    #                qi = q[idx]
    #                ffi = ff[idx]
    #                #sf_func = lambda ph,th: ffi*np.exp(2j*np.pi*\
    #                #( (qi*np.sin(th)*np.cos(ph))*site_coord[0]
    #                #+ (qi*np.sin(th)*np.sin(ph))*site_coord[1]
    #                #+ (qi*np.cos(th))*site_coord[2]))
    #                site_sf[idx] = quadpy.sphere.integrate_spherical(
    #                    sf_func,rule=quadpy.sphere.Lebedev(19))
    #            sf += site_sf
    #    #from matplotlib import pyplot as plt
    #    #for q_pk, I_pk in I_pks.items():
    #    #    plt.plot([q_pk,q_pk],[0.,10*I_pk],'r')
    #    #plt.plot(q,sf.real,'g')
    #    #plt.plot(q,sf.imag,'r')
    #    #plt.show()
    #return sf

