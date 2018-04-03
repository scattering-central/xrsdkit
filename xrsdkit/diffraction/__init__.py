from collections import OrderedDict

import numpy as np
from pymatgen import Lattice

from . import structure_factors as xrsf
from .peak_math import peak_profile

# list of structures that are crystalline
crystalline_structure_names = ['fcc']

def fcc_intensity(q,popd,source_wavelength):
    n_q = len(q)
    I = np.zeros(n_q)
    basis = popd['basis']
    profile_name = popd['settings']['profile']
    q_min = popd['settings']['q_min']
    q_max = popd['settings']['q_max']
    lat_a = popd['parameters']['a']
    # get d-spacings corresponding to the q-range limits
    d_min = 2*np.pi/q_max
    if q_min > 0.:
        d_max = 2*np.pi/q_min
    else:
        d_max = float('inf')
    # get the corresponding G_hkl lengths, i.e. 1/d
    if q_min > 0.:
        G_min = 1./d_max
    else:
        G_min = 0
    G_max = 1./d_min
    # create the fcc lattice
    lat = Lattice.cubic(lat_a)
    r_lat = lat.reciprocal_lattice_crystallographic
    # keep only reciprocal lattice points within our G_hkl limits
    r_pts = r_lat.get_points_in_sphere([[0,0,0]],[0,0,0],G_max)
    r_pts = [pt for pt in r_pts if pt[1] >= G_min]

    g_pks = OrderedDict()
    q_pks = OrderedDict()
    I_pks = OrderedDict()
    mult = OrderedDict()
    for hkl, g_hkl, idx in sorted(r_pts, 
    key=lambda pt: (pt[1], -pt[0][0], -pt[0][1], -pt[0][2])):
        # cast hkl as tuple for use as dict key
        immhkl = tuple(hkl)
        if g_hkl > 0.:
            q_hkl = 2*np.pi*g_hkl
            F_hkl = xrsf.fcc_sf(q_hkl,hkl.reshape(3,1),basis)
            I_hkl = (F_hkl * F_hkl.conjugate()).real
            # TODO: set this intensity threshold as a function input
            if I_hkl > 1.E-5:
                q_nearest_pk = float('inf') 
                if any(q_pks):
                    nearest_pk_idx = np.argmin([abs(q_hkl-qq) for qq in q_pks.values()])
                    q_nearest_pk = list(q_pks.values())[nearest_pk_idx]
                    hkl_nearest_pk = list(q_pks.keys())[nearest_pk_idx]
                dq_nearest_pk = q_hkl - q_nearest_pk
                # TODO: set this dq_nearest_peak threshold as a function input
                if abs(dq_nearest_pk) > 1.E-5:
                    mult[immhkl] = 1
                    I_pks[immhkl] = I_hkl
                    q_pks[immhkl] = q_hkl
                    g_pks[immhkl] = g_hkl
                else:
                    I_pks[hkl_nearest_pk] += I_hkl
                    mult[hkl_nearest_pk] += 1
    for hkl, q_pk in q_pks.items():
        # compute the structure factor
        # along the line connecting (000) to (hkl)
        hkl_range = np.outer(q/q_pk,hkl).T
        F_along_hkl = xrsf.fcc_sf(q_pk,hkl_range,basis)
        # compute a line shape 
        line_shape = peak_profile(q,q_pk,profile_name,popd['parameters'])
        I += (F_along_hkl*F_along_hkl.conjugate()).real\
            *mult[hkl]*line_shape

    # compute the Lorentz factor 
    # 

    # compute Debye-Waller factors if parameters given
    #

    # multiply correction factors into the intensity
    #

    return I

