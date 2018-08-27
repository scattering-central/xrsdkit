from collections import OrderedDict

import numpy as np
from pymatgen import Lattice

from . import form_factors as xrff
from . import structure_factors as xrsf
from ..tools import peak_math

def diffuse_intensity(q,popd,source_wavelength):
    I0 = popd['parameters']['I0']['value']
    F_q = xrff.compute_ff_squared(q,popd['basis'])
    return I0*F_q

def disordered_intensity(q,popd,source_wavelength):
    if popd['settings']['interaction'] == 'hard_spheres':
        return hard_sphere_intensity(q,popd,source_wavelength)
    else:
        msg = 'interaction specification {} is not supported'\
            .format(popd['settings']['interaction'])
        raise ValueError(msg)

def crystalline_intensity(q,popd,source_wavelength):
    if popd['settings']['lattice'] == 'fcc':
        return fcc_intensity(q,popd,source_wavelength)
    else:
        msg = 'lattice specification {} is not supported'\
            .format(popd['settings']['lattice'])
        raise ValueError(msg)

#def _specie_count(species_dict):
#    n_distinct = OrderedDict.fromkeys(species_dict)
#    for specie_name,specie_params in species_dict.items():
#        if not isinstance(specie_params,list):
#            n_distinct[specie_name] = 1
#        else:
#            n_distinct[specie_name] = len(specie_params)
#    return n_distinct

def hard_sphere_intensity(q,popd,source_wavelength):
    basis = popd['basis']
    r = popd['parameters']['r_hard']['value']
    p = popd['parameters']['v_fraction']['value']
    I0 = popd['parameters']['I0']['value']
    F_q = xrsf.hard_sphere_sf(q,r,p)
    P_q = xrff.compute_ff_squared(q,basis)

    #th = np.arcsin(source_wavelength * q/(4.*np.pi))
    # compute the polarization factor 
    #pz = 1. + np.cos(2.*th)**2 
    # compute the Lorentz factor 
    #ltz = 1. / (np.sin(th)*np.sin(2*th))
    #return I0*pz*ltz * F_q * P_q 
    return I0 * F_q * P_q 

# TODO: refactor to generalize crystalline intensities
def fcc_intensity(q,popd,source_wavelength):
    n_q = len(q)
    I = np.zeros(n_q)
    basis = popd['basis']
    profile_name = popd['settings']['profile']
    q_min = popd['settings']['q_min']
    q_max = popd['settings']['q_max']
    lat_a = popd['parameters']['a']['value']
    I0 = popd['parameters']['I0']['value']
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
    for hkl, g_hkl, idx, hkl_again in sorted(r_pts, 
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
        if profile_name == 'gaussian':
            hwhm_g = popd['parameters']['hwhm_g']['value']  
            line_shape = peak_math.gaussian_profile(q,q_pk,hwhm_g)
        elif profile_name == 'lorentzian':
            hwhm_l = popd['parameters']['hwhm_l']['value']  
            line_shape = peak_math.lorentzian_profile(q,q_pk,hwhm_l)
        elif profile_name == 'voigt':
            hwhm_g = popd['parameters']['hwhm_g']['value']  
            hwhm_l = popd['parameters']['hwhm_l']['value']  
            line_shape = peak_math.voigt_profile(q,q_pk,hwhm_g,hwhm_l)
        else:
            raise ValueError('peak profile {} is not supported'.format(profile_name))
        I += (F_along_hkl*F_along_hkl.conjugate()).real\
            *mult[hkl]*line_shape

    th = np.arcsin(source_wavelength * q/(4.*np.pi))
    # compute the polarization factor 
    pz = 1. + np.cos(2.*th)**2 
    # compute the Lorentz factor 
    ltz = 1. / (np.sin(th)*np.sin(2*th))
    # TODO: compute Debye-Waller factors if parameters given
    dbw = np.ones(n_q)
    # multiply correction factors into the intensity
    I = I0*I*pz*ltz*dbw 
    return I


