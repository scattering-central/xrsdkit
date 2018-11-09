from collections import OrderedDict
import copy

import numpy as np

from . import form_factors as xrff
from . import lattices
from . import space_groups as sgs
from ..tools import peak_math

def diffuse_intensity(q,popd,source_wavelength):
    I0 = popd['parameters']['I0']['value']
    F_q = xrff.compute_ff_squared(q,popd['basis'])
    # TODO: consider whether polz factors are needed here... 
    #pz = 1. + np.cos(2.*th)**2 
    return I0*F_q

def disordered_intensity(q,popd,source_wavelength):
    basis = popd['basis']
    I0 = popd['parameters']['I0']['value']
    P_q = xrff.compute_ff_squared(q,basis)
    if popd['settings']['interaction'] == 'hard_spheres':
        r = popd['parameters']['r_hard']['value']
        p = popd['parameters']['v_fraction']['value']
        F_q = hard_sphere_sf(q,r,p)
        # TODO: consider whether polz factors are needed here... 
        #pz = 1. + np.cos(2.*th)**2 
        return I0 * F_q * P_q 
    else:
        msg = 'interaction specification {} is not supported'\
            .format(popd['settings']['interaction'])
        raise ValueError(msg)

# TODO: vectorize and parallelize where possible
def crystalline_intensity(q,popd,source_wavelength):

    n_q = len(q)
    I = np.zeros(n_q)
    th = np.arcsin(source_wavelength*q/(4.*np.pi))
    # polarization factor 
    pz = 1. + np.cos(2.*th)**2 
    profile_name = popd['settings']['profile']
    q_min = popd['settings']['q_min']
    q_max = popd['settings']['q_max']
    I0 = popd['parameters']['I0']['value']
    sf_mode = popd['settings']['structure_factor_mode']
    lattice_id = popd['settings']['lattice']

    latparams = dict.fromkeys(['a','b','c','alpha','beta','gamma'])
    for k in latparams.keys(): 
        if k in popd['parameters']: latparams[k] = popd['parameters'][k]['value']
    a1,a2,a3 = lattices.get_lattice_vectors(lattice_id,**latparams)
    b1,b2,b3 = lattices.reciprocal_lattice_vectors(a1,a2,a3)
    absa1 = np.linalg.norm(a1)
    absa2 = np.linalg.norm(a2)
    absa3 = np.linalg.norm(a3)

    # NOTE: the following assumes full spherical integration of reciprocal space.
    # TODO: implement settings to select the reciprocal space regions to integrate.
    # Get d-spacings corresponding to the q-range limits,
    # and get the corresponding G_hkl lengths (G=1/d).
    # NB: for the crystallographic reciprocal lattice,
    # G_hkl-vectors are h*b1+k*b2+l*b3,
    # and the corresponding q-vectors are 2*pi*G_hkl
    d_min = 2*np.pi/q_max
    G_max = 1./d_min
    if q_min > 0.:
        d_max = 2*np.pi/q_min
        G_min = 1./d_max
    else:
        d_max = float('inf')
        G_min = 0

    # Find all reciprocal lattice points in the cored sphere from G_min to G_max.
    # Candidate points are in the minimal parallelepiped that encompasses the G_max sphere.
    # This paralleliped is found by projecting each b vector 
    # onto the unit normal to the basis plane defined by the other two b vectors,
    # and counting how many of these projected vectors fit within G_max.
    # Note, the reciprocal lattice basis plane unit normals
    # are simply the real space lattice vectors, normalized. 
    n1 = np.ceil(G_max*absa1/np.dot(b1,a1)) 
    n2 = np.ceil(G_max*absa2/np.dot(b2,a2)) 
    n3 = np.ceil(G_max*absa3/np.dot(b3,a3)) 
    h_range = np.arange(-1*n1+1,n1)
    k_range = np.arange(-1*n2+1,n2)
    l_range = np.arange(-1*n3+1,n3)

    # TODO: vectorize this?
    # NOTE: this is designed to leave out hkl=000 
    all_hkl = np.array([(h,k,l) for l in l_range for k in k_range for h in h_range \
            if (G_min < np.linalg.norm(np.dot((h,k,l),(b1,b2,b3))) <= G_max)])
    
    # symmetrize the hkl sampling, save the multiplicities 
    # TODO: implement 'space_group' as a setting
    space_group = None
    #space_group = popd['settings']['space_group']
    # TODO: make sure the specie coordinates agree with the space group selection
    # NOTE: this should be done when the space group is set...
    # and maybe also here if it's fast

    if not space_group:
        # select a space group, given lattice_id
        if len(popd['basis']) <= 1:
            # take the highest symmetry space group
            space_group = lattices.default_high_sym_space_groups[lattice_id]
        else:
            # take the lowest symmetry space group
            space_group = lattices.default_low_sym_space_groups[lattice_id]

    point_group = sgs.sg_point_groups[space_group]
    reduced_hkl,hkl_mults = lattices.symmetrize_points(all_hkl,np.array([b1,b2,b3]),point_group)  

    # g-vector magnitude for all hkl
    absg_hkl = np.linalg.norm(np.dot(reduced_hkl,[b1,b2,b3]),axis=1)
    # q-vector magnitude for all hkl
    absq_hkl = 2*np.pi*absg_hkl
    absq_set = set(absq_hkl)
    absq_set_list = list(absq_set)
    # diffraction angle theta for all hkl
    th_hkl = np.arcsin(source_wavelength*absq_hkl/(4.*np.pi))
    # Lorentz factors for all hkl
    ltz_hkl = 1. / (np.sin(th_hkl)*np.sin(2*th_hkl))

    # peak profiles for all abs(q) values
    # TODO: vectorize this?
    pks = {}
    for qval in absq_set: 
        if profile_name == 'gaussian':
            hwhm_g = popd['parameters']['hwhm_g']['value']  
            pks[qval] = peak_math.gaussian_profile(q,qval,hwhm_g)
        elif profile_name == 'lorentzian':
            hwhm_l = popd['parameters']['hwhm_l']['value']  
            pks[qval] = peak_math.lorentzian_profile(q,qval,hwhm_l)
        elif profile_name == 'voigt':
            hwhm_g = popd['parameters']['hwhm_g']['value']  
            hwhm_l = popd['parameters']['hwhm_l']['value']  
            pks[qval] = peak_math.voigt_profile(q,qval,hwhm_g,hwhm_l)
        else:
            raise ValueError('profile specification {} not understood'.format(profile_name))

    # form factors and coords for all species
    ffs = {}
    coords = {}
    for specie_nm, specd in popd['basis'].items():
        coords[specie_nm] = [specd['coordinates'][ic]['value'] for ic in range(3)]
        if sf_mode == 'radial':
            # form factor values are needed for all q
            ffs[specie_nm] = xrff.site_ff(q,specd)
        elif sf_mode == 'local':
            # form factor values are needed only at the q values
            # corresponding to the set of q_hkl points
            ff_set = xrff.site_ff(np.array(absq_set_list),specd)
            ffs[specie_nm] = dict([(qq,ff) for qq,ff in zip(absq_set_list,ff_set)]) 
        else:
            raise ValueError('structure factor mode "{}" not understood'.format(sf_mode))

    # structure factors for all hkl
    # TODO: can this be vectorized? 
    sf_hkl = np.zeros((reduced_hkl.shape[0],n_q),dtype=complex) 
    latcoords = lattices.centering_coords[lattices.centering_map[lattice_id]]
    for ihkl,absq in zip(range(reduced_hkl.shape[0]),absq_hkl):
        for lc in latcoords:
            for specie_nm,ff in ffs.items():
                if sf_mode == 'radial':
                    hkl_range = np.outer(q/absq,reduced_hkl[ihkl,:]).T
                    g_dot_r = np.dot(lc+coords[specie_nm],hkl_range)
                    sf_hkl[ihkl,:] += ff * np.exp(2j*np.pi*g_dot_r)
                elif sf_mode == 'local':
                    g_dot_r = np.dot(lc+coords[specie_nm],reduced_hkl[ihkl,:])
                    sf_hkl[ihkl,:] += ff[absq] * np.exp(2j*np.pi*g_dot_r)

    for ihkl,absq,ltz,mult in zip(range(reduced_hkl.shape[0]),absq_hkl,ltz_hkl,hkl_mults):
        I += mult*ltz*(sf_hkl[ihkl,:]*sf_hkl[ihkl,:].conjugate()).real*pks[absq] 
    return I0*pz*I
            
def local_structure_factor(lattice_id,q_hkl,hkl,basis):
    F_hkl = 0
    try: 
        float(q_hkl)
    except:
        raise TypeError('input q_hkl(={}) should be a single float'.format(q_hkl))
    coords = lattices.centering_coords[lattices.centering_map[lattice_id]]
    for coord in coords:
        for site_name, site_def in basis.items():
            c = [site_def['coordinates'][ic]['value'] for ic in range(3)]
            g_dot_r = np.dot(coord+c,hkl)
            F_hkl += xrff.site_ff(np.array([q_hkl]),site_def) \
            * np.exp(2j*np.pi*g_dot_r)#[0] 
    return F_hkl

def radial_structure_factor(lattice_id,q,hkl,basis):
    hkl_range = np.outer(q,np.array(hkl)/np.linalg.norm(hkl)).T
    F_hkl = np.zeros(hkl_range.shape[1],dtype=complex) 
    coords = lattices.centering_coords[lattices.centering_map[lattice_id]]
    for coord in coords:
        for site_name, site_def in basis.items():
            c = [site_def['coordinates'][ic]['value'] for ic in range(3)]
            g_dot_r = np.dot(coord+c,hkl)
            F_hkl += xrff.site_ff(q,site_def) \
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
    if q[0] == 0.:
        nc = np.zeros(len(q))
        nc[0] = -2*p*( 4*l1 + 18*p*l2 + p*l1 )
        nc[1:] = -24*p*(
            l1*( (sinqd[1:] - qd[1:]*cosqd[1:]) / qd3[1:] )
            -6*p*l2*( (qd2[1:]*cosqd[1:] - 2*qd[1:]*sinqd[1:] - 2*cosqd[1:] + 2) / qd4[1:] )
            -p*l1/2*( (qd4[1:]*cosqd[1:] - 4*qd3[1:]*sinqd[1:] - 12*qd2[1:]*cosqd[1:] 
                        + 24*qd[1:]*sinqd[1:] + 24*cosqd[1:] - 24) / qd6[1:] )
            )
    else:
        nc = -24*p*(
            l1*( (sinqd - qd*cosqd) / qd3 )
            -6*p*l2*( (qd2*cosqd - 2*qd*sinqd - 2*cosqd + 2) / qd4 )
            -p*l1/2*( (qd4*cosqd - 4*qd3*sinqd - 12*qd2*cosqd 
                        + 24*qd*sinqd + 24*cosqd - 24) / qd6 )
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

