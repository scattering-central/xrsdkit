from collections import OrderedDict
import copy

import numpy as np

from . import form_factors as xrff
from . import structure_factors as xrsf
from . import symmetries as xrsdsym
from ..tools import peak_math, positive_normal_sampling
from .. import definitions as xrsdefs

def compute_intensity(q,source_wavelength,structure,form,settings,parameters):
    nq = len(q)
    if structure == 'crystalline':
        coords = [[0.,0.,0.]]
        occs = [1.]
        if form == 'spherical':
            ff_funcs = [xrff.spherical_ff_func(parameters['r']['value'])]
        elif form == 'atomic':
            ff_funcs = [xrff.atomic_ff_func(settings['symbol'])]
        if form == 'polyatomic':
            coords = []
            for iat in range(settings['n_atoms']):
                crds_i = [  parameters['u_{}'.format(iat)]['value'],\
                            parameters['v_{}'.format(iat)]['value'],\
                            parameters['w_{}'.format(iat)]['value'] ]
                coords.append(crds_i)
            occs = [parameters['occupancy_{}'.format(iat)]['value'] for iat in range(settings['n_atoms'])]
            ff_funcs = [xrff.atomic_ff_func(settings['symbol_{}'.format(iat)]) for iat in range(settings['n_atoms'])]
        latparams = {}
        for param_nm,param_def in xrsdefs.structure_params('crystalline',{'lattice':settings['lattice']}).items():
            latparams[param_nm] = parameters[param_nm]['value']
        if settings['profile'] == 'voigt': 
            pk_func = peak_math.voigt_function(parameters['hwhm_g']['value'],parameters['hwhm_l']['value'])
        if settings['profile'] == 'gaussian': 
            pk_func = peak_math.gaussian_function(parameters['hwhm']['value'])
        if settings['profile'] == 'lorentzian': 
            pk_func = peak_math.lorentzian_function(parameters['hwhm']['value'])
        I_xtal = integrated_isotropic_diffraction_intensity(
            q,source_wavelength,settings['lattice'],latparams,coords,ff_funcs,pk_func,occs,
            q_min=settings['q_min'],q_max=settings['q_max'],
            space_group=settings['space_group'],
            sf_mode=settings['structure_factor_mode']
            )
        return parameters['I0']['value'] * I_xtal
    else:
        if form == 'atomic':
            ff_sqr = xrff.atomic_ff(q,settings['symbol']) ** 2
        if form == 'spherical':
            if settings['distribution'] == 'single':
                ff_sqr = xrff.spherical_ff(q,parameters['r']['value']) ** 2
            if settings['distribution'] == 'r_normal':
                ff_sqr = spherical_normal_intensity(q,
                    parameters['r']['value'],parameters['sigma']['value'],
                    settings['sampling_width'],settings['sampling_step']
                    )
        if form == 'guinier_porod':
            if settings['distribution'] == 'single':
                ff_sqr = guinier_porod_intensity(q,parameters['rg']['value'],parameters['D']['value']) 
        if structure == 'disordered':
            sf = xrsf.hard_sphere_sf(q,parameters['r_hard']['value'],parameters['v_fraction']['value'])
            return parameters['I0']['value'] * sf * ff_sqr 
        if structure == 'diffuse':
            return parameters['I0']['value'] * ff_sqr 


def guinier_porod_intensity(q,rg,porod_exponent):
    """Compute a Guinier-Porod scattering intensity.

    Returned array of intensities is normalized such that I(0)=1.    

    Parameters
    ----------
    q : array
        array of q values
    rg : float
        radius of gyration
    porod_exponent : float
        high-q Porod's law exponent

    Returns
    -------
    I : array
        Array of intensities for all q 

    Reference
    ---------
    B. Hammouda, J. Appl. Cryst. (2010). 43, 716-719.
    """
    guinier_factor = 1.
    # q-domain boundary q_splice:
    q_splice = 1./rg * np.sqrt(3./2*porod_exponent)
    idx_guinier = (q <= q_splice)
    idx_porod = (q > q_splice)
    # porod prefactor D:
    porod_factor = guinier_factor*np.exp(-1./2*porod_exponent)\
                    * (3./2*porod_exponent)**(1./2*porod_exponent)\
                    * 1./(rg**porod_exponent)
    I = np.zeros(q.shape)
    # Guinier equation:
    if any(idx_guinier):
        I[idx_guinier] = guinier_factor * np.exp(-1./3*q[idx_guinier]**2*rg**2)
    # Porod equation:
    if any(idx_porod):
        I[idx_porod] = porod_factor * 1./(q[idx_porod]**porod_exponent)
    return I 


def spherical_normal_intensity(q,r0,sigma,sampling_width=3.5,sampling_step=0.05):  
    """Compute the form factor for a normally-distributed sphere population.

    The returned form factor is normalized 
    such that its value at q=0 is 1.
    The current version samples the distribution 
    from r0*(1-sampling_width*sigma) to r0*(1+sampling_width*sigma)
    in steps of sampling_step*sigma*r0
    Additional info about sampling_width and sampling_step:
    https://github.com/scattering-central/saxskit/examples/spherical_normal_saxs_benchmark.ipynb

    Parameters
    ----------
    q : array
        array of scattering vector magnitudes
    r0 : float
        mean radius of the sphere population
    sigma : float
        fractional standard deviation of the sphere population radii
    sampling_width : float
        sampling width in units of sigma-
        samples are taken from below and above the mean,
        unless this would require sampling negative values,
        in which case the region below zero is truncated. 
    sampling_step : float
        spacing between samples in units of sigma 

    Returns
    -------
    I : array
        Array of intensity values for all q
    """
    if sigma < 1.E-9:
        x = q*r0
        V_r0 = float(4)/3*np.pi*r0**3
        I_0 = V_r0**2
        I = I_0*xrff.spherical_ff(q,r0)**2
    else:
        I = np.zeros(q.shape)
        rmin,rmax,dr = positive_normal_sampling(r0,sigma,sampling_width,sampling_step)
        sigma_r = sigma*r0
        I_0 = 0
        for ri in np.arange(rmin,rmax,dr):
            V_ri = float(4)/3*np.pi*ri**3
            # The normal-distributed density of particles with radius r_i:
            rhoi = 1./(np.sqrt(2*np.pi)*sigma_r)*np.exp(-1*(r0-ri)**2/(2*sigma_r**2))
            I0_i = V_ri**2*rhoi*dr
            I_0 += I0_i
            I += I0_i*xrff.spherical_ff(q,ri)**2
    I = I/I_0 
    return I 


# TODO: vectorize and parallelize where possible
def integrated_isotropic_diffraction_intensity(
    q,source_wavelength,lattice,latparams,coords,ff_funcs,pk_func,
    occupancies=None,q_min=0.,q_max=None,space_group='',sf_mode='local'):
    """Compute integrated diffraction pattern for an isotropic (powder-like) system.

    Parameters
    ----------
    q : numpy.array
        Vector of q-values where intensity will be computed
    lattice : str
        Lattice specification (one of xrsdkit.scattering.space_groups.all_lattices).
    latparams : dict
        Dict defining lattice parameters in Angstroms and degrees 
        (dict keys: ['a','b','c','alpha','beta','gamma'])
    coords : list 
        List of 3-element iterables defining fractional coordinates
        of specie positions on the lattice vector basis
    ff_funcs : list
        List of functions that compute form factors for all species at any q,
        in order corresponding to `coords`
    pk_func : callable 
        Function that yields a peak profile, as pk_func(q,q_center)
    source_wavelength : float
        Light source wavelength in Angstroms
    q_min : float
        Minimum q-value (>0) for reciprocal space integration
    q_max : float
        Maximum q-value (>`q_min`) for reciprocal space integration-
        if not provided, automatically set to the highest value in `q`.
    space_group : str
        Space group designation used for symmetrizing the reciprocal space summation,
        should be one of xrsdkit.scattering.space_groups.lattice_space_groups[`lattice`]
    sf_mode : str
        Either 'local' or 'radial'. 
        If 'local', for each reciprocal lattice point,
        the crystal structure factor is computed exactly at the lattice point.
        If 'radial', for each reciprocal lattice point,
        the crystal structure factor is computed along a line 
        from the reciprocal space origin through the lattice point.
        The 'radial' mode is meant to capture the effects of 
        form factors that vary considerably within peak widths.

    Returns
    -------
    numpy.array
        Diffracted intensity, normalized such that I(q=0) is equal to 1.
    """
    #import pdb; pdb.set_trace()
    if not source_wavelength:
        raise ValueError('cannot compute diffraction with source wavelength of {}'.format(source_wavelength))
    if not q_max: q_max = q[-1]
    n_species = len(coords)
    if not occupancies: occupancies = np.ones(n_species)
    if space_group and not space_group in xrsdefs.lattice_space_groups[lattice].values():
        raise ValueError('space group {} not valid for {} lattice'.format(space_group,lattice))

    n_q = len(q)
    I = np.zeros(n_q)
    th = np.arcsin(source_wavelength*q/(4.*np.pi))
    # polarization factor 
    pz = 0.5*(1.+np.cos(2.*th)**2) 
    I0 = 0.
    q0 = np.array([0.])

    a1,a2,a3 = xrsdefs.lattice_vectors(lattice,**latparams)
    b1,b2,b3 = xrsdefs.reciprocal_lattice_vectors(a1,a2,a3)
    absa1 = np.linalg.norm(a1)
    absa2 = np.linalg.norm(a2)
    absa3 = np.linalg.norm(a3)

    # Get d-spacings corresponding to the q-range limits,
    # and get the corresponding G_hkl lengths (G=1/d, q=2pi*G).
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
    if not all_hkl.any():
        return np.zeros(n_q)
    
    # symmetrize the hkl sampling, save the multiplicities 
    # TODO: determine whether or not this can be done solely based on the point group
    #point_group = xrsdefs.sg_point_groups[space_group]
    reduced_hkl,hkl_mults = xrsdsym.symmetrize_points(all_hkl,np.array([b1,b2,b3]),space_group)  

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
    pk_q = {}
    pk_0 = {}
    for qval in absq_set: 
        pk_q[qval] = pk_func(q,qval)
        pk_0[qval] = pk_func(q0,qval)[0]
    
    # structure factors for all hkl
    # TODO: can this be vectorized? 
    sf_hkl = np.zeros((reduced_hkl.shape[0],n_q),dtype=complex) 
    sf_0 = np.zeros(reduced_hkl.shape[0],dtype=complex)
    latcoords = xrsdefs.lattice_coords(lattice)
    for ccc,fff in zip(coords,ff_funcs):
        if sf_mode == 'radial':
            ff = fff(q)
            for ihkl,absq in zip(range(reduced_hkl.shape[0]),absq_hkl):
                for lc in latcoords:
                    g_dot_r = np.dot(lc+ccc,reduced_hkl[ihkl,:])
                    sf_hkl[ihkl,:] += fff(q) * np.exp(2j*np.pi*g_dot_r)
                    sf_0[ihkl] += fff(q0)[0] * np.exp(2j*np.pi*g_dot_r)
        elif sf_mode == 'local':
            ff_set = fff(np.array(absq_set_list))
            ff_absq = dict([(qq,ff) for qq,ff in zip(absq_set_list,ff_set)]) 
            for ihkl,absq in zip(range(reduced_hkl.shape[0]),absq_hkl):
                for lc in latcoords:
                    g_dot_r = np.dot(lc+ccc,reduced_hkl[ihkl,:])
                    sf_hkl[ihkl,:] += ff_absq[absq] * np.exp(2j*np.pi*g_dot_r)
                    sf_0[ihkl] += fff(q0)[0] * np.exp(2j*np.pi*g_dot_r)

    for ihkl,absq,ltz,mult in zip(range(reduced_hkl.shape[0]),absq_hkl,ltz_hkl,hkl_mults):
        I += mult*ltz*(sf_hkl[ihkl,:]*sf_hkl[ihkl,:].conjugate()).real*pk_q[absq] 
        I0 += mult*ltz*(sf_0[ihkl]*sf_0[ihkl].conjugate()).real*pk_0[absq]
    
    return pz*I/I0
            

