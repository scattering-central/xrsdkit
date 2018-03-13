import os

import yaml
import numpy as np

fpath = os.path.join(os.path.dirname(__file__),'atomic_scattering_params.yaml')
atomic_params = yaml.load(open(fpath,'r'))

def compute_ff(q,populations):
    nq = len(q)
    ff = np.zeros(nq)
    if not isinstance(populations,list):
        populations = [populations]
    for popd in populations:
        if 'flat' in popd.keys():
            ff_flat = float(popd['flat']['amplitude'])*np.ones(nq)
            ff += ff_flat
        if 'guinier_porod' in popd.keys():
            ff_gp = guinier_porod(q,popd['guinier_porod'])
            ff += ff_gp
        if 'spherical_normal' in popd.keys():
            ff_sph = spherical_normal_ff(q,popd['spherical_normal']) 
            ff += ff_sph
        if 'atomic' in popd.keys():
            if 'atom_name' in popd['atomic']:
                ff_atom = standard_atomic_ff(q,popd['atomic']['atom_name'])
            else:
                ff_atom = atomic_ff(q,popd['atomic'])
            ff += ff_atom
    return ff

def standard_atomic_ff(q,atom_name):
    pars = atomic_params[atom_name]
    return atomic_ff(q,pars)

def atomic_ff(q,params):
    g = q*1./(2*np.pi)
    s = g*1./2
    s2 = s**2
    Z = params['Z']
    a = params['a']
    b = params['b']
    ff = Z - 41.78214 * s2 * np.sum([aa*np.exp(-1*bb*s2) for aa,bb in zip(a,b)],axis=0)
    return ff

def spherical_normal_ff(q,r0,sigma,sampling_width=3.5,sampling_step=0.1):
    """Compute the form factor for a normally-distributed sphere population.

    The returned intensity is normalized 
    such that I(q=0) is equal to 1.
    The current version samples the distribution 
    from r0*(1-sampling_width*sigma) to r0*(1+sampling_width*sigma)
    in steps of sampling_step*sigma*r0
    Additional info about sampling_width and sampling_step:
    https://github.com/scattering-central/saxskit/examples/spherical_normal_saxs_benchmark.ipynb

    Originally contributed by Amanda Fournier.

    Parameters
    ----------
    q : array
        array of scattering vector magnitudes
    r0 : float
        mean radius of the sphere population
    sigma : float
        fractional standard deviation of the sphere population radii
    sampling_width : float
        number of standard deviations of radius for sampling
    sampling_step : float
        fraction of standard deviation to use as sampling step size    

    Returns
    -------
    ff : array
        Array of form factor amplitudes for each of the input q values
    """
    q_zero = (q == 0)
    q_nz = np.invert(q_zero) 
    ff = np.zeros(q.shape)
    if sigma < 1E-9:
        x = q*r0
        V_r0 = float(4)/3*np.pi*r0**3
        ff_0 = V_r0 
        ff[q_nz] = ff_0 * (3.*(np.sin(x[q_nz])-x[q_nz]*np.cos(x[q_nz]))*x[q_nz]**-3)
    else:
        sigma_r = sigma*r0
        dr = sigma_r*sampling_step
        rmin = np.max([r0-sampling_width*sigma_r,dr])
        rmax = r0+sampling_width*sigma_r
        ff_0 = 0
        for ri in np.arange(rmin,rmax,dr):
            xi = q*ri
            V_ri = float(4)/3*np.pi*ri**3
            # The normal-distributed density of particles with radius r_i:
            rhoi = 1./(np.sqrt(2*np.pi)*sigma_r)*np.exp(-1*(r0-ri)**2/(2*sigma_r**2))
            # NOTE: assume that the scattering intensities
            # from all the different sphere sizes
            # are superposed in the measured intensity.
            # If the intensity is proportional to scatterer density,
            # (which is represented in this case as rhoi*dr),
            # the form factor contribution must scale with sqrt(rhoi*dr). 
            ff_0 += V_ri * np.sqrt(rhoi*dr)
            ff[q_nz] += ff_0*(3.*(np.sin(xi[q_nz])-xi[q_nz]*np.cos(xi[q_nz]))*xi[q_nz]**-3)
    if any(q_zero):
        ff[q_zero] = ff_0 
    ff = ff/ff_0 
    return ff 

def guinier_porod(q,r_g,porod_exponent,guinier_factor):
    """Compute the form factor amplitude for a Guinier-Porod scattering population.
    
    Parameters
    ----------
    q : array
        array of q values
    r_g : float
        radius of gyration
    porod_exponent : float
        high-q Porod's law exponent
    guinier_factor : float
        low-q Guinier prefactor (equal to intensity at q=0)

    Returns
    -------
    ff : array
        Array of form factor amplitudes for each of the input q values

    Reference
    ---------
    B. Hammouda, J. Appl. Cryst. (2010). 43, 716-719.
    """
    # q-domain boundary q_splice:
    q_splice = 1./r_g * np.sqrt(3./2*porod_exponent)
    idx_guinier = (q <= q_splice)
    idx_porod = (q > q_splice)
    # porod prefactor D:
    porod_factor = guinier_factor*np.exp(-1./2*porod_exponent)\
                    * (3./2*porod_exponent)**(1./2*porod_exponent)\
                    * 1./(r_g**porod_exponent)
    I = np.zeros(q.shape)
    # Guinier equation:
    if any(idx_guinier):
        I[idx_guinier] = guinier_factor * np.exp(-1./3*q[idx_guinier]**2*r_g**2)
    # Porod equation:
    if any(idx_porod):
        I[idx_porod] = porod_factor * 1./(q[idx_porod]**porod_exponent)
    ff = np.sqrt(I)
    return ff



