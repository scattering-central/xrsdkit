import os

import yaml
import numpy as np

fpath = os.path.join(os.path.dirname(__file__),'atomic_scattering_params.yaml')
atomic_params = yaml.load(open(fpath,'r'))

def compute_ff(q,specie_name,params):
    nq = len(q)
    ff = np.zeros(nq)
    if specie_name == 'flat':
        return float(params['amplitude'])*np.ones(nq)
    if specie_name == 'guinier_porod':
        ff_gp = guinier_porod_ff(q,params['G'],params['r_g'],params['D'])
        return ff_gp 
    #if specie_name == 'spherical_normal':
    #    # TODO: come up with an expression for the form factor?
    #    I_sph = spherical_normal_intensity(q,params['r0'],params['sigma']) 
    #    # TODO: determine whether or not this square root approach is ok.
    #    ff_sph = np.sqrt(I_sph) 
    #    return ff_sph
    if specie_name == 'atomic':
        if 'atom_name' in params:
            ff_atom = standard_atomic_ff(q,params['atom_name'])
        else:
            ff_atom = atomic_ff(q,params)
        return ff_atom

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

def spherical_ff(q,r):
    x = q*r
    return 3.*(np.sin(x)-x*np.cos(x))*x**-3

def spherical_normal_ff(q,r0,sigma,sampling_width=3.5,sampling_step=0.1):  
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
        number of standard deviations of radius for sampling
    sampling_step : float
        fraction of standard deviation to use as sampling step size    

    Returns
    -------
    ff : array
        Array of form factor values for all q
    """
    q_zero = (q == 0)
    q_nz = np.invert(q_zero) 
    ff = np.zeros(q.shape)
    if sigma < 1E-9:
        x = q*r0
        V_r0 = float(4)/3*np.pi*r0**3
        ff_0 = V_r0 
        ff[q_nz] = ff_0*spherical_ff(q[q_nz],r0)
    else:
        sigma_r = sigma*r0
        dr = sigma_r*sampling_step
        rmin = np.max([r0-sampling_width*sigma_r,dr])
        rmax = r0+sampling_width*sigma_r
        ff_0 = 0
        for ri in np.arange(rmin,rmax,dr):
            V_ri = float(4)/3*np.pi*ri**3
            # The normal-distributed density of particles with radius r_i:
            rhoi = 1./(np.sqrt(2*np.pi)*sigma_r)*np.exp(-1*(r0-ri)**2/(2*sigma_r**2))
            # NOTE: this is a mistake.
            ff_0 += V_ri * np.sqrt(rhoi*dr)
            ff[q_nz] += ff_0*spherical_ff(q[q_nz],ri)
    if any(q_zero):
        ff[q_zero] = ff_0 
    ff = ff/ff_0 
    return ff 

def guinier_porod_ff(q,guinier_factor,r_g,porod_exponent):
    """Compute the form factor for a Guinier-Porod population.
    
    Parameters
    ----------
    q : array
        array of q values
    guinier_factor : float
        low-q Guinier prefactor (equal to intensity at q=0)
    r_g : float
        radius of gyration
    porod_exponent : float
        high-q Porod's law exponent

    Returns
    -------
    ff : array
        Array of form factor values for all q 

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
    ff = np.zeros(q.shape)
    # Guinier equation:
    if any(idx_guinier):
        ff[idx_guinier] = np.sqrt(guinier_factor) * np.exp(-1./6*q[idx_guinier]**2*r_g**2)
    # Porod equation:
    if any(idx_porod):
        ff[idx_porod] = np.sqrt(porod_factor) * 1./(2*q[idx_porod]**porod_exponent)
    return ff 


