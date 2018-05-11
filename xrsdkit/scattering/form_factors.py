import os

import yaml
import numpy as np

fpath = os.path.join(os.path.dirname(__file__),'atomic_scattering_params.yml')
atomic_params = yaml.load(open(fpath,'r'))
    
def compute_ff(q,basis):
    n_q = len(q)
    ff_q = np.zeros(n_q)
    for site_name, site_def in basis.items():
        ff_q += site_ff(q,site_def)
    return ff_q

def compute_ff_squared(q,basis):
    n_q = len(q)
    F_q = np.zeros(n_q)
    for site_name, site_def in basis.items():
        F_q += site_ff_squared(q,site_def)
    return F_q 

def site_ff(q,site_def):
    ff = site_def['form']
    if ff == 'flat':
        nq = len(q)
        return np.ones(nq)
    elif ff == 'atomic':
        if 'symbol' in site_def['settings']:
            ff_atom = standard_atomic_ff(q,site_def['settings']['symbol'])
        else:
            ff_atom = atomic_ff(q,site_def['settings']['Z'],site_def['parameters'])
        return ff_atom
    elif ff == 'spherical':
        return spherical_ff(q,site_def['parameters']['r'])

def site_ff_squared(q,site_def):
    ff = site_def['form'] 
    if ff == 'guinier_porod':
        p = site_def['parameters']
        ff2_gp = guinier_porod_intensity(q,p['G'],p['rg'],p['D'])
        return ff2_gp 
    elif ff == 'spherical_normal':
        p = site_def['parameters']
        ff2_sph = spherical_normal_intensity(q,p['r0'],p['sigma']) 
        return ff2_sph
    else:
        return site_ff(q,site_def)**2

def standard_atomic_ff(q,atom_symbol):
    pars = atomic_params[atom_symbol]
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
    ff = np.zeros(q.shape)
    x = q*r
    if q[0] == 0:
        ff[0] = 1.
        ff[1:] = 3.*(np.sin(x[1:])-x[1:]*np.cos(x[1:]))*x[1:]**-3
        return ff
    else:
        return 3.*(np.sin(x)-x*np.cos(x))*x**-3

def spherical_normal_intensity(q,r0,sigma,sampling_width=3.5,sampling_step=0.1):  
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
    I : array
        Array of intensity values for all q
    """
    I = np.zeros(q.shape)
    if sigma < 1E-9:
        x = q*r0
        V_r0 = float(4)/3*np.pi*r0**3
        I_0 = V_r0**2
        I = I_0*spherical_ff(q,r0)**2
    else:
        sigma_r = sigma*r0
        dr = sigma_r*sampling_step
        rmin = np.max([r0-sampling_width*sigma_r,dr])
        rmax = r0+sampling_width*sigma_r
        I_0 = 0
        for ri in np.arange(rmin,rmax,dr):
            V_ri = float(4)/3*np.pi*ri**3
            # The normal-distributed density of particles with radius r_i:
            rhoi = 1./(np.sqrt(2*np.pi)*sigma_r)*np.exp(-1*(r0-ri)**2/(2*sigma_r**2))
            I0_i = V_ri**2*rhoi*dr
            I_0 += I0_i
            I += I0_i*spherical_ff(q,ri)**2
    I = I/I_0 
    return I 

def guinier_porod_intensity(q,guinier_factor,rg,porod_exponent):
    """Compute a Guinier-Porod scattering intensity.
    
    Parameters
    ----------
    q : array
        array of q values
    guinier_factor : float
        low-q Guinier prefactor (equal to intensity at q=0)
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


