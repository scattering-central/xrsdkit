"""Computation and analysis of scattering and diffraction patterns.

A scattering/diffraction pattern is assumed to represent
one or more populations of scattering objects.
A population is described by a dict with the following entries:

    - 'lattice' : string indicating the lattice (e.g. 'fcc', 'dilute'). 

    - 'parameters' : dict describing the lattice parameters
        as well as any other parameters used in the scattering computation.
        Some of the keys are expected to refer to standard lattice notations:

        - 'a', 'b', 'c' : a, b, and c lattice parameters
        - 'alpha' : angle between b and c lattice vectors
        - 'beta' : angle between a and c lattice vectors
        - 'gamma' : angle between a and b lattice vectors
    
        Other keys are reserved for parameterizing diffraction peaks:

        - 'q_min' : minimum q-value for reciprocal lattice points 
        - 'q_max' : maximum q-value for reciprocal lattice points 
        - 'profile' : 'gaussian', 'lorentzian', or 'voigt' 
        - 'hwhm' : half-width at half max of the diffraction peaks

    - 'basis' : dict containing fractional coordinates (as keys)
        and descriptions of site occupancy (as values).

        - The coordinate (key) is a tuple of three floats.

        - The occupancy is described by a dict containing  
            any number of form factor specifiers (as keys)
            and corresponding form factor parameters (as values).

        - Each set of form factor parameters is a dict (or list of dicts)
            containing the parameter names (as keys) and values (as values).
            If a list of dicts is given, 
            the contributions from the scatterer populations 
            described in the dicts will be summed in computing .

The following lattices are currently supported:

    - 'dilute' : a dilute, non-interacting ensemble,
        with no structure factor applied to the scattering.
        This lattice has no parameters, no angles, 
        and at least one basis site.

    - 'fcc' : fcc lattice, with one parameter, no angles, 
        and at least one basis site.


The 'basis' of the lattice maps any number of fractional coordinates
to a list of dicts describing the populations contributing to the form factor at that site.
The supported populations and associated parameters are:

    - 'flat': a flat form factor for all q
      
      - 'amplitude': amplitude of the flat scattering

    - 'guinier_porod': scatterer populations described 
        by the Guinier-Porod equations

      - 'G': Guinier prefactor 
      - 'r_g': radius of gyration 
      - 'D': Porod exponent 

    - 'spherical_normal': populations of spheres 
        with normal (Gaussian) size distribution 

      - 'sld_contrast': scattering length density contrast (Angstrom**2) 
      - 'r0': mean sphere size (Angstrom) 
      - 'sigma': fractional standard deviation of sphere size 

    - 'atomic': atomic form factors described by
        ff = Z - 41.78214 * s**2 * sum_i(a_i*exp(-b_i*s**2)),
        where Z is the atomic number,
        s = sin(theta)/lambda,
        and a_i, b_i are the form factor parameters.

      - 'symbol': Atomic symbol (as on the periodic table),
        for using the standard scattering parameters
        (see atomic_scattering_parameters.yaml).
        Atomic sites outside the standard parameter set
        must provide values for atomic number (Z),
        and up to four pairs of scattering parameters (a, b).
      - 'Z': atomic number
      - 'a': list of a coefficients 
      - 'b': list of b coefficients 
        
For example, a single Guinier-Porod scatterer 
is placed in a 4-Angstrom fcc lattice,
with peaks from q=0.1 to q=1.0 
included in the structure factor:
fcc_gp_population = dict(
    lattice='fcc',
    parameters=dict(
        a=4.0,
        q_min=0.1,
        q_max=1.,
        hwhm=0.01
        )
    basis=dict(
        (0,0,0)=dict(
            guinier_porod=dict(
                G = 5.E3,
                r_g = 15.,
                D = 4.
                )   
            )
        )
    )

The diffracted intensity would be obtained by:
I_array = compute_intensity(q_array,fcc_gp_population)
"""
from collections import OrderedDict

import numpy as np

from . import scattering, diffraction

lattices = list([
    'dilute',
    'fcc'])
form_factors = list([
    'flat',
    'guinier_porod',
    'spherical_normal',
    'atomic'])
parameters = OrderedDict(
    dilute = ['density'], 
    fcc = ['a'],
    flat = ['amplitude'],
    guinier_porod = list([
        'G',
        'r_g',
        'D']),
    spherical_normal = list([
        'sld_contrast',
        'r0',
        'sigma']),
    atomic = list([
        'atom_name',
        'Z',
        'a',
        'b']))

def compute_intensity(q,populations,source_wavelength):
    """Compute scattering/diffraction intensity for some `q` values.

    TODO: Document the equation.

    Parameters
    ----------
    q : array
        Array of q values at which intensities will be computed.
    populations : dict or list of dict
        Each dict in the list describes a population of scatterers.
        See the module documentation for the dict specifications. 
    source_wavelength : float 
        Wavelength of radiation source in Angstroms

    Returns
    ------- 
    I : array
        Array of scattering intensities for each of the input q values
    """
    if not isinstance(populations,list):
        populations = [populations]
    n_q = len(q)
    I = np.zeros(n_q)
    for popd in populations:
        lat = popd['lattice']
        if lat == 'dilute':
            I += scattering.dilute_intensity(q,popd,source_wavelength)
        elif lat == 'fcc':
            I += diffraction.fcc_intensity(q,popd,source_wavelength)
        else:
            msg = 'lattice specification {} is not supported'.format(lat)
            raise ValueError(msg)
    return I

