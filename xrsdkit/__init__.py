"""Computation and analysis of scattering and diffraction patterns.

A scattering/diffraction pattern is assumed to represent
one or more populations of scattering objects.
A population is described by a dict with the following entries:

    - 'name' : string population identifier 
        (e.g. 'noise', 'substrate', 'particles')

    - 'structure' : the structure of the population 
        (e.g. 'fcc', 'diffuse', 'condensed'). 

    - 'parameters' : dict describing the structure (lattice parameters, etc)
        as well as any other parameters used in the scattering computation.
        Some of the keys are used for structural parameters:

        - 'a', 'b', 'c' : a, b, and c lattice parameters
        - 'alpha' : angle between b and c lattice vectors
        - 'beta' : angle between a and c lattice vectors
        - 'gamma' : angle between a and b lattice vectors
        - 'N' : number of scatterers (for 'diffuse' structures)
    
        Other keys are used for parameterizing diffraction peaks:

        - 'q_min' : minimum q-value for reciprocal lattice points 
        - 'q_max' : maximum q-value for reciprocal lattice points 
        - 'profile' : 'gaussian', 'lorentzian', or 'voigt' 
        - 'hwhm' : half-width at half max of the diffraction peaks

    - 'basis' : dict containing fractional coordinates (as keys)
        and descriptions of site occupancy (as values).

        - The coordinate (key) is a tuple of three floats.

        - The occupancy is described by a dict containing  
            any number of form factor specifiers (as keys)
            and form factor parameters (as values).

        - Each set of form factor parameters is a dict (or list of dicts)
            containing the parameter names (as keys) and values (as values).
            A list of dicts is used to include 
            multiple scatterers of the same type.

The following structures are currently supported:

    - 'diffuse' : a diffuse (or dilute), 
        non-interfering scattering ensemble.
        This structure has one parameter, 
        the number of scatterers 'N',
        and at least one basis site.

    - 'fcc' : fcc lattice, 
        with one parameter, no angles, 
        and at least one basis site.

The supported form factors and their parameters are:

    - 'flat': a flat form factor for all q
      
      - 'amplitude': amplitude of the flat scattering

    - 'guinier_porod': scatterer populations described 
        by the Guinier-Porod equations.
        The square root of the Guinier-Porod intensity
        is used as the form factor.

      - 'G': Guinier prefactor 
      - 'r_g': radius of gyration 
      - 'D': Porod exponent 

    - 'spherical': solid spheres 

      - 'r': sphere radius (Angstrom) 

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

    - all form factors:

      - 'occupancy': occupancy fraction, used for sites 
        with multiple fractional occupancies

        
For example, a single Guinier-Porod scatterer 
is placed in a 40-Angstrom fcc lattice,
with peaks from q=0.1 to q=1.0 
included in the summation:
fcc_gp_population = dict(
    structure='fcc',
    parameters=dict(
        a=40.,
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

structures = list([
    'diffuse',
    'fcc'])
sf_parameters = OrderedDict(
    peaks = list([
        'hwhm',
        'q_min',
        'q_max',
        'profile']),
    diffuse = ['N'],
    fcc = ['a'])

form_factors = list([
    'flat',
    'guinier_porod',
    'spherical_normal',
    'atomic'])
ff_parameters = OrderedDict(
    general = ['occupancy'],
    diffuse = ['density'], 
    fcc = ['a'],
    flat = ['amplitude'],
    spherical = ['r'],
    guinier_porod = list([
        'G',
        'r_g',
        'D']),
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
        lat = popd['structure']
        if lat == 'diffuse':
            I += scattering.diffuse_intensity(q,popd,source_wavelength)
        elif lat == 'fcc':
            I += diffraction.fcc_intensity(q,popd,source_wavelength)
        else:
            msg = 'structure specification {} is not supported'.format(lat)
            raise ValueError(msg)
    return I

def fcc_crystal(atom_name,a_lat,q_min=None,q_max=None,pk_profile=None,hwhm=None):
    fcc_pop = dict(
        name='fcc_{}'+atom_name,
        structure='fcc',
        parameters={'a':a_lat},
        basis={(0,0,0):{'atomic':{'atom_name':atom_name}}}
        )
    if q_min:
        fcc_pop['parameters']['q_min'] = q_min
    if q_max:
        fcc_pop['parameters']['q_max'] = q_max
    if pk_profile:
        fcc_pop['parameters']['profile'] = pk_profile 
    if hwhm:
        fcc_pop['parameters']['hwhm'] = hwhm 
    return fcc_pop

# TODO: add more convenience constructors for various populations






