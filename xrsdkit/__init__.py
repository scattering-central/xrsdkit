"""Computation and analysis of scattering and diffraction patterns.

A scattering/diffraction pattern is assumed to represent
one or more populations of scattering objects.
The populations are described by a dict,
where each population has a name (key)
and a sub-dict of parameters (value).
Each population sub-dict should have the following entries: 

    - 'structure' : the structure of the population 
        (e.g. 'diffuse', 'disordered', 'fcc'). 

    - 'settings' : dict of parameters defining
        the computational treatment of the population,
        such as peak profile specifications and 
        the q-limits for reciprocal space analysis.

        - 'q_min' : minimum q-value for reciprocal lattice analysis 
        - 'q_max' : maximum q-value for reciprocal lattice analysis 
        - 'profile' : 'gaussian', 'lorentzian', or 'voigt' 

    - 'parameters' : dict describing the structure (lattice parameters, etc)
        as well as any other scalar parameters used for the computation.
        Some keys are used for parameterizing intensities and diffraction peaks:

        - 'I0' : the scattering or diffraction computed for each population 
            is multiplied by this intensity prefactor,
            assumed equal to 1 if not provided
        - 'hwhm_g' : half-width at half max of Gaussian functions 
        - 'hwhm_l' : half-width at half max of Lorentzian functions 
        - 'q_center' : center q-value for describing single 'disordered' peaks
    
        Other keys are used for structural parameters:

        - 'a', 'b', 'c' : a, b, and c lattice parameters
        - 'alpha' : angle between b and c lattice vectors
        - 'beta' : angle between a and c lattice vectors
        - 'gamma' : angle between a and b lattice vectors

    - 'basis' : dict containing site names (as keys)
        and dicts specifying site location and content (as values).
        The site content dicts are structured as:

        - 'coordinates' : list of three floating point numbers,
            the fractional coordinates relative to a lattice site.

        - The remaining entries are form factor specifiers,
            referring to dicts or lists of dicts 
            containing parameter names and values for that form factor.
            A list of dicts is used to specify 
            multiple scatterers of the same type,
            e.g. for implementing fractional occupancies.

The following structures are currently supported:

    - 'diffuse' : a diffuse (or dilute), 
        non-interfering scattering ensemble.
        This structure has no parameters, 
        and at least one basis site.

    - 'disordered' : condensed, disordered material, 
        characterized by a single (probably broad) peak,
        defined by a 'profile' setting and
        parameters 'q_center', 'hwhm_g', and 'hwhm_l'.

    - 'fcc' : crystalline fcc lattice,
        defined by one lattice parameter 'a'.
        Peaks computed for this population respect
        the settings 'profile', 'q_min', and 'q_max',
        with parameters 'hwhm_g' and 'hwhm_l'.

The supported form factors and their parameters are:

    - 'flat': a flat form factor for all q,
        implemented for simulating a noise floor.
  
      - 'amplitude': amplitude of the flat scattering

    - 'guinier_porod': scatterer populations described 
        by the Guinier-Porod equations.
        This is currently only supported for 'diffuse' structures.

      - 'G': Guinier prefactor 
      - 'r_g': radius of gyration 
      - 'D': Porod exponent 

    - 'spherical': solid spheres 

      - 'r': sphere radius (Angstrom) 

    - 'spherical_normal': solid spheres 
        with a normal size distribution.
        This is currently only supported for 'diffuse' structures.

      - 'r0': mean sphere radius (Angstrom) 
      - 'sigma': fractional standard deviation of radius

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

      - 'occupancy': occupancy fraction, used for basis sites 
        with multiple fractional occupancies.
        If a single specie is specified for a basis site, 
        an occupancy of 1 is assumed.

For example, a single flat scatterer 
is placed in a 40-Angstrom fcc lattice,
with peaks from q=0.1 to q=1.0 
included in the summation:

my_populations = dict(
    my_fcc_population = dict(
        structure='fcc',
        settings=dict(
            q_min=0.1,
            q_max=1.,
            profile='voigt'
            ),
        parameters=dict(
            a=40.,
            hwhm_g=0.01,
            hwhm_l=0.01
            ),
        basis=dict(
            my_flat_scatterer=dict(
                coordinates=[0,0,0],
                flat={'amplitude':10}
                )
            )
        )
    )
"""
from collections import OrderedDict

import numpy as np

from .scattering import \
    form_factor_names, \
    diffuse_form_factor_names, \
    diffuse_intensity
from .diffraction import \
    crystalline_structure_names, \
    fcc_intensity

# list of allowed structure specifications
structure_names = list([
    'unidentified',
    'diffuse',
    'disordered',
    'fcc'])



def compute_intensity(q,populations,source_wavelength):
    """Compute scattering/diffraction intensity for some `q` values.

    TODO: Document the equation.

    Parameters
    ----------
    q : array
        Array of q values at which intensities will be computed.
    populations : dict
        Each entry in the dict describes a population of scatterers.
        See the module documentation for the dict specifications. 
    source_wavelength : float 
        Wavelength of radiation source in Angstroms

    Returns
    ------- 
    I : array
        Array of scattering intensities for each of the input q values
    """
    n_q = len(q)
    I = np.zeros(n_q)
    for pop_name,popd in populations.items():
        st = popd['structure']
        if st == 'diffuse':
            I += scattering.diffuse_intensity(q,popd,source_wavelength)
        elif st == 'fcc':
            if any([ any([specie_name in diffuse_form_factor_names 
                for specie_name in specie_dict.keys()])
                for coord,specie_dict in popd['basis'].items()]):
                msg = 'Populations of type {} are currently not supported '\
                    'in crystalline arrangements.'.format(diffuse_form_factor_names)
                raise ValueError(msg)
            I += diffraction.fcc_intensity(q,popd,source_wavelength)
        elif st == 'disordered':
            I0 = 1.
            if 'I0' in popd['parameters']: I0 = popd['parameters']['I0']
            profile_name = popd['settings']['profile']
            q_c = popd['parameters']['q_center']
            I += I0 * peak_math.peak_profile(q,q_c,profile_name,popd['parameters'])
        else:
            msg = 'structure specification {} is not supported'.format(st)
            raise ValueError(msg)
    return I

def fcc_crystal(atom_symbol,a_lat,q_min=None,q_max=None,pk_profile=None,hwhm_g=None,hwhm_l=None):
    fcc_pop = dict(
        name='fcc_{}'+atom_symbol,
        structure='fcc',
        settings={},
        parameters={'a':a_lat},
        basis={'{}_atom'.format(atom_symbol):dict(
            coordinates=[0,0,0],
            atomic={'symbol':atom_symbol}
            )}
        )
    if q_min:
        fcc_pop['settings']['q_min'] = q_min
    if q_max:
        fcc_pop['settings']['q_max'] = q_max
    if pk_profile:
        fcc_pop['settings']['profile'] = pk_profile 
    if hwhm_g:
        fcc_pop['parameters']['hwhm_g'] = hwhm_g
    if hwhm_l:
        fcc_pop['parameters']['hwhm_l'] = hwhm_l
    return fcc_pop

# TODO: add more convenience constructors for various populations






