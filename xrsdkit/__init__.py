"""Computation and analysis of scattering and diffraction patterns.

A scattering/diffraction pattern is assumed to represent
one or more populations of scattering objects.
A population is described by a dict with the following entries:

    - 'name' : string population identifier 
        (e.g. 'noise', 'substrate', 'particles')

    - 'structure' : the structure of the population 
        (e.g. 'diffuse', 'disordered', 'fcc'). 

    - 'parameters' : dict describing the structure (lattice parameters, etc)
        as well as any other parameters used in the scattering computation.
        Some of the keys are used for structural parameters:

        - 'a', 'b', 'c' : a, b, and c lattice parameters
        - 'alpha' : angle between b and c lattice vectors
        - 'beta' : angle between a and c lattice vectors
        - 'gamma' : angle between a and b lattice vectors
    
        Other keys are used for parameterizing intensities and diffraction peaks:

        - 'I0' : the scattering or diffraction computed for each population 
            is multiplied by this intensity prefactor 
        - 'q_min' : minimum q-value for reciprocal lattice analysis 
        - 'q_max' : maximum q-value for reciprocal lattice analysis 
        - 'profile' : 'gaussian', 'lorentzian', or 'voigt' 
        - 'hwhm_g' : half-width at half max of Gaussian functions 
        - 'hwhm_l' : half-width at half max of Lorentzian functions 
        - 'q_center' : center q-value for describing single 'disordered' peaks

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
        This structure has no parameters, 
        and at least one basis site.

    - 'disordered' : condensed, disordered material, 
        characterized by a single (probably broad) peak,
        with parameters 'profile', 'q_center', 'hwhm_g', 'hwhm_l'.

    - 'fcc' : crystalline fcc lattice,
        defined by one lattice parameter 'a'.
        Peaks (with parameters 'profile', 'hwhm_g', 'hwhm_l')
        are analyzed for reciprocal space vectors
        between parameters 'q_min' and 'q_max'. 

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
fcc_gp_population = dict(
    structure='fcc',
    parameters=dict(
        a=40.,
        q_min=0.1,
        q_max=1.,
        profile='voigt'
        hwhm_g=0.01
        hwhm_l=0.01
        )
    basis=dict(
        (0,0,0)=dict(
            flat={'amplitude':10}
            )
        )
    )
"""
from collections import OrderedDict

import numpy as np

from . import scattering, diffraction

# list of allowed structure specifications
structures = [
    'unidentified',
    'diffuse',
    'disordered',
    'fcc'])

# dict of allowed structure parameters:
sf_parameters = OrderedDict(
    general = ['I0'],
    crystalline = ['profile','hwhm_g','hwhm_l','q_min','q_max'],
    fcc = ['a'])

# list of allowed form factors:
form_factors = list([
    'flat',
    'guinier_porod',
    'spherical',
    'spherical_normal',
    'atomic'])

# list of form factors that can only be used in a diffuse structure:
diffuse_form_factors = list([
    'spherical_normal',
    'guinier_porod'])

# dict of allowed form factor parameters
ff_parameters = OrderedDict(
    general = ['occupancy'],
    flat = ['amplitude'],
    spherical = ['r'],
    spherical_normal = ['r0','sigma'],
    guinier_porod = ['G','r_g','D'],
    atomic = ['symbol','Z','a','b'])

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
        st = popd['structure']
        if st == 'diffuse':
            I += popd['parameters']['I0']\
                * scattering.diffuse_intensity(q,popd,source_wavelength)
        elif st == 'fcc':
            if any([ any([specie_name in diffuse_form_factors 
                for specie_name in specie_dict.keys()])
                for coord,specie_dict in popd['basis'].items()]):
                msg = 'Populations of type {} are currently not supported '\
                    'in crystalline arrangements.'.format(diffuse_form_factors)
                raise ValueError(msg)
            I += popd['parameters']['I0']\
                * diffraction.fcc_intensity(q,popd,source_wavelength)
        elif st == 'disordered':
            profile_name = popd['parameters']['profile']
            q_c = popd['parameters']['q_center']
            line_shape = peak_math.peak_profile(q,q_c,profile_name,popd['parameters'])
            I += popd['parameters']['I0']\
                * diffraction.fcc_intensity(q,popd,source_wavelength)
        else:
            msg = 'structure specification {} is not supported'.format(lat)
            raise ValueError(msg)
    return I

def fcc_crystal(atom_name,a_lat,q_min=None,q_max=None,pk_profile=None,hwhm_g=None,hwhm_l=None):
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
    if hwhm_g:
        fcc_pop['parameters']['hwhm_g'] = hwhm_g
    if hwhm_l:
        fcc_pop['parameters']['hwhm_l'] = hwhm_l
    return fcc_pop

# TODO: add more convenience constructors for various populations






