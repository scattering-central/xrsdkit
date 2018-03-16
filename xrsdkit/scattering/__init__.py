import numpy as np

from . import form_factors as xrff

def diffuse_intensity(q,popd,source_wavelength):
    n_q = len(q)
    I = np.zeros(n_q)
    basis = popd['basis']
    # the basis has only one point.
    # the coordinate is ignored. 
    species = basis[basis.keys()[0]]
    n_species = _specie_count(species)
    if np.sum(n_species.values()) == 1:
        specie_name = species.keys()[0]
        specie_params = species[specie_name]
        if isinstance(specie_params,list):
            specie_params = specie_params[0]
        spff = xrff.compute_ff(q,specie_name,specie_params)
        ff += specie_params['occupancy']*spff
    I = popd['N']*ff**2
    return I

def _specie_count(species_dict):
    n_distinct = OrderedDict.fromkeys(species_dict)
    for specie_name,specie_params in species_dict.items():
        if not isinstance(specie_params,list):
            n_distinct[specie_name] = 1
        else:
            n_distinct[specie_name] = len(specie_params)
    return n_distinct

def spherical_normal_intensity(q,r0,sigma,sampling_width=3.5,sampling_step=0.1):
    ff = spherical_normal_ff(q,r0,sigma,sampling_width,sampling_step)
    return ff**2

def guinier_porod_intensity(q,guinier_factor,r_g,porod_exponent):
    ff = guinier_porod_ff(q,guinier_factor,r_g,porod_exponent)

