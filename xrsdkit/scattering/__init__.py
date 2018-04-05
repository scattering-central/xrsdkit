import numpy as np

from . import form_factors as xrff

# list of allowed form factors:
#form_factor_names = list([
#    'flat',
#    'guinier_porod',
#    'spherical',
#    'spherical_normal',
#    'atomic'])
#

def diffuse_intensity(q,popd,source_wavelength):
    basis = popd['basis']
    #species = basis[list(basis.keys())[0]]
    I0 = 1.
    if 'I0' in popd['parameters']: I0 = popd['parameters']['I0']
    F_q = xrff.compute_ff_squared(q,basis)
    return I0*F_q

#def _specie_count(species_dict):
#    n_distinct = OrderedDict.fromkeys(species_dict)
#    for specie_name,specie_params in species_dict.items():
#        if not isinstance(specie_params,list):
#            n_distinct[specie_name] = 1
#        else:
#            n_distinct[specie_name] = len(specie_params)
#    return n_distinct

