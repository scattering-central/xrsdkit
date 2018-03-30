import numpy as np

from . import form_factors as xrff

# list of allowed form factors:
form_factor_names = list([
    'flat',
    'guinier_porod',
    'spherical',
    'spherical_normal',
    'atomic'])

# list of form factors that can only be used in a diffuse structure:
diffuse_form_factor_names = list([
    'spherical_normal',
    'guinier_porod'])

def diffuse_intensity(q,popd,source_wavelength):
    n_q = len(q)
    I = np.zeros(n_q)
    basis = popd['basis']
    #species = basis[list(basis.keys())[0]]
    I0 = 1.
    if 'I0' in popd['parameters']: I0 = popd['parameters']['I0']
    for site_name, site_items in basis.items():
        for site_item_name, site_item in site_items.items():
            if not isinstance(site_item,list):
                site_item = [site_item]
            if site_item_name in diffuse_form_factor_names:
                for p in site_item:
                    I += I0 * xrff.compute_ff_squared(q,site_item_name,p)
            else:
                for p in site_item:
                    occ = 1.
                    if 'occupancy' in p: occ = p['occupancy']
                    I += I0 * (occ*xrff.compute_ff(q,site_item_name,p))**2
    return I

#def _specie_count(species_dict):
#    n_distinct = OrderedDict.fromkeys(species_dict)
#    for specie_name,specie_params in species_dict.items():
#        if not isinstance(specie_params,list):
#            n_distinct[specie_name] = 1
#        else:
#            n_distinct[specie_name] = len(specie_params)
#    return n_distinct

