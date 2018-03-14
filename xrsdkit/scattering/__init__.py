import numpy as np

from xrsdkit.scattering import form_factors as xrff

def dilute_intensity(q,popd,source_wavelength):
    n_q = len(q)
    I = np.zeros(n_q)
    basis = popd['basis']
    d_factor = popd['parameters']['density']
    # the intensity is the sum of the form factors, squared
    for coords,ffspec in basis.items():
        ff = xrff.compute_ff(q,ffspec)
        I += ff**2
    return d_factor*I


