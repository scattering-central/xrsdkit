from collections import OrderedDict

import yaml
import os

class SaxsRegressor(object):
    """A set of regression models to be used on SAXS spectra"""

    def __init__(self,yml_file=None):
        pass

    # helper function - to set parametrs for scalers and models
    def set_param(self, m_s, param):
        for k, v in param.items():
            if isinstance(v, list):
                setattr(m_s, k, np.array(v))
            else:
                setattr(m_s, k, v)

    def predict_params(self,populations,features):
        pass

def parameterize_spectrum(q_I,populations):
    """Determine scattering equation parameters for a given spectrum.

    Parameters
    ----------
    q_I : array
        n-by-2 array of q (scattering vector in 1/A) and I (intensity)
    populations : dict
        dictionary of populations, similar to input of saxs_math.compute_saxs()

    Returns
    -------
    params : dict
        dict of scattering equation parameters
        corresponding to `populations`,
        similar to the input of compute_saxs().

    See Also
    --------
    compute_saxs : computes saxs intensity from `populations` and `params`
    """
    d = OrderedDict()
    if bool(populations['unidentified']):
        warnings.warn('{}: attempting to parameterize '\
        'unidentified scattering- aborting.'.format(__name__))
        return d 
    if bool(flags['diffraction_peaks']):
        warnings.warn('{}: diffraction peak parameterization '\
        'is not supported yet: aborting.'.format(__name__))
        return d

    q = q_I[:,0]
    n_q = len(q)
    I = q_I[:,1]
    I_nz = np.invert((I<=0))
    n_gp = populations['guinier_porod']
    n_sph = populations['spherical_normal']

    # PART 1: Get a number for I(q=0)
    # NOTE: This is not designed to handle peaks.
    # TODO: Fit a sensible low-q region
    # when q_Imax is not at low q
    # (i.e. when there are high-q peaks).
    q_Imax = q[np.argmax(I)]
    idx_fit = (q>=q_Imax)&(q<2.*q_Imax)
    if q_Imax > q[int(n_q/2)]:
        idx_fit = (q<q[int(n_q/2)])
    q_fit = q[idx_fit]
    I_fit = I[idx_fit]
    qs_fit = (q_fit-np.mean(q_fit))/np.std(q_fit)
    # attempt a high-order constrained fit
    I_at_0,pI = fit_I0(q_fit,I_fit,4)
    # check the SNR
    I_sig = np.polyval(pI,qs_fit)*np.std(I_fit)+np.mean(I_fit)
    I_bg = I_fit-I_sig
    snr = np.mean(I_fit)/np.std(I_bg)
    if snr < 100:
        # if fit is noisy, try constrained fit on third order
        I_at_0,pI = fit_I0(q_fit,I_fit,3)

    # PART 2: Estimate parameters for flagged populations
    #TODO: add parameters for diffraction peaks
    #TODO: add parameters for non-spherical form factors
    #if n_sph:
    #    #r0_sphere, sigma_sphere = spherical_normal_heuristics(
    #    #    np.array(zip(q,I)),I_at_0)
    #    #d['r0_sphere'] = r0_sphere
    #    #d['sigma_sphere'] = sigma_sphere
    #if n_gp:
    #    rg_pre, G_pre = precursor_heuristics(
    #        np.array(zip(q,I)))
    #    I_pre = guinier_porod(q,rg_pre,4,G_pre)
    #    d['rg_precursor'] = rg_pre 
    #    d['G_precursor'] = G_pre 

    #if n_gp+n_sph > 1:
    #    Ifunc = lambda x: x[0]*np.ones(n_q) \
    #        + guinier_porod(q,rg_pre,4,x[1]) \
    #        + x[2]*spherical_normal_saxs(q,r0_sphere,sigma_sphere) 
    #    I_error = lambda x: np.sum( (np.log(Ifunc(x)[I_nz]) - np.log(I[I_nz]))**2 )
    #    # NOTE: this fit is constrained to hold I(q=0) constant
    #    res = scipimin(I_error,[0.0,G_pre,I_at_0],
    #        bounds=[(0.0,None),(1E-6,None),(1E-6,None)],
    #        constraints=[{'type':'eq','fun':lambda x:np.sum(x)-I_at_0}])
    #    d['I0_floor'] = res.x[0]
    #    d['G_precursor'] = res.x[1]
    #    d['I0_sphere'] = res.x[2]
    #elif f_pre:
    #    d['I0_floor'] = 0
    #elif f_form:
    #    I0_floor = np.mean(I[int(n_q*0.9):])
    #    d['I0_floor'] = I0_floor 
    #    d['I0_sphere'] = I_at_0 - I0_floor

    return d


