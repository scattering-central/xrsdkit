"""Modules for fitting intensity versus q profiles"""
import copy

import numpy as np

from ..tools import standardize_array

def fit_I0(q,I,order=4):
    """Find an estimate for I(q=0) by polynomial fitting.
    
    Parameters
    ----------
    q : array
        array of scattering vector magnitudes in 1/Angstrom
    I : array
        array of intensities corresponding to `q`

    Returns
    -------
    I_at_0 : float
        estimate of the intensity at q=0
    p_I0 : array
        polynomial coefficients for the polynomial 
        that was fit to obtain `I_at_0` (numpy format)
    """
    #TODO (maybe): add a sign constraint such that I(q=0) > 0?
    q_s,q_mean,q_std = standardize_array(q)
    I_s,I_mean,I_std = standardize_array(I)
    p_I0 = fit_with_slope_constraint(q_s,I_s,-1*q_mean/q_std,0,order) 
    I_at_0 = np.polyval(p_I0,-1*q_mean/q_std)*I_std+I_mean
    return I_at_0,p_I0

def fit_with_slope_constraint(q,I,q_cons,dIdq_cons,order,weights=None):
    """Fit scattering data to a polynomial with one slope constraint.

    This is performed by forming a Lagrangian 
    from a quadratic cost function 
    and the Lagrange-multiplied constraint function.
    Inputs q and I are not standardized in this function,
    so they should be standardized beforehand 
    if standardized fitting is desired.
    
    TODO: Document cost function, constraints, Lagrangian.

    Parameters
    ----------
    q : array
        array of scattering vector magnitudes in 1/Angstrom
    I : array
        array of intensities corresponding to `q`
    q_cons : float
        q-value at which a slope constraint will be enforced-
        because of the form of the Lagrangian,
        this constraint cannot be placed at exactly zero
        (it would result in indefinite matrix elements)
    dIdq_cons : float
        slope (dI/dq) that will be enforced at `q_cons`
    order : int
        order of the polynomial to fit
    weights : array
        array of weights for the fitting of `I`

    Returns
    -------
    p_fit : array
        polynomial coefficients for the fit of I(q) (numpy format)
    """
    Ap = np.zeros( (order+1,order+1),dtype=float )
    b = np.zeros(order+1,dtype=float)
    # TODO: vectorize the construction of Ap
    for i in range(0,order):
        for j in range(0,order):
            Ap[i,j] = np.sum( q**j * q**i )
        Ap[i,order] = -1*i*q_cons**(i-1)
    for j in range(0,order):
        Ap[order,j] = j*q_cons**(j-1)
        b[j] = np.sum(I*q**j)
    b[order] = dIdq_cons
    p_fit = np.linalg.solve(Ap,b) 
    p_fit = p_fit[:-1]  # throw away Lagrange multiplier term 
    p_fit = p_fit[::-1] # reverse coefs to get np.polyfit format
    return p_fit

def flatten_params(populations):
    pd = {} 
    for pop_name,popd in populations.items():
        if 'parameters' in popd:
            for param_name,param_val in popd['parameters'].items():
                pd[pop_name+'__'+param_name] = copy.deepcopy(param_val)
        if 'basis' in popd:
            for site_name, site_def in popd['basis'].items():
                if 'coordinates' in site_def:
                    pd[pop_name+'__'+site_name+'__coordinate_0'] = copy.deepcopy(site_def['coordinates'][0])
                    pd[pop_name+'__'+site_name+'__coordinate_1'] = copy.deepcopy(site_def['coordinates'][1])
                    pd[pop_name+'__'+site_name+'__coordinate_2'] = copy.deepcopy(site_def['coordinates'][2])
                if 'parameters' in site_def:
                    for ff_param_name, ff_param_val in site_def['parameters'].items():
                        pd[pop_name+'__'+site_name+'__'+ff_param_name] = \
                        copy.deepcopy(ff_param_val)
    return pd

def unflatten_params(flat_params):
    pd = {} 
    for pkey,pval in flat_params.items():
        ks = pkey.split('__')
        kdepth = len(ks)
        pop_name = ks[0]
        if not pop_name in pd:
            pd[pop_name] = {} 
        if kdepth == 2: 
            # a structure parameter 
            if not 'parameters' in pd[pop_name]:
                pd[pop_name]['parameters'] = {} 
            param_name = ks[1]
            pd[pop_name]['parameters'][param_name] = copy.deepcopy(pval)
        else:
            # a basis or form factor parameter
            site_name = ks[1]
            if not 'basis' in pd[pop_name]:
                pd[pop_name]['basis'] = {} 
            if not site_name in pd[pop_name]['basis']:
                pd[pop_name]['basis'][site_name] = {}
            if ks[2] in ['coordinate_0','coordinate_1','coordinate_2']:
                # a coordinate
                if not 'coordinates' in pd[pop_name]['basis'][site_name]:
                    pd[pop_name]['basis'][site_name]['coordinates'] = [None,None,None]
                coord_idx = int(ks[2][-1])
                pd[pop_name]['basis'][site_name]['coordinates'][coord_idx] = copy.deepcopy(pval) 
            else:
                # a parameter for a form factor
                if not 'parameters' in pd[pop_name]['basis'][site_name]:
                    pd[pop_name]['basis'][site_name]['parameters'] = {} 
                param_name = ks[2]
                pd[pop_name]['basis'][site_name]['parameters'][param_name] = copy.deepcopy(pval)
    return pd


