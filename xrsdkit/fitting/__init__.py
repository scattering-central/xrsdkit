"""Modules for fitting intensity versus q profiles"""
from collections import OrderedDict

import numpy as np
import lmfit

from ..tools import standardize_array

# dict of allowed form factor parameters
#ff_parameters = OrderedDict(
#    all=['occupancy'],
#    spherical=['r'],
#    spherical_normal=['r0','sigma'],
#    guinier_porod=['G','rg','D'],
#    atomic=['symbol','Z','a','b'])

# dict of allowed structure parameters:
#sf_parameters = OrderedDict(
#    all = ['I0'],
#    diffuse = [],
#    disordered = ['hwhm_g','hwhm_l','q_center'],
#    crystalline = ['hwhm_g','hwhm_l'],
#    fcc = ['a'])

all_params = list([
    'I0','occupancy',
    'coordinates',
    'G','rg','D',
    'r', 'r0', 'sigma',
    'v_fraction',
    'hwhm_g','hwhm_l','q_center',
    'a'])

param_defaults = OrderedDict(
    I0 = 1.E-3,
    occupancy = 1.,
    coordinates = 0.,
    G = 1.,
    rg = 10.,
    D = 4.,
    r = 20.,
    r0 = 20.,
    sigma = 0.05,
    v_fraction = 0.1,
    hwhm_g = 1.E-3,
    hwhm_l = 1.E-3,
    q_center = 1.E-1,
    a = 10.)

param_bound_defaults = OrderedDict(
    I0 = (0.,None),
    occupancy = (0.,1.),
    coordinates = (None,None),
    G = (0.,None),
    rg = (1.E-6,None),
    D = (0.,4.),
    r = (1.E-6,None),
    r0 = (1.E-6,None),
    sigma = (0.,0.5),
    v_fraction = (0.,0.7405),
    hwhm_g = (1.E-6,None),
    hwhm_l = (1.E-6,None),
    q_center = (0.,None),
    a = (0.,None))

fixed_param_defaults = OrderedDict(
    I0 = False,
    occupancy = True,
    coordinates = True,
    G = False,
    rg = False,
    D = False,
    r = False,
    r0 = False,
    sigma = False,
    v_fraction = False,
    hwhm_g = False,
    hwhm_l = False,
    q_center = False,
    a = False)

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

def compute_Rsquared(y1,y2):
    """Compute the coefficient of determination.

    Parameters
    ----------
    y1 : array
        an array of floats
    y2 : array
        an array of floats

    Returns
    -------
    Rsquared : float
        coefficient of determination between `y1` and `y2`
    """
    sum_var = np.sum( (y1-np.mean(y1))**2 )
    sum_res = np.sum( (y1-y2)**2 ) 
    return float(1)-float(sum_res)/sum_var

def compute_pearson(y1,y2):
    """Compute the Pearson correlation coefficient.

    Parameters
    ----------
    y1 : array
        an array of floats
    y2 : array
        an array of floats

    Returns
    -------
    pearson_r : float
        Pearson's correlation coefficient between `y1` and `y2`
    """
    y1mean = np.mean(y1)
    y2mean = np.mean(y2)
    y1std = np.std(y1)
    y2std = np.std(y2)
    return np.sum((y1-y1mean)*(y2-y2mean))/(np.sqrt(np.sum((y1-y1mean)**2))*np.sqrt(np.sum((y2-y2mean)**2)))

def compute_chi2(y1,y2,weights=None):
    """Compute sum of difference squared between two arrays.

    Parameters
    ----------
    y1 : array
        an array of floats
    y2 : array
        an array of floats
    weights : array
        array of weights to multiply each element of (`y2`-`y1`)**2 

    Returns
    -------
    chi2 : float
        sum of difference squared between `y1` and `y2`. 
    """
    if weights is None:
        return np.sum( (y1 - y2)**2 )
    else:
        weights = weights / np.sum(weights)
        return np.sum( (y1 - y2)**2*weights )

