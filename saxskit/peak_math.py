import numpy as np
from collections import OrderedDict

from scipy.special import wofz
from scipy.optimize import minimize as scimin

# get y value nearest xpk guess, use it to guess a scaling factor
#ypk = y[np.argmin((x-xpk)**2)]
#scl = ypk / self.voigt(0, hwhm, hwhm)
#xpk, hwhm_g, hwhm_l, scl = self.solve_voigt(x,y,xpk,hwhm,hwhm,scl)
#y_voigt = self.voigt(x,hwhm_g,hwhm_l)

def solve_voigt(x, y, xc, hwhm_g, hwhm_l, scl):
    """iteratively minimize an objective to fit x, y curve to a voigt profile"""
    res = scimin(partial(self.hann_voigt_fit,x,y),(xc,hwhm_g,hwhm_l,scl))

def hann_voigt_fit(x, y, xc, hwhm_g, hwhm_l, scl):
    # estimate hwhm of voigt
    # hwhm estimation params from https://en.wikipedia.org/wiki/Voigt_profile
    phi = hwhm_l / hwhm_g
    c0 = 2.0056; c1 = 1.0593 
    hwhm_voigt = hwhm_g * (1 - c0*c1 + np.sqrt(phi**2 + 2*c1*phi +c0**2*c1**2))
    # x,y values in the window region
    i_win = np.array([i for i in range(len(y)) if x[i] > xc-hwhm_voigt and x[i] < xc+hwhm_voigt])
    y_win = np.array([y[i] for i in i_win])
    x_win = np.array([x[i] for i in i_win])
    n_win = len(i_win)
    # window weights
    w_win = 0.5 * (1 - np.cos(2*np.pi*np.arange(n_win)/(n_win-1)) )
    # voigt profile in window, scaled by scl
    y_voigt = scl*self.voigt(x_win-xc,hwhm_g,hwhm_l)
    return np.sum(w_win * (y_voigt - y_win)**2)

def gaussian(x, hwhm_g):
    """
    gaussian distribution at points x, 
    center 0, half width at half max hwhm_g
    """
    return np.sqrt(np.log(2)/np.pi) / hwhm_g * np.exp(-(x/hwhm_g)**2 * np.log(2))

def lorentzian(x, hwhm_l):
    """
    lorentzian distribution at points x, 
    center 0, half width at half max hwhm_l
    """
    return hwhm_l / np.pi / (x**2+hwhm_l**2)

def voigt(x, hwhm_g, hwhm_l):
    """
    voigt distribution resulting from convolution 
    of a gaussian with hwhm hwhm_g 
    and a lorentzian with hwhm hwhm_l
    """
    sigma = hwhm_g / np.sqrt(2 * np.log(2))
    return np.real(wofz((x+1j*hwhm_l)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)


