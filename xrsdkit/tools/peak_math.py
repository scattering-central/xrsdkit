import numpy as np
from collections import OrderedDict

from scipy.special import wofz
from scipy.optimize import minimize as scimin

def peak_profile(q,q_pk,profile_name,params):
    if profile_name == 'voigt':
        hwhm_g = params['hwhm_g']
        hwhm_l = params['hwhm_l']
        line_shape = voigt(q-q_pk,hwhm_g,hwhm_l)
    elif profile_name == 'gaussian':
        hwhm_g = params['hwhm_g']
        line_shape = gaussian(q-q_pk,hwhm_g)
    elif profile_name == 'lorentzian':
        hwhm_l = params['hwhm_l']
        line_shape = lorentzian(q-q_pk,hwhm_l)
    return line_shape

def gaussian(x, hwhm_g):
    """
    gaussian (normal) distribution at points x, 
    center 0, half width at half max hwhm_g
    """
    return np.sqrt(np.log(2)/np.pi) / hwhm_g * np.exp(-(x/hwhm_g)**2 * np.log(2))

def lorentzian(x, hwhm_l):
    """
    lorentzian (cauchy) distribution at points x, 
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
    gamma = hwhm_l
    v = np.real(wofz((x+1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)
    return v 

def peaks_by_window(x,y,w=10,thr=0.):
    """Find peaks in x,y data by a window-scanning.

    TODO: introduce window shapes and make use of the x-values.

    Parameters
    ----------
    x : array
        array of x-axis values
    y : array
        array of y-axis values
    w : int
        half-width of window- each point is analyzed
        with the help of this many points in either direction
    thr : float
        for a given point xi,yi, if yi is the maximum within the window,
        the peak is flagged if yi/mean(y_window)-1. > thr

    Returns
    -------
    pk_idx : list of int
        list of indices where peaks were found
    pk_confidence : list of float
        confidence in peak labeling for each peak found 
    """
    pk_idx = []
    pk_confidence = []
    for idx in range(w,len(y)-w-1):
        pkflag = False
        ywin = y[idx-w:idx+w+1]
        if np.argmax(ywin) == w:
            conf = ywin[w]/np.mean(ywin)-1.
            pkflag = conf > thr
        if pkflag:
            pk_idx.append(idx)
            pk_confidence.append(conf)

    #from matplotlib import pyplot as plt
    #plt.figure(2)
    #plt.plot(x,y)
    #for ipk,cpk in zip(pk_idx,pk_confidence):
    #    qpk = x[ipk]
    #    Ipk = y[ipk]
    #    print('q: {}, I: {}, confidence: {}'.format(qpk,Ipk,cpk))
    #    plt.plot(qpk,Ipk,'ro')
    #plt.show()

    return pk_idx,pk_confidence
    
def peaks_by_window(x,y,w=10,thr=0.):
    """Find peaks in x,y data by a window-scanning.

    TODO: introduce window shapes and make use of the x-values.

    Parameters
    ----------
    x : array
        array of x-axis values
    y : array
        array of y-axis values
    w : int
        half-width of window- each point is analyzed
        with the help of this many points in either direction
    thr : float
        for a given point xi,yi, if yi is the maximum within the window,
        the peak is flagged if yi/mean(y_window)-1. > thr

    Returns
    -------
    pk_idx : list of int
        list of indices where peaks were found
    pk_confidence : list of float
        confidence in peak labeling for each peak found 
    """
    pk_idx = []
    pk_confidence = []
    for idx in range(w,len(y)-w-1):
        pkflag = False
        ywin = y[idx-w:idx+w+1]
        if np.argmax(ywin) == w:
            conf = ywin[w]/np.mean(ywin)-1.
            pkflag = conf > thr
        if pkflag:
            pk_idx.append(idx)
            pk_confidence.append(conf)

    #from matplotlib import pyplot as plt
    #plt.figure(2)
    #plt.plot(x,y)
    #for ipk,cpk in zip(pk_idx,pk_confidence):
    #    qpk = x[ipk]
    #    Ipk = y[ipk]
    #    print('q: {}, I: {}, confidence: {}'.format(qpk,Ipk,cpk))
    #    plt.plot(qpk,Ipk,'ro')
    #plt.show()

    return pk_idx,pk_confidence

