from __future__ import print_function

import numpy as np

from xrsdkit import compute_intensity 
from xrsdkit.diffraction import structure_factors,peak_math
    
fcc_Al = {'Al':dict(
    structure='fcc',
    diffraction_setup=dict(
        q_min=1.,
        q_max=5.,
        profile='voigt'
        ),
    parameters=dict(
        a=4.046,
        hwhm_g=0.002,
        hwhm_l=0.0018
        ),
    basis={(0,0,0):{'atomic':{'symbol':'Al'}}}
    )}

def test_gaussian():
    qvals = np.arange(0.01,4.,0.01)
    for hwhm in [0.01,0.03,0.05,0.1]:
        g = peak_math.gaussian(qvals-2.,hwhm)
        intg = np.sum(0.01*g)
        print('approx. integral of gaussian with hwhm {}: {}'\
            .format(hwhm,intg))

def test_lorentzian():
    qvals = np.arange(0.01,4.,0.01)
    for hwhm in [0.01,0.03,0.05,0.1]:
        l = peak_math.lorentzian(qvals-2.,hwhm)
        intl = np.sum(0.01*l)
        print('approx. integral of lorentzian with hwhm {}: {}'\
            .format(hwhm,intl))

def test_voigt():
    qvals = np.arange(0.01,4.,0.01)
    for hwhm_g in [0.01,0.03,0.05,0.1]:
        for hwhm_l in [0.01,0.03,0.05,0.1]:
            v = peak_math.voigt(qvals-2.0,hwhm_g,hwhm_l)
            intv = np.sum(0.01*v)
            print('approx. integral of voigt '\
                'with gaussian hwhm {} and lorentzian hwhm {}: {}'\
                .format(hwhm_g,hwhm_l,intv))

def test_fcc_sf():
    # take the q value of the (111) sphere
    q_111 = np.sqrt(3)

    sf_func = lambda qi,ph,th: structure_factors.fcc_sf(qi,
        np.array([
            qi*np.sin(th)*np.cos(ph),
            qi*np.sin(th)*np.sin(ph),
            qi*np.cos(th)]).reshape(3,1),
        fcc_Al['Al']['basis'])

    ph,th = np.meshgrid(np.arange(0,np.pi,0.1),np.arange(0,2*np.pi,0.1))
    sf = np.zeros(ph.shape,dtype=complex)
    for kk in range(ph.shape[0]):
        for ll in range(ph.shape[1]):
            sf[kk,ll] = sf_func(q_111,ph[kk,ll],th[kk,ll])

    ph_111 = []
    th_111 = []
    for hkl in [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
                (-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1)]:
        pphh = np.arctan(hkl[1]/hkl[0])
        #tthh = np.arctan(np.sqrt(2)/hkl[2])
        tthh = np.arccos(hkl[2]/np.sqrt(3))
        if pphh < 0:
            pphh = pphh + np.pi
        if hkl[1] < 0:
            tthh = 2*np.pi - tthh
        #if tthh < 0:
        #    tthh = tthh + 2*np.pi
        ph_111.append(pphh)
        th_111.append(tthh)

    #from matplotlib import pyplot as plt
    #plt.figure(1)
    #sfcont_real = plt.contourf(ph,th,sf.real)
    #plt.plot(ph_111,th_111,'ko')
    #plt.colorbar(sfcont_real)
    #plt.figure(2)
    #sfcont_imag = plt.contourf(ph,th,sf.imag)
    #plt.plot(ph_111,th_111,'ko')
    #plt.colorbar(sfcont_imag)
    #plt.show()

#def test_fcc_spherical_average_sf():
#    qvals = np.arange(1.,5.,0.001)
#    sf_avg = structure_factors.fcc_sf_spherical_average(qvals,fcc_Al)
    #from matplotlib import pyplot as plt
    #plt.figure(3)
    #plt.plot(q,sf_avg.real,'g')
    #plt.plot(q,sf_avg.imag,'r')
    #plt.plot(qvals,(sf_avg*sf_avg.conjugate()).real,'r')
    #plt.legend(['real','imaginary','magnitude'])
    #plt.show()

def test_fcc_Al():
    qvals = np.arange(1.,5.,0.001)
    Ivals = compute_intensity(qvals,fcc_Al,0.8265616)

    #from matplotlib import pyplot as plt
    #plt.figure(4)
    #plt.plot(qvals,Ivals)
    #plt.show()
    



