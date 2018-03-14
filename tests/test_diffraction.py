from __future__ import print_function

import numpy as np

from xrsdkit import compute_intensity 
from xrsdkit.diffraction import structure_factors,peak_math

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

#def test_fcc_flat():
#    qvals = np.arange(0.01,4.,0.001)
#    pops = dict(
#            lattice='fcc',
#            parameters=dict(
#                a=30.,
#                profile='voigt',
#                hwhm=0.002,
#                q_min=0.04,
#                q_max=2.,
#                ),
#            basis={(0,0,0):dict(
#                flat={'amplitude':1.}
#                )}
#            )
#    Ivals = compute_intensity(qvals,pops,0.8265616)

def test_fcc_sf():
    # take the q value of the (111) sphere
    q_111 = np.sqrt(3)
    pops = dict(
            lattice='fcc',
            parameters=dict(
                a=4.046,
                profile='voigt',
                hwhm=0.002,
                q_min=1.,
                q_max=5.,
                ),
            basis={(0,0,0):dict(
                atomic={'atom_name':'Al'}
                )}
            )

    sf_func = lambda qi,ph,th: structure_factors.fcc_sf(qi,
        np.array([
            qi*np.sin(th)*np.cos(ph),
            qi*np.sin(th)*np.sin(ph),
            qi*np.cos(th)]).reshape(3,1),
        pops)

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

def test_fcc_Al():
    qvals = np.arange(1.,5.,0.001)
    pops = dict(
            lattice='fcc',
            parameters=dict(
                a=4.046,
                profile='voigt',
                hwhm=0.002,
                q_min=1.,
                q_max=5.,
                ),
            basis={(0,0,0):dict(
                atomic={'atom_name':'Al'}
                )}
            )
    Ivals = compute_intensity(qvals,pops,0.8265616)
    sfvals = structure_factors.fcc_sf_spherical_average(qvals,pops)

    #from matplotlib import pyplot as plt
    #plt.figure(1)
    #plt.plot(qvals,Ivals)
    #plt.show()
    
    #plt.figure(2)
    #plt.plot(qvals,sfvals.real,'b')
    #plt.plot(qvals,sfvals.imag,'g')
    #plt.plot(qvals,(sfvals*sfvals.conjugate()).real,'r')





