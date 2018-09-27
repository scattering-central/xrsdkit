from __future__ import print_function

import numpy as np

from xrsdkit.scattering import structure_factors as xrsf
from xrsdkit.tools import peak_math
from xrsdkit.system import System, Population, Specie
   
Al_atom_dict = dict(form='atomic',settings={'symbol':'Al'})

fcc_Al = Population('crystalline',
    settings={'lattice':'fcc','q_max':5.},
    parameters=dict(
        a={'value':4.046},
        hwhm_g={'value':0.002},
        hwhm_l={'value':0.0018}
        ),
    basis={'Al':Al_atom_dict}
    )

fcc_Al_system = System({'fcc_Al':fcc_Al.to_dict()})

glassy_Al = Population('disordered',
    settings={'interaction':'hard_spheres'},
    parameters=dict(
        r_hard={'value':4.046*np.sqrt(2)/4},
        v_fraction={'value':0.6},
        I0={'value':1.E5}
        ),
    basis={'Al':Al_atom_dict}
    )

glassy_Al_system = System({'glassy_Al':glassy_Al.to_dict()})

mixed_Al_system = System({'glassy_Al':glassy_Al.to_dict(),'fcc_Al':fcc_Al.to_dict()})

def test_Al_scattering():
    qvals = np.arange(1.,5.,0.001)
    I_fcc = fcc_Al_system.compute_intensity(qvals,0.8265616)
    I_gls = glassy_Al_system.compute_intensity(qvals,0.8265616)
    I_mxd = mixed_Al_system.compute_intensity(qvals,0.8265616)

    #from matplotlib import pyplot as plt
    #plt.figure(4)
    #plt.plot(qvals,I_fcc)
    #plt.plot(qvals,I_gls)
    #plt.plot(qvals,I_mxd)
    #plt.legend(['fcc','glassy','mixed'])
    #plt.show()
    
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

    sf_func = lambda qi,ph,th: xrsf.fcc_sf(qi,
        np.array([
            qi*np.sin(th)*np.cos(ph),
            qi*np.sin(th)*np.sin(ph),
            qi*np.cos(th)]).reshape(3,1),
        fcc_Al.basis_to_dict())

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

    


