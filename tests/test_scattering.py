import numpy as np

from xrsdkit.scattering import form_factors as xrff

qvals = np.arange(0.,1.,0.01)

def test_guinier_porod():
    ff = xrff.guinier_porod_intensity(qvals,20,4,120)

def test_spherical_normal():
    ff2 = xrff.spherical_normal_intensity(qvals,20,0.2)
    ff_mono = xrff.spherical_ff(qvals,20)

qvals = np.arange(0.,2.,0.01)
def test_plot_ff():
    ff_H = xrff.standard_atomic_ff(qvals,'H')
    ff_Al = xrff.standard_atomic_ff(qvals,'Al')
    ff_U = xrff.standard_atomic_ff(qvals,'U')
    ff2_gp = xrff.guinier_porod_intensity(qvals,1,4,120)
    ff2_poly = xrff.spherical_normal_intensity(qvals,20,0.2)
    ff_mono = xrff.spherical_ff(qvals,20)
    #from matplotlib import pyplot as plt
    #plt.plot(qvals,ff_H/ff_H[0]) 
    #plt.plot(qvals,ff_Al/ff_Al[0]) 
    #plt.plot(qvals,ff_U/ff_U[0]) 
    #plt.plot(qvals,ff_gp) 
    #plt.plot(qvals,ff_mono) 
    #plt.legend(['hydrogen','aluminum','uranium','guinier-porod','solid sphere'])
    #plt.xlabel('q')
    #plt.ylabel('f(q)')
    #plt.title('form factors')
    #plt.show()

