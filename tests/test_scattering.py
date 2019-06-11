import numpy as np

from xrsdkit import scattering as xrsdscat 
from xrsdkit.scattering import form_factors as xrff

qvals = np.arange(0.,1.,0.01)

def test_guinier_porod():
    ff = xrsdscat.guinier_porod_intensity(qvals,20,4)

def test_spherical_normal():
    ff2 = xrsdscat.spherical_normal_intensity(qvals,20,0.2)
    ff_mono = xrff.spherical_ff(qvals,20)

qvals = np.arange(0.,2.,0.01)
def test_plot_ff():
    ff_H = xrff.atomic_ff_normalized(qvals,'H')
    ff_Al = xrff.atomic_ff_normalized(qvals,'Al')
    ff_U = xrff.atomic_ff_normalized(qvals,'U')
    ff2_gp = xrsdscat.guinier_porod_intensity(qvals,20,4)
    ff2_poly = xrsdscat.spherical_normal_intensity(qvals,20,0.2)
    ff_mono = xrff.spherical_ff(qvals,20)
    #from matplotlib import pyplot as plt
    #plt.plot(qvals,ff_H**2) 
    #plt.plot(qvals,ff_Al**2) 
    #plt.plot(qvals,ff_U**2) 
    #plt.plot(qvals,ff_gp) 
    #plt.plot(qvals,ff_mono) 
    #plt.legend(['hydrogen','aluminum','uranium','guinier-porod','solid sphere'])
    #plt.xlabel('q')
    #plt.ylabel('f(q)')
    #plt.title('form factors')
    #plt.show()

