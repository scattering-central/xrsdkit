.. _sec-usage:

Usage
-----

Basic Example
.............

This example profiles, parameterizes, 
and optimizes the fit of a scattering equation
to a measured saxs spectrum.

**Import numpy and some xrsdkit tools** ::

    import numpy as np
    from xrsdkit.tools import profiler as xrsdprof
    from xrsdkit import models as xrsdmods
    from xrsdkit import system as xrsdsys
    from xrsdkit import visualization as xrsdvis 

**Read and profile a scattering pattern** ::

    q_I = np.loadtxt('my_data/sample_0.dat')
    q = q_I[:,0]
    I = q_I[:,1]
    p = xrsdprof.profile_pattern(q,I)    

**Use statistical models to identify and parameterize the material system** ::

    pred = xrsdmods.predict(p)
    source_wavelength = 0.8
    sys = xrsdmods.system_from_prediction(pred,q,I,source_wavelength)

**Fit the real-valued parameters objectively and plot the result** ::

    sys_opt = xrsdsys.fit(sys,q,I,source_wavelength)
    mpl_fig, I_comp = xrsdvis.plot_xrsd_fit(sys,q,I,source_wavelength)
    mpl_fig.show()

