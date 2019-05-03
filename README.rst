xrsdkit: Statistical models for fast x-ray scattering and diffraction data analysis 
===================================================================================


Description
-----------

This package provides the tools 
for a X-ray scattering and diffraction data ecosystem,
including tools to curate datasets 
and train statistical models for fast and/or automated analysis.


Example
-------

# TODO: add training of models from a dataset

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

More specific and detailed examples can be found in the "examples" directory.


Installation
------------

This package is hosted on PyPI. Install it by `pip install xrsdkit`


Contribution
------------

To contribute code, please feel free to submit a pull request on this repository.

If you have a dataset that you would like to use with xrsdkit,
and you would like some help with it, please contact the development team at
paws-developers@slac.stanford.edu.
We cannot guarantee a solution, 
but we will be very interested to hear about your use case.


License
-------

The 3-clause BSD license attached to this software 
can be found in the LICENSE file 
in the source code root directory.

