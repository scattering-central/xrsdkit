xrsdkit: Python modules for x-ray scattering and diffraction data analysis 
==========================================================================


Description
-----------

This package supports a data ecosystem for 
X-ray scattering and diffraction patterns,
notably including tools to curate datasets 
and build statistical models for automated analysis.
The package includes a set of models
trained on data curated by the developers
at Stanford Synchrotron Radiaton Lightsource (SSRL),
a directorate of the Stanford Linear Accelerator (SLAC) laboratory.

Data employed for the packaged models 
are attributed to the following sources:

 - Wu, Liheng, et al. Nature 548, 197–201 (2017). doi: 10.1038/nature23308

As more patterns are added to the curated set, 
the models are expected to become more effective.
If you have a data set that you would like to volunteer
to add to the curated set, 
please contact the development team at
ssrl-citrination@slac.stanford.edu or paws-developers@slac.stanford.edu.


Example
-------

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

To contribute data for improving the models,
please contact the development team at
ssrl-citrination@slac.stanford.edu or paws-developers@slac.stanford.edu.


License
-------

The 3-clause BSD license attached to this software 
can be found in the LICENSE file 
in the source code root directory.

