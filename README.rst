saxskit: Python modules for small angle x-ray scattering data analysis 
======================================================================


Description
-----------

This package uses data-driven models to analyze SAXS data.
The models are trained from a set SAXS spectra
that have been analyzed and curated on the Citrination platform.

SAXS spectra employed in this package 
are attributed to the following sources:

 - Wu, Liheng, et al. Nature 548, 197–201 (2017). doi: 10.1038/nature23308

As more spectra are added to the curated set, 
the models behind this package are expected to become more effective.
If you have a SAXS data set that you would like to volunteer
to add to the curated set, 
please contact the development team at
ssrl-citrination@slac.stanford.edu or paws-developers@slac.stanford.edu.


Example
-------

This example profiles, parameterizes, 
and optimizes the fit of a scattering equation
to a measured saxs spectrum.

Read intensivity n-by-2 array `q_I` from a csv file: ::

    import numpy as np
    q_i = np.genfromtxt ('my_data/sample_0.csv', delimiter=",")


Import saxskit: ::

    import saxskit

Profile a saxs spectrum: ::

    from saxskit.saxskit.saxs_math import profile_spectrum
    features = profile_spectrum(q_i)
To predict scatters populations we can use SAXSKIT models (built on Sklearn) or Citrination models.

**Using SAXSKIT models:**

* Initialize SaxsClassifier and predicted scatterer populations: ::

    from saxskit.saxskit.saxs_classify import SaxsClassifier
    m = SaxsClassifier()
    flags, propability = m.classify(features)
    print(flags)

OrderedDict([('unidentified', 0), ('guinier_porod', 1), ('spherical_normal', 1), ('diffraction_peaks', 0)]) ::

    print(propability)

OrderedDict([('unidentified', 0.99110040176950032), ('guinier_porod', 0.55612076431031976), ('spherical_normal', 0.74962303617945247), ('diffraction_peaks', 0.99999999999999989)])


* Initialize SaxsRegressor and predict counting scatterer populations: ::

    from saxskit.saxskit.saxs_regression import SaxsRegressor
    r = SaxsRegressor()
    population_keys = r.predict_params(flags,features, q_i)
    print(population_keys)

OrderedDict([('I0_floor', 0.0), ('I0_sphere', 0.0), ('r0_sphere', 11.041806824106182), ('sigma_sphere', 0.048352866927024042), ('rg_gp', 4.5950722385040859), ('D_gp', 4.0), ('G_gp', 0.0)])

Installation
------------

This package is hosted on PyPI. Install it by `pip install pysaxs`


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

