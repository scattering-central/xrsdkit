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

```python

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import saxskit

from saxskit.saxskit.saxs_classify import SaxsClassifier
from saxskit.saxskit.saxs_regression import SaxsRegressor
from saxskit.saxskit.saxs_math import profile_spectrum

q_i = np.genfromtxt ('my_data/sample_0.csv', delimiter=",")

features = profile_spectrum(q_i)

m = SaxsClassifier()

flags = m.run_classifier(features)
print(flags)

r = SaxsRegressor()

population_keys = r.predict_params(flags,features, q_i)
print(population_keys)

```


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

