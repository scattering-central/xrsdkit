xrsdkit: Python modules for x-ray scattering and diffraction data analysis 
==========================================================================


Description
-----------

This package uses data-driven models to analyze 
X-ray scattering and diffraction patterns.
The models are trained from a set of patterns 
that have been analyzed and curated on the Citrination platform.
Two flavors of model are available: 
one that is evaluated locally (based on scikit-learn),
and another that is built and evaluated on Citrination,
for users with access to the Citrination platform.

Scattering patterns employed in this package 
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

**Read n-by-2 array of scattering vectors and intensities (`q_I`) from a csv file:** ::

    import numpy as np
    q_I = np.genfromtxt ('my_data/sample_0.csv', delimiter=",")

**Import xrsdkit:** ::

    import xrsdkit

**Numerically profile the data:** ::

    (TODO: update this example to new API) 

To predict scatters populations we can use local scikit-learn models or remote Citrination models.

**Using xrsdkit (scikit-learn) models:** ::

    (TODO: update this example to new API) 

**Using Citrination models:** ::

    (TODO: update this example to new API) 

**Fit the remaining parameters:** ::

    (TODO: update this example to new API) 

The full script for this example can be found here:
https://github.com/scattering-central/saxskit/blob/dev/examples/predict.py

The output should look like this:
https://github.com/scattering-central/saxskit/blob/dev/examples/output.png

There are some more detailed examples of predictions, 
training and updating of models,
and least-squares fitting, 
in the "examples" directory.


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

