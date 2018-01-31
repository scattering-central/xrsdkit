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

**Read intensivity n-by-2 array `q_I` from a csv file:** ::

    import numpy as np
    q_i = np.genfromtxt ('my_data/sample_0.csv', delimiter=",")


**Import saxskit:** ::

    import saxskit

**Profile a saxs spectrum:** ::

    from saxskit.saxskit.saxs_math import profile_spectrum
    features = profile_spectrum(q_i)
To predict scatters populations we can use SAXSKIT models (built on Sklearn) or Citrination models.

**Using SAXSKIT models:**

* Initialize SaxsClassifier and **predicted scatterer populations**: ::

    from saxskit.saxskit.saxs_classify import SaxsClassifier
    populations, propability = m.classify(features)
    print("scatterer populations: ")
    for k,v in populations.items():
        print(k, ":", v, "  with propability: %1.3f" % (propability[k]))
    print()

| unidentified : 0 with propability: 0.984
| guinier_porod : 0 with propability: 0.772
| spherical_normal : 1 with propability: 0.995
| diffraction_peaks : 0 with propability: 0.996


* Initialize SaxsRegressor and **predict counting scatterer parameters**: ::

    from saxskit.saxskit.saxs_regression import SaxsRegressor
    r = SaxsRegressor()
    param = r.predict_params(flags,features, q_i)


* Initialize SaxsFitter and **update scatterer parameters with intensity parametes**: ::

    from saxskit import saxs_fit
    sxf = saxs_fit.SaxsFitter(q_i,populations)
    params, report = sxf.fit_intensity_params(params)
    print("scattering and intensity parameters: ")
    for k,v in params.items():
        print(k, ":", end="")
        for n in v:
            print(" %10.3f" % (n))
    print()

| I0_floor :      0.618
| I0_sphere :   2427.587
| r0_sphere :     26.771
| sigma_sphere :      0.048
|


**Using Citrination models:**

*  Create SaxsCitrination using Citrination credentials: ::

    from saxskit.saxskit.saxs_citrination import CitrinationSaxsModels

    api_key_file = '../../api_key.txt'
    saxs_models = CitrinationSaxsModels(api_key_file,'https://slac.citrination.com')

* Predict scatterer populations::

    flags, uncertainties = saxs_models.classify(features)
    for k,v in flags.items():
        print(k, ":", v, "  with uncertainties: %1.3f" % (uncertainties[k]))

| unidentified : 0 with uncertainties: 0.008
| guinier_porod : 0 with uncertainties: 0.051
| spherical_normal : 1 with uncertainties: 0.009
| diffraction_peaks : 0 with uncertainties: 0.006


* Predict counting scatterer parameters: ::

    params,uncertainties = saxs_models.predict_params(flags, features, q_i)

* Initialize SaxsFitter and **update scatterer parameters with intensity parametes**: ::

    sxf = saxs_fit.SaxsFitter(q_i,populations)
    params, report = sxf.fit_intensity_params(params)
    print("scattering and intensity parameters: ")
    for k,v in params.items():
        print(k, ":", end="")
        for n in range(len(v)):
            print(" %10.3f" % (v[n]))
    print()

| I0_floor :      0.545
| I0_sphere :   3022.984
| r0_sphere :     27.929
| sigma_sphere :      0.099

::

    for k,v in uncertainties.items():
        print(k, ": %1.3f" % (v))

| r0_sphere : 0.789
| sigma_sphere : 0.092
|


The full version of this code:
https://github.com/scattering-central/saxskit/blob/examples/examples/predict.py

Output:
https://github.com/scattering-central/saxskit/blob/examples/examples/output.png

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

