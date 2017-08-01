scattered: Python modules for scattering and diffraction data processing 
========================================================================


Description
-----------

This repository is a temporary solution
for collecting modules that should eventually be rolled uphill
into a more mature repository, such as scikit-beam
(https://github.com/scikit-beam).


Example
-------

This is sort of how it should work.
This example uses the `saxs` subpackage of `scattered`
to profile, parameterize, and optimize the fit of a scattering equation
to a measured saxs spectrum.

NOTE: this code block is not visible on github.

.. code-block:: python
    from scattered import saxs
    
    img = # read in an image array
    q_I = # use some other library to calibrate/reduce
    saxs_metrics = saxs.spectrum_profile(img1d)
    saxs_params = saxs.parameterize_spectrum(img1d,saxs_metrics)
    saxs_params = saxs.fit_spectrum(img1d,saxs_params)
    q_I_model = saxs.compute_saxs(saxs_params)
    # now plot q_I versus q_I_model


Installation
------------

This package is not yet distributed in any way.
It can be downloaded from this repository-
please take note of the attached LICENSE.


Attribution
-----------

If you use `scattered` in any original work,
the authors of the modules you use
would appreciate a citation.
Every subpackage and module in `scattered` 
has authorship and citation information attached
(not yet implemented). 
You can read it from the source code, 
or access it through a Python interpreter:

NOTE: this code block is not visible on github.

.. code-block:: python
    from scattered import saxs
    print saxs.__doc__


Contribution
------------

This repository is a temporary solution
for collecting code that will be moved eventually
to a more mature package, like scikit-beam 
(https://github.com/scikit-beam).
Please feel free to submit pull requests for minor contributions,
and add yourself to the authorship and citation information 
for the modules that you contribute to.
If you are looking to develop substantive scattering and diffraction modules,
we suggest that you start a temporary repository of your own,
and also aim to merge your modules into scikit-beam.


License
-------

The 3-clause BSD license attached to this software 
can be found in the LICENSE file 
in the source code root directory.

