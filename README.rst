pysaxs: Python modules for small angle x-ray scattering data analysis 
=====================================================================


Description
-----------

This repository is a staging area for some SAXS analysis modules.
We hope they will eventually be roll uphill
into a more mature repository, such as scikit-beam
(https://github.com/scikit-beam).


Example
-------

This is sort of how it should work.
This example uses the `saxs` subpackage 
to profile, parameterize, and optimize the fit of a scattering equation
to a measured saxs spectrum.

NOTE: this code block is not visible on github.

.. code-block:: python
    from pysaxs.saxs import ssrlsaxsfit as ssf
    
    img = #TODO: use some IO library (e.g. FabIO) to read an image 
    q, I = #TODO: use some mature package (e.g. PyFAI) to calibrate/reduce

    saxs_metrics = ssf.profile_spectrum(q,I)
    saxs_params = ssf.parameterize_spectrum(q,I,saxs_metrics)
    fit_params = ['r0_form','sigma_form'] # keys from saxs_params to optimize
    fit_constraints = ['fix_I0'] # any desired constraints
    saxs_params = ssf.fit_spectrum(q,I,'chi2_log',saxs_params,fit_params,fit_constraints)
    I_model = ssf.compute_saxs(q,saxs_params)

    # now plot q,I versus q,I_model
    from matplotlib import pyplot as plt
    plt.plot(q,I)
    plt.plot(q,I_model,'g')
    plt.legend(['measured','parameterized'])
    plt.show()

Installation
------------

This package is not yet distributed in any way.
It can be downloaded from this repository as is.


Attribution
-----------

If you use this in any original work,
the authors of the modules you use
would appreciate a citation.

To get a citation through a Python interpreter,
follow this procecdure for whatever modules you use:

NOTE: this code block is not visible on github.

.. code-block:: python
    from pysaxs.saxs import ssrlsaxsfit as ssf
    print ssf.__doc__


Contribution
------------

This repository is a staging area 
for code that aspires to a more mature package, 
such as scikit-beam (https://github.com/scikit-beam).
Please feel free to submit pull requests for contributions,
and add yourself to the authorship and citation information 
in the docstrings of the modules that you contribute to.


License
-------

The 3-clause BSD license attached to this software 
can be found in the LICENSE file 
in the source code root directory.

