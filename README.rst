pysaxs: Python modules for small angle x-ray scattering data analysis 
=====================================================================


Description
-----------

This repository is a staging area for some SAXS analysis modules.
We hope they will eventually be roll uphill
into a more mature repository.


Example
-------

This example profiles, parameterizes, 
and optimizes the fit of a scattering equation
to a measured saxs spectrum.

NOTE: this code block is not visible on github.

.. code-block:: python
    from pysaxs.saxs import saxsfit as sf
    
    img = #TODO: use some IO library (e.g. FabIO) to read an image 
    q, I = #TODO: use some mature package (e.g. PyFAI) to calibrate/reduce

    saxs_metrics = sf.profile_spectrum(q,I)
    saxs_params = sf.parameterize_spectrum(q,I,saxs_metrics)
    fit_params = ['r0_form','sigma_form'] # keys from saxs_params to optimize
    fit_constraints = ['fix_I0'] # any desired constraints
    saxs_params = sf.fit_spectrum(q,I,'chi2_log',saxs_params,fit_params,fit_constraints)
    I_model = sf.compute_saxs(q,saxs_params)

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


Contribution
------------

This repository is a staging area 
for code that aspires to a more mature package. 
Please feel free to submit pull requests for contributions,
and add yourself to the authorship and citation information 
in the docstrings of the modules that you contribute to.


License
-------

The 3-clause BSD license attached to this software 
can be found in the LICENSE file 
in the source code root directory.

