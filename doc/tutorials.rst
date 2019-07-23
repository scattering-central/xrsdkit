.. _sec-tutorials:


Tutorials
=========

The following four tutorials should be enough 
to get most users started with xrsdkit.
After setup and installation (tutorial 0),
each tutorial includes instructions for using the programming interface 
(for users importing xrsdkit and using it in scripts or notebooks),
followed by the same functionality via the xrsdkit GUI.


Tutorial 0: setup and installation
----------------------------------

To ensure that the package is installed correctly, 
invoke a Python interpreter and import it::

    $ python 
    >>> import xrsdkit

For gui usage, start the GUI from the command line::

    $ xrsdkit-gui

This should bring up the (empty) GUI.

#.. image:: graphics/gui_init.png

Note the panels on the right hand side
and their respective functionalities.

    - I/O: The topmost panel contains all controls for reading raw data and writing analysis results.
    - Fit control: The second panel is used to enter sample metadata and control the fitting process.
    - Noise model: The third panel is used to specify a noise model and control its parameters.
    - Populations: Populations are added by entering a new name and pressing the "+" button. Each new population creates a panel where its settings and parameters are controlled. 


Tutorial 1: data analysis and dataset curation
----------------------------------------------

In this tutorial, a measured pattern is read from a file,
some populations are specified,
and the parameters of the populations 
are optimized to fit the measured intensities.
The results are then visualized and saved to an output file.

For this tutorial to be executed as-written,
`download the xrsdkit-modeling repository <https://github.com/slaclab/xrsdkit_modeling>`_,
and run it from the repository's root directory.


Programming interface
.....................

**Import numpy and some xrsdkit tools** ::

    import numpy as np
    from xrsdkit import system as xrsdsys
    from xrsdkit import visualization as xrsdvis 

**Read the scattering pattern** ::

    q_I = np.loadtxt('my_data/sample_0.dat')
    q = q_I[:,0] 
    I = q_I[:,1] 

This produces two 1-d arrays.
One contains q-values, 
the other contains the corresponding intensities.
If this fails, the data file probably 
does not work nicely with numpy.loadtxt.
This can happen, for example, 
if the data file has a complicated header at the top.
Use whatever method works to read in the data,
as long as the q and I arrays are produced.

The pattern can be now be inspected 
via xrsdkit.visualization functions. ::

    xrsdvis.plot_etc(etc,etc)

**Specify some populations for fitting the pattern** 

Two populations and a flat noise model will be used.
One population is a dilute Guinier-Porod scatterer,
the other is a crystalline arrangement of spheres. ::

    sys = xrsdsys.System(etc,etc,etc)

For more information about how to specify populations, see:

    - :ref:`xrsdkit.system`
    - :ref:`xrsdkit.definitions`

**Fit the parameters objectively and plot the result** ::

    sys_opt = xrsdsys.fit(sys,q,I,source_wavelength)
    mpl_fig, I_comp = xrsdvis.plot_xrsd_fit(sys,q,I,source_wavelength)
    mpl_fig.show()


Graphical interface
...................

In the I/O panel, click the Browse button 
to open the data loader interface.

#.. image:: graphics/gui_init.png

In the data loader, use the Browse button
to find the directory containing the file.

Browse to the directory containing the data file(s).
This will load all of the data files that match the provided regular expression.
Each input data file will be automatically assigned to an output data file,
and the output files will be populated with any information 
entered into the "experiment metadata" input fields.

#.. image:: graphics/gui_init.png

Note: the GUI uses numpy.loadtxt internally,
so GUI users should format their data files accordingly. 

In the GUI, the populations and noise model 
are specified via the widgets on the right-hand side:

The fit is controlled and executed with from the  

After carrying out this process for several samples,
the outputs can be curated in a dataset for training models.
To curate a dataset, use this directory structure:

Graphic: directory structure


Tutorial 2: model training 
--------------------------

In this tutorial, a curated dataset of fit results
is used to train a set of models 
that can be used for automated analysis.
After training the models,
their performance is inspected 
by cross-validation metrics that are collected during training.
To optimize performance,
the training process can be tuned,
the modeling algorithms can be altered,
and the model hyperparameters can be tuned.
After the models are trained, they are saved to disk
so that they can be re-used without re-training.

For this tutorial to be executed as-written,
`download the xrsdkit-modeling repository <https://github.com/slaclab/xrsdkit_modeling>`_,
and run it from the repository's root directory.


Programming interface
.....................


Graphical interface
...................



Tutorial 3: model application 
-----------------------------

In this tutorial, a ready-trained set of models
is used to quickly analyze a few samples of previously unlabeled data.

For this tutorial to be executed as-written,
`download the xrsdkit-modeling repository <https://github.com/slaclab/xrsdkit_modeling>`_,
and run it from the repository's root directory.


Programming interface
.....................


Graphical interface
...................


