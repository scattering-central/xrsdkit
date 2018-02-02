.. _sec-usage:

Usage
-----


Make a prediction
.................

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

Train the models
................

**SAXSKIT has seven pretrained models**:

four classifiers that predict "True" or "False" for:

- data are identifiable
- the scatterers include one population of a normal size distribution of spherical scatterers
- the scatters include diffraction peaks
- the scatters include Guinier-Porod like terms

three regression models that predict:

- the mean sphere size (in Angstroms)
- the standard deviation (fractional), assuming a normal size distribution
- the estimated intensity of the spherical scattering at q=0

**SAXSKIT provides two options for training**:

- training from scratch
- updating existing models using additional data

"training from scratch" is useful for initial training or when we have a lot of new data (more than 30%). It is recommended to use "hyper_parameters_search = True."

Updating existing models is recommended when we have some new data (less than 30%). Updating existing models takes significant less time than "training from scratch"


Training from "scratch"
'''''''''''''''''''''''
Let's assume that initially we have only two datasets: 1 and 15. We want to use them to train the models.

::

    import saxskit
    from citrination_client import CitrinationClient
    from saxskit.saxskit.saxs_models import get_data_from_Citrination
    from saxskit.saxskit.saxs_models import train_classifiers, train_regressors

Step 1. Get data from Citrination using Citrination credentials ::

    path = os.getcwd()
    api_key_file = path + '/citrination_api_key_ssrl.txt'

    with open(api_key_file, "r") as g:
        a_key = g.readline().strip()
    cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)

    data = get_data_from_Citrination(client = cl, dataset_id_list= [1,15]) # [1,15] is a list of datasets ids


**data** is a pandas data frame that contains:

- experiment_id

    - It will be used for grouping for creating crossvaligdation folders during the training. Often samples from the same experiment are very similar and we should avoid to to have the samples from the same experiment in training and validation sets

- Twenty features:

    - 'Imax_over_Imean', 'Imax_sharpness', 'I_fluctuation',
    - 'logI_fluctuation', 'logI_max_over_std', 'r_fftIcentroid', 'r_fftImax',
    - 'q_Icentroid', 'q_logIcentroid', 'pearson_q', 'pearson_q2',
    - 'pearson_expq', 'pearson_invexpq', 'I0_over_Imean', 'I0_curvature',
    - 'q_at_half_I0', 'q_at_Iq4_min1', 'pIq4_qwidth', 'pI_qvertex',
    - 'pI_qwidth'

- Four True / False labels (for classification models):

    -  'unidentified', 'guinier_porod', 'spherical_normal','diffraction_peaks'

If a sample have 'unidentified = True', it also have "False" for all other labels.

- Ten continuouse labels (for regression models):

    -  'I0_floor', 'G_gp', 'rg_gp', 'D_gp', 'I0_sphere',
    -  'r0_sphere', 'sigma_sphere', 'I_pkcenter', 'q_pkcenter', 'pk_hwhm'.

Some samples have "None" for some of these labels. For example, only samples with 'spherical_normal = True' have some value for 'sigma_sphere'

Step 2. Train Classifiers ::

    train_classifiers(data,  hyper_parameters_search = True)

Scalers and models will be saved in 'saxskit/saxskit/modeling_data/scalers_and_models.yml'.
Accuracy will be saved in 'saxskit/saxskit/modeling_data/accuracy.txt'.
We can use yaml_filename='file_name.yml' as additional parametrs to save scalers and models in it.

Since often the data form the same experiment is highly correlated, "Leave N Group Out" technique is used to calculate accuracy. Data from two experiments is excluded from training and used as testing set. For example, if we have experiments 1,2,3,5,and 5:

- train the model on 1,2 3; test on 4,5
- train the model on 1,2,5; test on 3,4
- try all combinations...
- calculate average accuracy

Step 3. Train Regression models ::

    train_regressors(data,  hyper_parameters_search = True)

For the regression models, "Leave N Group Out" technique is also used. The accuracy is calculated as absolute mean error divided by standard derivation.

Updating the models
'''''''''''''''''''

Assume that we got a new dataset and now we want to update our models using new data. Since training "from scratch" took significant amount of time (specially, for regression models) we will use train_classifiers_partial() and train_regressors_partial().

::

    import saxskit
    from citrination_client import CitrinationClient
    from saxskit.saxskit.saxs_models import get_data_from_Citrination
    from saxskit.saxskit.saxs_models import train_classifiers_partial, train_regressors_partial

Step 1. Get data from Citrination using Citrination credentials ::

