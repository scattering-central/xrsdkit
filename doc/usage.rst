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

    from saxskit.saxs_math import profile_spectrum
    features = profile_spectrum(q_i)

To predict scatters populations we can use SAXSKIT models (built on Sklearn) or Citrination models.

**Using SAXSKIT models:**

* Initialize SaxsClassifier and **predicted scatterer populations**: ::

    from saxskit.saxs_classify import SaxsClassifier
    m = SaxsClassifier()
    populations, propability = m.classify(features)
    print("scatterer populations: ")
    for k,v in populations.items():
        print(k, ":", v, "  with propability: %1.3f" % (propability[k]))

| unidentified : 0   with propability: 0.998
| guinier_porod : 0   with propability: 0.775
| spherical_normal : 1   with propability: 0.996
| diffraction_peaks : 0   with propability: 1.000


* Initialize SaxsRegressor and **predict counting scatterer parameters**: ::

    from saxskit.saxs_regression import SaxsRegressor
    r = SaxsRegressor()
    params = r.predict_params(populations,features, q_i)


* Initialize SaxsFitter and **update scatterer parameters with intensity parametes**: ::

    from saxskit import saxs_fit
    sxf = saxs_fit.SaxsFitter(q_i,populations)
    params, report = sxf.fit_intensity_params(params)
    for k,v in params.items():
        print(k, ":", end="")
        for n in v:
            print(" %10.3f" % (n))

| I0_floor :      0.663
| I0_sphere :   2601.258
| r0_sphere :     26.690
| sigma_sphere :      0.048
|


**Using Citrination models:**

*  Create SaxsCitrination using Citrination credentials: ::

    from saxskit.saxs_citrination import CitrinationSaxsModels

    api_key_file = 'path_to_my_api_key_txt'
    saxs_models = CitrinationSaxsModels(api_key_file,'https://slac.citrination.com')

* Predict scatterer populations::

    populations, uncertainties = saxs_models.classify(features)
    for k,v in populations.items():
        print(k, ":", v, "  with uncertainties: %1.3f" % (uncertainties[k]))

| unidentified : 0   with uncertainties: 0.008
| guinier_porod : 0   with uncertainties: 0.034
| spherical_normal : 1   with uncertainties: 0.005
| diffraction_peaks : 0   with uncertainties: 0.010


* Predict counting scatterer parameters: ::

    params,uncertainties = saxs_models.predict_params(populations, features, q_i)
    for k,v in params.items():
        print(k, ":", v, " +/- %1.3f" % (uncertainties[k]))

| r0_sphere : [28.055669297520605]  +/- 0.907
| sigma_sphere : [0.094776319721295]  +/- 0.087
|
* Initialize SaxsFitter and **update scatterer parameters with intensity parametes**: ::

    sxf = saxs_fit.SaxsFitter(q_i,populations)
    params, report = sxf.fit_intensity_params(params)
    for k,v in params.items():
        print(k, ":", end="")
        for n in range(len(v)):
            print(" %10.3f" % (v[n]) )

| I0_floor :      0.540
| I0_sphere :   3202.553
| r0_sphere :     28.056
| sigma_sphere :      0.095
|
::



The full version of this code:
https://github.com/scattering-central/saxskit/blob/dev/examples/predict.py

Output:
https://github.com/scattering-central/saxskit/blob/dev/examples/output.png

IPython notebooks:
https://github.com/scattering-central/saxskit/blob/dev/examples/predict_using_saxskit.ipynb
https://github.com/scattering-central/saxskit/blob/dev/examples/predict_using_citrination_models.ipynb

Train the models
................

**SAXSKIT has seven pretrained models**:

four binary classifiers:

- 'unidentified': True if the scatterers cannot be identified easily from the data.
- 'spherical_normal': True if there are one or more normal distributions of spherical scatterers.
- 'diffraction_peaks': True if there are one or more diffraction peaks.
- 'guinier_porod': One or more scatterers described by a Guinier-Porod equatio

three regression models:

- 'r0_sphere': the mean sphere size (in Angstroms) for 'spherical_normal' scatterers
- 'sigma_sphere': the fractional standard deviation of sphere size for 'spherical_normal' scatterers
- 'rg_gp': the radius of gyration for 'guinier_porod' scatterers

Users with Citrination accounts can pull SAXS data from Citrination to train custom models. The SAXS records used for
 training must have been generated with saxskit.saxs_piftools, preferably by the same version of saxskit.


**SAXSKIT provides two options for training**:

- training from scratch: useful for initial training or when we have a lot of new data (around 30% of the dataset
 or more).
- updating existing models with additional data: takes less time than training new models, especially when the existing
 model was trained on a large data set. This is recommended when there is some new data, but the new data are less than
  about 30% of the dataset.

"training from scratch" is useful for initial training or when we have a lot of new data (more than 30%). It is
recommended to use "hyper_parameters_search = True."

Updating existing models is recommended when we have some new data (less than 30%). Updating existing models takes
significant less time than "training from scratch"


Training from "scratch"
'''''''''''''''''''''''
Let's assume that initially we have only two datasets: 1 and 15. We want to use them to train the models.

::

    import saxskit
    from citrination_client import CitrinationClient
    from saxskit.saxs_models import get_data_from_Citrination
    from saxskit.saxs_models import train_classifiers, train_regressors, save_models

Step 0 (optional). Specify full path to the YAML file where the models will be saved.

Scalers, models, sklearn version, and cross-validation errors will be saved at this path, and the cross-validation
errors are also saved in a .txt file of the same name, in the same directory. If the path is not specified,
the models will be saved at'modeling_data/custom_models/some_number.yml'and the cross-validation errors are
also saved in a .txt file of the same name, in the same directory.

::

    p = os.path.abspath(__file__)
    d = os.path.dirname(os.path.dirname(p))
    classifiers_path = os.path.join(d,'saxskit','modeling_data','scalers_and_models.yml')
    regressors_path = os.path.join(d,'saxskit','modeling_data','scalers_and_models_regression.yml')

Step 1. Get data from Citrination using Citrination credentials ::

    api_key_file = os.path.join(d, 'api_key.txt')
    if not os.path.exists(api_key_file):
        print("Citrination api key file did not find")

    with open(api_key_file, "r") as g:
        a_key = g.readline().strip()
    cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)

    data = get_data_from_Citrination(client = cl, dataset_id_list= [1,15])


At this point, **data** is a pandas data frame that contains (for each SAXS record):

- experiment_id - The experiment ID is used for cross-validation grouping. Often, samples from the same experiment are
very similar, so cross-validating in this way avoids overtraining.

- An array of numerical features that describe the shape of the spectrum (invariant with respect to intensity scaling).

- Four True / False labels (for classification models):

    - 'unidentified'
    - 'guinier_porod'
    - 'spherical_normal'
    - 'diffraction_peaks' Note, if a sample has 'unidentified = True', it implies False for all other labels.

- An array of scattering parameters (previously least-squares fit with saxskit):

    - For any record that is not 'unidentified':

        - 'I0_floor': flat noise floor intensity

    - For 'guinier_porod' scatterers:

        - 'G_gp': Guinier prefactors
        - 'rg_gp': radii of gyration
        - 'D_gp': Porod exponents

    - For 'spherical_normal' scatterers:

        - 'I0_sphere': Intensity scaling prefactors
        - 'r0_sphere': Mean sphere radii
        - 'sigma_sphere': Fractional standard deviations

    - For 'diffraction_peaks':

        - 'I_pkcenter': Intensities of the peaks at their maxima
        - 'q_pkcenter': q-values of the peak maxima
        - 'pk_hwhm': peak half-widths at half-max

Note that not every record contains a value for every parameter. For example, only samples with 'spherical_normal'
populations will have values for 'sigma_sphere'.

Step 2. Train Classifiers and Save The Models ::

    scalers, models, accuracy = train_classifiers(data, hyper_parameters_search = True, model='all')
    save_models(scalers, models, accuracy, classifiers_path)

For training from scratch, we use train_classifiers() with hyper_parameters_search = True. This will seek a set of
model hyperparameters that optimizes the model. The final set of hyperparameters is the set that provides the highest
 mean accuracy on the given test data and labels.

Since samples from the same experiment are often highly correlated, saxskit uses a "Leave-N-Groups-Out" technique to
evaluate training error. Saxskit leaves two groups (experiment_ids) out for each training cycle. For example, if we
 have experiments 1 through 5:

- train the model on 1,2 3; test on 4,5
- train the model on 1,2,5; test on 3,4
- try all combinations...
- calculate average accuracy

A set of serialized scalers and models will be saved in the package's source directory at:

    - saxskit/modeling_data/scalers_and_models.yml

The accuracy of the trained models will also be reported in:

    - saxskit/modeling_data/scalers_and_models.txt

To calculate the reported accuracy "Leave-N-Groups-Out" technique is also used. Every cycle data from two experiments used
for testing and the other data for training. The average accuracy is reported.

train_classifiers() has an optional argument 'model' which can be used to specify the model to train. For example ::

    scalers, models, accuracy = train_classifiers(data, hyper_parameters_search = True, model='spherical_normal')

The names of models to train :"unidentified", "spherical_normal","guinier_porod", "diffraction_peaks", or "all" to train all models.



Step 3. Train and Save Regression models ::

    scalers, models, accuracy = train_regressors(data, hyper_parameters_search = True, model= 'all')
    save_models(scalers, models, accuracy, regressors_path)

The approach is the same as above, but for a different set of models. These are the three regression models for the
scattering spectrum parameters affecting curve shape. In the current version, the regression model output is
one-dimensional, so these are mostly useful for spectra containing **one** 'guinier_porod' and/or **one** 'spherical_normal'
scatterer population.


A set of serialized scalers and models will be saved in the package's source directory at:

    - saxskit/modeling_data/scalers_and_models_regression.yml

Note, for the regression models, the "Leave-N-Groups-Out" cross validation is used, also with N=2. The reported error for each model is the mean absolute validation error divided by the standard deviation of the training data. The accuracy of the trained models will also be reported in:

    - saxskit/modeling_data/scalers_and_models_regression.txt

train_regressors() has an optional argument 'model' which can be used to specify the model to train. For example ::

    scalers, models, accuracy = train_regressors(data, hyper_parameters_search = False, model= 'r0_sphere')

The names of models to train :"r0_sphere", "sigma_sphere", "rg_gp", or "all" to train all models.

The full version of this code:
https://github.com/scattering-central/saxskit/blob/dev/examples/train.py

IPython notebook:
https://github.com/scattering-central/saxskit/blob/dev/examples/train_models.ipynb


Updating the models
'''''''''''''''''''

Assume that we got a new dataset and now we want to update our models using new data. Since training "from scratch" took a significant amount of time (specially for the regression models) we will use train_classifiers_partial() and train_regressors_partial() to update the models with the new data.

::

    import saxskit
    from citrination_client import CitrinationClient
    from saxskit.saxs_models import get_data_from_Citrination
    from saxskit.saxs_models import train_classifiers_partial, train_regressors_partial, save_models

Step 1. Specify full path to the YAML file where the models was saved.

The cross-validation errors were also saved in a .txt file of the same name, in the same directory. ::

    p = os.path.abspath(__file__)
    d = os.path.dirname(os.path.dirname(p))
    classifiers_path = os.path.join(d,'saxskit','modeling_data','scalers_and_models.yml')
    regressors_path = os.path.join(d,'saxskit','modeling_data','scalers_and_models_regression.yml')

Step 2. Get data from Citrination using Citrination credentials ::

    api_key_file = os.path.join(d, 'api_key.txt')
    if not os.path.exists(api_key_file):
        print("Citrination api key file did not find")

    with open(api_key_file, "r") as g:
        a_key = g.readline().strip()
    cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)

    new_data = get_data_from_Citrination(client = cl, dataset_id_list= [16]) # [16] is a list of datasets ids

Step 3 (optional). Get all available data from Citrination

If we want to know the accuracy of the updated models, it is recommended to calculate it against the full training set. To calculate the reported accuracy "Leave-N-Groups-Out" technique is used. Every cycle data from two experiments used for testing and the other data for training. The average accuracy is reported.
::

    all_data = get_data_from_Citrination(client = cl, dataset_id_list= [1,15,16])

Step 3. Update Classifiers ::

    scalers, models, accuracy = train_classifiers_partial(
        new_data, classifiers_path, all_training_data=all_data, model='all')

train_classifiers_partial() has an optional argument 'model' which can be used to specify the model to train.
For example ::

    scalers, models, accuracy = train_classifiers_partial(data, hyper_parameters_search = True, model='spherical_normal')

The names of models to train :"unidentified", "spherical_normal","guinier_porod", "diffraction_peaks", or "all" to train all models.

Accuracy after updating ::

    for model_name, acc in new_accuracy.items():
        print('{}: {:.4f}'.format(model_name,acc))

| diffraction_peaks: 0.9802
| guinier_porod: 0.7321
| spherical_normal: 0.9765
| unidentified: 0.9886

If we are not satisfied with new accuracy, we can train the models "from scratch" ::

    scalers, models, new_accuracy = train_classifiers(all_data, hyper_parameters_search = True, model='all')

Step 4. Save the models

Scalers, models, sklearn version, and cross-validation errors will be saved at "classifiers_path", and the cross-validation errors are also saved in a .txt file of the same name, in the same directory. If the path is not specified, the models will be saved at'modeling_data/custom_models/some_number.yml'and the cross-validation errors are also saved in a .txt file of the same name, in the same directory. ::

    save_models(scalers, models, accuracy, classifiers_path)

Step 5. Update rergession models ::

    scalers, models, accuracy = train_regressors_partial(
        new_data, regressors_path, all_training_data=all_data, model='all')

train_regressors_partial() has an optional argument 'model' which can be used to specify the model to train. For example ::

    scalers, models, accuracy = train_regressors_partial(data, hyper_parameters_search = False, model= 'r0_sphere')

The names of models to train :"r0_sphere", "sigma_sphere", "rg_gp", or "all" to train all models.

Accuracy after updating ::

    for model_name, acc in new_accuracy.items():
        print('{}: {:.4f}'.format(model_name,acc))

| r0_sphere: 0.2642
| rg_gp: 1.1316
| sigma_sphere: 0.5594

Again, if we are not satisfied with new accuracy, we can train the models "from scratch" ::

    scalers, models, new_accuracy = train_regressors(all_data, hyper_parameters_search = True, model='all')

Step 6. Save updated regression models.

Scalers, models, sklearn version, and cross-validation errors will be saved at "regressors_path", and the cross-validation errors are also saved in a .txt file of the same name, in the same directory. If the path is not specified, the models will be saved at'modeling_data/custom_models/some_number.yml'and the cross-validation errors are also saved in a .txt file of the same name, in the same directory.
::

    save_models(scalers, models, new_accuracy, regressors_path)


The full version of this code:
https://github.com/scattering-central/saxskit/blob/dev/examples/update_models.py

IPython notebook:
https://github.com/scattering-central/saxskit/blob/dev/examples/update_models.ipynb