import numpy as np
import pandas as pd
from sklearn import linear_model, model_selection, preprocessing

from .xrsd_model import XRSDModel
from ..tools.profiler import guinier_porod_profile, spherical_normal_profile
from ..tools import profiler

class Regressor(XRSDModel):
    """Class for generating models to predict real-valued parameters."""

    def __init__(self,label,yml_file):
        super(Regressor,self).__init__(label, yml_file)
        self.grid_search_hyperparameters = dict( 
            loss = ['huber', 'squared_loss'], # huber with epsilon = 0 yields mean abs error (MAE)
            epsilon = [1, 0.1, 0.01, 0.001, 0],  
            penalty = ['none', 'l2', 'l1', 'elasticnet'], # default: l2  
            alpha = [0.0001, 0.001, 0.01], # regularisation coef, default 0.0001 
            l1_ratio = [0, 0.15, 0.5, 0.95] # default 0.15, only valid for elasticnet penalty 
            )

    def build_model(self, model_hyperparams={}):
        if all([p in model_hyperparams for p in ['alpha','loss','penalty','l1_ratio','epsilon']]):
            new_model = linear_model.SGDRegressor(
                    alpha=model_hyperparams['alpha'], 
                    loss=model_hyperparams['loss'],
                    penalty=model_hyperparams['penalty'],
                    l1_ratio=model_hyperparams['l1_ratio'],
                    epsilon=model_hyperparams['epsilon'],
                    max_iter=1000)
        else:
            # NOTE: max_iter is about 10^6 / number of tr samples 
            new_model = linear_model.SGDRegressor(max_iter=1000) 
        return new_model

    def predict(self, sample_features, q_I):
        """Predict this model's scalar target for a given sample. 

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of xrsdkit.tools.profiler.profile_spectrum()
        q_I : array
            n-by-2 array of scattering vector (1/Angstrom) and intensities.

        Returns
        -------
        prediction : float
            predicted parameter value
        """
        if 'rg' in self.target:
            additional_features = guinier_porod_profile(q_I)
            my_features = np.append(np.array(list(sample_features.values())),
                                    np.array(list(additional_features.values()))).reshape(1,-1)

        elif 'r0' in self.target or 'sigma' in self.target:
            additional_features = spherical_normal_profile(q_I)
            my_features = np.append(np.array(list(sample_features.values())),
                                    np.array(list(additional_features.values()))).reshape(1,-1)

        x = self.scaler.transform(my_features)
        return float(self.model.predict(x)[0])

    def run_cross_validation(self,model,data,group_cv):
        label_std = pd.to_numeric(data[self.target]).std()
        if group_cv:
            new_accuracy = self.cross_validate_by_experiments(model,data,label_std)
        else:
            new_accuracy = self.cross_validate(model,data,label_std)
        return new_accuracy

    def cross_validate(self,model,df,label_std):
        """Test a model using scikit-learn 5-fold crossvalidation

        Parameters
        ----------
        model : sklearn model
            model to be cross-validated 
        df : pandas.DataFrame
            pandas dataframe of features and labels
        label_std : float
            standard deviation of training data for the model label 

        Returns
        -------
        scores : object
            TODO: describe scores output
        """
        scaler = preprocessing.StandardScaler()
        scaler.fit(df[profiler.profile_keys])
        scores = model_selection.cross_val_score(
                model,scaler.transform(df[profiler.profile_keys]), df[self.target],
                cv=5, scoring='neg_mean_absolute_error')
        return scores 

    def cross_validate_by_experiments(self, model, df, label_std):
        """Test a model by LeaveOneGroupOut cross-validation.

        Parameters
        ----------
        df : pandas.DataFrame
            pandas dataframe of features and labels
        model : sk-learn
            with specific parameters
        label_std : float
            standard deviation of training data for the model label 
            
        Returns
        -------
        test_scores_by_ex : object
            TODO: describe scores output
        """
        experiments = df.experiment_id.unique()
        test_scores_by_ex = []
        for i in range(len(experiments)):
            tr = df[(df['experiment_id']!= experiments[i])]
            test = df[(df['experiment_id']== experiments[i])]

            scaler = preprocessing.StandardScaler()
            scaler.fit(tr[profiler.profile_keys])
            model.fit(scaler.transform(tr[profiler.profile_keys]), tr[self.target])
            transformed_data = scaler.transform(test[profiler.profile_keys])
            pr = model.predict(transformed_data)
            test_score = mean_absolute_error(pr, test[self.target])
            test_scores_by_ex.append(test_score/label_std)

        return test_scores_by_ex

