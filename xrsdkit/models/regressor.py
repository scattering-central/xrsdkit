import numpy as np
from .general_model import XRSDModel
from ..tools import profiler
from xrsdkit.tools.profiler import guinier_porod_profile, spherical_normal_profile

class Regressor(XRSDModel):
    """To create regressor for prediction continuous value from scattering/diffraction data;
    train, update, and save it; make a prediction."""

    def __init__(self,label,yml_file=None):

        super(Regressor,self).__init__(label, yml_file, False)

        self.n_groups_out = 1

        f = []
        f.extend(profiler.profile_keys_1)
        if 'rg' in label:
            f.extend(profiler.gp_profile_keys)
        elif 'r0' or 'sigma' in label:
            f.extend(profiler.spherical_profile_keys)
        self.features = f


    def predict(self, sample_features, population, q_I):
        """Determine the types of structures represented by the sample
        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of xrsdkit.tools.profiler.profile_spectrum()
        population : str
            Scatterer population.
        q_I : array
            n-by-2 array of scattering vector (1/Angstrom) and intensities.
        Returns
        -------
        prediction : float or None
            predicted parameter
            None is reterned for models that was not trained yet
        """

        # for now the regressor works only for data with duffuse only populations:
        if population=='Noise' or population=='pop0_unidentified':
            return None


        if self.target == 'rg_0':
            additional_features = guinier_porod_profile(q_I)
            my_features = np.append(np.array(list(sample_features.values())),
                                    np.array(list(additional_features.values()))).reshape(1,-1)

        else:
            additional_features = spherical_normal_profile(q_I)
            my_features = np.append(np.array(list(sample_features.values())),
                                    np.array(list(additional_features.values()))).reshape(1,-1)

        if self.scaler: # we have a saved model
            x = self.scaler.transform(my_features)
            prediction = float(self.model.predict(x)[0])

            return prediction
        else:
            return None