import numpy as np
from .general_model import XrsdModel
from ..tools import profiler
from xrsdkit.tools.profiler import guinier_porod_profile, spherical_normal_profile

class Regressor(XrsdModel):
    """Models for prediction continuous values from scattering/diffraction data"""

    def __init__(self,label,yml_file_cl=None):

        XrsdModel.__init__(self, label, yml_file=yml_file_cl, classifier = False)

        self.n_groups_out = 1

        f = []
        f.extend(profiler.profile_keys_1)
        if label.startswith( 'r_g' ): # 'r_g_0', 'r_g_1', 'r_g_2'....
            f.extend(profiler.gp_profile_keys)
            self.features = f
        else:
            f.extend(profiler.spherical_profile_keys)
            self.features = f


    def predict(self, sample_features, populations, q_I):
        """Determine the types of structures represented by the sample

        Parameters
        ----------
        sample_features : OrderedDict
            OrderedDict of features with their values,
            similar to output of saxs_math.profile_spectrum()
        populations : dict
            dictionary counting scatterer populations,
            similar to output of SaxsClassifier.classify()
        q_I : array
            n-by-2 array of scattering vector (1/Angstrom) and intensities.

        Returns
        -------
        prediction : float or None
            predicted parameter
            None is reterned for models that was not trained yet
        """

        # for now the regressor works only for data with duffuse only populations:
        if populations['diffuse_structure_flag'][0]== 0:
            return None
        if populations['crystalline_structure_flag'][0]==1: # diffuse and crystaline pops
            return None

        if self.target == 'r_g_0' and populations['guinier_porod_population_count'][0]==0:
            return None

        if (self.target == 'sigma_0' or self.target == 'r0_0')\
                and populations['spherical_normal_population_count'][0]==0:
            return None

        if self.target == 'r_g_0':
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