import os
import copy
from collections import OrderedDict

import yaml
import numpy as np
from sklearn import preprocessing,linear_model

from . import saxs_math, saxs_fit
from . import parameter_keys, all_parameter_keys 
from . import peak_math, peak_finder

class SaxsRegressor(object):
    """A set of regression models to be used on SAXS spectra"""

    def __init__(self,yml_file=None):
        if yml_file is None:
            p = os.path.abspath(__file__)
            d = os.path.dirname(p)
            yml_file = os.path.join(d,'modeling_data','scalers_and_models_regression.yml')

        s_and_m_file = open(yml_file,'rb')
        s_and_m = yaml.load(s_and_m_file)

        reg_models_dict = s_and_m['models']
        scalers_dict = s_and_m['scalers']

        self.models = OrderedDict.fromkeys(all_parameter_keys)
        self.scalers = OrderedDict.fromkeys(all_parameter_keys)
        reg_models = reg_models_dict.keys()
        for model_name in reg_models:
            model_params = reg_models_dict[model_name]
            scaler_params = scalers_dict[model_name]
            if scaler_params is not None:
                s = preprocessing.StandardScaler()
                self.set_param(s,scaler_params)
                m = linear_model.SGDRegressor()
                self.set_param(m,model_params)
            self.models[model_name] = m
            self.scalers[model_name] = s

    # helper function - to set parameters for scalers and models
    def set_param(self, m_s, param):
        for k, v in param.items():
            if isinstance(v, list):
                setattr(m_s, k, np.array(v))
            else:
                setattr(m_s, k, v)

    def predict_params(self,populations,features,q_I):
        """Evaluate the scattering parameters of a sample.

        Parameters
        ----------
        populations : dict
            dictionary counting scatterer populations,
            similar to output of SaxsClassifier.classify()
        features : dict
            dictionary of sample numerical features,
            similar to output of saxs_math.profile_spectrum().
        q_I : array 
            n-by-2 array of scattering vector (1/Angstrom) and intensities. 

        Returns
        -------
        params : dict
            dictionary of with predicted parameters
        """
        feature_array = np.array(list(features.values())).reshape(1,-1)

        params = OrderedDict()    
        fixed_params = OrderedDict()    
        if bool(populations['unidentified']):
            return params 

        # TODO: The predictions need to handle 
        # multiple populations of the same type:
        # include this once we have training data

        params['I0_floor'] = [saxs_fit.param_defaults['I0_floor']]

        if bool(populations['spherical_normal']):
            params.update(OrderedDict.fromkeys(parameter_keys['spherical_normal']))
            #if self.scalers['r0_sphere'] != None:
            x = self.scalers['r0_sphere'].transform(feature_array)
            r0sph = self.models['r0_sphere'].predict(x)
            params['r0_sphere'] = [r0sph[0]]
            fixed_params['r0_sphere'] = [r0sph[0]]

            #if self.scalers['sigma_sphere'] != None:
            additional_features = saxs_math.spherical_normal_profile(q_I)
            ss_features = np.append(feature_array, np.array(list(additional_features.values()))).reshape(1,-1)
            x = self.scalers['sigma_sphere'].transform(ss_features)
            sigsph = self.models['sigma_sphere'].predict(x)
            params['sigma_sphere'] = [sigsph[0]]
            params['I0_sphere'] = [saxs_fit.param_defaults['I0_sphere']]

        if bool(populations['guinier_porod']):
            #if self.scalers['rg_gp'] != None:
            additional_features = saxs_math.guinier_porod_profile(q_I)
            rg_features = np.append(feature_array, np.array(list(additional_features.values()))).reshape(1,-1)
            x = self.scalers['rg_gp'].transform(rg_features)
            rg = self.models['rg_gp'].predict(x)
            params['rg_gp'] = [rg[0]]
            params['D_gp'] = [4.]
            params['G_gp'] = [saxs_fit.param_defaults['G_gp']]

        if bool(populations['diffraction_peaks']):
            
            # 1) walk the spectrum, collect best diff. pk. candidates
            pk_idx, pk_conf = peak_finder.peaks_by_window(q_I[:,0],q_I[:,1],20,0.)
            conf_idx = np.argsort(pk_conf)[::-1]
            params['q_pkcenter'] = []
            params['I_pkcenter'] = []
            params['pk_hwhm'] = []
            npk = 0
            # 2) for each peak (from best candidate to worst),
            for idx in conf_idx:
                if npk < populations['diffraction_peaks']:
                    # a) record the q value
                    q_pk = q_I[:,0][pk_idx[idx]]
                    # b) estimate the intensity
                    I_at_qpk = q_I[:,1][pk_idx[idx]]
                    I_pk = I_at_qpk * 0.1
                    #I_pk = I_at_qpk - I_nopeaks[pk_idx[idx]] 
                    # c) estimate the width
                    idx_around_pk = (q_I[:,0]>0.95*q_pk) & (q_I[:,0]<1.05*q_pk)
                    qs,qmean,qstd = saxs_math.standardize_array(q_I[idx_around_pk,0])
                    Is,Imean,Istd = saxs_math.standardize_array(q_I[idx_around_pk,1])
                    p_pk = np.polyfit(qs,Is,2,None,False,np.ones(len(qs)),False)
                    # quadratic vertex horizontal coord is -b/2a
                    #qpk_quad = -1*p_pk[1]/(2*p_pk[0])
                    # quadratic focal width is 1/a 
                    p_pk_fwidth = abs(1./p_pk[0])*qstd
                    params['q_pkcenter'].append(q_pk)
                    params['I_pkcenter'].append(I_pk)
                    params['pk_hwhm'].append(p_pk_fwidth*0.5)
                    npk += 1    

        sxf = saxs_fit.SaxsFitter(q_I,populations)
        p_fit, rpt = sxf.fit_intensity_params(params) 
        params = saxs_fit.update_params(params,p_fit)

        return params

