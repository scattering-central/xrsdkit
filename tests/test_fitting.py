from collections import OrderedDict
import os

import numpy as np

#from xrsdkit.models.saxs_classify import SaxsClassifier
#from xrsdkit.models.saxs_regression import SaxsRegressor
from xrsdkit.tools import profiler 
from xrsdkit.fitting.xrsd_fitter import XRSDFitter 

def test_fit():
    pass
    #datapath = os.path.join(os.path.dirname(__file__),
    #    'test_data','solution_saxs','spheres','spheres_0.csv')
    #print('testing SaxsFitter on {}'.format(datapath))
    #datapath = os.path.join(os.path.dirname(__file__),
    #    'test_data','solution_saxs','peaks','peaks_0.csv')
    #test_data = open(datapath,'r')
    #q_I = np.loadtxt(test_data,dtype=float,delimiter=',')
    #p = os.path.dirname(os.path.abspath(__file__))
    #d = os.path.dirname(p)
    #pmod_cls = os.path.join(d,'xrsdkit','models','modeling_data','scalers_and_models.yml')
    #pmod_reg = os.path.join(d,'xrsdkit','models','modeling_data','scalers_and_models_regression.yml')
    #sxc = SaxsClassifier(pmod_cls)
    #prof = profiler.profile_spectrum(q_I)

    # TODO: make all models work with profile_spectrum() output
    #tmp_prof = OrderedDict()
    #for k in prof.keys():
    #    if prof[k] is not None:
    #        tmp_prof[k] = prof[k]

    #pops,certs = sxc.classify(tmp_prof)
    #sxr = SaxsRegressor(pmod_reg)
    #params = sxr.predict_params(pops,tmp_prof,q_I)
    #sxf = XRSDFitter(q_I,pops)
    #params,rpt = sxf.fit_intensity_params(params)
    #p_opt,rpt_opt = sxf.fit(params)
    #obj_init = sxf.evaluate(params)
    #obj_opt = sxf.evaluate(p_opt)
    #print('optimization objective: {} --> {}'.format(obj_init,obj_opt))
    #for k, v in params.items():
    #    print('\t{}: {} --> {}'.format(k,v,p_opt[k]))


