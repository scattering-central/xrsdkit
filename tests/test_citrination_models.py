import os
import numpy as np

from xrsdkit.tools import profiler
from xrsdkit.models.citrination_models import CitrinationSystemClassifier
from xrsdkit.models import root_dir

# TODO: replace example data files with non-delimited .dat files

def test_citrination_models():
    api_key_file = os.path.join(root_dir, 'api_key.txt')
    if not os.path.exists(api_key_file):
        print('api_key.txt not found: skipping Citrination test')
        return
    test_path = os.path.join(root_dir,'tests','test_data','solution_saxs','spheres','spheres_1.csv')
    q_I = np.loadtxt(test_path, delimiter=',')
    features = profiler.profile_spectrum(q_I)
    sys_cls_model = CitrinationSystemClassifier(api_key_file,'https://slac.citrination.com')
    print('testing Citrination system classifier on {}'.format(test_path))
    sys_cls, unc = sys_cls_model.classify(features)
    print('system class: {}, uncertainty: {}'.format(sys_cls,unc))



