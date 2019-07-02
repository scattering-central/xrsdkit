import shutil
import os

import numpy as np
import pandas as pd

from xrsdkit.tools import ymltools as xrsdyml 
from xrsdkit.tools import profiler
from xrsdkit import models as xrsdmods
from xrsdkit.models.train import train_from_dataframe
from xrsdkit.models.predict import predict, system_from_prediction 
from xrsdkit.visualization import visualize_dataframe

data_dir = os.path.join(os.path.dirname(__file__),'test_data')
ds1_path = os.path.join(data_dir,'dataset_1')
ds2_path = os.path.join(data_dir,'dataset_2')
temp_models_dir = os.path.join(data_dir,'modeling_data')

df,idxs = xrsdyml.read_local_dataset([ds1_path,ds2_path],downsampling_distance=1.)
df_ds = df 

def test_visualization():
    if df is not None and 'DISPLAY' in os.environ:
        visualize_dataframe(df)

# test prediction on loaded models
def test_predict_0():
    datapath = os.path.join(data_dir,
        'solution_saxs','spheres','spheres_0.dat')
    sysfpath = os.path.splitext(datapath)[0]+'.yml'
    f = open(datapath,'r')
    q_I = np.loadtxt(f,dtype=float)
    feats = profiler.profile_pattern(q_I[:,0],q_I[:,1])
    try:
        pred = predict(feats)
        sys = system_from_prediction(pred,q_I[:,0],q_I[:,1],source_wavelength=0.8265617)
    except RuntimeError:
        pass

# train new models
def test_training():
    if df_ds is not None:
        train_from_dataframe(df_ds,train_hyperparameters=False,select_features=False,output_dir=temp_models_dir)

# test prediction on newly trained models
def test_predict_1():
    datapath = os.path.join(data_dir,
        'solution_saxs','spheres','spheres_0.dat')
    sysfpath = os.path.splitext(datapath)[0]+'.yml'
    f = open(datapath,'r')
    q_I = np.loadtxt(f,dtype=float)
    feats = profiler.profile_pattern(q_I[:,0],q_I[:,1])
    # models will only be trained if a dataframe was downloaded
    if df_ds is not None:
        # load new models
        xrsdmods.load_models(temp_models_dir)
        pred = predict(feats)
        sys = system_from_prediction(pred,q_I[:,0],q_I[:,1],source_wavelength=0.8265617)
        xrsdyml.save_sys_to_yaml(sysfpath,sys)
        os.remove(sysfpath)
        # throw away the temporary modeling files
        shutil.rmtree(temp_models_dir)

