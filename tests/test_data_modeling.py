import os

import numpy as np
import pandas as pd

from xrsdkit.tools import ymltools as xrsdyml 
from xrsdkit.tools import profiler
from xrsdkit.models.train import train_from_dataframe
from xrsdkit.models.predict import predict, system_from_prediction 
from xrsdkit.visualization import visualize_dataframe
from xrsdkit.visualization.gui import run_fit_gui 

datapath = os.path.join(os.path.dirname(__file__),
        'test_data','dataset.csv')
df = None
if os.path.exists(datapath):
    print('loading cached dataset from {}'.format(datapath))
    df = pd.read_csv(datapath)

def test_visualization():
    if df is not None and 'DISPLAY' in os.environ:
        visualize_dataframe(df)

def downsample_df():
    df_ds = None
    if df is not None:
        df_ds = xrsdyml.downsample_by_group(df) 
    return df_ds

df_ds = downsample_df()

def test_training():
    if df_ds is not None:
        train_from_dataframe(df_ds,train_hyperparameters=False,save_models=True,test=True)

def test_predict_spheres():
    datapath = os.path.join(os.path.dirname(__file__),
        'test_data','solution_saxs','spheres','spheres_0.dat')
    sysfpath = os.path.splitext(datapath)[0]+'.yml'
    f = open(datapath,'r')
    q_I = np.loadtxt(f,dtype=float)
    feats = profiler.profile_pattern(q_I[:,0],q_I[:,1])
    # models will only be trained if a dataframe was downloaded
    if df_ds is not None:
        pred = predict(feats,test=True)
        sys = system_from_prediction(pred,q_I[:,0],q_I[:,1],source_wavelength=0.8265617)
        xrsdyml.save_sys_to_yaml(sysfpath,sys)
        if 'DISPLAY' in os.environ:
            fit_sys = run_fit_gui({datapath:sysfpath})

