import os

import numpy as np
from citrination_client import CitrinationClient

from xrsdkit.tools import profiler
from xrsdkit.tools.citrination_tools import get_data_from_Citrination 
from xrsdkit.models import predict, root_dir, model_dsids, downsample_by_group, train_from_dataframe
from xrsdkit.visualization import visualize_dataframe

# TODO: package a small dataframe for use in testing,
# so that we don't have to download during tests 

def download_pifs():
    api_key_file = os.path.join(root_dir, 'api_key.txt')
    df = None
    if os.path.exists(api_key_file):
        a_key = open(api_key_file, 'r').readline().strip()
        cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)
        #df, _ = get_data_from_Citrination(cl,[model_dsids['system_classifier']]) 
        df, _ = get_data_from_Citrination(cl,[21,22,23,24,25]) 
    return df

df = download_pifs()

def test_training():
    if df is not None:
        train_from_dataframe(df,train_hyperparameters=False,save_models=True,test=True)

def test_visualization():
    if df is not None:
        visualize_dataframe(df)

def test_downsampling():
    if df is not None:
        downsample_by_group(df) 

def test_predict_spheres():
    datapath = os.path.join(os.path.dirname(__file__),
        'test_data','solution_saxs','spheres','spheres_0.csv')
    f = open(datapath,'r')
    q_I = np.loadtxt(f,dtype=float,delimiter=',')
    feats = profiler.profile_spectrum(q_I)
    # models will only be trained if a dataframe was downloaded
    if df is not None:
        pred = predict(feats,test=True)




