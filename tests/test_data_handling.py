import os

from citrination_client import CitrinationClient
from xrsdkit.tools.citrination_tools import get_data_from_Citrination 
from xrsdkit.models import root_dir, model_dsids, train_from_dataframe
from xrsdkit.visualization import visualize_dataframe

def test_download():
    api_key_file = os.path.join(root_dir, 'api_key.txt')
    df = None
    if os.path.exists(api_key_file):
        a_key = open(api_key_file, 'r').readline().strip()
        cl = CitrinationClient(site='https://slac.citrination.com',api_key=a_key)
        #df, _ = get_data_from_Citrination(cl,[model_dsids['system_classifier']]) 
        df, _ = get_data_from_Citrination(cl,[22,23,30]) 
    return df

df = test_download()

def test_training():
    if df is not None:
        train_from_dataframe(df,train_hyperparameters=False,test=True)

def test_visualization():
    if df is not None:
        visualize_dataframe(df,show_plots=False,save_plots=False)



