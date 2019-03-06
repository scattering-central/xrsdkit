from .ymltools import read_local_dataset, create_modeling_dataset
from ..db import gather_remote_dataset
from ..models.train import train_from_dataframe

def train_models_from_local_dataset(dataset_dir,downsampling_distance=1.):
    df = read_local_dataset(dataset_dir,downsampling_distance=downsampling_distance) 
    train_from_dataframe(df,train_hyperparameters=True,save_models=True,test=False)

def train_models_from_remote_dataset(dataset_dir,downsampling_distance=1.):
    df = gather_remote_dataset(dataset_dir,downsampling_distance=downsampling_distance)
    train_from_dataframe(df,train_hyperparameters=True,save_models=True,test=False)

