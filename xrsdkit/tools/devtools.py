import os

from .ymltools import read_local_dataset
from ..db import gather_remote_dataset
from ..models.train import train_from_dataframe
from ..models import modeling_data_dir, load_models

def train_models_from_local_dataset(dataset_dir, output_dir, downsampling_distance=1.):
    df = read_local_dataset(dataset_dir,downsampling_distance=downsampling_distance) 
    train_from_dataframe(df,output_dir,train_hyperparameters=True,select_features=True,save_models=True)

def train_models_from_remote_dataset(dataset_dir, output_dir, downsampling_distance=1.):
    df = gather_remote_dataset(dataset_dir,downsampling_distance=downsampling_distance)
    train_from_dataframe(df,output_dir,train_hyperparameters=True,select_features=True,save_models=True)

def dataset_to_csv(dataset_dir,downsampling_distance=1.):
    df = read_local_dataset(dataset_dir,downsampling_distance=downsampling_distance) 
    output_path = os.path.join(dataset_dir,'dataset.csv')
    df.to_csv(output_path)

def update_models(models_dir):
    load_models(models_dir, modeling_data_dir)

