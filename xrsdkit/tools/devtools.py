from __future__ import print_function
import os

from .ymltools import read_local_dataset
from ..db import gather_remote_dataset
from ..models.train import train_from_dataframe
from ..models import load_models

def train_on_local_dataset(dataset_dir, output_dir=None, model_config_path=None, downsampling_distance=1.):
    df = read_local_dataset(dataset_dir, downsampling_distance=downsampling_distance) 
    reg_models, cls_models = train_from_dataframe(df, 
            train_hyperparameters=True, select_features=True, 
            output_dir=output_dir, model_config_path=model_config_path, message_callback=print)
    return reg_models, cls_models

def train_on_remote_dataset(dataset_dir, output_dir, conf_file=None, downsampling_distance=1.):
    df = gather_remote_dataset(dataset_dir, downsampling_distance=downsampling_distance)
    train_from_dataframe(df, train_hyperparameters=True, select_features=True, 
            output_dir=output_dir, model_config_path=model_config_path, message_callback=print)

def dataset_to_csv(dataset_dir, downsampling_distance=1.):
    df = read_local_dataset(dataset_dir, downsampling_distance=downsampling_distance) 
    output_path = os.path.join(dataset_dir, 'dataset.csv')
    df.to_csv(output_path)

