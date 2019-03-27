import os
from xrsdkit.models import root_dir
from xrsdkit.tools.ymltools import downsample_by_group
from xrsdkit.models.train import train_from_dataframe
from xrsdkit.db import load_yml_to_file_table, load_from_files_table_to_samples_table
from xrsdkit.db import load_from_samples_to_training_table, get_training_dataframe
from xrsdkit.db import storage_client, storage_path, test_db_connector

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
testing_data_dir = os.path.join(src_dir,'xrsdkit','models','modeling_data', 'test')
if not os.path.exists(testing_data_dir): os.mkdir(testing_data_dir)

def test_load_yml_to_file_table():
    if test_db_connector and storage_client and storage_path:
        load_yml_to_file_table(test_db_connector, storage_path)

def test_load_from_files_table_to_samples_table():
    if test_db_connector and storage_client:
        load_from_files_table_to_samples_table(test_db_connector)

def test_load_from_samples_to_training_table():
    if test_db_connector:
        load_from_samples_to_training_table(test_db_connector)

df = None
def test_get_training_dataframe():
    if test_db_connector:
        df = get_training_dataframe(test_db_connector)

def test_if_the_result_is_trainable():
    if df:
        df_sample = downsample_by_group(df)
        train_from_dataframe(df_sample,testing_data_dir)

