import os
from pg import DB, connect
import paramiko
from xrsdkit.tools.psql_tools import load_yml_to_file_table, load_from_files_table_to_samples_table
from xrsdkit.tools.psql_tools import load_from_samples_to_training_table, get_training_dataframe
from xrsdkit.models import root_dir
from xrsdkit.tools.ymltools import downsample_by_group
from xrsdkit.models.train import train_from_dataframe

path_to_dir = "/afs/slac.stanford.edu/g/ssrl_dev/new_dataset_test"

api_passwords_file = os.path.join(root_dir, 'api_passwords.txt')
print(api_passwords_file)
db = None
client = None
df = None
if os.path.exists(api_passwords_file):
    afs_u_name, afs_pasw, db_u_name, db_pasw = open(api_passwords_file, 'r').readline().strip().split(",")
    db = DB(dbname='test_data', host='134.79.98.141', port=5432, user=db_u_name, passwd=db_pasw)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('134.79.34.167', username=afs_u_name, password=afs_pasw)


def test_load_yml_to_file_table():
    if db and client:
        load_yml_to_file_table(db, client, path_to_dir)

def test_load_from_files_table_to_samples_table():
    if db and client:
        load_from_files_table_to_samples_table(db, client)

def test_load_from_samples_to_training_table():
    if db:
        load_from_samples_to_training_table(db)

def test_get_training_dataframe():
    if db:
        df = get_training_dataframe(db)

def test_if_the_result_is_trainable():
    if df:
        df_sample = downsample_by_group(df)
        train_from_dataframe(df_sample)
