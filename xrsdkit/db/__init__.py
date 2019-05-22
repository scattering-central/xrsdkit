"""This subpackage defines the API to communicate with the xrsdkit database (PostgreSQL).

This API requires the pygresql Python package.
In turn, pygresql requires PostgreSQL packages to be installed on the local host.
At the time of development, Unix platforms require libpq and libpq-dev.
The setup of PostgreSQL is outlined in the main xrsdkit package documentation.

The operations for managing this database involve up to three hosts.
One, the localhost (your machine) running the program.
Two, the storage host, where YAML-formatted xrsdkit results are saved.
Three, the database host, where tables of xrsdkit data are managed.
The localhost machine may serve all three of these purposes at small scale.
For installations at scale, separate specialized hosts are probably necessary.

To facilitate connections between the local host and the storage and db hosts,
the localhost user must have two files in their home directory:
'.xrsdkit_db_host' and '.xrsdkit_storage_host'.
Developers planning on running tests against a test database
should also have a '.xrsdkit_test_db_host' file.

.xrsdkit_db_host / .xrsdkit_test_db_host:
    line 1: db_host_name:db_server_port
    line 2: database title 
    line 3: database username
    line 4: database user password

example:
192.99.99.999:7653
test_db
db_user
clever_password

.xrsdkit_storage_host:
    line 1: storage_host_name
    line 2: path to training dataset directory on storage host
    line 3: path to test dataset directory on storage host (used only for testing, may be left empty)
    line 4: username on storage host
    line 5: path to user's private ssh key file on local host

example:
192.99.99.998
/path/to/training/dataset
/path/to/test/dataset
storage_user
/home/.ssh/id_rsa

The database server must be running on a host 
that shares a network connection with the local machine.
This database host can be the localhost, or a remote server/virtual machine.

Communications are facilitated by DB connector objects.
To create a connector:
    from pg import DB, connect
    db = DB(dbname='PSQL_DB_NAME', host='PSQL_HOST_ADDRESS', port=PSQL_PORT, user='PSQL_USERNAME', passwd='PSQL_PASSWORD')

File operations are facilitated by SSH clients 
(requiring the paramiko Python package).
Instructions on using paramiko to set up SSH clients 
can be found in the paramiko documentation.
At the time of development:
https://docs.paramiko.org/en/2.4/api/client.html

The data pipeline supported by this module is as follows:
    directory of training data (on any host) 
    -> DB files table (on DB host)
    -> DB samples table (on DB host)
    -> DB training table (on DB host)
    -> pandas.DataFrame (on localhost)
Before using this pipeline, the local host 
must have access to a remote (or local) directory 
that contains the training dataset.
This directory should have sub-directories for each experiment.

The "files" table has the following columns:
    sample_id     | character varying
    experiment_id | character varying
    good_fit      | boolean
    yml_path      | text

The "samples" table has the following columns:
    samples columns:
    sample_id             | character varying
    experiment_id         | character varying
    features              | json
    regression_labels     | json
    classification_labels | json

The "training" table has the following columns:
    sample_id              | character varying
    experiment_id          | character varying
    <feature_name_1>       | numeric
    <feature_name_2>       | numeric
    ...                    | ... 
    <feature_name_N>       | numeric
    <class_name_0>         | character varying
    <class_name_1>         | character varying
    ...                    | ... 
    <class_name_N>         | character varying
    <parameter_name_1>     | numeric
    <parameter_name_2>     | numeric
    ...                    | ... 
    <parameter_name_N>     | numeric

The "training" table has numeric columns for each feature, 
character columns for each classification label,
and numeric columns for each regression (parameter) label.
When we are inserting a new sample with labels that are not in the table,
the new colums are appended to the table.

Assuming a user has collected data from an experiment,
processed the data into .yml files,
and stored the .yml files in the dataset directory, 
the user should do the following to include their data into the database:
    > load_yml_to_file_table(db, client, path_to_dir)
    > load_from_files_table_to_samples_table(db, client)
    > load_from_samples_to_training_table(db)

To train or re-train xrsdkit models from the "training" table,
the user should then download the data from the database:
    > df = get_training_dataframe(db)

The DataFrame `df` is then used to train new classifiers and regression models
with methods in the xrsdkit.models.train subpackage.
"""
import os
from collections import OrderedDict
import warnings

import yaml
import pandas as pd
import paramiko
use_pg = True
try:
    from pg import DB, connect
except:
    use_pg = False

from ..tools.ymltools import unpack_sample, create_modeling_dataset
from ..tools.profiler import profile_keys

user_home_dir = os.path.expanduser('~')
storage_host_info_file = os.path.join(user_home_dir,'.xrsdkit_storage_host')
db_host_info_file = os.path.join(user_home_dir,'.xrsdkit_db_host')
test_db_host_info_file = os.path.join(user_home_dir,'.xrsdkit_test_db_host')

storage_client = None
storage_path = None
storage_path_test = None
try:
    if os.path.exists(storage_host_info_file):
        storage_host_lines = open(storage_host_info_file,'r').readlines()
        storage_host = storage_host_lines[0].strip()
        storage_path = storage_host_lines[1].strip()
        storage_path_test = storage_host_lines[2].strip()
        storage_user = storage_host_lines[3].strip()
        private_key_file = storage_host_lines[4].strip()
        storage_client = paramiko.SSHClient()
        storage_client.load_system_host_keys()
        storage_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        storage_client.connect(storage_host, username=storage_user, key_filename=private_key_file)
    else:
        warnings.warn('storage host info file not found')
except:
    warnings.warn('unable to establish connection to storage host')

db_connector = None
try:
    if os.path.exists(db_host_info_file): 
        db_host_lines = open(db_host_info_file,'r').readlines()
        db_host,db_port = db_host_lines[0].split(':')
        db_name = db_host_lines[1].strip()
        db_user = db_host_lines[2].strip()
        db_key = db_host_lines[3].strip()
        db_connector = DB(dbname=db_name, host=db_host, port=int(db_port), user=db_user, passwd=db_key)
    else:
        warnings.warn('database host info file not found')
except:
    warnings.warn('unable to establish connection to database host')

test_db_connector = None
try:
    if os.path.exists(test_db_host_info_file): 
        db_host_lines = open(db_host_info_file,'r').readlines()
        db_host,db_port = db_host_lines[0].split(':')
        db_name = db_host_lines[1].strip()
        db_user = db_host_lines[2].strip()
        db_key = db_host_lines[3].strip()
        test_db_connector = DB(dbname=db_name, host=db_host, port=int(db_port), user=db_user, passwd=db_key)
    else:
        warnings.warn('test database host info file not found')
except:
    warnings.warn('unable to establish connection to test database host')


def load_yml_to_file_table(db, path_to_dir, drop_table=False):
    """Add data to the 'files' table from a directory on any remote machine.

    The data directory should contain one or more subdirectories,
    where each subdirectory contains .yml files,
    where each .yml file describes one sample,
    as saved by save_sys_to_yaml().

    Parameters
    ----------
    db : pg.DB 
        A database connector (DB object from PyGreSQL)
    path_to_dir : str
        absolute path to the directory with the training set
        Precondition: dataset directory includes directories named
        by the name of the experiments; each experiment directory
        holds a set of .yml files that contain all the data from this experiment. 
        Each .yml file has all the features and labels 
        to describe the System object that was fit to the sample.
    drop_table : bool
        If True, the existing table will be dropped,
        and a new table will be created from scratch,
        else, only data that are not already in the table will be added.
    """
    if drop_table:
        db.query("DROP TABLE files")
    db.query("CREATE TABLE IF NOT EXISTS files(sample_id VARCHAR PRIMARY KEY, "
                                            "experiment_id VARCHAR, good_fit BOOLEAN, "
                                            "yml_path TEXT)")
    # get the list of experiments that are already in the table
    exp_from_table = db.query('SELECT DISTINCT experiment_id FROM files').getresult()
    exp_from_table = [row[0] for row in exp_from_table]
    all_sys_dicts = download_sys_data(path_to_dir)
    all_sample_ids = []
    for file_path,sys_dict in all_sys_dicts.items():
        expt_id = sys_dict['sample_metadata']['experiment_id']
        sample_id = sys_dict['sample_metadata']['sample_id']
        # make sure the experiment_id is not yet in the table
        if expt_id in exp_from_table:
            warnings.warn('Skipping duplicate experiment id: {}'.format(expt_id))
        else:
            if sample_id in all_sample_ids:
                warnings.warn('Skipping duplicate sample id: {}'.format(sample_id)) 
            else: 
                all_sample_ids.append(sample_id)
                # add attributes and file path to the files table 
                db.insert('files', sample_id=sample_id, experiment_id=expt_id,
                    yml_path=file_path, good_fit=sys_dict['fit_report']['good_fit'])


def download_sys_data(path_to_dir):
    """Download xrsdkit.system.System data from remote directory.

    Parameters
    ----------
    path_to_dir : str
        absolute path to the directory with the training set

    Returns
    -------
    all_sys_dicts : OrderedDict
        Dictionary mapping file paths (keys) to 
        dicts (values) describing xrsdkit.system.System objects
    """
    # get the list of experiments that are in the dataset directory
    all_sys_dicts = OrderedDict() 
    stdin, stdout, stderr = storage_client.exec_command('ls '+path_to_dir)
    for experiment_id in stdout:
        experiment_id = experiment_id.strip('\n')
        exp_data_dir = os.path.join(path_to_dir,experiment_id)
        # make sure exp_data_dir is a directory
        if storage_client.exec_command("os.path.isdir(exp_data_dir)"):
            # get the list of files in the experiment directory
            stdin2, stdout2, stderr2 = storage_client.exec_command('ls '+exp_data_dir)
            for s_data_file in stdout2:
                s_data_file = s_data_file.strip('\n')
                # if the file is a .yml file, attempt to load it 
                if s_data_file.endswith('.yml'):
                    print('downloading sample data from {}'.format(s_data_file))
                    file_path = os.path.join(exp_data_dir, s_data_file)
                    # use cat to grab the file content and dump it to a stream
                    stdin, stdout, stderr = storage_client.exec_command('cat ' + file_path)
                    net_dump = stdout.readlines()
                    str_d = "".join(net_dump)
                    # load the stream to data as yaml, grab key attributes
                    sys_dict = yaml.load(str_d)
                    all_sys_dicts[file_path] = sys_dict
    return all_sys_dicts


def load_from_files_table_to_samples_table(db, drop_table=False):
    """Process the data from a the "files" table and insert corresponding rows into the "samples" table.

    Parameters
    ----------
    db : pg.DB 
        a database connector (DB object from PyGreSQL)
    drop_table : bool
        If True, the existing table will be dropped,
        and a new table will be created from scratch,
        else, only data that are not already in the table will be added.
    """
    if drop_table:
        db.query("DROP TABLE samples")
    db.query("CREATE TABLE IF NOT EXISTS samples(sample_id VARCHAR PRIMARY KEY, "
                                                "experiment_id VARCHAR, "
                                                "features JSON, "
                                                "regression_labels JSON, "
                                                "classification_labels JSON )")

    # get the list of the experiments that are not in the "samples" table
    # NOTE: should we allow this function to add samples to an existing experiment_id?
    new_experiments = db.query("SELECT DISTINCT experiment_id "
                               " FROM files "
                               " WHERE experiment_id NOT IN ("
                                    "SELECT DISTINCT experiment_id "
                                    "FROM samples "
                                    "WHERE experiment_id IS NOT NULL)").getresult()
    new_experiments = [row[0] for row in new_experiments]

    for ex in new_experiments:
        print('reading data from {}'.format(ex))
        q = "SELECT yml_path FROM files WHERE experiment_id = '{}' AND good_fit = true".format(ex)
        experiment_files = [row[0] for row in db.query(q).getresult()]
        print('done - found {} records'.format(len(experiment_files)))

        for f in experiment_files:
            # get the content of this file
            stdin, stdout, stderr = storage_client.exec_command('cat ' + f)
            net_dump = stdout.readlines()
            str_d = "".join(net_dump)
            pp = yaml.load(str_d)
            expt_id, sample_id, good_fit, features, \
                classification_labels, regression_labels = unpack_sample(pp)
            # add a new row to the table "samples"
            db.insert('samples', sample_id=sample_id, experiment_id = expt_id,
                                  features=features, regression_labels=regression_labels,
                                    classification_labels=classification_labels)


def load_from_samples_to_training_table(db, drop_table=False):
    """Process the data from a the "samples" table and insert corresponding rows into the "training" table.

    This unpacks JSON columns from the samples table,
    so that the training table has distinct columns for each feature and label.
    Only data from new experiments will be added.

    Parameters
    ----------
    db : db.pg object
        a database connection - DB object from PyGreSQL
    drop_table : bool
        If True, the existing table will be dropped,
        and a new table will be created from scratch,
        else, only data that are not already in the table will be added.
    """
    if drop_table:
            db.query("DROP TABLE training")

    # get all existing classification labes:
    q = 'SELECT json_object_keys(classification_labels) FROM samples'
    all_cl_labels = set([r[0] for r in db.query(q).getresult()])

    # get all existing regression labels:
    q = 'SELECT json_object_keys(regression_labels) FROM samples'
    all_reg_labels = set([r[0] for r in db.query(q).getresult()])

    # notes: this table will be used for creating training dataframe;
    # the columns of this table must have the exactly the same formatting
    # as profiler.profiler_keys including low/upper cases
    q = 'CREATE TABLE IF NOT EXISTS training(sample_id VARCHAR PRIMARY KEY, experiment_id VARCHAR, "' + \
        '" NUMERIC, "'.join(profile_keys) + '" NUMERIC)'
    db.query(q)

    # add new columns if needed (for new classification and regression labels)
    q = 'ALTER TABLE training ADD COLUMN IF NOT EXISTS "' + \
        '" VARCHAR, ADD COLUMN IF NOT EXISTS "'.join(all_cl_labels) + '" VARCHAR, ADD COLUMN IF NOT EXISTS "' + \
        '" NUMERIC, ADD COLUMN IF NOT EXISTS "'.join(all_reg_labels) + '" NUMERIC'
    db.query(q)

    # get the list of experiments that are not in the 'training' table
    new_experiments = db.query("SELECT DISTINCT experiment_id "
                               " FROM samples "
                               " WHERE experiment_id NOT IN ("
                                    "SELECT DISTINCT experiment_id "
                                    "FROM training "
                                    "WHERE experiment_id IS NOT NULL)").getresult()
    new_experiments = [row[0] for row in new_experiments]

    # get data from the "samples" table for all experiments that are not in the "training" table:
    for ex in new_experiments:
        print('reading data from {}'.format(ex))
        q = "SELECT * FROM samples WHERE experiment_id = '{}'".format(ex)
        experiment_data = db.query(q).dictresult() # list of dict
        print('done - found {} records'.format(len(experiment_data)))

        for f in experiment_data:
            features = OrderedDict(f['features'])
            cl_labels = OrderedDict(f['classification_labels'])
            reg_labels = OrderedDict(f['regression_labels'])
            # add a new row to the table "samples"
            feature_values_as_str = [str(v) for v in features.values()]
            reg_values_as_str = [str(v) for v in reg_labels.values()]

            # the next query: 'INSERT INTO training (sample_id, experiment_id, "Feature_1", "Feature_2", ...
            # , "Classification_label_1", .... , "Regression_label_1") VALUES ('R1_000_34', 'R1', '0.7855869',...'
            # We need to use double quotes for feature names since we want to preserve the letter cases.
            q_row = 'INSERT INTO training (sample_id, experiment_id, "' + '", "'.join(features.keys()) +\
                    '", "'+'", "'.join(cl_labels.keys()) +'", "'+ '", "'.join(reg_labels.keys()) + \
                    '") VALUES (' +"'"+ f['sample_id'] +"'" + ", " + "'"+  f['experiment_id'] +"'" + ", " + \
                    ', '.join(feature_values_as_str) +", '"+ "', '".join(f['classification_labels'].values()) + \
                    "', "+', '.join(reg_values_as_str) + ")"
            db.query(q_row)


def get_training_dataframe(db):
    """Create a Pandas DataFrame from a postgreSQL table.

    Parameters
    ----------
    db : pg.DB
        a database connector (DB object from PyGreSQL)

    Returns
    -------
    df : pandas.DataFrame
        dataframe containing features and labels
    """
    data = db.query('SELECT * FROM training').dictresult()
    df = pd.DataFrame(data)
    return df


def gather_remote_dataset(dataset_dir,downsampling_distance=None):
    # use storage_client to gather system dicts
    sys_data = download_sys_data(dataset_dir)
    all_sys_dicts = list(sys_data.values())
    # build modeling dataframe from system dicts
    df = create_modeling_dataset(all_sys_dicts,downsampling_distance=downsampling_distance)
    return df

