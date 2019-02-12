"""This module contains the API to communicate with the xrsdkit database (PostgreSQL).

This API requires the pygresql Python package.
The database server must be running on a machine 
that shares a network connection with the local machine.
This can be the localhost itself, a server, or a virtual machine.

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
    directory of training data (on DB host) 
    -> DB files table 
    -> DB samples table 
    -> DB training table 
    -> pandas.DataFrame on localhost
Before using this pipeline, the database host machine
must have access to a directory containing the training dataset.
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
and stored the .yml files in a directory on the DB host machine,
the user should do the following to include their data into the database:
    > load_yml_to_file_table(db, client, path_to_dir)
    > load_from_files_table_to_samples_table(db, client)
    > load_from_samples_to_training_table(db)

To train or re-train xrsdkit models from the "training" table,
the user should then download the data from the database:
    > df = get_training_dataframe(db)

The DataFrame `df` is then used to train new classifiers and regression models
with xrsdkit.models.train.
"""
import os
from collections import OrderedDict

import yaml
import pandas as pd

from .ymltools import unpack_sample
from .profiler import profile_keys

def load_yml_to_file_table(db, ssh_client, path_to_dir, drop_table=False):
    """Add data to the 'files' table from a directory on any remote machine.

    The data directory should contain one or more subdirectories,
    where each subdirectory contains .yml files,
    where each .yml file describes one sample,
    as saved by save_sys_to_yaml().

    Parameters
    ----------
    db : pg.DB 
        A database connector (DB object from PyGreSQL)
    ssh_client : paramiko.SSHClient
        ssh client connected with the database host machine
    path_to_dir : str
        absolute path to the folder with the training set
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

    # get the list of experiments that are in the dataset directory
    stdin, stdout, stderr = ssh_client.exec_command('ls '+path_to_dir)

    for experiment in stdout:
        experiment = experiment.strip('\n')
        exp_data_dir = os.path.join(path_to_dir,experiment)
        # make sure exp_data_dir is a directory, and that the experiment is not yet in the table 
        if ssh_client.exec_command("os.path.isdir(exp_data_dir)") and experiment not in exp_from_table:
            # get the list of files in the experiment directory
            stdin2, stdout2, stderr2 = ssh_client.exec_command('ls '+exp_data_dir)
            for s_data_file in stdout2:
                s_data_file = s_data_file.strip('\n')
                # if the file is a .yml file, attempt to load it into the files table
                if s_data_file.endswith('.yml'):
                    file_path = os.path.join(exp_data_dir, s_data_file)
                    # use cat to grab the file content and dump it to a stream
                    stdin, stdout, stderr = ssh_client.exec_command('cat ' + file_path)
                    net_dump = stdout.readlines()
                    str_d = "".join(net_dump)
                    # load the stream to data as yaml, grab key attributes
                    pp = yaml.load(str_d)
                    expt_id_yml = pp['sample_metadata']['experiment_id']
                    sample_id_yml = pp['sample_metadata']['sample_id']
                    fit = pp['fit_report']['good_fit']
                    # add attributes and file path to the DB
                    db.insert('files', sample_id=sample_id_yml, experiment_id = expt_id_yml,
                                  yml_path=file_path, good_fit=fit)
        print('FINISHED loading experiment {} to files table'.format(experiment))

def load_from_files_table_to_samples_table(db, ssh_client, drop_table=False):
    """Process the data from a the "files" table and insert corresponding rows into the "samples" table.

    Parameters
    ----------
    db : pg.DB 
        a database connector (DB object from PyGreSQL)
    ssh_client : paramiko.SSHClient
        ssh client connected with the database host machine
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
            stdin, stdout, stderr = ssh_client.exec_command('cat ' + f)
            net_dump = stdout.readlines()
            str_d = "".join(net_dump)
            pp = yaml.load(str_d)
            expt_id, sample_id, features, classification_labels, \
                    regression_labels = unpack_sample(pp)
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
