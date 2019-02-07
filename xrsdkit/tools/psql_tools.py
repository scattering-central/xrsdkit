"""This module contains API to communicate with scattering_data database on pawsweb.
    The database server must be running on 134.79.98.141, port=5432.
    The user should have pygresql installed on his local machine.
    To crate a connector:
    from pg import DB, connect
    db = DB(dbname='scattering_data', host='134.79.98.141', port=5432, user=PSQL_USERNAME, passwd=PSQL_PASSWORD)

    To run load_yml_to_file_table() and load_from_files_table_to_samples_table() the user also need
    to paramiko installed on his local machine.
    To create a ssh_clint:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=REMOUTE_USERNAME, password=REMOUT_PASSWORD)

    Data pipeline supported by this module:
    dir with training data -> files table -> samples table -> training table -> pandas dataframe

    files columns:
    sample_id     | character varying
    experiment_id | character varying
    good_fit      | boolean
    yml_path      | text

    samples columns:
    sample_id             | character varying
    experiment_id         | character varying
    features              | json
    regression_labels     | json
    classification_labels | json

    training columns:
    sample_id              | character varying
    experiment_id          | character varying
    Imax_over_Imean        | numeric
    Ilowq_over_Imean       | numeric
    Imax_sharpness         | numeric
    I_fluctuation          | numeric
    logI_fluctuation       | numeric
    ...
    pop0_lattice           | character varying
    pop2_form              | character varying
    pop0_form              | character varying
    ...
    Training has a column for each feature, and each label. When we are
    inserting a new sample with labels that are not in "training",
    the new colums are appening.

    When user got data from a new experiment, he should run:
    load_yml_to_file_table(db, client, path_to_dir)
    load_from_files_table_to_samples_table(db, client)
    load_from_samples_to_training_table(db)

    Then, when he wants to retrain the models, he can get data from the database:
    df = get_training_dataframe(db)
    """

import os
import yaml
import pandas as pd
from collections import OrderedDict
from .ymltools import unpack_sample
from .profiler import profile_keys
import pprint

def load_yml_to_file_table(db, ssh_client, path_to_dir, drop_table=False):
    """Add data from a remote directory to the "files" table.

    The data directory should contain one or more subdirectories,
    where each subdirectory contains .yml files,
    where each .yml file describes one sample,
    as saved by save_sys_to_yaml().

    Parameters
    ----------
    db : db.pg object
        a database connection - DB object from PyGreSQL
    ssh_client : paramiko.SSHClient
        ssh client connected with the host machine
    path_to_dir : str
        absolute path to the folder with the training set
        Precondition: dataset directory includes directories named
        by the name of the experiments; each experiment directory
        holds data from this experiment. For each sample there is one yml file
        with System object ... and one dat file with q and I arrays
        (array of scattering vector magnitudes, array of integrated
        scattering intensities).
    drop_table : bool
        if True, the existing table will be dropped and a new table will be created
        from scratch;
        if False, only data that is not already in the "file" table will be added.
    """
    if drop_table:
            db.query("DROP TABLE files")
    db.query("CREATE TABLE IF NOT EXISTS files(sample_id VARCHAR PRIMARY KEY, "
                                            "experiment_id VARCHAR, good_fit BOOLEAN, "
                                            "yml_path TEXT)")

    # get the list of experiments that are already in the table
    exp_from_table = db.query('SELECT DISTINCT experiment_id FROM files').getresult()
    exp_from_table = [row[0] for row in exp_from_table]

    stdin, stdout, stderr = ssh_client.exec_command('ls '+path_to_dir)

    for experiment in stdout:
        experiment = experiment.strip('\n')
        exp_data_dir = os.path.join(path_to_dir,experiment)
        # add only new experiments
        if ssh_client.exec_command("os.path.isdir(exp_data_dir)") and experiment not in exp_from_table:
            stdin2, stdout2, stderr2 = ssh_client.exec_command('ls '+exp_data_dir)
            for s_data_file in stdout2:
                s_data_file = s_data_file.strip('\n')
                if s_data_file.endswith('.yml'):
                    file_path = os.path.join(exp_data_dir, s_data_file)
                    stdin, stdout, stderr = ssh_client.exec_command('cat ' + file_path)
                    net_dump = stdout.readlines()
                    str_d = "".join(net_dump)
                    pp = yaml.load(str_d)
                    expt_id_yml = pp['sample_metadata']['experiment_id']
                    sample_id_yml = pp['sample_metadata']['sample_id']
                    fit = pp['fit_report']['good_fit']

                    db.insert('files', sample_id=sample_id_yml, experiment_id = expt_id_yml,
                                  yml_path=file_path, good_fit=fit)
        print(experiment, "DONE")

def load_from_files_table_to_samples_table(db, ssh_client, drop_table=False):
    """Process the data from a the "files" table and
    insert corresponding rows into the "samples" table.

    Parameters
    ----------
    db : db.pg object
        a database connection - DB object from PyGreSQL
    ssh_client : paramiko.SSHClient
        ssh client connected with the host machine:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect('134.79.34.167', username='my_user_name', password='my_password')
    drop_table : bool
        if True, the existing table will be dropped and a new table will be created
        from scratch;
        if False, only data that is not already in the "samples" table will be added.
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
    """Process the data from a the "samples" table and
    insert corresponding rows into the "training" table
    in the format sutable for training
    (each feature and labels has its own column).
    Only data from new experiment directories will be added.

    Parameters
    ----------
    db : db.pg object
        a database connection - DB object from PyGreSQL
    drop_table : bool
        if True, the existing table will be dropped and a new table will be created
        from scratch;
        if False, only data that is not already in the "samples" table will be added.
    """
    if drop_table:
            db.query("DROP TABLE training")

    #db.query("DROP TABLE training")
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

    #add new columns if it is needed (new classification and regression labels)
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

    # get data from the "samples" table from the experiments that are not in "training" table:
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
    """Create a Pandas DataFrame object from
    a postgreSQL table.

    Parameters
    ----------
    db : db.pg object
        a database connection - DB object from PyGreSQL
    df : pandas.DataFrame
        dataframe containing features and labels
    """
    data = db.query('SELECT * FROM training').dictresult()
    df = pd.DataFrame(data)

    return df