.. _sec-installation:


Database Installation
=====================

The xrsdkit package can do many things on a personal computer,
but usage at scale will require a database.
The database used in developing xrsdkit is PostgreSQL (version >=10).
Version 10 was used because it supports columns of .json data,
which are used by xrsdkit. 
Version 9 may work, but it has not been tested.

This section outlines the process of establishing a PostgreSQL instance 
and configuring it for xrsdkit.

To install PostgreSQL on your server machine, 
you need super-user (`su` or `sudo`) privileges.


1. Install PostgreSQL
---------------------
Install PostgreSQL on your server machine by following these instructions:
https://www.postgresql.org/download/linux/redhat/  
(for RedHat family).
This site also includes instructions for other architectures,
and many public resources are available by searching on the web.


2. Configure your database connection routes 
--------------------------------------------
Locate your database configuration files. 
On RedHat servers with PostgreSQL 10, 
the files are located at: 

    `/var/lib/pgsql/10/data/pg_hba.conf`
    `/var/lib/pgsql/10/data/postgresql.conf`

Note that other platforms and database version 
may place these files in different locations.

**2.1. Update "listen_addresses"** in postgresql.conf. 

"listen_addresses" specifies the TCP/IP address(es) on which the server is to listen for connections
from client applications. The special entry "*" corresponds to all available IP interfaces.
For more information: 
https://www.postgresql.org/docs/10/runtime-config-connection.html
::

    listen_addresses = "*"

**2.2. Update "host"** in pg_hba.conf.
If the database is not being set up on the local host,
client authentication over TCP/IP is controlled by the "host" line,
which should be formatted as follows:

    host    database    user    IP-address  IP-mask auth-method [auth-options]

For example, to authenticate all clients and all users 
with any IP adress by unencrypted passwords: 
::

    host all all 0.0.0.0/0 password

Note, if you employ this configuration for your database,
it should only be used on a secure network,
because database user passwords will be communicated in plain text.
For more information:
https://www.postgresql.org/docs/9.1/auth-pg-hba-conf.html


3. Start the PostgreSQL service
-------------------------------
On RedHat:
::

    sudo /sbin/service postgresql start


4. Set up your database(s) 
--------------------------
You will need to switch to the database administrator account
(by default, "postgres"), and then start the database management program (psql).
::

    sudo su - postgres
    psql

Create a main database:
::

    CREATE DATABASE main_db;

For developers who will be running tests against a database,
a separate database should be created for that purpose:
::

    CREATE DATABASE test_db;


You do not need to create any tables at this point.
The tables will be created by methods in the `xrsdkit.db` subpackage.
See, for example:
::

    `load_yml_to_file_table()`
    `load_from_files_table_to_samples_table()`
    `load_from_samples_to_training_table()`

For developers who will be running tests,
The tables in the test database will be created by running `tests/test_db.py`.


5. Create users
---------------

From the `psql` prompt: 
::

    CREATE USER user1 WITH ENCRYPTED PASSWORD 'password_for_user1';
    GRANT ALL PRIVILEGES ON DATABASE main_db TO user1;
    GRANT ALL PRIVILEGES ON DATABASE test_db TO user1;

    CREATE USER test WITH ENCRYPTED PASSWORD 'password_for_test';
    GRANT ALL PRIVILEGES ON DATABASE test_db TO test;

After these users are created,
xrsdkit is ready to use the database!
The user will have to place the database user authentication information
in a hidden file in their home directory.
The `xrsdkit.db` subpackage documentation describes this in detail.
TODO: provide link to `xrsdkit.db` doc.

For setting up the database on platforms 
not covered in this document, this tutorial may help:
https://www.tutorialspoint.com/postgresql/postgresql_environment.htm

