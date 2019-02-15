.. _sec-installation:


Database Installation
=====================

The xrsdkit can do many things as a local software package,
but usage of xrsdkit at large scale will require a database.
The database used in developing xrsdkit is PostgreSQL. 
This section outlines the process of establishing a PostgreSQL instance 
and configuring it for xrsdkit.

To install PostgresSQL on your server machine, you should have "super user" privilege.

1. Install PostgresSQL
----------------------
Install PostgresSQL>=10 on your server machine by instructions on:
https://www.postgresql.org/download/linux/redhat/  (for Red Hat family).

2. Configurate your database (open for listening from the internet)
-------------------------------------------------------------------

**2.1. Update "listen_addresses"** in postgresql.conf file. For PostgresSQL 10, the path is \n
/var/lib/pgsql/10/data/postgresql.conf
but the other version have this file in different locations!

"listen_addresses" specifies the TCP/IP address(es) on which the server is to listen for connections
from client applications. The special entry  * corresponds to all available IP interfaces.
More info: https://www.postgresql.org/docs/10/runtime-config-connection.html
::

    listen_addresses = "*"

**2.2. Update "host"** in pg_hba.conf.
Client authentication is controlled by values in this file.
For PostgresSQL 10, the path is
/var/lib/pgsql/10/data/pg_hba.conf.
"Host" line has format:

host       database  user  IP-address  IP-mask  auth-method  [auth-options]

You can update it so that all clients and all users with any IP adress will be asked to supply an unencrypted password for authentication.
::

    host all all 0.0.0.0/0 password


3. Start the service
--------------------
::

    sudo /sbin/service postgresql start


4. Connect with PostgreSQL as a superuser
-----------------------------------------
::

    sudo su - postgres
    psql

5. Create two databases
-----------------------
Create a main database and a test database.
You do not need to create any tables at this points.
The tables in the main db will be created by running functions from xrsdkit.db:

- load_yml_to_file_table()
- load_from_files_table_to_samples_table()
- load_from_samples_to_training_table().

The tables in the test db with be created by running xrsdkit.test.test_db.py

6. Create users
---------------

Form psql prompt: ::

    CREATE USER user1 WITH ENCRYPTED PASSWORD 'password1';
    GRANT ALL PRIVILEGES ON DATABASE main_db TO user1;
    GRANT ALL PRIVILEGES ON DATABASE test TO user1;

    CREATE USER test WITH ENCRYPTED PASSWORD 'test';
    GRANT ALL PRIVILEGES ON DATABASE test_db TO test;


For the instalation of PostgreSQL on the other version of OS:
https://www.tutorialspoint.com/postgresql/postgresql_environment.htm



