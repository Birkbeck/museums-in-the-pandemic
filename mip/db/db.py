# -*- coding: utf-8 -*-

"""
DB management

POSTGRESQL config
- http://www.project-open.com/en/howto-postgresql-port-secure-remote-access

"""

import logging
logger = logging.getLogger(__name__)

from tabulate import tabulate
import sqlite3
import psycopg2
from sqlalchemy import create_engine
import pandas as pd
import json

# load postgres configuration
try: 
    with open('.secrets.json') as f:
        pg_config = json.load(f)['postgres']
except:
    # look for other location
    with open('../../.secrets.json') as f:
        pg_config = json.load(f)['postgres']


def open_sqlite(db_fn):
    """ Open connection to sqlite local DB """
    conn = sqlite3.connect(db_fn)
    logger.debug("open_sqlite: "+ db_fn)
    return conn


def create_alchemy_engine_sqlite_corpus():
    print("create_alchemy_engine_sqlite_corpus")
    local_search_db = 'tmp/mip_corpus_search.db'
    local_engine = create_engine('sqlite:///'+local_search_db, echo=False)
    return local_search_db, local_engine

def create_alchemy_engine_posgresql():
    print("create_alchemy_engine_posgresql")
    ipaddress = pg_config['ip']
    dbname=pg_config['dbname']
    username=pg_config['user']
    password=pg_config['pwd']
    port=5432
    #postgres_str = f'postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'
    postgres_str = f'postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'
          
    # Create the connection
    db_engine = create_engine(postgres_str)
    return db_engine

def create_page_dump(db_conn):
    """ create table for Google page dump """
    c = db_conn.cursor()

    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS google_pages_dump 
            (url text NOT NULL, 
            search text NOT NULL,
            search_type text NOT NULL,
            muse_id text NOT NULL, 
            page_content text NOT NULL,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP)
            ;
            ''')
    db_conn.commit()
    print('create_page_dump')


def insert_google_page(db_conn, url, search, search_type, muse_id, page_content):
    c = db_conn.cursor()
    sql = '''INSERT INTO google_pages_dump(url, search, search_type, muse_id, page_content)
              VALUES(?,?,?,?,?) '''
    cur = db_conn.cursor()
    cur.execute(sql, [url, search, search_type, muse_id, page_content])
    db_conn.commit()
    return db_conn


def insert_google_id(db_conn, musename, muse_id):
    c = db_conn.cursor()
    musename = musename.replace("'","''")
    #musename = musename.replace('"','')
    if "Search" in musename:
        print("'")
    sql ="UPDATE google_pages_dump SET muse_id='"+muse_id+"' WHERE search ='" +musename+"'"
    cur = db_conn.cursor()
    cur.execute(sql)
    db_conn.commit()
    #print(".")
    return db_conn


def run_select_sql(sql, db_conn):
    """ Run SQL select on db and return data frame. """
    assert sql
    assert db_conn
    df = pd.read_sql(sql, db_conn)
    return df


def check_dbconnection_status(db_conn):
    assert db_conn
    if not db_conn.closed == 0:
        raise RuntimeError('Connection to Postgresql DB is closed, it should be open!')
    return True


def url_exists(con, targeturl):
    
    df = pd.read_sql("SELECT * from google_pages_dump WHERE url==\""+targeturl+"\"", con)

    if(df.empty): 
        return False
    
    return True
    # TODO Val check if URL exists in database


def is_postgresql_db_accessible():
    """ @returns True if the DB is accessible. """
    try:
        conn = connect_to_postgresql_db()
        conn.close()
        return True
    except Exception as e:
        logger.warn(str(e))
        return False


def count_all_db_rows():
    assert is_postgresql_db_accessible()
    db_conn = connect_to_postgresql_db()
    #sql = """
    #SELECT pgClass.relname, pgClass.reltuples AS n_rows
    #FROM
    #    pg_class pgClass
    #LEFT JOIN
    #    pg_namespace pgNamespace ON (pgNamespace.oid = pgClass.relnamespace)
    #WHERE
    #    pgNamespace.nspname NOT IN ('pg_catalog', 'information_schema') 
    #    AND pgClass.relkind='r'
    #order by pgClass.relname;
    #"""

    sql = """select schemaname, relname, n_live_tup, n_dead_tup
        from pg_stat_user_tables
        order by schemaname, relname;"""

    df = pd.read_sql(sql, db_conn)
    print('DB stats')
    print(tabulate(df,headers='firstrow'))
    return df

def connect_to_postgresql_db():
    """ Connect to central DB using a singleton """
    logger.debug("connect_to_postgresql_db")
    db_conn = psycopg2.connect(host=pg_config['ip'], dbname=pg_config['dbname'], 
        user=pg_config['user'], password=pg_config['pwd'], connect_timeout=3)
    check_dbconnection_status(db_conn)
    return db_conn


def make_string_sql_safe(s):
    """ Escape quotes """
    s = s.replace("'","''")
    return s

def scan_table_limit_offset(db_conn, select_sql, block_sz, funct):
    print('scan_table_limit_offset')
    offset = 0
    keep_scanning = True
    while keep_scanning:
        print('  offset =',offset)
        sql = "{} limit {} offset {};".format(select_sql, block_sz, offset)
        results_df = pd.read_sql(sql, db_conn)
        funct(results_df)
        # scanning logic
        offset += block_sz
        if len(results_df) < block_sz:
            keep_scanning = False
    
    