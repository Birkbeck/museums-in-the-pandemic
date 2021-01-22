# -*- coding: utf-8 -*-

"""
DB management
"""

import sqlite3
import pandas as pd

def open_sqlite(db_fn):
    conn = sqlite3.connect(db_fn)
    return conn


def create_page_dump(db_conn):
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

def extract_google_urls(db_conn):
    return db_conn

def url_exists(db, targeturl):
    con = sqlite3.connect(db)
    df = pd.read_sql("SELECT * from google_pages_dump", con)
    for row in df.iterrows():
        if(row[1]['url']==targeturl):
            return True
    
    return False
    # TODO Val check if URL exists in database
