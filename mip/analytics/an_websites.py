# -*- coding: utf-8 -*-

"""
Analyse scraped websites
"""

from db.db import open_sqlite, run_select_sql
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Comment
import re
from utils import remove_empty_elem_from_list, remove_multiple_spaces_tabs
import logging
logger = logging.getLogger(__name__)

field_sep = '\n'

def get_scraping_sessions(db):
    df = run_select_sql("select session_id, count(*) as page_n from web_pages_dump group by session_id; ", db)
    return df

def get_scraping_sessions_by_museum(db):
    df = run_select_sql(
        """select session_id, muse_id, count(*) as page_n, sum(page_content_length) as data_size
        from web_pages_dump 
        group by session_id, muse_id;
        """, db)
    return df

def create_webpage_attribute_table(db_con):
    c = db_con.cursor()
    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS web_page_attributes
            (page_attr_id integer PRIMARY KEY AUTOINCREMENT,
            page_id integer NOT NULL,
            session_id text NOT NULL,
            attrib_name text NOT NULL,
            attrib_val text,
            UNIQUE(page_id, attrib_name));
            ''')
    db_con.commit()
    logger.debug('create_webpage_attribute_table')


def clear_attribute_table(db_con):
    logger.debug("clear_attribute_table")
    c = db_con.cursor()
    c.execute('''DELETE from web_page_attributes;''')
    db_con.commit()

def insert_page_attribute(db_con, page_id, session_id, attrib_name, attrib_val):
    assert attrib_name in ['title','headers','all_text']
    assert page_id >= 0
    if attrib_val == '':
        attrib_val = None
    
    c = db_con.cursor()
    sql = '''INSERT INTO web_page_attributes(page_id, session_id, attrib_name, attrib_val)
              VALUES(?,?,?,?);'''
    
    cur = db_con.cursor()
    cur.execute(sql, [page_id, session_id, attrib_name, attrib_val])
    db_con.commit()


def tag_visible(element):
    """ Returns True if html tag is a visible one """
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def clean_text(s):
    s = s.strip()
    s = s.replace('\t',' ')
    return s


def extract_attributes_from_page_html(db_con, page_id, session_id, page_html):
    """ Extract text attributes from HTML code of a museum web page """
    assert page_id >= 0
    assert len(page_html) > 0
    soup = BeautifulSoup(page_html, 'html.parser')

    # get page title
    ptitle = soup.title.string
    if ptitle: ptitle = ptitle.strip()
    insert_page_attribute(db_con, page_id, session_id, 'title', ptitle)
    
    # get page headers
    headers = soup.find_all(re.compile('^h[1-6]$'))
    headers = [clean_text(h.text).strip() for h in headers if h.text]
    headers = remove_empty_elem_from_list(headers)
    headers = field_sep.join(headers)
    insert_page_attribute(db_con, page_id, session_id, 'headers', remove_multiple_spaces_tabs(headers))

    # get all text
    if soup.body:
        texts = soup.body.findAll(text=True)
        visible_texts = remove_empty_elem_from_list(filter(tag_visible, texts))
        page_all_text = field_sep.join(t.strip() for t in visible_texts if t)
        page_all_text = remove_multiple_spaces_tabs(page_all_text)
        insert_page_attribute(db_con, page_id, session_id, 'all_text', page_all_text)
    else:
        insert_page_attribute(db_con, page_id, session_id, 'all_text', None)
        logger.debug("page ID "+str(page_id)+" has no HTML content")


    # TODO: extract page fields/links here

def extract_text_from_websites(in_website_db, out_attr_db):
    """ Scan all pages in in_website_db and generate attributes in out_attr_db"""
    clear_attribute_table(out_attr_db)
    block_sz = 50
    offset = 0
    keep_scanning = True
    while keep_scanning:
        print(offset)
        sql = "select * from web_pages_dump limit "+str(block_sz)+ " offset "+str(offset)+";"
        pages_df = run_select_sql(sql, in_website_db)
        for index, row in pages_df.iterrows():
            page_id = row['page_id']
            session_id = row['session_id']
            page_html = row['page_content']
            extract_attributes_from_page_html(out_attr_db, page_id, session_id, page_html)
        # process pages
        # scanning logic
        offset += block_sz
        if len(pages_df) < block_sz:
            keep_scanning = False

    # close DBs
    in_website_db.close()
    out_attr_db.close()

def analyse_museum_websites(museums_df):
    """ Main function """
    # input data (museum sample)
    sample_df = pd.read_csv("data/museums/mip_data_sample_2020_01.tsv", sep='\t')

    #in_db = 'tmp/websites-sample-2020-01-26.db'
    in_db = 'tmp/websites.db'
    logger.info("extract_text_from_websites: "+in_db)
    db = open_sqlite(in_db)

    # get session stats
    print(get_scraping_sessions(db))
    
    stats_df = get_scraping_sessions_by_museum(db)
    df = sample_df.merge(stats_df, how='left', left_on='mm_id', right_on='muse_id')
    df.to_excel('tmp/analytics/websites-sample-stats.xlsx')

    # extract page attributes
    create_webpage_attribute_table(db)
    