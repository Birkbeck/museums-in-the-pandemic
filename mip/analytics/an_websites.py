# -*- coding: utf-8 -*-

"""
Analyse scraped websites
"""

from db.db import connect_to_postgresql_db, check_dbconnection_status, make_string_sql_safe
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Comment
from scrapers.scraper_websites import get_scraping_session_tables, get_scraping_session_stats_by_museum, get_webdump_table_name
import re
from utils import remove_empty_elem_from_list, remove_multiple_spaces_tabs
import logging
import difflib

logger = logging.getLogger(__name__)

# constants
field_sep = '\n'
table_suffix = '_attr'


def get_webdump_attr_table_name(session_id):
    tablen = get_webdump_table_name(session_id) + table_suffix
    return tablen


def create_webpage_attribute_table(table_name, db_con):
    """ create table for attributes """
    assert table_name
    attr_table = table_name + table_suffix
    c = db_con.cursor()
    # Create table
    # #page_attr_id integer PRIMARY KEY AUTOINCREMENT,
    sql = '''CREATE TABLE IF NOT EXISTS {0}
            (page_id integer NOT NULL REFERENCES {1}(page_id),
            session_id text NOT NULL,
            attrib_name text NOT NULL,
            attrib_val text,
            PRIMARY KEY(page_id, attrib_name));
            CREATE INDEX IF NOT EXISTS {2}_session_idx ON {0} USING btree(session_id);
            CREATE INDEX IF NOT EXISTS {2}_page_idx ON {0} USING btree(page_id);
            CREATE INDEX IF NOT EXISTS {2}_attrib_idx ON {0} USING btree(attrib_name);
            '''.format(attr_table, table_name, attr_table.replace('.','_'))
    c.execute(sql)
    db_con.commit()
    
    logger.info('create_webpage_attribute_table: '+attr_table)
    return attr_table


def clear_attribute_table(table_name, db_con):
    """ This does not delete the source pages, only the extracted attributes """
    logger.debug("clear_attribute_table: "+ table_name)
    assert table_suffix in table_name
    c = db_con.cursor()
    c.execute('''DELETE from {};'''.format(table_name))
    db_con.commit()
    return True


def insert_page_attribute(db_con, table_name, page_id, session_id, attrib_name, attrib_val):
    """  """
    assert attrib_name in ['title','headers','all_text']
    assert page_id >= 0
    if attrib_val == '':
        attrib_val = None
    
    c = db_con.cursor()
    sql = '''INSERT INTO {}(page_id, session_id, attrib_name, attrib_val)
              VALUES(%s,%s,%s,%s);'''.format(table_name)
    
    cur = db_con.cursor()
    try:
        cur.execute(sql, [page_id, session_id, attrib_name, attrib_val])
        db_con.commit()
    except UnicodeEncodeError as e:
        logger.warn(str(page_id))
        logger.warn(str(e))
        print("broken page_id",page_id)
        # TODO: fix utf 
        raise e
        #cur.execute(sql, [page_id, session_id, attrib_name, attrib_val])
        #db_con.commit()

    return True


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


def extract_attributes_from_page_html(page_id, session_id, page_html, attr_table, db_con):
    """ Extract text attributes from HTML code of a museum web page """
    assert page_id >= 0
    if len(page_html)==0: 
        logger.warning("page_id {} is empty".format(page_id))
        return False

    assert len(page_html) > 0
    soup = BeautifulSoup(page_html, 'html.parser')

    # get page title
    if soup.title:
        ptitle = soup.title.string
        if ptitle: ptitle = ptitle.strip()
        insert_page_attribute(db_con, attr_table, page_id, session_id, 'title', ptitle)
    
    # get page headers
    headers = soup.find_all(re.compile('^h[1-6]$'))
    headers = [clean_text(h.text).strip() for h in headers if h.text]
    headers = remove_empty_elem_from_list(headers)
    headers = field_sep.join(headers)
    insert_page_attribute(db_con, attr_table, page_id, session_id, 'headers', remove_multiple_spaces_tabs(headers))

    # get all text
    if soup.body:
        texts = soup.body.findAll(text=True)
        visible_texts = remove_empty_elem_from_list(filter(tag_visible, texts))
        page_all_text = field_sep.join(t.strip() for t in visible_texts if t)
        page_all_text = remove_multiple_spaces_tabs(page_all_text)
        insert_page_attribute(db_con, attr_table, page_id, session_id, 'all_text', page_all_text)
    else:
        insert_page_attribute(db_con, attr_table, page_id, session_id, 'all_text', None)
        logger.debug("page ID "+str(page_id)+" has no HTML content")

    # TODO: extract other page fields/links here
    return True

def extract_text_from_websites(in_table, out_table, db_conn, target_museum_id=None):
    """ Scan all pages in in_website_db and generate attributes in out_attr_db"""
    logger.info("extract_text_from_websites {} -> {}".format(in_table, out_table))
    assert in_table
    assert table_suffix in out_table
    #clear_attribute_table(out_table, db_conn)
    block_sz = 100
    offset = 0
    keep_scanning = True
    
    while keep_scanning:
        print(offset)
        where = ''
        if target_museum_id:
            where = " where muse_id ='{}'".format(target_museum_id)
        sql = "select * from {} {} limit {} offset {};".format(in_table, where, str(block_sz), str(offset))
        pages_df = pd.read_sql(sql, db_conn)
        
        for index, row in pages_df.iterrows():
            if index % 100 == 0: print('\t',index)
            page_id = row['page_id']
            url = row['url']
            session_id = row['session_id']
            page_html = row['page_content']
            if not exists_attrib_page(page_id, session_id, db_conn):
                extract_attributes_from_page_html(page_id, session_id, page_html, out_table, db_conn)
                
        # process pages
        # scanning logic
        offset += block_sz
        if len(pages_df) < block_sz:
            keep_scanning = False
        else: 
            print("next block")
    return True


def exists_attrib_page(page_id, session_id, db_conn):
    check_dbconnection_status(db_conn)
    table = get_webdump_attr_table_name(session_id)
    sql = "select page_id from {} where page_id = {}".format(table, page_id)
    df = pd.read_sql(sql, db_conn)
    if len(df) > 0:
        return True
    return False


def analyse_museum_websites():
    """ Main function analyse_museum_websites"""
    # input data (museum sample)
    db_conn = connect_to_postgresql_db()

    logger.info("extract_text_from_websites")

    # get session stats
    tables = get_scraping_session_tables(db_conn)

    for tab in tables:
        df = get_scraping_session_stats_by_museum(tab, db_conn)
        #df = sample_df.merge(stats_df, how='left', left_on='mm_id', right_on='muse_id')
        df.to_excel('tmp/analytics/websites-stats-{}.xlsx'.format(tab), index=False)
        # prepare table
        out_table = create_webpage_attribute_table(tab, db_conn)
        # extract attributes
        extract_text_from_websites(tab, out_table, db_conn)
    