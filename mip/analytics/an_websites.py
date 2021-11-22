# -*- coding: utf-8 -*-

"""
Analyse scraped websites
"""

from db.db import connect_to_postgresql_db, check_dbconnection_status, make_string_sql_safe
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Comment
from scrapers.scraper_websites import get_scraping_session_tables, get_scraping_session_stats_by_museum, get_webdump_table_name, get_session_id_from_table_name,get_previous_session_tables, check_for_url_redirection
from utils import get_url_domain
import re
from utils import remove_empty_elem_from_list, remove_multiple_spaces_tabs, get_soup_from_html, get_all_text_from_soup, garbage_collect, parallel_dataframe_apply
import logging
import difflib
import unicodedata
import constants

logger = logging.getLogger(__name__)


def get_webdump_attr_table_name(session_id):
    """
    get full table name from session id e.g. websites.web_pages_dump_20210420_attr
    """
    tablen = get_webdump_table_name(session_id) + constants.table_suffix
    return tablen


def create_webpage_attribute_table(table_name, db_con):
    """ create table for attributes """
    assert table_name
    attr_table = table_name + constants.table_suffix
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
    assert constants.table_suffix in table_name
    c = db_con.cursor()
    c.execute('''DELETE from {};'''.format(table_name))
    db_con.commit()
    return True


def insert_page_attribute(db_con, table_name, page_id, session_id, attrib_name, attrib_val):
    """ insert page attribute into DB """
    assert attrib_name in ['title','headers','all_text']
    assert page_id >= 0
    attrib_val = clean_unicode_issues_string(attrib_val)
    if attrib_val == '':
        attrib_val = None
    
    c = db_con.cursor()
    sql = '''INSERT INTO {}(page_id, session_id, attrib_name, attrib_val)
              VALUES(%s,%s,%s,%s);'''.format(table_name)
    
    cur = db_con.cursor()
    try:
        cur.execute(sql, [page_id, session_id, attrib_name, attrib_val])
        db_con.commit()
    except (UnicodeEncodeError,ValueError) as e:
        logger.warn(str(page_id))
        logger.warn(str(e))
        msg = "broken page_id, fixing unicode"+str(page_id)
        print(msg)
        logger.warn(msg)
        raise e
        # reinsert cleaned string
        #clean_attrib_val = unicodedata.normalize("NFKD", attrib_val)
        #cur.execute(sql, [page_id, session_id, attrib_name, clean_attrib_val])
        #db_con.commit()

    return True


def clean_unicode_issues_string(s):
    """ Bug fix for unicode issues """
    if s is None: return s
    if s == '': return ''

    cs = s.replace("\x00", "\uFFFD").strip()
    # normalise unicode string
    cs2 = unicodedata.normalize("NFKD", cs)
    try:
        cs2.encode()
    except UnicodeEncodeError as e:
        cs2 = cs2.encode('utf-8', "backslashreplace").decode('utf-8')
        cs2.encode()
    #cs = s.decode("utf-8", errors="replace").replace("\x00", "\uFFFD")
    return cs2


def clean_text(s):
    s = s.strip()
    s = s.replace('\t',' ')
    return s


def extract_attributes_from_page_html(page_id, session_id, page_html, attr_table, db_con):
    """ Extract text attributes from HTML code of a museum web page """
    assert page_id >= 0
    if page_html is None or len(page_html)==0:
        logger.warning("page_id {} is empty".format(page_id))
        return False
    
    msg = "\textract page ID="+str(page_id)
    logger.debug(msg)
    print(msg)

    assert len(page_html) > 0
    soup = get_soup_from_html(page_html)

    # get page title
    if soup.title:
        ptitle = soup.title.string
        if ptitle: ptitle = ptitle.strip()
        insert_page_attribute(db_con, attr_table, page_id, session_id, 'title', ptitle)
    
    # get page headers
    headers = soup.find_all(re.compile('^h[1-6]$'))
    headers = [clean_text(h.text).strip() for h in headers if h.text]
    headers = remove_empty_elem_from_list(headers)
    headers = constants.field_sep.join(headers)
    insert_page_attribute(db_con, attr_table, page_id, session_id, 'headers', remove_multiple_spaces_tabs(headers))

    # get all text
    if soup.body:
        page_all_text = get_all_text_from_soup(soup)
        insert_page_attribute(db_con, attr_table, page_id, session_id, 'all_text', page_all_text)
    else:
        insert_page_attribute(db_con, attr_table, page_id, session_id, 'all_text', None)
        logger.debug("page ID "+str(page_id)+" has no HTML content")
    
    # TODO: extract other page fields/links here
    
    # clear up soup
    soup.decompose()
    del soup
    return True


def extract_text_from_websites(in_table, out_table, db_conn, target_museum_id=None):
    """ Scan all pages in in_website_db and generate attributes in out_attr_db"""
    msg = "extract_text_from_websites {} -> {}".format(in_table, out_table)
    logger.info(msg)
    print(msg)
    assert in_table
    assert constants.table_suffix in out_table
    #clear_attribute_table(out_table, db_conn)
    block_sz = 8000
    offset = 0
    keep_scanning = True
    
    while keep_scanning:
        print('offset =', offset)
        where = ''
        if target_museum_id:
            where = " where muse_id ='{}'".format(target_museum_id)
        sql = "select * from {} {} limit {} offset {};".format(in_table, where, str(block_sz), str(offset))
        pages_df = pd.read_sql(sql, db_conn)
        
        for index, row in pages_df.iterrows():
            # print progress
            if index % block_sz == 0: print('\tidx=',index)
            page_id = row['page_id']
            url = row['url']
            session_id = row['session_id']
            page_html = row['page_content']
            # extract all
            to_extract = True
            if 'prev_session_diff_b' in row:
                new_page = row['new_page_b']
                prev_session_diff_b = row['prev_session_diff_b']
                if not new_page and not prev_session_diff_b:
                    to_extract = False
            if to_extract and not exists_attrib_page(page_id, session_id, db_conn):
                print('\t extracting page_id=', page_id)
                extract_attributes_from_page_html(page_id, session_id, page_html, out_table, db_conn)
            del page_html, page_id, session_id

        # process pages
        # scanning logic
        offset += block_sz
        if len(pages_df) < block_sz:
            keep_scanning = False
        else:
            print("next block")
        #garbage_collect() # TODO: memory leak here
    return True


def exists_attrib_page(page_id, session_id, db_conn):
    """ @returns True if page exists in table """
    check_dbconnection_status(db_conn)
    table = get_webdump_attr_table_name(session_id)
    sql = "select page_id from {} where page_id = {}".format(table, page_id)
    df = pd.read_sql(sql, db_conn)
    if len(df) > 0:
        return True
    return False


def __extract_text_from_websites(tables):
    print('__extract_text_from_websites')
    db_conn2 = connect_to_postgresql_db()
    for tab in tables['table']:
        print('\t',tab)
        df = get_scraping_session_stats_by_museum(tab, db_conn2)
        #df = sample_df.merge(stats_df, how='left', left_on='mm_id', right_on='muse_id')
        df.to_excel('tmp/analytics/websites-stats-{}.xlsx'.format(tab), index=False)
        # prepare table
        out_table = create_webpage_attribute_table(tab, db_conn2)
        # extract attributes
        extract_text_from_websites(tab, out_table, db_conn2) # DEBUG mm.musa.016
    db_conn2.close()
    # empty dataframe
    return pd.DataFrame()



def analyse_museum_websites():
    """ Main function analyse_museum_websites """
    # input data (museum sample)
    db_conn = connect_to_postgresql_db()
    logger.info("extract_text_from_websites")

    tables = get_scraping_session_tables(db_conn)
    tables_df = pd.DataFrame({'table':tables})
    # extract fields from websites (PARALLEL)
    parallel_dataframe_apply(tables_df, __extract_text_from_websites, n_cores=6)

    return True


def get_attribute_for_webpage_url(url, session_id, attrib_name, db_conn):
    """
    @returns attribute value (e.g. all_text) for a URL in a target scraping session;
        None if URL or attribute does not exist 
    """
    page_tbl_name = get_webdump_table_name(session_id)
    attr_tbl_name = get_webdump_attr_table_name(session_id)
    
    #print("get_attribute_for_webpage_url", attr_tbl_name)
    
    sql = """select url, a.page_id, attrib_name, attrib_val from {} p, {} a where a.page_id = p.page_id 
        and p.url = '{}' and a.attrib_name = '{}';""".format(page_tbl_name, attr_tbl_name, make_string_sql_safe(url), attrib_name)

    attr_df = pd.read_sql(sql, db_conn)
    df = attr_df[['url', 'page_id', 'attrib_name', 'attrib_val']]
    
    if len(df) > 0:
        assert len(df) == 1
        val = df['attrib_val'].tolist()[0]
        if len(val) == 0: val = None
        return val
    else: 
        return None



def get_attribute_for_webpage_id(page_id, session_id, attrib_name, db_conn):
    """
    @returns attribute value (e.g. all_text) for a URL in a target scraping session;
        None if URL or attribute does not exist 
    """
    #print("get_attribute_for_webpage_id", page_id)
    page_tbl_name = get_webdump_table_name(session_id)
    attr_tbl_name = get_webdump_attr_table_name(session_id)
    #if page_id==60967:
    #    print('ok')
    sql = """select url, a.page_id, attrib_name, attrib_val from {} p, {} a where a.page_id = p.page_id 
        and p.page_id = {} and a.attrib_name = '{}';""".format(page_tbl_name, attr_tbl_name, page_id, attrib_name)
    #print(sql)
    assert db_conn.closed == 0
    attr_df = pd.read_sql(sql, db_conn)
    df = attr_df[['url', 'page_id', 'attrib_name', 'attrib_val']]
    #print(df)
    if len(df) > 0:
        assert len(df) == 1
        val = df['attrib_val'].tolist()[0]
        if val is not None:
            if len(val) == 0: val = None
        return val
    else: 
        return None


def get_page_id_for_webpage_url(url, session_id, db_conn):
    """
    @returns a list of page IDs for a URL in a target scraping session;
        None if URL or attribute does not exist 
    """
    assert url
    assert session_id
    page_tbl_name = get_webdump_table_name(session_id)
    
    # generate variants 
    url_variants = []
    redirected_url = check_for_url_redirection(url, True, db_conn)
    if redirected_url != url:
        url_variants.append(redirected_url)    
    url_variants.append(url)

    # fix for URL from google results
    google_text_syntax = '#:~:text'
    if google_text_syntax in url:
        url2 = url.split(google_text_syntax)[0]
        assert url2
        url_variants.append(url2)
        url = url2

    if url[-1] == '/':
        url_variants.append(url[:-1])
        url_variants.append(url + 'index.html')
        url_variants.append(url + 'index.htm')
        url_variants.append(url + 'index.php')
    else:
        url_variants.append(url + '/')

    for u in url_variants.copy():
        url_variants.append(u.replace('https://','http://'))
        url_variants.append(u.replace('http://','https://'))

    url_variants = ','.join(["'"+make_string_sql_safe(x)+"'" for x in set(url_variants)])
    sql = """select url, page_id from {} p where p.url in ({});""".format(page_tbl_name, url_variants)
    df = pd.read_sql(sql, db_conn)

    if len(df) > 0:
        val = df['page_id'].tolist()
        return val
    else:
        return []


def get_attribute_for_webpage_url_lookback(url, session_id, attrib_name, db_conn):
    ''' get attribute for webpage. If attribute empty, look back to previous sessions until the page is found '''
    ## DEBUG
    # 20210304 745284; # page here
    # 20210420 29599;
    # 20210914 197;
    #input_text1 = get_attribute_for_webpage_id_lookback(745284, '20210304', 'all_text', db_conn)
    #input_text2 = get_attribute_for_webpage_id_lookback(29599, '20210420', 'all_text', db_conn)
    #input_text3 = get_attribute_for_webpage_id_lookback(197, '20210914', 'all_text', db_conn)
    
    #page_ids = get_page_id_for_webpage_url(url, session_id, db_conn)
    #for page_id in page_ids:
    #    attr = get_attribute_for_webpage_id(page_id, session_id, attrib_name, db_conn)
    #    if attr:
    #        return page_id, attr
    # DEBUG     
    #get_previous_session_tables('20210601', db_conn)
    #get_previous_session_tables('20210401', db_conn)
    #get_previous_session_tables('20210901', db_conn)
    #res111 = get_page_id_for_webpage_url("https://www.lincolnshirelife.co.uk/posts/view/st-katherines-900-years-of-history#:~:text=St%20Katherine's%20is%20housed%20in,wealth%20of%20stories%20to%20tell.", session_id, db_conn)
    #res112 = get_page_id_for_webpage_url("https://www.lincolnshirelife.co.uk/posts/view/st-katherines-900-years-of-history#:~:text=St%20Katherine's%20is%20housed%20in,wealth%20of%20stories%20to%20tell.", session_id, db_conn)

    # scan all sessions from current one back
    session_tables = get_previous_session_tables(session_id, db_conn)
    session_tables.insert(0, get_webdump_table_name(session_id))
    
    for tab in session_tables:
        #print('   get_attribute_for_webpage_url_lookback:',tab)
        prev_session = get_session_id_from_table_name(tab)
        page_ids = get_page_id_for_webpage_url(url, prev_session, db_conn)
        for page_id in page_ids:
            sql = 'select * from {} where page_id = {} and page_content_length > 0;'.format(tab, page_id)
            # prev_session_diff_b, new_page_b, prev_session_page_id, prev_session_table
            df = pd.read_sql(sql, db_conn)
            if len(df) == 0: 
                # check previous page
                continue
            # page found
            d_res = df.iloc[0].to_dict()
            assert d_res['url']
            attr = get_attribute_for_webpage_id(d_res['page_id'], prev_session, attrib_name, db_conn)
            if not attr:
                continue
            #print('   get_attribute_for_webpage_url_lookback: found attr page_id =',d_res['page_id'],tab)
            return page_id, attr

    msg = 'warning: get_attribute_for_webpage_url_lookback: attribute not found for url={} session_id={}'.format(url, session_id)
    print(msg) 
    logger.warn(msg)
    return None, None