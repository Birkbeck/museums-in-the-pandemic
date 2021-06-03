# -*- coding: utf-8 -*-
#
# Web scraper

import time
import uuid
import json
import os
import random
import calendar
import datetime
from datetime import date
import pandas as pd
import urllib
import numpy as np
import constants
from urllib.parse import urlparse
from museums import get_museums_w_web_urls, load_all_google_results
from scrapy.exceptions import CloseSpider
from db.db import connect_to_postgresql_db, check_dbconnection_status, make_string_sql_safe
# scrapy imports
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.dupefilters import RFPDupeFilter
import re
import difflib
from utils import is_url, get_url_domain, get_app_settings, split_dataframe, parallel_dataframe_apply, get_soup_from_html, get_all_text_from_soup, garbage_collect, _is_number

import logging
logger = logging.getLogger(__name__)

page_counter = 0

def scrape_websites():
    """ Main """
    app_settings = get_app_settings()

    #url_df = load_urls_for_wide_scrape()
    url_df = load_urls_for_narrow_scrape()

    # shuffle URLs
    url_df = url_df.sample(frac=1)
    print("scrape_websites urls =", len(url_df))
    # generate session and timestamp for scraping
    global session_id
    session_id = gen_scraping_session_id() # first scrape ID "20210304" 
    logger.info("scraping session_id: "+str(session_id))

    # open DB
    global db_conn
    db_conn = connect_to_postgresql_db()
    init_url_redirection_db(db_conn)
    init_website_dump_db(db_conn, session_id)
    
    # init crawler pool
    crawler_process = CrawlerProcess()

    # DEBUG
    #url_df = url_df[url_df.url=='https://marblebar.org.au/company/st-peters-heritage-centre-hall-1460398/']
    #url_df = url_df[url_df.url=='https://www.nvr.org.uk/']
    #url_df = url_df.sample(15, random_state=7134)
    #url_df.to_pickle('tmp/museum_scraping_input_debug.pik')
    #url_df = pd.read_pickle('tmp/museum_scraping_input_debug.pik')
    #url_df.to_excel("tmp/museum_scraping_input_debug.xlsx",index=False)
    # END DEBUG

    assert len(url_df) > 0
    max_urls_single_crawler = 5000
    # split df and create a new crawler for each chunk
    chunks = split_dataframe(url_df, max_urls_single_crawler)
    for df in chunks:
        logger.debug("urls chunk urls={} chunks={}".format(len(df),len(chunks)))
        assert len(df)>0
        #assert df['url'].is_unique
        # find redirections
        redirected_url_df = parallel_dataframe_apply(df, check_redirections_before_scraping, n_cores=8)
        # set up crawler
        start_urls = redirected_url_df.url.tolist()
        msg = "start crawler with start_urls={}".format(len(start_urls))
        logger.info(msg)
        print(msg)
        donotfollow_domains = app_settings['website_scraper']['donotfollow_domains']
        allowed_domains = redirected_url_df.domain.tolist()
        # add crawler to pool
        crawler_process.crawl(MultiWebsiteSpider, 
            session_id=session_id,
            table_name=get_webdump_table_name(session_id),
            museums_df=redirected_url_df,
            allowed_domains=allowed_domains, 
            donotfollow_domains=donotfollow_domains,
            start_urls=start_urls,
            db_con=db_conn)
    
    garbage_collect()
    # start crawling (hangs here)
    crawler_process.start()
    global page_counter
    logger.info("Scraped ended page_counter={}".format(page_counter))


def check_redirections_before_scraping(df):
    """ called in parallel """
    msg = "check_redirections_before_scraping urls={}".format(len(df))
    logger.info(msg)
    print(msg)
    local_db_conn = connect_to_postgresql_db()
    redirected_url_df = pd.DataFrame()

    for ind in range(len(df)):
        if ind % 200 == 0:
            logger.debug("   redir n={}".format(ind))
        row = df.iloc[ind]
        redirect_url = check_for_url_redirection(row.url, True, local_db_conn)
        if redirect_url != 'timeout':
            row['url'] = redirect_url
        
        redirected_url_df = redirected_url_df.append(row)

    assert len(redirected_url_df) == len(df)
    assert len(redirected_url_df)>=0
    redirected_url_df['domain'] = redirected_url_df['url'].apply(get_url_domain)
    local_db_conn.close()
    return redirected_url_df


def gen_scraping_session_id():
    """ Session id is the current date """
    session_id = str(date.today()).replace('-','')
    logger.info("scraping session_id: "+str(session_id))
    return session_id


def load_urls_for_narrow_scrape():
    """ Load only relevant websites. Combine sample and predicted links """
    # generate results
    df = get_museums_w_web_urls()
    df = df[df.url_source!='no_pred']
    # keep only real URLs
    validdf = df[df['url'].apply(is_url)]
    #validdf['id_duplicated'] = validdf.duplicated(subset=['url'])
    #validdf.to_csv('tmp/urls.tsv',sep='\t')
    
    #assert validdf['url'].is_unique
    #df = df.drop_duplicates(subset=['url'])
    #assert df['url'].is_unique
    msg = "load_urls_for_narrow_scrape Museums={}; all rows={}; valid URLs={}".format(df.muse_id.nunique(), len(df), len(validdf))
    print(msg)
    logger.info(msg)
    return validdf


def load_urls_for_wide_scrape():
    """ load Google top 15-20 urls for each museum for wide scraping """
    print("load_urls_for_wide_scrape")
    # load google results
    google_df = load_all_google_results()
    #for vars, subdf in google_df.groupby(['search_variety','search_type','scrape_target']):
    #    print(vars, len(subdf))
    google_df = google_df[google_df.google_rank < 11]
    #print(len(google_df))
    df = google_df[['muse_id','url']].drop_duplicates()
    #print(len(df))

    # keep only non-platform websites
    df = df[google_df['url'].apply(is_valid_website)]
    
    df = df.drop_duplicates(subset=['url'])
    assert df['url'].is_unique
    msg = "load_urls_for_wide_scrape Museums={} URLs={}".format(df.muse_id.nunique(), len(df))
    print(msg)
    logger.info(msg)
    return df


def init_url_redirection_db(db_con):
    check_dbconnection_status(db_con)
    c = db_con.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS websites.url_redirections (
            url text PRIMARY KEY,
            redirected_to_url text,
            ts timestamp DEFAULT CURRENT_TIMESTAMP);
            ''')
    # TODO: add indices on url and muse_id
    # CREATE INDEX IF NOT EXISTS idx1 ON websites.web_pages_dump_20210304 USING btree(muse_id);
    # CREATE INDEX IF NOT EXISTS idx2 ON websites.web_pages_dump_20210304 USING btree(url);
    # TODO: add website ranking
    db_con.commit()
    logger.debug('init_url_redirection_db')


def init_website_dump_db(db_con, session_id):
    assert session_id
    check_dbconnection_status(db_con)
    c = db_con.cursor()
    table_name = get_webdump_table_name(session_id)
    
    # drop table (CAREFUL: DESTRUCTIVE)
    #c.execute('DROP TABLE IF EXISTS {};'.format(table_name))

    # SELECT table_name FROM information_schema.tables WHERE table_schema='public'

    # Create table
    sql = '''CREATE TABLE IF NOT EXISTS {0} (
            page_id serial PRIMARY KEY,
            url text NOT NULL,
            referer_url text,
            session_id text NOT NULL,
            is_start_url boolean NOT NULL,
            url_domain text NOT NULL,
            muse_id text NOT NULL,
            page_content text,
            page_content_length numeric NOT NULL,
            depth numeric NOT NULL,
            google_rank numeric,
            new_page_b boolean,
            prev_session_diff_b boolean,
            prev_session_diff json,
            prev_session_table text,
            prev_session_page_id numeric,
            ts timestamp DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(url, session_id));

            CREATE INDEX IF NOT EXISTS idx1 ON {0} USING btree(muse_id);
            CREATE INDEX IF NOT EXISTS idx2 ON {0} USING btree(url);
            CREATE INDEX IF NOT EXISTS idx3 ON {0} USING btree(prev_session_diff_b);
            CREATE INDEX IF NOT EXISTS idx4 ON {0} USING btree(prev_session_page_id);
            CREATE INDEX IF NOT EXISTS idx5 ON {0} USING btree(new_page_b);
            '''.format(table_name)
    c.execute(sql)
    
    db_con.commit()
    logger.debug('init_website_dump_db')


def is_valid_website(url):
    """ This is to find simple museum websites and not platforms """
    valid = True
    assert url != 'timeout'
    if not(isinstance(url, str) and len(url)>5):
        valid = False
    else:
        dom = get_url_domain(url)
        for blocked_site in ['facebook.','twitter.','tripadvisor.','.google.co','linkedin.','expedia.']:
            if blocked_site.lower() in dom.lower():
                valid = False
    return valid


#def get_museum_ids_from_session(session_id, db_conn):
#    TODO "select distinct muse_id as muse_id from {};".format()

def insert_website_page_in_db(table_name, muse_id, url, referer_url, b_base_url, page_content, response_status, 
                            session_id, depth, db_conn, prev_session_diff_b,
                            prev_session_diff, prev_session_table, prev_session_page_id, new_page_b):
    """ Insert page dump """

    if not new_page_b and not prev_session_diff_b:
        assert page_content is None

    if page_content is None:
        page_length = 0
    else:
        page_length = len(page_content)

    sql = '''INSERT INTO {} (url, referer_url, is_start_url, url_domain, muse_id, 
                            page_content, page_content_length, depth, session_id, 
                            prev_session_diff, prev_session_table, prev_session_page_id, prev_session_diff_b, new_page_b)
              VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);'''.format(table_name)
    cur = db_conn.cursor()
    try:
        cur.execute(sql, [url, referer_url, b_base_url, get_url_domain(url), muse_id, page_content, 
                    page_length, depth, session_id, prev_session_diff, prev_session_table, prev_session_page_id, 
                    prev_session_diff_b, new_page_b])
        db_conn.commit()
    except Exception as e:
        logger.error('error while inserting in insert_website_page_in_db')
        logger.error(str(e))
        raise e
    return True


def get_webdump_table_name(session_id):
    """ Table for a single scraping session """
    table_name = "websites.web_pages_dump_"+str(session_id)
    return table_name


def get_session_id_from_table_name(table):
    session_id = table.replace("websites.web_pages_dump_",'')
    assert len(session_id) == 8
    return session_id


def url_session_exists(url, session_id, db_conn):
    """ Check if the page has been scraped already in this session """
    cur = db_conn.cursor()
    sql = "select page_id from {} where url = '{}';".format(get_webdump_table_name(session_id), make_string_sql_safe(url))
    res = pd.read_sql(sql, db_conn)
    if len(res) > 0: 
        return True
    else: 
        return False


def compare_html_with_previous_versions(url, html, current_session_id, db_conn):
    """ Scraper Logic """
    res = {}
    # check if URL is present in the previous session
    prev_session_tables = get_previous_session_tables(current_session_id, db_conn)
    assert len(prev_session_tables) > 0

    def get_url_from_session(url, tab, db_conn):
        cur = db_conn.cursor()
        sql = """select page_id, page_content
            from {} where url = '{}' and page_content_length > 0;""".format(tab, make_string_sql_safe(url))
        res = pd.read_sql(sql, db_conn)
        if len(res) == 0:
            return None
        return res
    
    # init variables
    changed = None
    prev_session_diff = None
    prev_page_id = None
    new_page_b = True
    html_found = False
    page_content = None
    session_table = None

    # check if url exists in previous sessions in reverse order
    # find session where page was new
    # look for previous version of page (HTML)
    for tab in prev_session_tables:
        page_d = get_url_from_session(url, tab, db_conn)
            # break
        if page_d is not None and len(page_d) > 0:
            # found
            prev_page_id = page_d['page_id'].tolist()[0]
            page_content = page_d['page_content'].tolist()[0]
            html_found = True
            session_table = tab
            break
    
    new_page_b = not html_found
    if html_found:
        prev_page_id, prev_text = get_previous_version_of_page_text(url, session_table, db_conn)
        changed = False
        if prev_text is not None:
            soup = get_soup_from_html(html)
            new_text = get_all_text_from_soup(soup)
            changed = new_text != prev_text
            if changed:
                # prev_session_diff prev_session_id prev_session_page_id
                prev_session_diff = diff_texts(prev_text, new_text)
                if prev_session_diff is not None:
                    # save as json
                    assert len(prev_session_diff) > 0
                    prev_session_diff = json.dumps(prev_session_diff)
                else: 
                    # there are actually no differences between versions
                    changed = False
            soup.decompose()
            del soup

        # valid URL, save it in DB
        if not changed:
            # make sure not to save the same page more than once
            # page is identical to previous version, don't save (save storage space)
            html = None

    # pack results
    res['changed'] = changed
    res['prev_session_diff'] = prev_session_diff
    res['prev_session_table'] = session_table
    res['prev_page_id'] = prev_page_id
    res['new_page_b'] = new_page_b
    res['html'] = html
    return res


class CustomLinkExtractor(LinkExtractor):
    """ This is to avoid rescraping the same URL in the same session. """
    
    def _link_allowed(self, link):
        """ return True if URL has to be scraped (URL is not in the DB) """
        # check if URL is in DB
        allowed = super()._link_allowed(link)
        if not allowed:
            return False
        
        if not is_valid_website(link.url):
            return False

        global session_id
        global db_conn
        assert session_id
        already_in_session = url_session_exists(link.url, session_id, db_conn)
        if already_in_session:
            return False
        
        # run this again
        allowed = super()._link_allowed(link)
        if not allowed:
            return False
        return True


class MultiWebsiteSpider(CrawlSpider):
    """ 
    Spider to scrape museum pages
    https://docs.scrapy.org/en/latest/
    """
    custom_settings = {
        'USER_AGENT': constants.user_agent,
        'DOWNLOAD_DELAY': .2,
        'DEPTH_LIMIT': 1
        #'CLOSESPIDER_ERRORCOUNT': 1
    }
    name = 'website_scraper'
    rules = [
        Rule(CustomLinkExtractor(unique=True, deny=[]), callback='parse', follow=True),
    ]

    def get_museum_id_for_url(self, url):
        """  """
        cur_dom = get_url_domain(url)
        muse_ids = self.museums_df[self.museums_df.domain == cur_dom]['muse_id']
        if len(muse_ids)==0:
            raise ValueError('Museum ID not found '+url)
        muse_id = muse_ids.tolist()[0]
        return muse_id

    def parse(self, response):
        """ Parse cralwed HTML page and save it in DB """
        global page_counter
        page_counter += 1
        #try:
        # call back from scraper
        # extract fields
        # response.text and not response.body! body is a bytestring
        html = response.text 
        url = response.url
        assert self.db_con
        try: 
            muse_id = self.get_museum_id_for_url(url)
        except ValueError as e:
            logger.debug(e)
            # skip this url
            return 
        # extract fields from response
        depth = response.meta['depth']
        assert depth >= 1
        scraping_session_id = self.session_id    
        # get ancestor link
        referer_url = response.request.headers.get('Referer', None)
        if referer_url:
            referer_url = str(referer_url, 'utf-8')
        b_base_url = url in self.start_urls
        # check if url is in allowed domains
        b_allowed = True #False # DEBUG
        for dom in self.donotfollow_domains:
            if dom in url:
                b_allowed = False
        del dom
        # check if url is allowed
        allowed_doms = [get_url_domain(surl.lower()) for surl in self.start_urls]
        cur_dom = get_url_domain(url.lower())
        found_doms = [dom in cur_dom for dom in allowed_doms]
        if not any(found_doms):
            logger.warning(url+" not allowed. This should not happen, allowed domains: " + str(allowed_doms))
            b_allowed = False
            return
        del found_doms, cur_dom

        if not b_allowed:
            logger.debug("MIP debug: url "+url+" is not allowed. Skipping.")
            return
        if url_session_exists(url, scraping_session_id, self.db_con):
            return
        
        # get previous version
        version_comparison_d = compare_html_with_previous_versions(url, html, self.session_id, self.db_con)
        
        check_dbconnection_status(self.db_con)
        insert_website_page_in_db(self.table_name, muse_id, url, referer_url, b_base_url, 
                    version_comparison_d['html'], response.status, self.session_id, depth, self.db_con,
                    prev_session_diff_b=version_comparison_d['changed'],
                    new_page_b=version_comparison_d['new_page_b'],
                    prev_session_diff=version_comparison_d['prev_session_diff'],
                    prev_session_table=version_comparison_d['prev_session_table'],
                    prev_session_page_id=version_comparison_d['prev_page_id'])
        logger.debug('url saved: ' + url)
        logger.debug('page_counter: ' + str(page_counter))
        
        #except Exception as e:
        #    logger.error(e)
        #    print(e)
        #    raise CloseSpider(str(e))

def get_previous_session_tables(session_id, db_conn):
    """ to get previous version of pages """
    #global db_conn
    tabs = get_scraping_session_tables(db_conn)
    other_tabs = [t for t in tabs if not session_id in t]
    assert len(other_tabs)>0
    previous_session_tables = sorted(other_tabs, reverse=True)
    return previous_session_tables


def get_url_redirection_from_db(url, db_conn):
    """  """
    # check if redirection is in DB
    sql = "select redirected_to_url from websites.url_redirections where url=%s;"
    cur = db_conn.cursor()
    #res = cur.execute(sql, [url])
    res1 = pd.read_sql(sql, db_conn, params=[url])
    res = res1['redirected_to_url'].tolist()
    if len(res)>0:
        assert len(res)==1
        return res[0]
    else:
        return None


def check_for_url_redirection(url, check_db=False, db_conn=None):
    """ 
    Useful to include redirected domain for scraping. 
    Uses table in DB to cache redirections.
    NOTE: It does NOT work for Facebook/Twitter
    @returns redirected url or the same url if there is no redirection
    """
    assert url
    res = None
    if check_db:
        check_dbconnection_status(db_conn)
        db_res = get_url_redirection_from_db(url, db_conn)
        if db_res:
            if db_res != 'timeout':
                return db_res
            else:
                # timeout, return same URL
                return url

    new_url = 'timeout'
    try:
        # user agent must be defined, otherwise some sites says 403
        req = urllib.request.Request(url, headers={'User-Agent': user_agent})
        response = urllib.request.urlopen(req, timeout=5)
        new_url = response.geturl()
        redirected = new_url != url
    except:
        logger.warning("MIP check_for_url_redirection: could not check redirection for "+url)
    
    if check_db:
        sql = "insert into websites.url_redirections(url, redirected_to_url) VALUES(%s, %s);"
        cur = db_conn.cursor()
        cur.execute(sql, [url, new_url])
        db_conn.commit() 
    return new_url


def get_scraping_session_tables(db_conn):
    """ table names with schema in in reversed order """
    sql = "SELECT table_name FROM information_schema.tables WHERE table_schema='websites';"
    res = pd.read_sql(sql, db_conn)['table_name'].tolist()
    res = [r for r in res if 'web_pages' in r]
    res = ["websites."+r for r in res if not 'attr' in r]
    res = sorted(res, reverse=True)
    return res


def get_scraping_session_stats_by_museum(table_name, db_conn):
    """ Get stats from website scraping table """
    assert not "attr" in table_name
    sql = """select session_id, muse_id, count(page_id) as page_n, sum(page_content_length) as data_size
        from {} group by session_id, muse_id;
        """.format(table_name)
    df = pd.read_sql(sql, db_conn)
    return df


def get_url_content_from_latest_session(url, db_conn):
    """
    get URL from website DB starting from latest table
    """
    #print(url)
    if pd.isnull(url) or not is_url(url): 
        return None
    
    tables = get_scraping_session_tables(db_conn)
    for tbl in tables:
        sql = """select d.url, d.page_content from {} as d 
            where url = '{}';""".format(tbl, make_string_sql_safe(url))
        resdf = pd.read_sql(sql, db_conn)
        if len(resdf) > 0:
            html = resdf.loc[0, 'page_content']
            if html is not None and len(html) > 0:
                return html
    return None

def get_previous_version_of_page_text(url, table_name, db_conn):
    """ look for page in previous scraping session """
    assert table_name
    sql = """select d.page_id, d.url, d.session_id, a.attrib_name, a.attrib_val from {} d left join {} a 
        on d.page_id = a.page_id 
        where url = '{}';""".format(table_name, table_name + constants.table_suffix, make_string_sql_safe(url))
    df = pd.read_sql(sql, db_conn)
    #print(df.columns, len(df))
    if len(df) == 0:
        return None, None
    # get all_text for HTML page
    resdf = df.loc[df['attrib_name']=='all_text']
    
    if len(resdf) == 1:
        # text found, return it
        page_id = resdf['page_id'].tolist()[0]
        text = resdf['attrib_val'].tolist()[0]
        return page_id, text
    # text not found
    return None, None


def diff_texts(text_a, text_b):
    """ find deltas between texts (to calculate page differences) """
    if text_a is None: text_a = ''
    if text_b is None: text_b = ''
    
    a = clean_text_for_diff(text_a)
    b = clean_text_for_diff(text_b)
    diffs = difflib.unified_diff(a, b)
    diff_lines = [line for line in diffs]
    assert diff_lines is not None
    if len(diff_lines) == 0:
        return None
    return diff_lines


def clean_text_for_diff(text):
    """ remove all special chars and flatten text """
    text = re.sub(r'\n+', '\n', text)
    lines = text.split('\n')
    res = []
    
    for l in lines:
        l_clean = re.sub('\W+', ' ', l.lower())
        l_clean = re.sub(' +', ' ', l_clean).strip()
        if l_clean is not None and len(l_clean)>0:
            res.append(l_clean)
    return res


def is_valid_facebook_url(url):
    url = url.lower().strip()
    if not 'facebook.com' in url: return False
    if 'sharer.php' in url: return False
    if 'share.php?' in url: return False
    if '/share?' in url: return False
    if '/dialog/' in url: return False
    if '/hashtag/' in url: return False
    if 'twitter.co' in url: return False
    if 'linkedin.' in url: return False
    return True


def is_valid_twitter_url(url):
    url = url.lower().strip()
    if not 'twitter.com' in url: return False
    if '/share?' in url: return False
    if '/intent/' in url: return False
    if 'search?' in url: return False
    if '/home?' in url: return False
    if url == 'http://twitter.com/share': return False
    if url == 'http://twitter.com/': return False
    if url == 'http://twitter.com': return False
    if '/hashtag/' in url: return False
    if 'facebook.co' in url: return False
    if 'linkedin.' in url: return False
    if url.count('/') < 3: 
        return False
    return True


def is_valid_social_url(url):
    return is_valid_facebook_url(url) or is_valid_twitter_url(url)


def extract_fb_tw_links_from_pages():
    """Look for facebook and twitter links in museum websites """

    def clean_fb_url(url, db_conn):
        assert is_valid_facebook_url(url)
        if "?" in url:
            url = url.split("?")[0]
        
        if url[-1] == '/':
            url = url[:-1]
        #rurl = check_for_url_redirection(url, True, db_conn)
        #if rurl != 'timeout':
        #    rurl = url
        return url

    def clean_tw_url(url, db_conn):
        if "?" in url:
            url = url.split("?")[0]
        if '/status/' in url:
            url = url.split('/status/')[0]
        if '/statuses/' in url:
            url = url.split('/statuses/')[0]

        if url[-1] == '/':
            url = url[:-1]
        
        url = url.replace('@', '')
        #rurl = check_for_url_redirection(url, True, db_conn)
        #if rurl != 'timeout':
        #    rurl = url
        return url

    print("extract_fb_tw_links_from_pages")
    df = get_museums_w_web_urls().sample(frac = 1)
    db_conn = connect_to_postgresql_db()

    rows_list = []

    for idx, mus in df.iterrows():
        if idx % 100 == 0:
            print(idx, end=' ')
        
        #if idx == 700: break # DEBUG
        muse_id = mus['muse_id']
        mname = mus['musname']
        url = mus['url']
        mus_links = []
        #print("\n", muse_id, mname, url)
        html = get_url_content_from_latest_session(url, db_conn)
        if html is not None:
            # look for links in HTML
            sp = get_soup_from_html(html)
            for link in sp.find_all('a', href=True):
                child_url = link['href']
                child_url = child_url.lower().strip()
                if is_valid_facebook_url(child_url):
                    mus_links.append({'museum_id':muse_id, 'type':'facebook', 'url': child_url, 
                        'clean_url': clean_fb_url(child_url, db_conn)})
                if is_valid_twitter_url(child_url):
                    mus_links.append({'museum_id':muse_id, 'type':'twitter', 'url': child_url, 
                        'clean_url': clean_tw_url(child_url, db_conn)})
        
        if len(mus_links)==0 or not pd.DataFrame(mus_links).type.isin(['facebook']).any():
            # facebook not found
            mus_links.append({'museum_id':muse_id, 'type':'facebook', 'url':'no_resource', 'clean_url':'no_resource'})
        if len(mus_links)==0 or not pd.DataFrame(mus_links).type.isin(['twitter']).any():
            # twitter not found
            mus_links.append({'museum_id':muse_id, 'type':'twitter', 'url':'no_resource', 'clean_url':'no_resource'})
        
        rows_list.extend(mus_links)
    print('')
    sociallinks_df = pd.DataFrame(rows_list)
    sociallinks_df = sociallinks_df.drop_duplicates(['museum_id','type','clean_url'])

    #sociallinks_df['url_valid'] = sociallinks_df['url'].apply(is_valid_social_url)
    valid_sociallinks_df = sociallinks_df
    #valid_sociallinks_df = sociallinks_df[sociallinks_df['url'].apply(is_valid_social_url)]

    print("sociallinks_df n =",len(valid_sociallinks_df))
    valid_sociallinks_df.to_csv('tmp/websites_social_links.tsv', sep='\t', index_label='row_id')
    return valid_sociallinks_df

