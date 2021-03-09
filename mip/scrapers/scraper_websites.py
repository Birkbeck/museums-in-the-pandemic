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
from urllib.parse import urlparse
from museums import load_all_google_results
from db.db import connect_to_postgresql_db
# scrapy imports
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.dupefilters import RFPDupeFilter
from utils import get_url_domain, get_app_settings, split_dataframe, parallel_dataframe_apply

import logging
logger = logging.getLogger(__name__)

page_counter = 0
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"

def scrape_websites():
    """ Main """
    app_settings = get_app_settings()

    url_df = load_urls_for_wide_scrape()
    print("scrape_websites urls =", len(url_df))

    # generate session and timestamp for scraping
    global session_id
    # DEBUG <<<< REMOVE <<<<<< IMPORTANT
    session_id = "20210304" #gen_scraping_session_id() 
    logger.info("scraping session_id: "+str(session_id))

    # open DB
    global db_conn
    db_conn = connect_to_postgresql_db()
    init_website_dump_db(db_conn, session_id)
    
    # init crawler pool
    crawler_process = CrawlerProcess()

    # DEBUG
    #url_df = url_df[url_df.url=='https://marblebar.org.au/company/st-peters-heritage-centre-hall-1460398/']
    #url_df.to_excel("tmp/museum_scraping_input.xlsx",index=False)
    #url_df = url_df.sample(10, random_state=7134)
    
    max_urls_single_crawler = 5000
    # split df and create a new crawler for each chunk
    chunks = split_dataframe(url_df, max_urls_single_crawler)
    for df in chunks:
        logger.debug("urls chunk urls={} chunks={}".format(len(df),len(chunks)))
        assert len(df)>0
        assert df['url'].is_unique
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

    # start crawling (hangs here)
    crawler_process.start()
    global page_counter
    logger.info("Scraped ended page_counter={}".format(page_counter))


def check_redirections_before_scraping(df):
    msg = "check_redirections_before_scraping urls={}".format(len(df))
    logger.info(msg)
    print(msg)
    redirected_url_df = pd.DataFrame()

    for ind in range(len(df)):
        if ind % 200 == 0:
            logger.debug("   redir n={}".format(ind))
        row = df.iloc[ind]
        redirect_url = check_for_url_redirection(row.url)
        if redirect_url:
            row['url'] = redirect_url
        redirected_url_df = redirected_url_df.append(row)

    assert len(redirected_url_df) == len(df)
    assert len(redirected_url_df)>0
    redirected_url_df['domain'] = redirected_url_df['url'].apply(get_url_domain)
    return redirected_url_df


def gen_scraping_session_id():
    """ Session id is the current date """
    session_id = str(date.today()).replace('-','')
    logger.info("scraping session_id: "+str(session_id))
    return session_id


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
    msg = "load_urls_for_wide_scrape Museums={} URLs={}".format(df.muse_id.nunique(),len(df))
    print(msg)
    logger.info(msg)
    return df


def init_website_dump_db(db_con, session_id):
    assert session_id
    c = db_con.cursor()
    table_name = get_webdump_table_name(session_id)
    
    # drop table (CAREFUL: DESTRUCTIVE)
    #c.execute('DROP TABLE IF EXISTS {};'.format(table_name))

    # SELECT table_name FROM information_schema.tables WHERE table_schema='public'

    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS {} (
            page_id serial PRIMARY KEY,
            url text NOT NULL,
            referer_url text,
            session_id text NOT NULL,
            is_start_url boolean NOT NULL,
            url_domain text NOT NULL,
            muse_id text NOT NULL, 
            page_content text NOT NULL,
            page_content_length numeric NOT NULL,
            depth numeric NOT NULL,
            ts timestamp DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(url, session_id));
            '''.format(table_name))
    db_con.commit()
    logger.debug('init_website_dump_db')


def is_valid_website(url):
    """ This is to find simple museum websites and not platforms """
    valid = True
    if not(isinstance(url, str) and len(url)>5):
        valid = False
    else:
        dom = get_url_domain(url)
        for blocked_site in ['facebook','twitter','tripadvisor','.google.co']:
            if blocked_site.lower() in dom.lower():
                valid = False
    return valid


def insert_website_page_in_db(table_name, muse_id, url, referer_url, b_base_url, page_content, response_status, 
                            session_id, depth, db_conn):
    """ Insert page dump """
    c = db_conn.cursor()
    sql = '''INSERT INTO {} (url, referer_url, is_start_url, url_domain, muse_id, 
                                        page_content, page_content_length, depth, session_id)
              VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s);'''.format(table_name)
    cur = db_conn.cursor()
    try:
        cur.execute(sql, [url, referer_url, b_base_url, get_url_domain(url), muse_id, page_content, 
                    len(page_content), depth, session_id])
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


def url_session_exists(url, session_id, db_conn):
    """ Check if the page has been scraped already in this session """
    cur = db_conn.cursor()
    sql = "select page_id from {} where url = '{}';".format(get_webdump_table_name(session_id), url)
    res = pd.read_sql(sql, db_conn)
    if len(res) > 0: 
        return True
    else: 
        return False


class CustomLinkExtractor(LinkExtractor):
    """ This is to avoid rescraping the same URL in the same session. """
    
    def _link_allowed(self, link):
        """ return True if URL has to be scraped (URL is not in the DB) """
        # check if URL is in DB
        allowed = super()._link_allowed(link)
        if not allowed:
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
        'USER_AGENT': user_agent,
        'DOWNLOAD_DELAY': .2,
        'DEPTH_LIMIT': 1
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
        # check if url is 
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
        # valid URL, save it in DB
        insert_website_page_in_db(self.table_name, muse_id, url, referer_url, b_base_url, 
                    html, response.status, self.session_id, depth, self.db_con)
        logger.debug('url saved: ' + url)
        logger.debug('page_counter: ' + str(page_counter))


def check_for_url_redirection(url):
    """ Useful to include redirected domain for scraping """
    assert url
    try:
        # user agent must be defined, otherwise some sites says 403
        req = urllib.request.Request(url, headers={'User-Agent': user_agent})
        response = urllib.request.urlopen(req, timeout=5)
        new_url = response.geturl()
        redirected = new_url != url
        if redirected:
            return new_url
    except:
        logger.warning("MIP check_for_url_redirection: could not check redirection for "+url)
    return None


def get_scraping_session_tables(db_conn):
    """  """
    sql = "SELECT table_name FROM information_schema.tables WHERE table_schema='websites';"
    res = pd.read_sql(sql, db_conn)['table_name'].tolist()
    res = ["websites."+r for r in res if not 'attr' in r]
    return res


def get_scraping_session_stats_by_museum(table_name, db_conn):
    """ Get stats from website scraping table """
    assert not "attr" in table_name
    sql = """select session_id, muse_id, count(page_id) as page_n, sum(page_content_length) as data_size
        from {} group by session_id, muse_id;
        """.format(table_name)
    df = pd.read_sql(sql, db_conn)
    return df