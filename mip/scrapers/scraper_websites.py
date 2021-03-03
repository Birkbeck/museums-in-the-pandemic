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
from utils import get_url_domain, get_app_settings

import logging
logger = logging.getLogger(__name__)

page_counter = 0
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"

def scrape_websites():
    """ Main """
    app_settings = get_app_settings()

    url_df = load_urls_for_wide_scrape()
    print("scrape_websites urls =", len(url_df))
    print("scrape_websites urls =", len(url_df))

    # generate session and timestamp for scraping
    session_id = gen_scraping_session_id()
    logger.info("scraping session_id: "+str(session_id))
        
    # open DB
    db_conn = connect_to_postgresql_db()
    init_website_dump_db(db_conn, session_id)
    
    # DEBUG problem cases
    #sample_df = sample_df[sample_df.mm_id.isin(['mm.aim.0172'])] #'mm.New.102','mm.domus.WM042', 'mm.New.39'
    #url_df = url_df.sample(10)
    # start crawler
    crawler_process = CrawlerProcess()
    websites_counter = 0
    # add all scrapers
    for i in url_df.index:
        row = url_df.loc[i]
        assert row.muse_id
        websites_counter += 1
        msg = ">> {} of {} websites processed".format(websites_counter, len(url_df))
        logger.debug(msg)
        print(msg)
        if is_valid_website(row.url):
            # valid URL
            scrape_website_scrapy(crawler_process, row.muse_id, row.url, session_id, db_conn, app_settings)
            db_conn.commit()
            # delay
            time.sleep(1)
        else:
            logger.debug("MIP issue: empty URL for museum: "+row.mm_id)
    # start crawling
    crawler_process.start()
    
    global page_counter
    logger.info("Scraped "+str(page_counter)+" pages.")


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
    google_df = google_df[google_df.google_rank < 16]
    #print(len(google_df))
    df = google_df[['muse_id','url']].drop_duplicates()
    #print(len(df))

    # keep only non-platform websites
    df = df[google_df['url'].apply(is_valid_website)]

    #for vars, subdf in df.groupby('muse_id'):
        #print(vars, len(subdf))
        #print(subdf)
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
            root_url text NOT NULL,
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
        for blocked_site in ['facebook','twitter','tripadvisor']:
            if blocked_site.lower() in dom.lower():
                valid = False
    return valid


def insert_website_page_in_db(table_name, muse_id, url, referer_url, b_base_url, root_url, page_content, response_status, 
                            session_id, depth, db_conn):
    """ Insert page dump """
    c = db_conn.cursor()
    sql = '''INSERT INTO {} (url, referer_url, is_start_url, root_url, url_domain, muse_id, 
                                        page_content, page_content_length, depth, session_id)
              VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);'''.format(table_name)
    cur = db_conn.cursor()
    try:
        cur.execute(sql, [url, referer_url, b_base_url, root_url, get_url_domain(url), muse_id, page_content, 
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
    sql = "select page_id from {} where url = '{}';".format(get_webdump_table_name(session_id),url)
    res = pd.read_sql(sql, db_conn)
    if len(res) > 0: 
        return True
    else: 
        return False


class WebsiteSpider(CrawlSpider):
    """ 
    Simple scraper
    https://docs.scrapy.org/en/latest/
    """
    custom_settings = {
        'USER_AGENT': user_agent,
        'DOWNLOAD_DELAY': .5,
        'DEPTH_LIMIT': 2
    }
    name = 'website_scraper'
    rules = [
        Rule(LinkExtractor(unique=True, deny=[]), callback='parse', follow=True),
    ]

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
        # extract fields from response
        depth = response.meta['depth']
        assert depth >= 1
        scraping_session_id = self.session_id
        # get ancestor link
        referer_url = response.request.headers.get('Referer', None)
        if referer_url:
            referer_url = str(referer_url,'utf-8')
        b_base_url = url in self.start_urls
        root_url = ''.join(self.start_urls[0])
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
        del found_doms, cur_dom

        if b_allowed:
            # valid URL, save it in DB
            if url_session_exists(url, scraping_session_id, self.db_con):
                return
            insert_website_page_in_db(self.table_name, self.muse_id, url, referer_url, b_base_url, 
                    root_url, html, response.status, self.session_id, depth, self.db_con)
            logger.debug('url saved: ' + url)
            logger.debug('page_counter: ' + str(page_counter))
        else:
            logger.debug("MIP debug: url "+url+" is not allowed. Skipping.")


def check_for_url_redirection(url):
    """ Useful to include redirected domain for scraping """
    assert url
    try:
        # user agent must be defined, otherwise some sites says 403
        req = urllib.request.Request(url, headers={'User-Agent': user_agent})
        response = urllib.request.urlopen(req) #, timeout=5
        new_url = response.geturl()
        redirected = new_url != url
        if redirected:
            return new_url
    except:
        logger.warning("MIP check_for_url_redirection: could not check redirection for "+url)
    return None


def scrape_website_scrapy(crawler_process, muse_id, start_url, session_id, db_con, app_settings):
    """ TODO: document """
    assert muse_id
    assert start_url
    logger.debug("scrape_website_scrapy: "+muse_id+' '+start_url)
    muse_id = muse_id.strip()
    start_url = start_url.strip()
    
    # find allowed domains, including redirections
    allowed_domains = [get_url_domain(start_url)]
    start_urls = [start_url]
    redirect_url = check_for_url_redirection(start_url)
    if redirect_url:
        start_urls.append(redirect_url)
        allowed_domains.append(get_url_domain(redirect_url))
    assert len(allowed_domains)>0
    assert len(start_urls)>0
    donotfollow_domains = app_settings['website_scraper']['donotfollow_domains']
    # scrape
    crawler_process.crawl(WebsiteSpider, 
        session_id=session_id,
        table_name=get_webdump_table_name(session_id),
        muse_id=muse_id, 
        allowed_domains=allowed_domains, 
        donotfollow_domains=donotfollow_domains,
        start_urls=start_urls,
        db_con=db_con)


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