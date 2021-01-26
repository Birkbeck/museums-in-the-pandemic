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
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from db.db import open_sqlite, create_page_dump
# scrapy imports
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from utils import get_url_domain, get_app_settings

import logging
logger = logging.getLogger(__name__)

def scrape_websites(museums_df):
    """ Main """
    print("scrape_websites", len(museums_df))
    app_settings = get_app_settings()
    
    # set up local DB
    db_fn = 'tmp/websites.db'
    db = open_sqlite(db_fn)
    init_website_db(db)

    # generate session and timestamp for scraping
    session_id = str(uuid.uuid1())
    ts = datetime.datetime.now()
    logger.info("scraping session_id: "+session_id)

    # load input data
    sample_df = pd.read_csv("data/museums/mip_data_sample_2020_01.tsv", sep='\t') #.head(4) # DEBUG
    logger.debug("sample_df", len(sample_df))
    
    # start crawler
    crawler_process = CrawlerProcess()

    # add all scrapers 
    for i in sample_df.index:
        row = sample_df.loc[i]
        assert row.mm_id
        if isinstance(row.website, str) and len(row.website)>5:
            # valid URL
            scrape_website_scrapy(crawler_process, row.mm_id, row.website, session_id, ts, db, app_settings)
        else:
            logger.debug("empty URL for museum"+row.mm_id)
    # start crawling
    crawler_process.start()

def init_website_db(db_con):
    c = db_con.cursor()

    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS web_pages_dump 
            (url text NOT NULL,
            url_domain text NOT NULL,
            muse_id text NOT NULL, 
            page_content text NOT NULL,
            page_content_length numeric NOT NULL,
            depth numeric NOT NULL,
            session_id text NOT NULL,
            session_ts DATETIME NOT NULL,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP);
            ''')
    db_con.commit()
    logger.debug('init_website_db')


def insert_website_page_in_db(muse_id, url, page_content, response_status, session_id, session_ts, depth, db_conn):
    c = db_conn.cursor()
    sql = '''INSERT INTO web_pages_dump(url, url_domain, muse_id, page_content, page_content_length, depth, session_id, session_ts)
              VALUES(?,?,?,?,?,?,?,?) '''
    cur = db_conn.cursor()
    
    cur.execute(sql, [url, get_url_domain(url), muse_id, page_content, len(page_content), depth, session_id, session_ts])
    db_conn.commit()


class WebsiteSpider(CrawlSpider):
    """ 
    Simple scraper
    https://docs.scrapy.org/en/latest/
    """
    custom_settings = {'DOWNLOAD_DELAY': .5, 'DEPTH_LIMIT': 2}
    name = 'website_scraper'
    rules = [
        Rule(LinkExtractor(unique=True), callback='parse', follow=True),
    ]

    def parse(self, response):
        # call back from scraper
        html = response.body
        url = response.url
        assert self.db_con
        depth = response.meta['depth']
        assert depth >= 1
        scraping_session_id = self.session_id
        scraping_session_ts = self.session_ts
        logger.debug('URL ' + url)
        insert_website_page_in_db(self.muse_id, url, html, response.status, self.session_id, self.session_ts, depth, self.db_con)


def scrape_website_scrapy(crawler_process, muse_id, start_url, session_id, session_ts, db_con, app_settings):
    logger.debug("scrape_website_scrapy: "+muse_id+' '+start_url)
    assert muse_id
    assert start_url
    # find allowed domains
    allowed_domains = app_settings['scraper']['allowed_domains']
    allowed_domains.append(get_url_domain(start_url))
    assert len(allowed_domains)>0
    
    # scrape
    crawler_process.crawl(WebsiteSpider, session_id=session_id, muse_id=muse_id, 
        allowed_domains=allowed_domains, start_urls=[start_url], 
        db_con=db_con, session_ts=session_ts)
