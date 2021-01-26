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
import scrapy
from scrapy.crawler import CrawlerProcess

import logging
logger = logging.getLogger(__name__)

def scrape_websites(museums_df):
    """ """
    print("scrape_websites", len(museums_df))

    db_fn = 'tmp/websites.db'
    db = open_sqlite(db_fn)
    create_page_dump(db)
    # generate session for scraping
    session_id = str(uuid.uuid1())
    logger.info("scraping session_id: "+session_id)

    start_scrapy(db, session_id)
    return

    #sample_df = pd.read_csv("data/museums/mip_data_sample_2020_01.tsv", sep='\t')
    #print("sample_df", len(sample_df))

    #for i in sample_df.index:
    #    row = sample_df.loc[i]
    #    scrape_website(row.mm_id, row.mm_name, row.website)

    
def scrape_website(muse_id, muse_name, url):
    assert muse_id
    assert muse_name
    assert url
    print("scrape_website",url)


class WebsiteSpider(scrapy.Spider):
    """ 
    https://docs.scrapy.org/en/latest/
    """
    def __init__(self, category=None, *args, **kwargs):
        super(WebsiteSpider, self).__init__(*args, **kwargs)
    
    custom_settings = {'DOWNLOAD_DELAY': 1, 'DEPTH_LIMIT': 3}
    name = 'museum_website_scraper'
    
    #def start_requests(self):
    #    logger.debug("Museum id:"+self.muse_id)
    #    yield scrapy.Request(url) #, headers=headers, params=params,callback = self.parse)
    
    def parse(self,response):
        # write result in DB
        html = response.body
        url = response.url
        #url_exists(google_db_fn, )
        muse_id = self.muse_id
        assert self.db_con
        scrape_session_id = self.session_id
        # TODO: insert scraped page


def start_scrapy(db_con, session_id):
    process = CrawlerProcess()
    process.crawl(WebsiteSpider, session_id=session_id, muse_id='test', start_urls=['https://www.bbc.co.uk/'], db_con=db_con)
    process.start()