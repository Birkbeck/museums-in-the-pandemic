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
from db.db import *
import scrapy
from scrapy.crawler import CrawlerProcess

import logging
logger = logging.getLogger(__name__)

def scrape_websites(museums_df):
    """ """
    print("scrape_websites", len(museums_df))
    start_scrapy()
    return

    sample_df = pd.read_csv("data/museums/mip_data_sample_2020_01.tsv", sep='\t')
    print("sample_df", len(sample_df))

    for i in sample_df.index:
        row = sample_df.loc[i]
        scrape_website(row.mm_id, row.mm_name, row.website)

    
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
    
    custom_settings = {'DOWNLOAD_DELAY': 1}
    #start_urls = 
    name = 'museum_website_scraper'
    
    #def start_requests(self):
    #    logger.debug("Museum id:"+self.muse_id)
    #    yield scrapy.Request(url) #, headers=headers, params=params,callback = self.parse)
    
    def parse(self,response):
        # write result in DB
        print(response.body)


def start_scrapy():
    process = CrawlerProcess()
    process.crawl(WebsiteSpider, muse_id='test', start_urls=['https://www.bbc.co.uk/'])
    process.start()