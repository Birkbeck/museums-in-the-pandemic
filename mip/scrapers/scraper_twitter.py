# -*- coding: utf-8 -*-

import logging
import twint
logger = logging.getLogger(__name__)

"""
Twitter scraper
"""

def scrape_twitter_account(url, muse_id, session_id):
    print("scrape_twitter_account: " + url)
    urllist = url.split("/")
    print(urllist[3])
    t = twint.Config()
    t.Search = "from:@"+urllist[3]
    t.Store_object = True
    t.Since = "2019-01-01" 
    twint.run.Search(t)
    tlist = t.search_tweet_list
    return tlist
