# -*- coding: utf-8 -*-

import logging
import twint
import json
import datetime
from vpn import vpn_random_region
from facebook_scraper import get_posts
logger = logging.getLogger(__name__)

"""
Twitter scraper
"""
def scrape_facebook(df):
    scrape_facebook_url("https://www.facebook.com/groups/InstantPotCommunity/", "mm001.New", "12", 500)
    return None

def scrape_facebook_url(url, muse_id, session_id, postnumber):
    print("scrape_facebook_url: " + url)
    urllist = url.split("/")
    print(urllist[3])
    if urllist[3]=='groups':
        lastpost=""
        
        for post in get_posts(group=urllist[4], pages=postnumber, extra_info=True):
            print(post)
            lastpost=post
        if lastpost=="":
            vpn_random_region()
            return scrape_facebook_url(url, muse_id, session_id, postnumber)

        d2 = datetime.datetime(2019, 1, 1)
        print(lastpost['time'])
        print(d2)
        print(lastpost['time']>d2)
        if lastpost['time']>d2:
            return scrape_facebook_url(url, muse_id, session_id, postnumber+500)
        else:
            print(lastpost)
        ##finalpost=json.dumps(lastpost)
        ##print(finalpost)
        ##print(finalpost[datetime.datetime])

    else:
        for post in get_posts(urllist[4], pages=1, extra_info=True):
            print(post)
            lastpost=post
        if lastpost=="":
            vpn_random_region()
            return scrape_facebook_url(url, muse_id, session_id, postnumber)

        d2 = datetime.datetime(2019, 1, 1)
        print(lastpost['time'])
        print(d2)
        print(lastpost['time']>d2)
        if lastpost['time']>d2:
            return scrape_facebook_url(url, muse_id, session_id, postnumber+500)
        else:
            print(lastpost)

    
    return tlist
