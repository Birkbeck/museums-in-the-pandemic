# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import unittest
import urllib
#import twint
import pandas as pd
import requests
import json
from scrapers.scraper_twitter import scrape_twitter_account, scrape_twitter_accounts
from scrapers.scraper_websites import *
from analytics.an_websites import *
from museums import get_fb_tw_links
import datetime
#import nest_asyncio


class TestVPN(unittest.TestCase):
    def setUp(self):        
        self.somevar = 0

    def test_1(self):
        assert 1 + 1 == 2

    def test_2(self):
        assert 1 + 1 == 2

    def test_vpn(self):
        from vpn import vpn_random_region
        print("testing vpn")
        cmd = ""
        vpn_random_region()

class TestFacebookScraper(unittest.TestCase):
    def setUp(self):        
        self.somevar = 0
    
    def test_scrape_facebook(self):
        i = 0


class TestTwitterScraper(unittest.TestCase):
    def setUp(self):        
        self.somevar = 0
    
    def test_scrape_twitter(self):
        scrape_twitter_accounts(None)

    
class TestWebsiteScraper(unittest.TestCase):
    def setUp(self):        
        self.somevar = 0

    def test_delta_calculation(self):
        db_conn = connect_to_postgresql_db()
        session_id = '20210323'
        tab = get_previous_session_table(session_id, db_conn)
        old_text = """Things to Do What’s On Where to stay Conferences & Exhibitions Getting Around Our Story News Suppliers Contact Us Get Listed Search Things to Do What’s On Where to stay Conferences & Exhibitions Getting Around Our Story News Suppliers Contact Us Get Listed Bed & Breakfast Home
                    /
                    Where to Stay
                    /
                    """
        for url in ["https://www.destinationmiltonkeynes.co.uk/where-to-stay/bed-breakfast/",
            "https://www.broughtonhouse.com/"]:
            prev_text = get_previous_version_of_page_text(url, tab, db_conn)
            print(prev_text)
            diffs = diff_texts(prev_text, old_text)
            diffs_json = json.dumps(diffs)

        aa = ['thesame','this is a test\npiach\nnonpiach','this is another test\ntestino testaccio','this is another test\n   testino testaccio\n  s s travell  ']
        bb = ['thesame','this is a not test\npiach\nnonpiach','this is another test\n  s travell  \n   testino testaccio','this is new text','this is old text']
        for a in aa:
            for b in bb:
                print("="*50)
                print(a)
                print("-"*30)
                print(b)
                print("-"*30)
                diffs = diff_texts(a,b)
                print(diffs)

        pass


    def test_get_fb_tw_links(self):
        return
        get_fb_tw_links()

    def test_problematic_sites(self):
        return
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"
        #user_agent = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36"

        for url in ['https://museumofcarpet.org','https://northhertsmuseum.org/','https://waterandsteam.org.uk/']:
            print("test:",url)
            req = urllib.request.Request(url, headers={'User-Agent': user_agent})
            response = urllib.request.urlopen(req)
            new_url = response.geturl()
            print(new_url)