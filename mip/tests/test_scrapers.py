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

    def test_1(self):
        assert 1 + 1 == 2

    def test_2(self):
        assert 1 + 1 == 2

    def test_get_fb_tw_links(self):
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