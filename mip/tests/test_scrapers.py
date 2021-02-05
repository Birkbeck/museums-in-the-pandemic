# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import unittest
import urllib
import twint
from scrapers.scraper_twitter import scrape_twitter_account
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


class TestTwitterScraper(unittest.TestCase):
    def setUp(self):        
        self.somevar = 0
    
    def test_scrape_twitter(self):
        #nest_asyncio.apply()
        
        scrape_twitter_account("https://twitter.com/adlingtonhall/status/745626936714661890&amp;ved=2ahUKEwi4v8aQu77uAhUQCawKHaS_Bu8QFjARegQIPBAC", "mm001.New", "12")
        #t = twint.Config()
        #t.Search = "from:@adlingtonhall"
        #t.Store_object = True
        #t.Limit = 1000 
        #twint.run.Search(t)
        #tlist = t.search_tweet_list
        #for item in tlist:
            #print(item)

class TestWebsiteScraper(unittest.TestCase):
    def setUp(self):        
        self.somevar = 0

    def test_1(self):
        assert 1 + 1 == 2

    def test_2(self):
        assert 1 + 1 == 2

    def test_problematic_sites(self):
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"
        #user_agent = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36"

        for url in ['https://museumofcarpet.org','https://northhertsmuseum.org/','https://waterandsteam.org.uk/']:
            print("test:",url)
            req = urllib.request.Request(url, headers={'User-Agent': user_agent})
            response = urllib.request.urlopen(req)
            new_url = response.geturl()
            print(new_url)