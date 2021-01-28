# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import unittest
import urllib


class TestVPN(unittest.TestCase):
    def setUp(self):        
        self.somevar = 0

    def test_1(self):
        assert 1 + 1 == 2

    def test_2(self):
        assert 1 + 1 == 2

    def test_vpn(self):
        print("testing vpn")
        cmd = ""
        vpn_random_region()


class TestTwitterScraper(unittest.TestCase):
    def setUp(self):        
        self.somevar = 0

    def test_scrape_twitter(self):
        from scrapers.scraper_twitter import scrape_twitter_account

        urls = ["https://twitter.com/britishmuseum"]
        session_id = "test"
        muse_id = "fake_id"
        for u in urls:
            html = scrape_twitter_account(u, muse_id, session_id)


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