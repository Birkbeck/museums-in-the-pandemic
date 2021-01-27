# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import unittest
import urllib

""" 
Unit Tests for all classes
"""

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

        for url in ['https://museumofcarpet.org','https://waterandsteam.org.uk/']:
            req = urllib.request.Request(url, headers={'User-Agent': user_agent})
            response = urllib.request.urlopen(req) #, timeout=5
            new_url = response.geturl()
            print(new_url)