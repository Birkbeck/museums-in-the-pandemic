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
import unicodedata
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
            page_id, prev_text = get_previous_version_of_page_text(url, tab, db_conn)
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

    def test_unicode_issues(self):
        clean_unicode_issues_string("test 123")

        odd_str = "I interned in the fashion section of a magazine, and I took garment construction classes every chance I could.\nWhen I\xa0graduated, I found the postgraduate course at London College of Fashion, it seemed like the perfect next step for me to get to London and really focus on creativity.\nLuckily, I’d built up enough work in a portfolio by then, and I\xa0was accepted\xa0after a pretty tough interview. View this post on Instagram It's not always easy finding the perfect gift - if you have any questions about sizing feel free to get in touch #loungeinluxury A post shared by\nGilda & Pearl\n(@gildapearl)"
        odd_str2 = unicodedata.normalize("NFKD", odd_str)

        # str with surrogates
        odd_str = "When I used this craft I was paying a tribute to my country and the diversity of craft we have and the communities that make them, whilst representing them in a different way here and I would like to carry on with that idea. Showing that we can bring 200 year old century craft into the fashion industry. Harikrishnan‘s inflatable latex trousers brought “anatomically impossible“ proportions to the runway this week\ud83c\udf88Here‘s why inflatable fashion is blowing TF up:\nhttps://t.co/9k7O1z8SyS pic.twitter.com/V8xy0btNvz\n— Dazed (@Dazed)\nFebruary 26, 2020 Follow\nHari\non Instagram\nWatch the full\nMA20 catwalk show\nWhat’s on at LCF:\nopen days and events\nMore\nLCF Stories Related content Collections by MA Menswear students for LCFMA20. "
        odd_str2 = unicodedata.normalize("NFKD", odd_str)
        #s3 = odd_str2.decode('utf8','replace')
        
        s3 = odd_str2.encode(errors='surrogatepass').decode('utf8',errors='surrogatepass')
        s4 = odd_str2.encode('utf-8', "backslashreplace").decode('utf-8')
        #s3.encode()
        s4.encode()
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