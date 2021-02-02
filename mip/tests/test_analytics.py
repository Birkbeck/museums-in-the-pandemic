# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import unittest
import urllib
from db.db import open_sqlite
from analytics.an_websites import *


class TestTextExtraction(unittest.TestCase):
    def setUp(self):        
        self.db_con = open_sqlite('data/test_data/websites_sample_2020-02-01.db')
        self.out_db_con = open_sqlite('tmp/websites_sample_textattr.db')


    def test_create_attr_db(self):
        create_webpage_attribute_table(self.out_db_con)
        extract_text_from_websites(self.db_con, self.out_db_con)


    def test_2(self):
        assert 1 + 1 == 2

    