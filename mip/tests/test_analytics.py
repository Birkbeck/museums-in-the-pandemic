# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import unittest
import urllib
from db.db import open_sqlite, connect_to_postgresql_db, is_postgresql_db_accessible
from analytics.an_websites import *
from analytics.text_models import *
from museums import *

class TestTextExtraction(unittest.TestCase):
    def setUp(self):        
        self.db_con = open_sqlite('data/test_data/websites_sample_2020-02-01.db')
        self.out_db_con = open_sqlite('tmp/websites_sample_textattr.db')


    def test_create_attr_db(self):
        create_webpage_attribute_table(self.out_db_con)
        extract_text_from_websites(self.db_con, self.out_db_con)


    def test_2(self):
        assert 1 + 1 == 2


class TestTextModel(unittest.TestCase):

    def setUp(self):        
        i = 0 

    def test_linguistic_model(self):
        i = 0
        setup_ling_model()

class TestVal(unittest.TestCase):
    generate_combined_dataframe()
    
    
class TestCentralDB(unittest.TestCase):
    def setUp(self):        
        i = 0

    def test_db_connection(self):
        print(is_postgresql_db_accessible())
        connect_to_postgresql_db()

    def test_load_manual_links(self):
        #load_manual_museum_urls()
        generate_stratified_museum_sample()
        