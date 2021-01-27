# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import unittest

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

