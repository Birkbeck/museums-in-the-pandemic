# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import unittest
import tests
        
def get_all_tests():
    test_classes = []
    
    # import all tests
    from tests.test_scrapers import TestWebsiteScraper, TestVPN, TestTwitterScraper
    from tests.test_analytics import TestTextExtraction, TestTextModel
    
    # ========================================================
    # add tests to run here
    # ========================================================
    #test_classes.append(TestVPN)
    test_classes.append(TestTwitterScraper)
    #test_classes.append(TestWebsiteScraper)
    #test_classes.append(TestVPN)
    #test_classes.append(TestTextExtraction)
    #test_classes.append(TestTextModel)
    
    
    # build suite
    suite = unittest.TestSuite()
    for tclass in test_classes:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(tclass))
    return suite