# -*- coding: utf-8 -*-

# MIP project

# %% Setup
import logging
logger = logging.getLogger(__name__)

import os
import logging
import sys
import datetime
import pandas as pd
import openpyxl
from utils import StopWatch
from scrapers.scraper_google_selenium import scrape_google_museum_names
from scrapers.scraper_google_selenium import extract_google_results
from scrapers.scraper_twitter import scrape_twitter_account
from scrapers.scraper_websites import scrape_websites
from scrapers.scraper_facebook import scrape_facebook
from analytics.an_websites import analyse_museum_websites
from museums import load_input_museums, load_extracted_museums, combinedatasets, get_fuzzy_string_match_ranks, load_fuzzy_museums
from db.db import is_postgresql_db_accessible
from tests.run_tests import get_all_tests
import unittest

COMMANDS = ["help","tests","scrape_google","extract_google",'scrape_twitter','scrape_websites','an_websites', 'scrape_facebook', 'fuzzy_string_match', 'combine_datasets']
cmd = None

# %% Operations

def init_app():
    print("Init App\n")
    # set up folders
    folders = ['tmp','tmp/logs','tmp/analytics']
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)

    # check if config files exist
    assert os.path.exists('.secrets.json')

    # check if DB is accessible
    print("is_postgresql_db_accessible =",is_postgresql_db_accessible())

    # enable logger
    logging.basicConfig(
        filename=get_log_file_name(),
        format='%(asctime)s:%(levelname)s:%(filename)s:\t%(message)s',
        level=logging.DEBUG
    )

def get_log_file_name():
    fn = datetime.datetime.now().strftime('%Y%m%d')
    fn = 'tmp/logs/log-' + fn + '.txt'
    return fn
    
def cleanup():
    logger.info("Cleanup")

# %% Main
def main():
    sw = StopWatch("app")
    init_app()
    logger.info("\n"*3)
    logger.info("== MIP App ==")
    logger.info("Parameters: " + str(sys.argv[1:]))
    logger.info("N CPUs =" + str(os.cpu_count()))

    # check input
    cmd_list = sys.argv[1:]
    if len(sys.argv) <= 1:
            raise RuntimeError("No parameter. Valid parameters: " + str(COMMANDS))
    for cmd in cmd_list:
        if len(sys.argv) < 2 or not sys.argv[1] in COMMANDS:
            raise RuntimeError("Invalid parameter. Valid parameters: " + str(COMMANDS))
    
    for cmd in cmd_list:
        if cmd == "scrape_google":
            print("scrape_google")
            df = load_input_museums()
            scrape_google_museum_names(df)

        if cmd=="combine_datasets":
            combinedatasets()  

        if cmd=="fuzzy_string_match":
            df = load_fuzzy_museums()
            get_fuzzy_string_match_ranks(df)

        if cmd == "extract_google":
            print("extract_google")                      
            df = load_input_museums()
            extract_google_results(df)
            #load_extracted_museums()
        
        if cmd == "scrape_twitter":
            print("scrape_twitter")
            df = load_input_museums()
            scrape_twitter(df)

        if cmd == "scrape_facebook":
            print("scrape_facebook")
            df = load_input_museums()
            scrape_facebook(df)

        if cmd == "scrape_websites":
            print("scrape_websites")
            df = load_input_museums()
            scrape_websites(df)

        if cmd == "an_websites":
            print("an_websites")
            df = load_input_museums()
            analyse_museum_websites(df)

        if cmd == "add_mus_ids":
            print("add_mus_ids")
            df = load_input_museums()
            add_museum_ids(df)

        if cmd == "tests":
            logger.info("tests")
            unittest.TextTestRunner().run(get_all_tests())

    cleanup()
    logger.info(sw.tick("OK"))
    logger.info("OK")

if __name__ == '__main__':
    main()
    