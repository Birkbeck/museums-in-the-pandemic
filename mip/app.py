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
from scrapers.scraper_google_selenium import scrape_google_museum_names, extract_google_results
from scrapers.scraper_websites import scrape_websites
from scrapers.scraper_twitter import scrape_twitter_accounts
from scrapers.scraper_facebook import scrape_facebook
from analytics.an_websites import analyse_museum_websites, website_size_analysis
from analytics.text_models import analyse_museum_text, make_text_corpus, make_social_media_corpus, make_corpus_sqlite
from museums import load_input_museums, load_input_museums_wattributes, load_extracted_museums, \
    generate_stratified_museum_sample,generate_stratified_museum_urls, combinedatasets, \
    get_fuzzy_string_match_scores, load_fuzzy_museums, compare_result_to_sample, get_museums_w_web_urls, \
    get_twitter_facebook_links
from db.db import is_postgresql_db_accessible, count_all_db_rows

from tests.run_tests import get_all_tests
import unittest

COMMANDS = ["tests","scrape_google","extract_google",'scrape_twitter','scrape_websites','ex_txt_fields',
    'scrape_facebook','compare_sample','db_stats','an_text','corpus']
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
            raise RuntimeError("No parameter. Valid parameters: " + str(sorted(COMMANDS)))
    for cmd in cmd_list:
        if len(sys.argv) < 2 or not sys.argv[1] in COMMANDS:
            raise RuntimeError("Invalid parameter. Valid parameters: " + str(sorted(COMMANDS)))
    
    for cmd in cmd_list:
        if cmd == "scrape_google":
            print("scrape_google")
            df = load_input_museums()
            scrape_google_museum_names(df)

        if cmd == "extract_google":
            print("extract_google")                      
            df = load_input_museums()
            extract_google_results(df)
            load_extracted_museums()
        
        if cmd == "scrape_twitter":
            print("scrape_twitter")
            df = get_twitter_facebook_links()
            scrape_twitter_accounts(df)

        if cmd == "scrape_facebook":
            print("scrape_facebook")
            df = get_twitter_facebook_links()
            scrape_facebook(df)

        if cmd == "scrape_websites":
            logger.info("scrape_websites")
            assert is_postgresql_db_accessible()
            scrape_websites()

        if cmd == "ex_txt_fields":
            print("ex_txt_fields")
            assert is_postgresql_db_accessible()
            analyse_museum_websites()

        if cmd == "an_text":
            print("an_text")
            # RUN in terminal:
            ## python -m spacy download en_core_web_sm
            # python -m spacy download en_core_web_lg
            assert is_postgresql_db_accessible()
            analyse_museum_text()

        if cmd == "corpus":
            print("corpus")
            # RUN in terminal:
            assert is_postgresql_db_accessible()
            #make_text_corpus()
            #make_social_media_corpus()
            #make_corpus_sqlite()
            website_size_analysis()

        if cmd == "compare_sample":
            print("compare_sample")
            search="twitter"
            percentage=compare_result_to_sample(search)
            print(percentage)
            
        if cmd == "tests":
            logger.info("tests")
            unittest.TextTestRunner().run(get_all_tests())

        if cmd == "db_stats":
            logger.info("db_stats")
            assert is_postgresql_db_accessible()
            count_all_db_rows()

    cleanup()
    logger.info(sw.tick("OK"))
    logger.info("OK")
    print('OK')

if __name__ == '__main__':
    main()
    