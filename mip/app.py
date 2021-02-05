# -*- coding: utf-8 -*-

# MIP project

# %% Setup
#To Import twint from the mip_v1 environemnt terminal enter the following commands in the same precise order:
#pip install twint
#pip3 install --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint
#pip3 install --upgrade aiohttp_socks


import logging
logger = logging.getLogger(__name__)

import os
import logging
import sys
import datetime
import pandas as pd
from utils import StopWatch

from scrapers.scraper_google_selenium import *
from scrapers.scraper_twitter import scrape_twitter_account
from scrapers.scraper_websites import scrape_websites
from analytics.an_websites import analyse_museum_websites

from tests.run_tests import get_all_tests
import unittest

COMMANDS = ["help","tests","scrape_google","extract_google",'scrape_twitter','scrape_websites','an_websites','extract_google_results']
cmd = None

# %% Operations

def init_app():
    print("Init App")
    # set up folders
    folders = ['tmp','tmp/logs','tmp/analytics']
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)

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


def load_input_museums():
    df = pd.read_csv('data/museums/museum_names_and_postcodes-2020-01-26.tsv', sep='\t')
    df = exclude_closed(df)
    if(pd.Series(df["Museum_Name"]).is_unique):
        print("All museum names unique.")
    else:
        raise ValueError("Duplicate museum names exist.")
    print("loaded museums:",len(df))
    return df


def load_extracted_museums():
    df = pd.read_csv('tmp/google_extracted_results.tsv', sep='\t')
    df = exclude_closed(df)
    print(df.columns.values)
    array=['3','2','1']
    df=df[df.google_rank<4]
    df.to_csv('tmp/google_extracted_results_top3.tsv', index=None, sep='\t')
    return None


def exclude_closed(df):
    df=df[df.year_closed == '9999:9999']
    print(df)
    return df

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
        
        if cmd == "extract_google":
            print("extract_google")            
            df = load_input_museums()
            extract_google_results(df)
            load_extracted_museums()
        
        if cmd == "scrape_twitter":
            print("scrape_twitter")
            df = load_input_museums()
            scrape_twitter(df)

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
    