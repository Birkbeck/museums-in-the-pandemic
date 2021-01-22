# -*- coding: utf-8 -*-

# MIP project

# %% Setup

import os
import sys
import pandas as pd
from utils import StopWatch
from scrapers.scraper_google_selenium import *
from scrapers.scraper_twitter import scrape_twitter

COMMANDS = ["help","tests","scrape_google","extract_google",'scrape_twitter']
cmd = None

# %% Operations
def init_app():
    print("Init App")
    # set up folders
    folders = ['tmp']
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)
    
def cleanup():
    print("Cleanup")

def load_input_museums():
    df = pd.read_csv('data/museums/museum_names_and_postcodes-2020-01-13.csv')
    print("loaded museums:",len(df))
    return df

# %% Main
def main():
    sw = StopWatch("app")
    init_app()
    print("== MIP App ==")
    print("Parameters: " + str(sys.argv[1:]))
    print("N CPUs =", os.cpu_count())

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
            extract_google_results()
        
        if cmd == "scrape_twitter":
            print("scrape_twitter")
            df = load_input_museums()
            scrape_twitter(df)

        elif cmd == "tests":
            
            print("run tests")
            # TODO

    cleanup()
    sw.tick("OK")
    print("OK")

if __name__ == '__main__':
    main()
