# -*- coding: utf-8 -*-
#
# Web scraper

import time
import uuid
import json
import os
import random
import calendar
import datetime
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from db.db import *

def scrape_websites(museums_df):
    """ """
    print("scrape_websites", len(museums_df))

    sample_df = pd.read_csv("data/museums/mip_data_sample_2020_01.tsv", sep='\t')
    print("sample_df", len(sample_df))

    for i in sample_df.index:
        row = sample_df.loc[i]
        scrape_website(row.mm_id,row.mm_name,row.website)

    
def scrape_website(muse_id, muse_name, url):
    assert muse_id
    assert muse_name
    assert url
    print("scrape_website",url)