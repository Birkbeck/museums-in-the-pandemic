# -*- coding: utf-8 -*-

"""
Analyse scraped websites
"""

from db.db import open_sqlite, run_select_sql
import pandas as pd
import logging
logger = logging.getLogger(__name__)

def get_scraping_sessions(db):
    df = run_select_sql("select session_id, count(*) as page_n from web_pages_dump group by session_id; ", db)
    return df

def get_scraping_sessions_by_museum(db):
    df = run_select_sql("select session_id, muse_id, count(*) as page_n, sum(page_content_length) as data_size from web_pages_dump group by session_id, muse_id; ", db)
    return df

def extract_text_from_websites(museums_df):

    # input data (museum sample)
    sample_df = pd.read_csv("data/museums/mip_data_sample_2020_01.tsv", sep='\t')

    in_db = 'tmp/websites-sample-2020-01-26.db'
    logger.info("extract_text_from_websites: "+in_db)
    db = open_sqlite(in_db)
    print(get_scraping_sessions(db))

    stats_df = get_scraping_sessions_by_museum(db)
    df = sample_df.merge(stats_df, how='left', left_on='mm_id', right_on='muse_id')
    df.to_excel('tmp/analytics/websites-sample-stats.xlsx')
    i = 0
