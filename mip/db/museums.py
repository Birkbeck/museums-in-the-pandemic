# -*- coding: utf-8 -*-

"""
Museum data
"""

import pandas as pd


def load_input_museums():
    """ Load MM museum data that includes ALL museums """
    df = pd.read_csv('data/museums/museum_names_and_postcodes-2020-01-26.tsv', sep='\t')
    df = exclude_closed(df)
    if(pd.Series(df["Museum_Name"]).is_unique):
        print("All museum names unique.")
    else:
        raise ValueError("Duplicate museum names exist.")
    print("loaded museums:",len(df))
    return df


def get_fb_tw_links():
    """ Extract FB and TW links from Google results """
    df = pd.read_csv('data/google_results/museum_searches-2021-02-05.tsv', sep='\t')
    print(df.columns)
    grouped = df.groupby('muse_id')
    for muse_id, muse_df in grouped:
        tw_df = muse_df[muse_df['domain'].str.match('twitter')]
        fb_df = muse_df[muse_df['domain'].str.match('facebook')]
        print(muse_id, len(muse_df), len(tw_df), len(fb_df))