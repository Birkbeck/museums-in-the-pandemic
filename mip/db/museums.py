# -*- coding: utf-8 -*-

"""
Museum data
"""

import pandas as pd


def load_input_museums():
    df = pd.read_csv('data/museums/museum_names_and_postcodes-2020-01-26.tsv', sep='\t')
    df = exclude_closed(df)
    if(pd.Series(df["Museum_Name"]).is_unique):
        print("All museum names unique.")
    else:
        raise ValueError("Duplicate museum names exist.")
    print("loaded museums:",len(df))
    return df