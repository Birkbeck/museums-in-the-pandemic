# -*- coding: utf-8 -*-

"""
functions to handle museum data
"""

import pandas as pd

def load_museums_df_complete():
    """ Load and combine master list of museums for scraping and analysis """
    # TODO
    print("load_museums_df_complete")

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
        # TODO


def load_extracted_museums():
    """ TODO: document """
    df = pd.read_csv('data/google_results/museum_searches-2021-02-05.tsv', sep='\t')
    
    comparedf = pd.read_csv('data/websites_to_flag.tsv', sep='\t')
    
    urldict={}
    
    dfaccurate=pd.DataFrame(columns=["url","search", "muse_id", "location"])
    dfcheck=pd.DataFrame(columns=["google_rank","url","search", "muse_id", "location"])
    
    dfaccurate=pd.DataFrame(columns=["url","search", "muse_id", "location"])
    addedtocheck = False
    for item in df.iterrows():
        
        urlstring = item[1].url.split("/")[2]
        if item[1].google_rank ==1:
            #print(item)
            if comparedf['website'].str.contains(urlstring).any():
                

                list1=[item[1].google_rank, item[1].url, item[1].search, item[1].muse_id, item[1].location]
                dftoadd=pd.DataFrame([list1],columns=["google_rank","url","search", "muse_id", "location"])
                dfcheck=dfcheck.append(dftoadd)
                addedtocheck=True
            else:
                list1=[item[1].url, item[1].search, item[1].muse_id, item[1].location]
                dftoadd=pd.DataFrame([list1],columns=["url","search", "muse_id", "location"])
                dfaccurate=dfaccurate.append(dftoadd)
                addedtocheck=False  
        else:
            if addedtocheck == True and (item[1].google_rank ==2 or item[1].google_rank ==3):
                list1=[item[1].google_rank, item[1].url, item[1].search, item[1].muse_id, item[1].location]
                dftoadd=pd.DataFrame([list1],columns=["google_rank","url","search", "muse_id", "location"])
                dfcheck=dfcheck.append(dftoadd)

    dfaccurate.to_excel("tmp/accurate_results_view.xlsx", index=False)  
    dfcheck.to_excel("tmp/tocheck_results_view.xlsx", index=False)
    return None


def exclude_closed(df):
    """ TODO: document """
    df=df[df.year_closed == '9999:9999']
    assert len(df) > 0
    return df