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
    assert df["id"].is_unique
    if df["Museum_Name"].is_unique:
        print("All museum names unique.")
    else:
        raise ValueError("Duplicate museum names exist.")
    print("loaded museums:",len(df))
    return df


def load_input_museums_wattributes():
    """  """
    df = pd.read_csv('data/museums/museums_wattributes-2020-02-23.tsv', sep='\t')
    print(df.columns)
    # remove closed museums
    df = df[df.closing_date.str.lower() == 'still open']
    assert len(df) > 0 
    #print(df[~df.duplicated(subset=['muse_id'])])
    print("loaded museums w attributes:",len(df))
    assert df["muse_id"].is_unique
    return df


def get_fb_tw_links():
    """ Extract FB and TW links from Google results """
    df = pd.read_csv('data/google_results/museum_searches_all-2021-02-16.tsv', sep='\t')
    df = df[df.search_type == 'website']
    print(df.columns)
    stats_df = pd.DataFrame()
    res_df = pd.DataFrame()
    for muse_id, muse_df in df.groupby('muse_id'):
        tw_df = muse_df[muse_df['domain'].str.contains('twitter')]
        fb_df = muse_df[muse_df['domain'].str.contains('facebook')]
        muse_name = muse_df['Museum_Name'].tolist()[0]
        assert muse_name
        tw_urls = tw_df['url'].tolist()
        fb_urls = fb_df['url'].tolist()
        row = pd.DataFrame({'muse_id':muse_id, 'links_n':len(muse_df), 'twitter_link_n':len(tw_df), 'facebook_link_n':len(fb_df)}, index=[muse_id])
        stats_df = stats_df.append(row)
        # save fb/tw links
        row = pd.DataFrame({'muse_id':muse_id, 'muse_name': muse_name, 'top_facebook': None, 'top_twitter': None}, index=[muse_id])
        if len(tw_urls)>0: row['top_twitter'] = tw_urls[0]
        if len(fb_urls)>0: row['top_facebook'] = fb_urls[0]
        res_df = res_df.append(row)
    print(res_df.describe())
    print(res_df.sum())
    print(res_df.isnull().sum(axis = 0))
    res_df.to_csv('tmp/google_tw_fb_links_df.tsv', index=False, sep='\t')


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


def load_manual_museum_urls():
    """ Load manually selected URLs for difficult museums """
    fn = 'data/museums/manual_google_url_results_top10_2021-02-19.xlsx'
    df = pd.read_excel(fn)
    df.muse_id
    print(df.columns, len(df))
    print("museum IDs:",len(df.muse_id.value_counts()))
    manual_sites_df = df[df.correct_site.notnull()][['muse_id','correct_site']]
    manual_sites_df = manual_sites_df[manual_sites_df.correct_site.str.contains('http')]
    manual_sites_df.rename(columns={"correct_site":"url"})
    
    # get valid websites
    valid_websites_df = df[df.valid=='T'][['muse_id','url']]
    # concat results
    valid_websites_df = pd.concat([valid_websites_df, manual_sites_df], axis=0)
    print("valid_websites_df", len(valid_websites_df))
    assert len(valid_websites_df) > 100
    return valid_websites_df


def generate_stratified_museum_sample():
    print("generate_stratified_museum_sample")
    df = load_input_museums()
    print(df.columns)
    df = load_input_museums_wattributes()
    
    manual_museums_df = load_manual_museum_urls()

    print(manual_museums_df.columns)
    df = df[~df.id.isin(manual_museums_df.muse_id)]
    print(len(df))
    # generate sample
    # TODO load mus attributes


def generate_string_pool_from_museum_name(mname):
    """ @returns variants of strings for fuzzy match on museum names """
    assert len(mname)>2
    pool = []
    # TODO: generate rules


def fuzzy_string_match(a, b):
    """ @returns a similarity score based on the extent to which a is found in b"""
    assert len(a) > 0
    assert len(b) > 0
    score = None
    # TODO: use this function here 
    # https://towardsdatascience.com/fuzzy-string-matching-in-python-68f240d910fe


    return score


def match_museum_name_with_string(mname, str_from_url):
    """@returns max similarity score between variants of mname and str_from_url)"""
    pool = generate_string_pool_from_museum_name(mname)
    scores = []
    for name_variant in pool:
        score = fuzzy_string_match(name_variant, str_from_url)
        if score is not None:
            scores.append(score)
    max_score = max(scores)
    return max_score