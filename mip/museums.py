# -*- coding: utf-8 -*-

"""
functions to handle museum data
"""

import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def load_museums_df_complete():
    """ Load and combine master list of museums for scraping and analysis """
    # TODO
    print("load_museums_df_complete")


def load_input_museums():
    """ Load MM museum data that includes ALL museums """
    fn = 'data/museums/museum_names_and_postcodes-2020-01-26.tsv'
    df = pd.read_csv(fn, sep='\t')
    df = exclude_closed(df)
    assert df["id"].is_unique
    if not df["Museum_Name"].is_unique:
        raise ValueError("Duplicate museum names exist.")
    print("loaded museums:",len(df), fn)
    return df

def load_fuzzy_museums():
    """ Load MM museum data that includes ALL museums """
    df = pd.read_csv('data/google_results/google_extracted_results_twitter_noloc.tsv', sep='\t')
    
    #df = exclude_closed(df)
    print("loaded museums:",len(df))
    return df


def load_input_museums_wattributes():
    """  """
    fn = 'data/museums/museums_wattributes-2020-02-23.tsv'
    df = pd.read_csv(fn, sep='\t')
    print(df.columns)
    # remove closed museums
    df = df[df.closing_date.str.lower() == 'still open']
    assert len(df) > 0 
    print("loaded museums w attributes (open):",len(df), fn)
    assert df["muse_id"].is_unique
    assert df["musname"].is_unique
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
    df = pd.read_csv('data/google_results/museum_searches_all-2021-02-18.tsv', sep='\t')
    dfcheck1=pd.DataFrame(columns=["google_rank","url","search", "muse_id"])
    museidlist=["mm.fcm.186","mm.domus.YH153","mm.domus.SE481","mm.wiki.220","mm.aim82NM.130","mm.ace.1101","mm.musa.285","mm.mald.111","mm.domus.NW191","mm.fcm.130","mm.ace.1163","mm.domus.SE270","mm.domus.EM031","mm.ace.1258","mm.misc.015","mm.musa.295","mm.New.26","mm.ace.685","mm.aim82NM.012","mm.domus.SC280","mm.domus.SC314","mm.domus.WA017","mm.musa.345","mm.domus.SC074","mm.hha.119","mm.domus.NW031","mm.domus.YH123","mm.domus.SC096","mm.musa.349","mm.wiki.105"]
    for item in df.iterrows():
        if item[1].muse_id in museidlist:
            list1=[item[1].google_rank, item[1].url, item[1].search, item[1].muse_id]
            dftoadd=pd.DataFrame([list1],columns=["google_rank","url","search", "muse_id"])
            dfcheck1=dfcheck1.append(dftoadd)
    dfcheck1.to_excel("tmp/tocheck_sample.xlsx", index=False)

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
            if addedtocheck == True and (item[1].google_rank >1 and item[1].google_rank <11):
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
    fn = 'data/samples/manual_google_url_results_top10_2021-02-19.xlsx'
    fn2 = 'data/samples/sample_museum_search_with_loc'
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


def generate_derived_attributes_muse_df(df):
    print("generate_derived_attributes_muse_df")

    def get_region(x):
        x = x.replace("/England",'')
        r = x.split('/')
        reg = r[1]
        assert reg
        return reg

    def get_gov(x):
        s = x.split(":")
        assert s[0]
        return s[0]

    df['region'] = df['admin_area'].apply(get_region)
    df['gov'] = df['governance'].apply(get_gov)
    #print(df['gov'].value_counts())
    #print(df.sample(10))
    return df


def generate_stratified_museum_sample():
    """
    How to calculate SE/CI for this sample size: 
     http://sample-size.net/confidence-interval-proportion
    """
    print("generate_stratified_museum_sample")

    df1 = load_input_museums()
    print(df1.columns)
    df2 = load_input_museums_wattributes()
    print(df2.columns)
    print("difference between datasets:", set(df1.id).symmetric_difference(set(df2.muse_id)))
    # only select museums present in the initial dataset
    #df2 = df2[df2.muse_id.str.isin(df1.id.str)]

    # remove manual museums


    manual_museums_df = load_manual_museum_urls()
    print(manual_museums_df.columns)
    df = df2[~df2.muse_id.isin(manual_museums_df.muse_id)]
    df = generate_derived_attributes_muse_df(df)
    print("selected museums for sampling:", len(df))
    
    # generate sample
    fraction = .03
    sample_n = int(len(df) * fraction)
    print("sample_n", sample_n)
    cols = ["region","size","accreditation","gov"]
    sample_df = pd.DataFrame()
    for val, subdf in df.groupby(cols):
        sub_smpl_f = len(subdf) * fraction
        sub_smpl_n = int(round(sub_smpl_f,0))
        print(val, len(subdf), sub_smpl_f, sub_smpl_n)
        sample_df = sample_df.append(subdf.sample(sub_smpl_n, random_state=32))
    
    print("sample_df", len(sample_df))
    sample_df['valid'] = ''
    assert sample_df.muse_id.is_unique
    fout = 'tmp/museums_stratified_sample_{}.tsv'.format(len(sample_df))
    sample_df.to_csv(fout, sep="\t", index=False)
    print(fout)


def generate_string_pool_from_museum_name(mname):
    """ @returns variants of strings for fuzzy match on museum names """
    assert len(mname)>2
    pool = []
    joiningwords=["or", "the", "a", "for", "th"]
    pool.append(mname)
    pool.append(mname+" museum")
    
    mnamelist= mname.rstrip().split(" ")
    newphrase=""
    for word in mnamelist:
        
        if word not in joiningwords and word != "and" :
            newphrase = newphrase+word+" "
    pool.append(newphrase.rstrip())
    pool.append(newphrase+"museum")
    pool.append(newphrase.replace(' ',''))
    pool.append(newphrase.replace(' ','')+"museum")
    
    newphrase=""
    for word in mnamelist:
        
        if word not in joiningwords:
            if word == "and":
                newphrase = newphrase+"& "
            else:
                newphrase = newphrase+word+" "
    pool.append(newphrase.rstrip())
    pool.append(newphrase+"museum")
    pool.append(newphrase.replace(' ',''))
    pool.append(newphrase.replace(' ','')+"museum")
    
    newphrase=""
    for word in mnamelist:
        
        if word not in joiningwords and word != "and" :
            newphrase = newphrase+word[0]
    pool.append(newphrase)
    pool.append(newphrase+" museum")
    pool.append(newphrase+"museum")
    
    newphrase=""
    for word in mnamelist:
        
        if word not in joiningwords:
            if word == "and":
                newphrase = newphrase+"&"
            else:
                newphrase = newphrase+word[0]
    pool.append(newphrase)
    pool.append(newphrase+" museum")
    pool.append(newphrase+"museum")
    
    return pool


def fuzzy_string_match(a, b):
    """ @returns a similarity score based on the extent to which a is found in b"""
    assert len(a) > 0
    assert len(b) > 0
    
    ratio = fuzz.token_sort_ratio(a, b)
    return ratio
    # https://towardsdatascience.com/fuzzy-string-matching-in-python-68f240d910fe


    
def generate_combined_dataframe():
    print("generate_combined_dataframe")
    scrapetarget=[]
    searchtype=[]
    df_mus = pd.read_csv('data/google_results/results_source_files/google_extracted_results_reg.tsv.gz', sep='\t')
    for item in df_mus.iterrows():
        scrapetarget.append('web')
        searchtype.append('regular')
    df_mus['scrape_target']=scrapetarget
    df_mus['search_variety']=searchtype
    scrapetarget=[]
    searchtype=[]

    df_mus_exact=pd.read_csv('data/google_results/results_source_files/google_extracted_results_exact.tsv.gz', sep='\t')
    for item in df_mus_exact.iterrows():
        scrapetarget.append('web')
        searchtype.append('exact')
    df_mus_exact['scrape_target']=scrapetarget
    df_mus_exact['search_variety']=searchtype
    scrapetarget=[]
    searchtype=[]
    df_facebook=pd.read_csv('data/google_results/results_source_files/google_extracted_results_facebook.tsv.gz', sep='\t')
    for item in df_facebook.iterrows():
        scrapetarget.append('facebook')
        searchtype.append('location')
    df_facebook['scrape_target']=scrapetarget
    df_facebook['search_variety']=searchtype
    scrapetarget=[]
    searchtype=[]
    df_facebook_noloc=pd.read_csv('data/google_results/results_source_files/google_extracted_results_facebook_noloc.tsv.gz', sep='\t')
    for item in df_facebook_noloc.iterrows():
        scrapetarget.append('facebook')
        searchtype.append('regular')
    df_facebook_noloc['scrape_target']=scrapetarget
    df_facebook_noloc['search_variety']=searchtype
    scrapetarget=[]
    searchtype=[]
    df_twitter=pd.read_csv('data/google_results/results_source_files/google_extracted_results_twitter.tsv.gz', sep='\t')
    for item in df_twitter.iterrows():
        scrapetarget.append('twitter')
        searchtype.append('location')
    df_twitter['scrape_target']=scrapetarget
    df_twitter['search_variety']=searchtype
    scrapetarget=[]
    searchtype=[]
    df_twitter_noloc=pd.read_csv('data/google_results/results_source_files/google_extracted_results_twitter_noloc.tsv.gz', sep='\t')
    for item in df_twitter_noloc.iterrows():
        scrapetarget.append('twitter')
        searchtype.append('regular')
    df_twitter_noloc['scrape_target']=scrapetarget
    df_twitter_noloc['search_variety']=searchtype
    finaldf=df_mus.append(df_mus_exact)
    finaldf=finaldf.append(df_facebook)
    finaldf=finaldf.append(df_facebook_noloc)
    finaldf=finaldf.append(df_twitter)
    finaldf=finaldf.append(df_twitter_noloc)
    finaldf.to_csv('data/google_results/google_results_all_02_03_2021.tsv', index=False, sep='\t')



def match_museum_name_with_string(mname, str_from_url):
    """@returns max similarity score between variants of mname and str_from_url)"""
    pool = generate_string_pool_from_museum_name(mname)
    scores = []
    print(mname)
    print(str_from_url)
    for name_variant in pool:
        
        score = fuzzy_string_match(name_variant, str_from_url)
        if score is not None:
            scores.append(score)
    max_score = max(scores)
    return max_score

def get_fuzzy_string_match_scores(musdf):
    scorerow=[]
    for row in musdf.iterrows():
        urlstring=row[1].url.split("/")[3].lower()
        if(urlstring=='events'):
            urlstring=row[1].url.split("/")[4].lower()
        musename = row[1].Museum_Name.lower()
        if urlstring !='':
            scorerow.append(match_museum_name_with_string(musename, urlstring))
        else:
            scorerow.append(0)
    musdf['score']=scorerow
    finaldf=pd.DataFrame()
    for score, muse_df in musdf.groupby('muse_id'):
        newdf=muse_df.sort_values(by=['score'], ascending=False)
        finaldf=pd.concat([finaldf, newdf])
    finaldf.to_csv('tmp/fuzzy_museum_scores.tsv', index=False, sep='\t')
    return None


def combinedatasets():
    df1 = pd.read_csv('tmp/all_museum_id.tsv', sep='\t')
    df2 = pd.read_csv('tmp/all_museum_data.csv', sep=',')
    df3=pd.merge(df1, df2, on='musname')
    df3.to_csv('tmp/museums_wattributes-2020-02-23.tsv', index=False, sep='\t')
    return None


def load_all_google_results():
    df = pd.read_csv('data/google_results/google_results_all_01_03_2021.tsv.gz', sep='\t')
    print("load_all_google_results", len(df))
    print(df.describe())
    print(df.columns)
    print(df.search_type.value_counts())
    print(df.search_variety.value_counts())
    print(df.year_closed.value_counts())
    print(df.scrape_target.value_counts())
    return df
