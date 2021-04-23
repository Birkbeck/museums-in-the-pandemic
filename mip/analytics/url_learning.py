# -*- coding: utf-8 -*-

"""
Linguistic models for text analytics


"""

import logging
logger = logging.getLogger(__name__)

import unittest
import urllib
from db.db import open_sqlite, connect_to_postgresql_db, is_postgresql_db_accessible, make_string_sql_safe
from analytics.an_websites import *
from analytics.text_models import *
from museums import *
import pandas as pd
from scrapers.scraper_websites import check_for_url_redirection

def assign_urls_to_sample():
        stratdf = pd.read_csv('data/museums/museums_wattributes-2020-02-23.tsv', sep='\t')
        googledf=pd.read_csv('data/google_results/results_source_files/google_extracted_results_reg.tsv.gz', sep='\t')
        facebookdf=pd.read_csv('data/google_results/results_source_files/google_extracted_results_facebook.tsv.gz', sep='\t')
        twitterdf=pd.read_csv('data/google_results/results_source_files/google_extracted_results_twitter.tsv.gz', sep='\t')
        sampledf=load_museum_samples()
        #sampledf.drop(['search'], axis=1)

        museweight = generate_weighted_museum_names()
        stratdf=stratdf.filter(['muse_id','musname','size', 'governance', 'town'], axis=1)
        googledf=googledf.filter(['muse_id','search_type', 'url','google_rank'], axis=1)
        facebookdf=facebookdf.filter(['muse_id','search_type', 'url', 'google_rank'], axis=1)
        twitterdf=twitterdf.filter(['muse_id','search_type', 'url', 'google_rank'], axis=1)

        print(len(googledf))
        outputdfg = pd.merge(stratdf,googledf,on='muse_id')
        print(len(googledf))
        outputdff = pd.merge(stratdf,facebookdf,on='muse_id')
        outputdft = pd.merge(stratdf,twitterdf,on='muse_id')
        outputdf=outputdfg.append(outputdff)
        outputdf=outputdf.append(outputdft)

        outputdf['url_size']=outputdf['url'].str.len() 
        outputdf['N_slash']=outputdf['url'].str.split('/')
        outputdf['N_slash']=outputdf['N_slash'].apply(lambda x: len(x))
        
        
        
        
        outputdf['has_visit']=outputdf.apply(lambda row: (hasvisit(row['url'],row['search_type'])), axis=1)
        outputdf['has_museum']=outputdf.apply(lambda row: (hasmuseum(row['url'],row['search_type'])), axis=1)
        outputdf['has_location']=outputdf.apply(lambda row: (haslocation(row['url'],row['town'])), axis=1)
        
        outputdf['fuzzy_score_full_url']=outputdf.apply(lambda row: (generate_weighted_fuzzy_scores(get_musname_pool(row['musname'], row['town']), row['url'], museweight, "", False, row['search_type'])), axis=1)
        outputdf['fuzzy_score_domain']=outputdf.apply(lambda row: (generate_weighted_fuzzy_scores(get_musname_pool(row['musname'], row['town']), row['url'], museweight, "", True, row['search_type'])), axis=1)
        outputdf['fuzzy_score_domain_inverse']=outputdf.apply(lambda row: (get_fuzzy_string_score(get_url_domain_with_search(row['url'], row['search_type']),row['musname'] )), axis=1)
        outputdf['withloc_fuzzy_score_full_url']=outputdf.apply(lambda row: (generate_weighted_fuzzy_scores(get_musname_pool(row['musname'], row['town']), row['url'], museweight, row['town'], False, row['search_type'])), axis=1)
        outputdf['withloc_fuzzy_score_domain']=outputdf.apply(lambda row: (generate_weighted_fuzzy_scores(get_musname_pool(row['musname'], row['town']), row['url'], museweight, row['town'], True, row['search_type'])), axis=1)
        
        outputdf['exact_score_url']=outputdf.apply(lambda row: (get_exact_match(row['musname'], row['url'])), axis=1)
        outputdf['exact_score_url_inverse']=outputdf.apply(lambda row: (get_exact_match(row['url'],row['musname'])), axis=1)
        outputdf['exact_score_domain']=outputdf.apply(lambda row: (get_exact_match(row['musname'], get_url_domain_with_search(row['url'], row['search_type']))), axis=1)
        outputdf['exact_score_domain_inverse']=outputdf.apply(lambda row: (get_exact_match(get_url_domain_with_search(row['url'], row['search_type']),row['musname'])), axis=1)
        print(len(outputdf))
        outputdf = pd.merge(outputdf,sampledf,on=['muse_id', 'search_type'],how='left')
        print(len(outputdf))
        #TO DO:
        
        
        
        #pickle
        #auto machine learning package (investigate
        # #look up creating test and validation(given we have a small sample))
        #check sample of 10 pages in andreas sample table
        #make url_learning.py under analytics
        #merge_dataset->museum url attribute table
        #random forest (sklearn) //get tutorial //
        #score=[]
        #for item in outputdf.iterrows():

            #score.append(generate_weighted_fuzzy_scores(item[1].musname, item[1].url, museweight, item[1].town))
        outputdf=outputdf[outputdf.google_rank<21]
        outputdf=outputdf.sort_values(['muse_id','search_type','google_rank'], ascending=[True,True,True])
        outputdf.to_excel("tmp/combined_museum_matrix.xlsx", index=False)
        outputdf.to_csv('tmp/combined_museum_matrix.tsv', index=False, sep='\t')
        outputdf.to_pickle("./combined_museum_matrix.pkl")
def repair_valid_column(valid):
    if valid=='T':
        return True
    else:
        return False

