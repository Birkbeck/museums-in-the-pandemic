# -*- coding: utf-8 -*-

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

class TestTextExtraction(unittest.TestCase):
    def setUp(self):        
        #self.db_con = open_sqlite('data/test_data/websites_sample_2020-02-01.db')
        #self.out_db_con = open_sqlite('tmp/websites_sample_textattr.db')
        pass

    def test_create_text_sample(self):

        s = make_string_sql_safe("urls with 's don't work")

        db_conn = connect_to_postgresql_db()
        
        table = get_scraping_session_tables(db_conn)[1]
        muse_ids = ('mm.domus.SE245','mm.domus.SW096','mm.misc.007')
        sql = "select * from {} where is_start_url and muse_id in ('{}','{}','{}');".format(table,muse_ids[0],muse_ids[1],muse_ids[2])
        df = pd.read_sql(sql, db_conn)
        assert len(df)>0
        
        out_table = create_webpage_attribute_table(table, db_conn)
        clear_attribute_table(out_table, db_conn)
        #clear_attribute_table(out_table, db_conn)
        # extract attributes
        for muse_id in muse_ids:
            extract_text_from_websites(table, out_table, db_conn, muse_id)

        sql = """select muse_id, url, is_start_url, a.* 
        from {} p, {} a 
        where p.page_id = a.page_id and p.is_start_url;""".format(table, out_table)
        attr_df = pd.read_sql(sql, db_conn)

        attr_df = join_museum_info(attr_df, 'muse_id')

        attr_df.to_excel('tmp/sample_museum_web_text-20210310.xlsx', index=False)
        pass


    def test_create_attr_db(self):
        pass
        #create_webpage_attribute_table(self.out_db_con)
        #extract_text_from_websites(self.db_con, self.out_db_con)


class TestTextModel(unittest.TestCase):

    def setUp(self):        
        i = 0 

    def test_linguistic_model(self):
        i = 0
        setup_ling_model()


class TestVal(unittest.TestCase):
    def setUp(self):        
        i = 0
    #def test_stratified_sample(self):
        #generate_stratified_museum_sample()
    def test_assign_urls_to_sample(self):
        stratdf = pd.read_csv('data/museums/museums_stratified_sample_400.tsv', sep='\t')
        googledf=pd.read_csv('data/google_results/results_source_files/google_extracted_results_reg.tsv.gz', sep='\t')
        facebookdf=pd.read_csv('data/google_results/results_source_files/google_extracted_results_facebook.tsv.gz', sep='\t')
        twitterdf=pd.read_csv('data/google_results/results_source_files/google_extracted_results_twitter.tsv.gz', sep='\t')
        stratdf=stratdf.filter(['muse_id','musname'], axis=1)
        googledf=googledf.filter(['muse_id','search_type', 'google_rank', 'url'], axis=1)
        facebookdf=facebookdf.filter(['muse_id','search_type', 'google_rank', 'url'], axis=1)
        twitterdf=twitterdf.filter(['muse_id','search_type', 'google_rank', 'url'], axis=1)
        outputdfg = pd.merge(stratdf,googledf,on='muse_id')
        outputdff = pd.merge(stratdf,facebookdf,on='muse_id')
        outputdft = pd.merge(stratdf,twitterdf,on='muse_id')
        outputdf=outputdfg.append(outputdff)
        outputdf=outputdf.append(outputdft)
        outputdf=outputdf[outputdf.google_rank<11]
        outputdf=outputdf.sort_values(['muse_id','search_type','google_rank'], ascending=[True,True,True])
        outputdf.to_excel("tmp/merged_stratified_sample_400.xlsx", index=False)
        outputdf.to_csv('tmp/merged_stratified_sample_400.tsv', index=False, sep='\t')
    #def test_loading_sample(self):
        #load_manual_museum_urls()
        #print(check_for_url_redirection("https://www.facebook.com/pages/The-Scottish-Fisheries-Museum/168840906463849?ref=hl"))
    #def test_fuzzy_string_match(self):
        #musdf = load_fuzzy_museums()
        #generate_samples()
        #musdf=load_extracted_museums(musdf)
        #get_fuzzy_string_match_scores(musdf, 'web')
    #def test_generate_samples(self):
        #generate_samples()
    #def test_combined_dataframe(self):
        #generate_combined_dataframe()
    

                
    
class TestCentralDB(unittest.TestCase):
    def setUp(self):        
        i = 0

    def test_db_connection(self):
        print(is_postgresql_db_accessible())
        connect_to_postgresql_db()

    def test_load_manual_links(self):
        #load_manual_museum_urls()
        generate_stratified_museum_sample()
        