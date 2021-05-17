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
from analytics.url_learning import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
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
    


    def test_ml(self):
        df_matrix=pd.read_csv('tmp/ml_museum_scores.tsv', sep='\t')
        
        print(df_matrix['muse_id'].nunique())
        df_matrix=df_matrix.loc[df_matrix['predicted'] == 1]
        print(df_matrix['muse_id'].nunique())
        resultset='website'
        df_matrix=pd.read_excel(r'tmp/combined_museum_matrix.xlsx')
        print(df_matrix['muse_id'].nunique())
        df_matrix=df_matrix.loc[df_matrix['search_type'] == resultset]
        if resultset=='website':
            df_matrix=df_matrix[~df_matrix['url'].str.contains("facebook")]
            df_matrix=df_matrix[~df_matrix['url'].str.contains("twitter")]
        df_matrix['google_rank']=df_matrix.groupby('muse_id').cumcount()+1
        df_matrix.to_excel(r'tmp/a_x_test_2.xlsx')
        apply_random_forest(df_matrix)
        #generate_ml_model()
        
    

                
    
class TestCentralDB(unittest.TestCase):
    def setUp(self):        
        i = 0

    def test_db_connection(self):
        print(is_postgresql_db_accessible())
        connect_to_postgresql_db()

    def test_load_manual_links(self):
        #load_manual_museum_urls()
        generate_stratified_museum_sample()
        