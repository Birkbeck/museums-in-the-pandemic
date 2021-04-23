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
from analytics.url_learning import repair_valid_column
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
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
        #df_matrix_1 = pd.read_excel(r'tmp/merged_stratified_sample_400_1.xlsx')
        #df_matrix_2 = pd.read_excel(r'tmp/merged_stratified_sample_400_2.xlsx')
        #df_matrix_3 = pd.read_excel(r'tmp/merged_stratified_sample_400_3.xlsx')
        #outputdf=pd.concat([df_matrix_1, df_matrix_2], ignore_index=True)
        #outputdf=pd.concat([outputdf, df_matrix_3], ignore_index=True)
        
        #outputdf['url_size']=outputdf['url'].str.len() 
        #outputdf['N_slash']=outputdf['url'].str.split('/')
        outputdf=pd.read_excel(r'tmp/outputdf.xlsx')
        outputdf['N_slash']=outputdf['N_slash'].apply(lambda x: len(x))
        
        
        
        
        #outputdf['has_visit']=outputdf.apply(lambda row: (hasvisit(row['url'],row['search_type'])), axis=1)
        #outputdf['has_museum']=outputdf.apply(lambda row: (hasmuseum(row['url'],row['search_type'])), axis=1)
        #outputdf['has_location']=outputdf.apply(lambda row: (haslocation(row['url'],row['town'])), axis=1)
        #museweight = generate_weighted_museum_names()
       # outputdf['fuzzy_score_full_url']=outputdf.apply(lambda row: (generate_weighted_fuzzy_scores(get_musname_pool(row['musname'], row['town']), row['url'], museweight, "", False, row['search_type'])), axis=1)
        #outputdf['fuzzy_score_domain']=outputdf.apply(lambda row: (generate_weighted_fuzzy_scores(get_musname_pool(row['musname'], row['town']), row['url'], museweight, "", True, row['search_type'])), axis=1)
        #outputdf['fuzzy_score_domain_inverse']=outputdf.apply(lambda row: (get_fuzzy_string_score(get_url_domain_with_search(row['url'], row['search_type']),row['musname'] )), axis=1)
        #outputdf['withloc_fuzzy_score_full_url']=outputdf.apply(lambda row: (generate_weighted_fuzzy_scores(get_musname_pool(row['musname'], row['town']), row['url'], museweight, row['town'], False, row['search_type'])), axis=1)
        #outputdf['withloc_fuzzy_score_domain']=outputdf.apply(lambda row: (generate_weighted_fuzzy_scores(get_musname_pool(row['musname'], row['town']), row['url'], museweight, row['town'], True, row['search_type'])), axis=1)
        
        #outputdf['exact_score_url']=outputdf.apply(lambda row: (get_exact_match(row['musname'], row['url'])), axis=1)
        #outputdf['exact_score_url_inverse']=outputdf.apply(lambda row: (get_exact_match(row['url'],row['musname'])), axis=1)
        #outputdf['exact_score_domain']=outputdf.apply(lambda row: (get_exact_match(row['musname'], get_url_domain_with_search(row['url'], row['search_type']))), axis=1)
        #outputdf['exact_score_domain_inverse']=outputdf.apply(lambda row: (get_exact_match(get_url_domain_with_search(row['url'], row['search_type']),row['musname'])), axis=1)

        #outputdf.to_excel(r'tmp/outputdftest.xlsx')
        df_matrix=pd.read_excel(r'tmp/outputdftest.xlsx')
        df_matrix['valid']=df_matrix.apply(lambda row: (repair_valid_column(row['valid'])), axis=1)
        #df_matrix.to_excel(r'tmp/outputdftest2.xlsx')
        df_matrix['iscorrect']=df_matrix['valid']
        df_ml_matrix=df_matrix.loc[df_matrix['iscorrect']!='',[	'google_rank',	'url_size',	'N_slash',	'has_visit',	'has_museum',	'has_location',	'fuzzy_score_full_url',	'fuzzy_score_domain',	'fuzzy_score_domain_inverse',	'withloc_fuzzy_score_full_url',	'withloc_fuzzy_score_domain',	'exact_score_url',	'exact_score_url_inverse',	'exact_score_domain',	'exact_score_domain_inverse',	'iscorrect']]
        df_ml_matrix=df_ml_matrix.round(4)
        df_ml_matrix=df_ml_matrix.fillna(0)
        
        #df_ml_matrix_2=pd.get_dummies(df_ml_matrix, columns=["size", "governance"], prefix=["size", "gov"])
        
        y=df_ml_matrix['iscorrect']
        y=y.astype('int')
        x=df_ml_matrix.drop(['iscorrect'], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=101)
        x_train.to_excel(r'tmp/xtrain.xlsx')
        x_test.to_excel(r'tmp/xtest.xlsx')
        df_ml_matrix.to_excel("tmp/ml_museum_matrix.xlsx", index=False)
        rf_Model=RandomForestClassifier()
        rf_Model.fit(x_train, y_train)
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
        print(f'Train Accuracy - :{rf_Model.score(x_train,y_train):.3f}')
        print(f'Test Accuracy - :{rf_Model.score(x_test,y_test):.3f}')
        print("end")
    

                
    
class TestCentralDB(unittest.TestCase):
    def setUp(self):        
        i = 0

    def test_db_connection(self):
        print(is_postgresql_db_accessible())
        connect_to_postgresql_db()

    def test_load_manual_links(self):
        #load_manual_museum_urls()
        generate_stratified_museum_sample()
        