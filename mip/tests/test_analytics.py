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
from analytics.text_models import *

class TestTextExtraction(unittest.TestCase):
    def setUp(self):        
        #self.db_con = open_sqlite('data/test_data/websites_sample_2020-02-01.db')
        #self.out_db_con = open_sqlite('tmp/websites_sample_textattr.db')
        pass

    def test_fb_tw_data(self):
        df_400 = pd.read_excel(r'tmp/outputdftest.xlsx')
        df_400=df_400.loc[df_400['valid'] == 'T']
        df_400=df_400.loc[df_400['search_type'] != 'website']
        df_400=df_400[['muse_id','search_type','url']].copy()
        df_400=df_400.rename(columns = {'muse_id': 'museum_id', 'search_type': 'type'}, inplace = False)
        socdf = pd.read_csv('data/museums/websites_social_links_2021-05-28.tsv', sep='\t')
        mltw = pd.read_csv('tmp/ml_museum_scores_tw.tsv', sep='\t')
        mltw=mltw.loc[mltw['predicted'] == 1]
        mlfb = pd.read_csv('tmp/ml_museum_scores_fb.tsv', sep='\t')
        mlfb=mlfb.loc[mlfb['predicted'] == 1]
        mlsoc=mltw.append(mlfb)
        #mlsoc.to_excel(r'tmp/a_x_test2.xlsx')
        socdf2=socdf.loc[socdf['url']=='no_resource']
        socdf=socdf.loc[socdf['url']!='no_resource']
        socdf2=socdf2[['row_id','museum_id','type']].copy()
        mlsoc=mlsoc[['muse_id','search_type','url']].copy()
        mlsoc = mlsoc.rename(columns = {'muse_id': 'museum_id', 'search_type': 'type'}, inplace = False)
        socdfml = pd.merge(socdf2, mlsoc,  how='left', left_on=['museum_id','type'], right_on = ['museum_id','type'])
        
        socdf=socdf.append(socdfml)
        
        socdf=pd.merge(socdf, df_400,  how='left', left_on=['museum_id','type'], right_on = ['museum_id','type'])
        socdf.to_excel(r'tmp/a_x_test2.xlsx')
        print('end')

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
    #def test_text_tokenization(self):
    #    print('hi')
    #    analyse_museum_text()

    def test_fb_tw_data(self):
        analyse_museum_text()
        targetsoc='twitter'
        df_400 = pd.read_excel(r'tmp/outputdftest.xlsx')
        df_400=df_400.loc[df_400['valid'] == 'T']
        df_400=df_400.loc[df_400['search_type'] != 'website']
        df_400=df_400[['muse_id','search_type','url']].copy()
        df_400["url"]=df_400["url"].str.replace("https://", "")
        df_400["url"]=df_400["url"].str.replace("http://", "")
        df_400["url"]=df_400["url"].str.rstrip("/")
        df_400=df_400.rename(columns = {'muse_id': 'museum_id', 'search_type': 'type', 'url':'urlMark'}, inplace = False)
        
        socdf = pd.read_csv('data/museums/websites_social_links_2021-06-03.tsv', sep='\t')
        socdf["clean_url"]=socdf["clean_url"].str.replace("https://", "")
        socdf["clean_url"]=socdf["clean_url"].str.replace("http://", "")
        socdf=socdf.rename(columns = {'clean_url':'urlScrape'}, inplace = False)
        socdf["occurances"]= socdf.groupby('urlScrape')['urlScrape'].transform('size')
        #socdf.to_excel(r'tmp/a_x_test2.xlsx')
        mltw = pd.read_csv('tmp/ml_museum_scores_tw.tsv', sep='\t')
        mltw=mltw.loc[mltw['predicted'] == 1]
        mlfb = pd.read_csv('tmp/ml_museum_scores_fb.tsv', sep='\t')
        mlfb=mlfb.loc[mlfb['predicted'] == 1]
        mlsoc=mltw.append(mlfb)
        
        mlsoc["url"]=mlsoc["url"].str.replace("https://", "")
        mlsoc["url"]=mlsoc["url"].str.replace("http://", "")
        mlsoc["url"]=mlsoc["url"].str.rstrip("/")
        #mlsoc.to_excel(r'tmp/a_x_test2.xlsx')
        #socdf2=socdf.loc[socdf['url']=='no_resource']
        #socdf=socdf.loc[socdf['url']!='no_resource']
        #socdf2=socdf2[['row_id','museum_id','type']].copy()
        mlsoc=mlsoc[['muse_id','musname', 'search_type','url']].copy()
        mlsoc = mlsoc.rename(columns = {'muse_id': 'museum_id', 'search_type': 'type', 'url':'urlML'}, inplace = False)
        socdf = socdf.merge(mlsoc,  on=['museum_id','type'],how='left')
        
       
        socdffb=socdf.loc[socdf['type']==targetsoc]
        #socdftw=socdf.loc[socdf['type']=='twitter']
        fbdict={}
        
        df_scraped = pd.DataFrame(columns=['museum_id','urllist'])
        for index, row in socdffb.iterrows():
            
            
            if(row['museum_id'] not in fbdict):
                scrapeurls=[]
                tempdict={}
                tempdict[row['urlScrape']]=row['occurances']
                scrapeurls.append(tempdict)
                fbdict[row['museum_id']]=scrapeurls
            else:
                scrapeurls = fbdict.get(row['museum_id'])
                tempdict={}
                tempdict[row['urlScrape']]=row['occurances']
                scrapeurls.append(tempdict)
                fbdict[row['museum_id']]=scrapeurls
        final_fbdict={}
        for index, row in socdffb.iterrows():
            #print(row['museum_id'])
            if(row['museum_id']=='mm.domus.SE350'):
                print('found')
            
            final_fbdict[row['museum_id']]=process_tw_fb_links(row['urlML'],fbdict[row['museum_id']])
        labeldict={}
        for key, value in final_fbdict.items():
            if isinstance(value, list):
                labeldict[key]='Scrape'
            else:
                if('(((both)))' in value):
                    labeldict[key]='Both'
                    final_fbdict[key]=value.split('(((')[0]
                elif(value=='no_resource'):
                    labeldict[key]='no_resource'
                else:
                    labeldict[key]='ML'


        df_facebook=pd.DataFrame(list(final_fbdict.items()),columns = ['museum_id','url'])
        df_source=pd.DataFrame(list(labeldict.items()),columns = ['museum_id','source'])
        df_facebook=df_facebook.merge(df_source,  on=['museum_id'],how='left')

        df_facebook.to_csv('tmp/a_x_test.tsv', index=False, sep='\t')

        df_facebook=df_facebook.explode('url')
        df_verified= pd.read_excel(r'tmp/verified_soc_links.xlsx')
        
        df_verified["url"]=df_verified["url"].str.replace("https://", "")
        df_verified["url"]=df_verified["url"].str.replace("http://", "")
        df_verified["url"]=df_verified["url"].str.rstrip("/")
        df_verified=df_verified.loc[df_verified['type'] == targetsoc]
        df_verified=df_verified[['url','value','museum_id']].copy()
        

        df_facebook = df_facebook.merge(df_verified,  on=['url','museum_id'],how='left')
        df_facebook.to_excel(r'tmp/a_x_test.xlsx')
        nan_value = float("NaN")
        df_facebook.replace("", nan_value, inplace=True)
        df_facebook.dropna(subset = ["value"], inplace=True)
        
        ####evaluation method 2
        
        eval2dict={}
        for index, row in df_facebook.iterrows():
            if(row['museum_id'] not in eval2dict):
                if(row['url']=='no_resource'):
                    eval2dict[row['museum_id']]='N'+row['value']
                else:
                    eval2dict[row['museum_id']]=row['value']
            elif((eval2dict[row['museum_id']]=='F' or eval2dict[row['museum_id']]=='NF') and row['value']=='T'):
                if(row['url']=='no_resource'):
                    eval2dict[row['museum_id']]='N'+row['value']
                else:
                    eval2dict[row['museum_id']]=row['value']
            elif(eval2dict[row['museum_id']]=='NT' and row['value']=='T' and row['url']!='no_resource'):
                eval2dict[row['museum_id']]=row['value']
            else:
                print('skip')
        true_pos=0
        true_neg=0
        false_pos=0
        false_neg=0
        number=0
        for key,value in eval2dict.items():
            number=number+1
            if(value=='T'):
                true_pos=true_pos+1
            elif(value=='F'):
                false_pos=false_pos+1
            elif(value=='NT'):
                true_neg=true_neg+1
            elif(value=='NF'):
                false_neg=false_pos+1
        accuracy=(true_pos+true_neg)/number
        sensitivity=true_pos/(true_pos+false_neg)
        #specificity=true_neg/(true_neg+false_pos)
        precision=true_pos/(true_pos+false_pos)
        print('evaluation method 2')

        print('accuracy')
        print(accuracy)
        print('sensitivity')
        print(sensitivity)
        print('specificity')
        #print(specificity)
        print('precision')
        print(precision)
        ###evaluation method 2 end
        print("")

        ###evaluation methon 1
        true_pos=0
        true_neg=0
        false_pos=0
        false_neg=0
        number=0
        for index,row in df_facebook.iterrows():
            number=number+1
            if(row['url']!='no_resource'):
                if(row['value']=='T'):
                    true_pos=true_pos+1
                else:
                    false_pos=false_pos+1
            else:
                if(row['value']=='T'):
                    true_neg=true_neg+1
                else:
                    false_neg=false_neg+1
        accuracy=(true_pos+true_neg)/number
        sensitivity=true_pos/(true_pos+false_neg)
        specificity=true_neg/(true_neg+false_pos)
        precision=true_pos/(true_pos+false_pos)
        print('evaluation method 1')

        print('accuracy')
        print(accuracy)
        print('sensitivity')
        print(sensitivity)
        print('specificity')
        print(specificity)
        print('precision')
        print(precision)
        ##evaluation method 1 end









        #socdf=df_400.merge(socdf,  on=['museum_id','type'],how='left')
        df_facebook.to_excel(r'tmp/a_x_test2.xlsx')

        
        print('end')



    
        
    

                
    
class TestCentralDB(unittest.TestCase):
    def setUp(self):        
        i = 0

    def test_db_connection(self):
        print(is_postgresql_db_accessible())
        connect_to_postgresql_db()

    def test_load_manual_links(self):
        #load_manual_museum_urls()
        generate_stratified_museum_sample()
        