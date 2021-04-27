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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle


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

def get_best_forest(df_ml_matrix, name):
        df_ml_matrix=df_ml_matrix.loc[df_ml_matrix['iscorrect']!='',[	'google_rank',	'url_size',	'N_slash',	'has_visit',	'has_museum',	'has_location',	'fuzzy_score_full_url',	'fuzzy_score_domain',	'fuzzy_score_domain_inverse',	'withloc_fuzzy_score_full_url',	'withloc_fuzzy_score_domain',	'exact_score_url',	'exact_score_url_inverse',	'exact_score_domain',	'exact_score_domain_inverse',	'iscorrect', 'huge',	'large',	'medium',	'small',	'unknown',	'Government:Cadw',	'Government:Local Authority',	'Government:National',	'Independent:English Heritage',	'Independent:Historic Environment Scotland',	'Independent:National Trust',	'Independent:National Trust for Scotland',	'Independent:Not for profit',	'Independent:Private',	'Independent:Unknown',	'University',	'Unknown'
]]

        
        df_ml_matrix=df_ml_matrix.round(4)
        df_ml_matrix=df_ml_matrix.fillna(0)
        
        #df_ml_matrix_2=pd.get_dummies(df_ml_matrix, columns=["size", "governance"], prefix=["size", "gov"])
        
        y=df_ml_matrix['iscorrect']
        y=y.astype('int')
        x=df_ml_matrix.drop(['iscorrect'], axis=1)
        #best_x_train=pd.DataFrame()
        #best_x_test=pd.DataFrame()
        #best_y_train=pd.DataFrame()
        #best_y_test=pd.DataFrame()
        p_best_precision=None
        p_best_forest=None
        p_best_confusemat=None
        p_best_sensitivity=None
        p_best_specificity=None
        p_best_accuracy=None
        p_x_test=None
        p_y_test=None

        sn_best_precision=None
        sn_best_forest=None
        sn_best_confusemat=None
        sn_best_sensitivity=None
        sn_best_specificity=None
        sn_best_accuracy=None
        sn_x_test=None
        sn_y_test=None

        sp_best_precision=None
        sp_best_forest=None
        sp_best_confusemat=None
        sp_best_sensitivity=None
        sp_best_specificity=None
        sp_best_accuracy=None
        sp_x_test=None
        sp_y_test=None

        a_best_precision=None
        a_best_forest=None
        a_best_confusemat=None
        a_best_sensitivity=None
        a_best_specificity=None
        a_best_accuracy=None
        a_x_test=None
        a_y_test=None
        for z in range (0,100):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
            x_train.to_excel(r'tmp/xtrain.xlsx')
            x_test.to_excel(r'tmp/xtest.xlsx')
            df_ml_matrix.to_excel("tmp/ml_museum_matrix.xlsx", index=False)
            rf_Model=RandomForestClassifier()
            rf_Model.fit(x_train, y_train)
            y_pred_test = rf_Model.predict(x_test)
            
            confusmat=confusion_matrix( y_pred_test, y_test)  
            

            precision=precision_score(y_test, y_pred_test, average=None)[1]
            
            sensetivity=recall_score(y_test, y_pred_test, average=None)[1]
            
            specificity=recall_score(y_test, y_pred_test, average=None)[0]
            
            accuracy=(confusmat[0][0]+confusmat[1][1])/(confusmat[0][0]+confusmat[0][1]+confusmat[1][0]+confusmat[1][1])
            
            #print(precision[1])
            #plot_confusion_matrix(rf_Model, x_test, y_test)  
            #plt.show() 
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
            tr_a=rf_Model.score(x_train,y_train)
            ts_a=rf_Model.score(x_test,y_test)
            if p_best_precision is not None and sn_best_sensitivity is not None and sp_best_specificity is not None and a_best_accuracy is not None:
                if precision>p_best_precision:
                    p_best_precision=precision
                    p_best_confusemat=confusmat
                    p_best_forest=rf_Model
                    p_best_sensitivity=sensetivity
                    p_best_specificity=specificity
                    p_best_accuracy=accuracy
                    p_x_test=x_test
                    p_y_test=y_test
                if accuracy>a_best_accuracy:
                    a_best_precision=precision
                    a_best_confusemat=confusmat
                    a_best_forest=rf_Model
                    a_best_sensitivity=sensetivity
                    a_best_specificity=specificity
                    a_best_accuracy=accuracy
                    a_x_test=x_test
                    a_y_test=y_test
                if sensetivity>sn_best_sensitivity:
                    sn_best_precision=precision
                    sn_best_confusemat=confusmat
                    sn_best_forest=rf_Model
                    sn_best_sensitivity=sensetivity
                    sn_best_specificity=specificity
                    sn_best_accuracy=accuracy
                    sn_x_test=x_test
                    sn_y_test=y_test
                if specificity>sp_best_specificity:
                    sp_best_precision=precision
                    sp_best_confusemat=confusmat
                    sp_best_forest=rf_Model
                    sp_best_sensitivity=sensetivity
                    sp_best_specificity=specificity
                    sp_best_accuracy=accuracy
                    sp_x_test=x_test
                    sp_y_test=y_test
            else:
                p_best_precision=precision
                p_best_confusemat=confusmat
                p_best_forest=rf_Model
                p_best_sensitivity=sensetivity
                p_best_specificity=specificity
                p_best_accuracy=accuracy
                p_x_test=x_test
                p_y_test=y_test

                sn_best_precision=precision
                sn_best_confusemat=confusmat
                sn_best_forest=rf_Model
                sn_best_sensitivity=sensetivity
                sn_best_specificity=specificity
                sn_best_accuracy=accuracy
                sn_x_test=x_test
                sn_y_test=y_test

                sp_best_precision=precision
                sp_best_confusemat=confusmat
                sp_best_forest=rf_Model
                sp_best_sensitivity=sensetivity
                sp_best_specificity=specificity
                sp_best_accuracy=accuracy
                sp_x_test=x_test
                sp_y_test=y_test

                a_best_precision=precision
                a_best_confusemat=confusmat
                a_best_forest=rf_Model
                a_best_sensitivity=sensetivity
                a_best_specificity=specificity
                a_best_accuracy=accuracy
                a_x_test=x_test
                a_y_test=y_test
        print("Results for best precision:")
        plot_confusion_matrix(p_best_forest, p_x_test, p_y_test)  
        plt.show() 
        print (p_best_confusemat)
        print ("precision:")
        print (p_best_precision)
        print ("accuracy: ")
        print (p_best_accuracy)
        print ("sensitivity: ")
        print (p_best_sensitivity)
        print ("specificity: ")
        print (p_best_specificity)
        print("")
        print("Results for best accuracy:")
        plot_confusion_matrix(a_best_forest, a_x_test, a_y_test)  
        plt.show() 
        print (a_best_confusemat)
        print ("precision:")
        print (a_best_precision)
        print ("accuracy: ")
        print (a_best_accuracy)
        print ("sensitivity: ")
        print (a_best_sensitivity)
        print ("specificity: ")
        print (a_best_specificity)
        print("")
        print("Results for best sensitivity:")
        plot_confusion_matrix(sn_best_forest, sn_x_test, sn_y_test)  
        plt.show() 
        print (sn_best_confusemat)
        print ("precision:")
        print (sn_best_precision)
        print ("accuracy: ")
        print (sn_best_accuracy)
        print ("sensitivity: ")
        print (sn_best_sensitivity)
        print ("specificity: ")
        print (sn_best_specificity)
        print("")
        print("Results for best specificity:")
        plot_confusion_matrix(sp_best_forest, sp_x_test, sp_y_test)  
        plt.show() 
        print (sp_best_confusemat)
        print ("precision:")
        print (sp_best_precision)
        print ("accuracy: ")
        print (sp_best_accuracy)
        print ("sensitivity: ")
        print (sp_best_sensitivity)
        print ("specificity: ")
        print (sp_best_specificity)
        print("")
        
        #filename = 'finalized_model'+name+'.sav'
        #pickle.dump(best_forest, open(filename, 'wb'))
       
        print("end")

