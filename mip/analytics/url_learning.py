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
        df_ml_matrix_out=df_ml_matrix.loc[df_ml_matrix['iscorrect']!='',[	'muse_id',	'url', 'musname']]
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

        #p_best_precision=None
        #p_best_forest=None
        #p_best_confusemat=None
        #p_best_sensitivity=None
        #p_best_specificity=None
        #p_best_accuracy=None
        #p_x_test=None
        #p_y_test=None

        #sn_best_precision=None
        #sn_best_forest=None
        #sn_best_confusemat=None
        #sn_best_sensitivity=None
        #sn_best_specificity=None
        #sn_best_accuracy=None
        #sn_x_test=None
        #sn_y_test=None

        #sp_best_precision=None
        #sp_best_forest=None
        #sp_best_confusemat=None
        #sp_best_sensitivity=None
        #sp_best_specificity=None
        #sp_best_accuracy=None
        #sp_x_test=None
        #sp_y_test=None

        a_best_precision=None
        a_best_forest=None
        a_best_confusemat=None
        a_best_sensitivity=None
        a_best_specificity=None
        a_best_accuracy=None
        a_x_test=None
        a_y_test=None
        a_x_train=None
        a_y_train=None
        a_y_pred_test=None
        split=0.2
        res_df = pd.DataFrame(columns=['precision','accuracy','sensetivity','specificity'])

        for z in range (0,300):
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split)
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
                    res_df.loc[z, ['precision']] = precision
                    res_df.loc[z, ['accuracy']] = accuracy
                    res_df.loc[z, ['sensetivity']] = sensetivity
                    res_df.loc[z, ['specificity']] = specificity
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
                    if  a_best_accuracy is not None:
                        #if precision>p_best_precision:
                            #p_best_precision=precision
                            #p_best_confusemat=confusmat
                            #p_best_forest=rf_Model
                            #p_best_sensitivity=sensetivity
                            #p_best_specificity=specificity
                            #p_best_accuracy=accuracy
                            #p_x_test=x_test
                            #p_y_test=y_test
                        if accuracy>a_best_accuracy:
                            a_best_precision=precision
                            a_best_confusemat=confusmat
                            a_best_forest=rf_Model
                            a_best_sensitivity=sensetivity
                            a_best_specificity=specificity
                            a_best_accuracy=accuracy
                            a_x_test=x_test
                            a_y_test=y_test
                            a_x_train=x_train
                            a_y_train=y_train
                            a_y_pred_test=y_pred_test
                        #if sensetivity>sn_best_sensitivity:
                            #sn_best_precision=precision
                            #sn_best_confusemat=confusmat
                            #sn_best_forest=rf_Model
                            #sn_best_sensitivity=sensetivity
                            #sn_best_specificity=specificity
                            #sn_best_accuracy=accuracy
                            #sn_x_test=x_test
                            #sn_y_test=y_test
                        #if specificity>sp_best_specificity:
                            #sp_best_precision=precision
                            #sp_best_confusemat=confusmat
                            #sp_best_forest=rf_Model
                            #sp_best_sensitivity=sensetivity
                            #sp_best_specificity=specificity
                            #sp_best_accuracy=accuracy
                            #sp_x_test=x_test
                            #sp_y_test=y_test
                    else:
                        #p_best_precision=precision
                        #p_best_confusemat=confusmat
                        #p_best_forest=rf_Model
                        #p_best_sensitivity=sensetivity
                        #p_best_specificity=specificity
                        #p_best_accuracy=accuracy
                        #p_x_test=x_test
                        #p_y_test=y_test

                        #sn_best_precision=precision
                        #sn_best_confusemat=confusmat
                        #sn_best_forest=rf_Model
                        #sn_best_sensitivity=sensetivity
                        #sn_best_specificity=specificity
                        #sn_best_accuracy=accuracy
                        #sn_x_test=x_test
                        #sn_y_test=y_test

                        #sp_best_precision=precision
                        #sp_best_confusemat=confusmat
                        #sp_best_forest=rf_Model
                        #sp_best_sensitivity=sensetivity
                        #sp_best_specificity=specificity
                        #sp_best_accuracy=accuracy
                        #sp_x_test=x_test
                        #sp_y_test=y_test

                        a_best_precision=precision
                        a_best_confusemat=confusmat
                        a_best_forest=rf_Model
                        a_best_sensitivity=sensetivity
                        a_best_specificity=specificity
                        a_best_accuracy=accuracy
                        a_x_test=x_test
                        a_y_test=y_test
                        a_x_train=x_train
                        a_y_train=y_train
                        a_y_pred_test=y_pred_test
                
                
                
        res_df.to_excel(r'tmp/300_splits_results.xlsx')
            
            #plot_confusion_matrix(a_best_forest, a_x_test, a_y_test)  
            #plt.show() 
        
        print ("avg precision:")
        print (a_best_precision)
        print ("avg accuracy: ")
        print (a_best_accuracy)
        print ("avg sensitivity: ")
        print (a_best_sensitivity)
        print ("avg specificity: ")
        print (a_best_specificity)
        print("")
            #a_x_test.to_excel(r'tmp/a_x_test.xlsx')
            #print(a_x_test.sample(n=10, random_state=1))
            #print(a_y_test.sample(n=10, random_state=1))
            #a_x_test=pd.concat([a_x_test, a_y_test], axis=1)
            
            #a_y_prediction = pd.DataFrame(a_y_pred_test)
            #print(a_y_prediction.sample(n=10, random_state=1))
            #print(a_x_test.sample(n=10, random_state=1))
            #a_x_test['predicted']=a_y_pred_test.tolist()
            #print(a_x_test.sample(n=10, random_state=1))
            #a_x_test=pd.concat([a_x_test, df_ml_matrix_out], axis=1)
            #print(a_x_test.sample(n=10, random_state=1))
            #a_x_test.to_excel(r'tmp/a_x_test_2.xlsx')
            #a_x_test_2=pd.read_excel(r'tmp/a_x_test_2.xlsx')
            
            #df_ml_matrix_out.to_excel(r'tmp/df_ml_matrix_out.xlsx')
        filename = 'finalized_model'+name+'.sav'
        pickle.dump(a_best_forest, open(filename, 'wb'))
        
        print("end")
        
            
        print("really end")
    
def apply_random_forest(df_matrix):
    print('applying forest')
    filename='finalized_modelwb.sav'
    ran_forest=pickle.load(open(filename, 'rb'))
    dummy1=pd.get_dummies(df_matrix['size'])
    dummy2=pd.get_dummies(df_matrix['governance'])
    df_matrix=pd.concat([df_matrix, dummy1], axis=1)
    df_matrix=pd.concat([df_matrix, dummy2], axis=1)
    df_ml_matrix_out=df_matrix[[	'muse_id',	'url', 'musname']].copy()
    df_ml_matrix=df_matrix[[	'google_rank',	'url_size',	'N_slash',	'has_visit',	'has_museum',	'has_location',	'fuzzy_score_full_url',	'fuzzy_score_domain',	'fuzzy_score_domain_inverse',	'withloc_fuzzy_score_full_url',	'withloc_fuzzy_score_domain',	'exact_score_url',	'exact_score_url_inverse',	'exact_score_domain',	'exact_score_domain_inverse',	 'huge',	'large',	'medium',	'small',	'unknown',	'Government:Cadw',	'Government:Local Authority',	'Government:National',	'Independent:English Heritage',	'Independent:Historic Environment Scotland',	'Independent:National Trust',	'Independent:National Trust for Scotland',	'Independent:Not for profit',	'Independent:Private',	'Independent:Unknown',	'University',	'Unknown'
]].copy()
    print('starting prediction')
    y_pred_test = ran_forest.predict(df_ml_matrix)
    a_y_prediction = pd.DataFrame(y_pred_test)
    print(a_y_prediction.sample(n=100, random_state=2))
    df_matrix['predicted']=y_pred_test.tolist()
    print(df_matrix.sample(n=100, random_state=2))
    df_matrix.to_csv('tmp/ml_museum_scores.tsv', index=False, sep='\t')
    print("end")


def generate_ml_model():
        #df_matrix_1 = pd.read_excel(r'tmp/merged_stratified_sample_400_1.xlsx')
        #df_matrix_2 = pd.read_excel(r'tmp/merged_stratified_sample_400_2.xlsx')
        #df_matrix_3 = pd.read_excel(r'tmp/merged_stratified_sample_400_3.xlsx')
        #outputdf=pd.concat([df_matrix_1, df_matrix_2], ignore_index=True)
        #outputdf=pd.concat([outputdf, df_matrix_3], ignore_index=True)
        
        #outputdf['url_size']=outputdf['url'].str.len() 
        #outputdf['N_slash']=outputdf['url'].str.split('/')
        #outputdf=pd.read_excel(r'tmp/outputdf.xlsx')
        #outputdf['N_slash']=outputdf['N_slash'].apply(lambda x: len(x))
        
        
        
        
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
        stratdf = pd.read_csv('data/museums/museums_wattributes-2020-02-23.tsv', sep='\t')
        stratdf=stratdf.filter(['muse_id','size', 'governance'], axis=1)
        df_matrix = pd.merge(stratdf,df_matrix,on='muse_id')
        dummy1=pd.get_dummies(df_matrix['size'])
        dummy2=pd.get_dummies(df_matrix['governance'])
        df_matrix=pd.concat([df_matrix, dummy1], axis=1)
        df_matrix=pd.concat([df_matrix, dummy2], axis=1)
        #df_matrix.to_excel(r'tmp/outputdftest2.xlsx')
        df_ml_matrix_wb=df_matrix.loc[df_matrix['search_type'] == 'website']
        df_ml_matrix_fb=df_matrix.loc[df_matrix['search_type'] == 'facebook']
        df_ml_matrix_tw=df_matrix.loc[df_matrix['search_type'] == 'twitter']
        print("websites")
        get_best_forest(df_ml_matrix_wb, 'wb')
        print("facebook")
        #get_best_forest(df_ml_matrix_fb, 'fb')
        print("twitter")
        #get_best_forest(df_ml_matrix_tw, 'tw')

