# -*- coding: utf-8 -*-

"""
Linguistic models for text analytics

"""
import logging

logger = logging.getLogger(__name__)

from db.db import connect_to_postgresql_db, create_alchemy_engine_posgresql
import pandas as pd
from utils import StopWatch
import numpy as np
import pickle
#import tensorflow as tf
from bs4 import BeautifulSoup
from bs4.element import Comment
import re
from analytics.an_websites import get_webdump_attr_table_name
from utils import remove_empty_elem_from_list, remove_multiple_spaces_tabs, _is_number, parallel_dataframe_apply
import matplotlib.pyplot as plt
from db.db import make_string_sql_safe
from museums import get_museums_w_web_urls, get_museums_sample_urls, load_input_museums_wattributes
from analytics.an_websites import get_page_id_for_webpage_url, get_attribute_for_webpage_id

# constants
tokens_table_name = 'analytics.mus_sentence_tokens'

def prep_training_data():
    """ 
    Columns: id, label
    """
    indic_df, ann_df = get_indicator_annotations()
    print(ann_df.columns)
    # format training set
    train_df = ann_df[['text_phrases','indicator_code']]
    # output file
    train_df.to_csv('tmp/indicator_train_df.tsv',sep='\t',index_label='id')
    return train_df


def get_indicator_annotations(data_folder=''):
    """ @returns indicators data frame and annotations data frame """
    in_fn = data_folder + "data/annotations/indicators_and_annotations-v4.xlsx"
    indic_df = pd.read_excel(in_fn, 0)
    ann_df = pd.read_excel(in_fn, 1)
    
    # select first 4 columns
    ann_df = ann_df.iloc[:, : 4]
    ann_df['example_id'] = range(0, len(ann_df))
    ann_df['example_id'] = ["ann_ex_{:05d}".format(x) for x in ann_df['example_id']]
    assert len(indic_df) > 0
    assert len(ann_df) > 0
    return indic_df, ann_df


def setup_ling_model():
    """ 
    - https://towardsdatascience.com/deep-learning-for-specific-information-extraction-from-unstructured-texts-12c5b9dceada
    - https://towardsdatascience.com/text-classification-in-python-dd95d264c802
    - supervised machine learning classification model
    - HuggingFace package: 
        https://github.com/huggingface/transformers#Migrating-from-pytorch-pretrained-bert-to-pytorch-transformers
    """
    logger.debug('setup_ling_model')
    
    train_df = prep_training_data()

    get_museum_text_sample()
    
    #train_ds, dicts = load_ds(os.path.join(data_dir,'atis.train.pkl'))
    #test_ds, dicts  = load_ds(os.path.join(data_dir,'atis.test.pkl'))
  
    #bert_model()
    logger.debug("end of NLP")


def get_museum_text_sample():
    """
    """
    print("get_museum_text_sample")


def bert_model():
    """
    - https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03
    - https://towardsdatascience.com/first-time-using-and-fine-tuning-the-bert-framework-for-classification-799def68a5e4
    """
    pass


def _OLD_create_token_table(db_con):
    return
    """ 
    Create table for tokens
    "sentence_id": sent_id, "token":token.text, 'lemma':token.lemma_,
    "pos_tag":token.pos_, 'is_stop': token.is_stop
     """
    c = db_con.cursor()
    # Create table
    
    # #page_attr_id integer PRIMARY KEY AUTOINCREMENT,
    sql = '''CREATE TABLE IF NOT EXISTS {0}
            (token_id SERIAL PRIMARY KEY,
            sentence_id integer NOT NULL,
            page_id integer NOT NULL,
            session_id text NOT NULL,
            token text NOT NULL,
            lemma text,
            pos_tag text,
            is_stop boolean);
            CREATE INDEX IF NOT EXISTS {1}_sent_idx ON {0} USING btree(sentence_id);
            CREATE INDEX IF NOT EXISTS {1}_session_idx ON {0} USING btree(session_id);
            CREATE INDEX IF NOT EXISTS {1}_page_idx ON {0} USING btree(page_id);
            '''.format(tokens_table_name, tokens_table_name.replace('.','_'))
    #print(sql)
    c.execute(sql)
    db_con.commit()
    
    logger.info('_create_token_table')
    return True


def spacy_extract_tokens(nlp, text):
    """ """
    tokens_df = pd.DataFrame()
    text_sentences = nlp(text)
    sent_id = 0
    # segment sentences
    for sentence in text_sentences.sents:
        sent_id += 1
        # for each sentence
        snt_text = sentence.text
        pos_df = pd.DataFrame()
        #print(colored('>', 'red'), snt_text)
        for token in sentence:
            # for each token
            tokens_df = tokens_df.append(pd.DataFrame(
                {"sentence_id": sent_id, 
                "token": token.text, 'lemma': token.lemma_,
                 "pos_tag":token.pos_, 'is_stop': token.is_stop}, 
                index=[0]), ignore_index=True)
    return tokens_df


def _preprocess_input_text(txt):
    """
    preprocess text from web to make tokenisation easier
    """
    t = txt.replace("'re ", 'are ').strip()
    t = txt.replace("'s ", ' ')
    if len(t) < 3:
        return None
    return t


def spacy_extract_tokens_page(session_id, page_id, nlp, text, db_conn, db_engine, insert_db=False):
    """ 
    [developed in notebook 01]
    Preprocess text and writes it into the token table
    @returns data frame with tokens with POS, lemma, stop words
    """
    print('spacy_extract_tokens_page')
    if page_id==60967:
        print("ok")
    logger.debug('spacy_extract_tokens_page')
    if text is None or len(text) < 3: 
        return None
    text = _preprocess_input_text(text)
    

    tokens_df = spacy_extract_tokens(nlp, text)
    tokens_df["session_id"] = session_id
    tokens_df["page_id"] = page_id

    # change sentence id format
    tokens_df['sentence_id'] = ["mus_page{}_sent{:05d}".format(page_id,x) for x in tokens_df['sentence_id']]
    
    # clear page
    try:
        sql = "delete from {} where session_id = '{}' and page_id = {}".format(tokens_table_name, session_id, page_id)
        c = db_conn.cursor()
        c.execute(sql)
        db_conn.commit()
        logger.debug('delete from '+tokens_table_name+' ok.')
    except:
        logger.warning('delete from '+tokens_table_name+' failed.')

    if insert_db:
        # insert tokens into DB
        print("spacy_extract_tokens_page insert tokens into DB (n={})...".format(len(tokens_df)))
        tokens_df.to_sql('mus_sentence_tokens', db_engine, schema='analytics', index=False, if_exists='append', method='multi')
    
    return tokens_df


def get_indicator_annotation_tokens(nlp):
    """
    returns annotation tokens from examples
    """
    print('get_indicator_annotation_tokens')
    indic_df, ann_df = get_indicator_annotations()
    assert len(ann_df)>0
    ann_tokens_df = pd.DataFrame()
    print('Annotations n =', len(ann_df))

    for index, row in ann_df.iterrows():
        txt = str(row['text_phrases']).strip()
        df = spacy_extract_tokens(nlp, txt)
        n = len(df)
        df = df.drop_duplicates(['lemma'])
        #if n != len(df1):
        #    i = 0
        #df = df1
        #print(df)
        df['example_id'] = row['example_id']
        df['indicator_code'] = row['indicator_code']
        assert len(df) > 1, txt
        ann_tokens_df = pd.concat([ann_tokens_df, df])

    # change sentence ID format
    ann_tokens_df['sentence_id'] = ["ex_sent_{}".format(x) for x in ann_tokens_df['sentence_id']]

    # fix PRON 
    mfilter = ann_tokens_df["lemma"] == '-PRON-'
    ann_tokens_df.loc[mfilter, "lemma"] = ann_tokens_df.loc[mfilter, "token"]
    ann_tokens_df["lemma"] = ann_tokens_df["lemma"].str.lower()
    # filter tokens
    ann_tokens_df = _filter_tokens(ann_tokens_df)
    assert len(ann_tokens_df) > 0
    return ann_tokens_df


def match_indicators_in_muse_page(muse_id, session_id, page_id, nlp, annotat_tokens_df, keep_stopwords, db_conn, db_engine):
    """
    Main function to perform matching for a target museum
    """
    logger.info('match_indicators_in_muse_page {} {} {} stopwords={}'.format(muse_id, session_id, page_id, keep_stopwords))
    
    input_text = get_attribute_for_webpage_id(page_id, session_id, 'all_text', db_conn)

    # save page tokens only once
    page_tokens_df = spacy_extract_tokens_page(session_id, page_id, nlp, input_text, db_conn, db_engine, insert_db=keep_stopwords)    
    
    #page_tokens_df.to_csv('tmp/debug_page_tokens_df.csv',index=False) # DEBUG
    #annotat_tokens_df.to_csv('tmp/annotat_tokens_df.csv',index=False) # DEBUG
    
    # filter tokens based on POS
    if page_tokens_df is not None:
        page_tokens_df = _filter_tokens(page_tokens_df, keep_stopwords)
        annotat_tokens_df = _filter_tokens(annotat_tokens_df, keep_stopwords)

        # add full text for DEBUG
        sent_full_txt_df = page_tokens_df.groupby('sentence_id').apply(lambda x: " ".join(x['token'].tolist())).to_frame().rename(columns={0:'page_tokens'})
        ann_full_txt_df = annotat_tokens_df.groupby('example_id').apply(lambda x: " ".join(x['token'].tolist())).to_frame().rename(columns={0:'ann_ex_tokens'})
        
        # this will read the tokens from the DB
        _match_musetext_indicators(muse_id, session_id, page_id, annotat_tokens_df, page_tokens_df, 
                                    ann_full_txt_df, sent_full_txt_df, keep_stopwords, db_conn, db_engine)


def analyse_museum_text():
    """
    Main command: 
    Preprocess and process museum text using NLP tools.
    """
    logger.info("analyse_museum_text")
    db_conn = connect_to_postgresql_db()
    db_engine = create_alchemy_engine_posgresql()
    #_create_token_table(db_conn)

    # set up the spacy environment
    import spacy
    from spacy import displacy
    from collections import Counter
    spacy.prefer_gpu()
    # load language model
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    
    # get indicator tokens and write them to the DB
    ann_tokens_df = get_indicator_annotation_tokens(nlp)
    if True:
        ann_tokens_df.to_sql('indicator_annotation_tokens', db_engine, schema='analytics', index=False, if_exists='replace', method='multi')
    
    # ann_tokens_df = ann_tokens_df.sample(100) # DEBUG

    df = get_museums_sample_urls()
    #df = df.sample(5) # DEBUG

    # load all museums
    #df = get_museums_w_web_urls()

    attr_df = load_input_museums_wattributes()
    #stratdf = stratdf.filter(['muse_id', 'governance', 'town'], axis=1) ##DEBUG town should be removed
    df = pd.merge(df, attr_df, on='muse_id')
    print("museum df len", len(df))
    #df=df.loc[df['governance'] == 'University'] ##DEBUG line should be unhashed to only use university museums
    session_id = '20210304'
    attrib_name = 'all_text'
    
    i = 0
    for index, row in df.iterrows():
        i += 1
        muse_id = row['muse_id']
        msg = ">>> Processing museum {} of {}, muse_id={}, session={}".format(i,len(df),muse_id,session_id)
        # get main page of a museum
        main_page_ids = get_page_id_for_webpage_url(row['url'], muse_id, session_id, attrib_name, db_conn)
        if main_page_ids is None:
            continue
        assert len(main_page_ids) == 1
        logger.info(msg)
        print(msg)
        for page_id in main_page_ids:
            for keep_stopwords in [True, False]:
                # match indicators with annotations
                match_indicators_in_muse_page(muse_id, session_id, page_id, nlp, ann_tokens_df, keep_stopwords, db_conn, db_engine)
                #spacy_extract_tokens(session_id, page_id, nlp, input_text, db_conn, db_engine)
    del i

    # add indices to table
    idx_sql = """
        ALTER TABLE analytics.text_indic_ann_matches  
            DROP CONSTRAINT IF EXISTS text_indic_ann_matches_pk;
        ALTER TABLE analytics.text_indic_ann_matches 
            ADD CONSTRAINT text_indic_ann_matches_pk 
            PRIMARY KEY (sentence_id, example_id, page_id, keep_stopwords);"""
    c = db_conn.cursor()
    c.execute(idx_sql)
    db_conn.commit()
    logger.info('matches written in table analytics.text_indic_ann_matches.')


def _filter_tokens(df, keep_stopwords=True):
    """ IMPORTANT METHOD: Remove tokens that do not carry semantic content. """
    
    if not keep_stopwords:
        df = df[~df['is_stop']]

    # filter all tokens shorter than 2
    df = df[df.lemma.str.len() > 1]

    # remove specific words
    filt_df = df[~df['lemma'].isin(['of','the','a','this','i','as'])]
    
    # remove based on POS
    pos_to_exclude = ['CCONJ','SCONJ','ADP','PUNCT','SYM','SPACE','NUM','AUX']
    filt_df = filt_df[~filt_df['pos_tag'].isin(pos_to_exclude)]

    return filt_df.copy()


def __OLD_match_tokens(musetxt_df, annot_df, case_sensitive, keep_stopwords):
    return
    """
    Match tokens between museum text and annotation tokens (indicators)
    
    sw = StopWatch("_match_tokens")
    assert len(musetxt_df) >= 0
    assert len(annot_df) > 0, annot_df
    # set up options
    prefix = 'var_'
    suffix = ''
    if case_sensitive: suffix = '_csens'
    else: suffix = '_cinsens'
    if case_sensitive: suffix = '_csens'
    else: suffix = '_cinsens'
    
    if keep_stopwords: suffix += '_wstopw'
    else: suffix += '_nostopw'
        
    # filter tokens
    filt_text_df = _filter_tokens(musetxt_df, keep_stopwords)
    filt_ann_df = _filter_tokens(annot_df, keep_stopwords)
    text_df_n = len(filt_text_df)
    ann_df_n = len(filt_ann_df)
    assert ann_df_n > 0, filt_ann_df
    
    # case in/sensitive
    if not case_sensitive:
        # make lower case
        filt_text_df.loc[:, 'lemma'] = filt_text_df['lemma'].str.lower()
        filt_text_df.loc[:, 'token'] = filt_text_df['token'].str.lower()
        filt_ann_df.loc[:, 'lemma'] = filt_ann_df.loc[:, 'lemma'].str.lower()
        filt_ann_df.loc[:, 'token'] = filt_ann_df.loc[:, 'token'].str.lower()
        
    # generate match variables for each example/text pair
    #lemmas_df = filt_text_df #.merge(filt_ann_df, on=['lemma'])
    #tokens_df = filt_text_df #.merge(filt_ann_df, on=['token'])
    shared_lemmas = set(filt_text_df.lemma.tolist()).intersection(filt_ann_df.lemma.tolist())
    shared_tokens = set(filt_text_df.token.tolist()).intersection(filt_ann_df.token.tolist())
    lemmas_m = " ".join(shared_lemmas)
    tokens_m = " ".join(shared_tokens)

    # empty overlap, skip
    if len(shared_lemmas) == 0 and len(shared_tokens) == 0: 
        return None

    ## calculate overlap for lemma
    if text_df_n == 0:
        lemma_text_overlap = 0
    else:
        lemma_text_overlap = len(shared_lemmas)/text_df_n
    lemma_ann_overlap = len(shared_lemmas)/ann_df_n

    ## calculate overlap for tokens
    if text_df_n == 0:
        token_text_overlap = 0
    else:
        token_text_overlap = len(shared_tokens)/text_df_n
    token_ann_overlap = len(shared_tokens)/ann_df_n
    
    assert token_text_overlap >= 0 and token_text_overlap <= 1
    assert token_ann_overlap >= 0 and token_ann_overlap <= 1
    
    vars_d = {
        prefix+'lemmas_n'+suffix:len(shared_lemmas), 
        prefix+'lemmas_txt_overlap'+suffix: round(lemma_text_overlap,5),
        prefix+'lemmas_ann_overlap'+suffix: round(lemma_ann_overlap,5),
        prefix+'lemmas_m'+suffix: lemmas_m, 
        prefix+'tokens_n'+suffix:len(shared_tokens),
        prefix+'tokens_txt_overlap'+suffix: round(token_text_overlap,5),
        prefix+'tokens_ann_overlap'+suffix: round(token_ann_overlap,5),
        prefix+'tokens_m'+suffix: tokens_m
    }
    #logger.debug(sw.tick())
    return vars_d
    """


def _count_unique_matches(df, col, target_col_name):
    """
    Aggregate token/lemma matches
    """
    # count rows by these columns
    aggr_cols = ['sentence_id_txt','example_id','indicator_code']
    
    # ignore repetitions in column 'col' to avoid overcounting tokens
    dupl_cols = aggr_cols + [col]
    match_df = df.drop_duplicates(dupl_cols).groupby(aggr_cols).size().reset_index().rename(columns={0:target_col_name})
    
    # keep repetitions
    match_dupl_df = df.groupby(aggr_cols).size().reset_index().rename(columns={0:target_col_name+"_wdupl"})
    
    # aggregate results
    assert len(match_dupl_df) == len(match_df), str(len(match_dupl_df)) + ' ' + str(len(match_df))
    match_df = match_df.merge(match_dupl_df, on=aggr_cols)
    assert len(match_dupl_df) == len(match_df)
    return match_df


def _match_musetext_indicators(muse_id, session_id, page_id, annot_df, page_tokens_df, 
                annotat_full_txt_df, sentences_full_txt_df, keep_stopwords, db_conn, db_engine):
    """ 
    Main match loop between set of sentences and set of annotations for a single museum 
    """
    assert muse_id
    assert session_id
    assert page_id
    assert len(annot_df) >= 0
    sw = StopWatch("_match_musetext_indicators")
    # get tokens from token table
    #sql = """select * from {} where session_id = '{}' and page_id = {}""".format(tokens_table_name, session_id, page_id)
    txt_df = page_tokens_df # pd.read_sql_query(sql, db_conn)
    
    df = pd.DataFrame()
    if len(page_tokens_df)==0:
        return df
    
    # count sentence length in input data
    txt_len_df = txt_df.groupby(['sentence_id']).size().reset_index().rename(columns={0:'sent_len'})
    ann_len_df = annot_df.groupby(['example_id']).size().reset_index().rename(columns={0:'example_len'})

    # match lemmas between page and annotations
    #  join
    lemma_df = txt_df.merge(annot_df, on='lemma', suffixes=['_txt','_ann'])
    lemma_match_df = _count_unique_matches(lemma_df, 'lemma', 'lemma_n')
    
    # match tokens between page and annotations
    #  join
    token_df = txt_df.merge(annot_df, on='token', suffixes=['_txt','_ann'])
    token_match_df = _count_unique_matches(token_df, 'token', 'token_n')
    
    # merge lemmas and tokens results
    match_df = lemma_match_df.merge(token_match_df, on=['sentence_id_txt','example_id','indicator_code'], how='outer')
    
    # fill with zero
    match_df['lemma_n'] = match_df['lemma_n'].replace(np.nan, 0)
    match_df['lemma_n_wdupl'] = match_df['lemma_n_wdupl'].replace(np.nan, 0)
    match_df['token_n'] = match_df['token_n'].replace(np.nan, 0)
    match_df['token_n_wdupl'] = match_df['token_n_wdupl'].replace(np.nan, 0)
    
    n = len(match_df)
    
    # add sentence length
    match_df = match_df.merge(txt_len_df, left_on='sentence_id_txt', right_on='sentence_id')
    match_df = match_df.merge(ann_len_df, left_on='example_id', right_on='example_id')
    assert n == len(match_df)

    match_df = match_df.drop(columns=['sentence_id_txt'])

    # calculate overlaps
    digits = 5
    match_df['ann_overlap_lemma'] = round(match_df['lemma_n'] / match_df['example_len'],digits)
    match_df['ann_overlap_token'] = round(match_df['token_n'] / match_df['example_len'],digits)
    match_df['txt_overlap_lemma'] = round(match_df['lemma_n'] / match_df['sent_len'],digits)
    match_df['txt_overlap_token'] = round(match_df['token_n'] / match_df['sent_len'],digits)
    
    # add texts for DEBUG
    if True:
        n1 = len(match_df)
        match_df = match_df.merge(annotat_full_txt_df, on='example_id')
        match_df = match_df.merge(sentences_full_txt_df, on='sentence_id')
        assert len(match_df) == n1

    # check overlap score ranges
    assert match_df.ann_overlap_lemma.between(0,1).all(), match_df.ann_overlap_lemma.sort_values()
    assert match_df.ann_overlap_token.between(0,1).all(), match_df.ann_overlap_token.sort_values()
    assert match_df.txt_overlap_lemma.between(0,1).all(), match_df.txt_overlap_lemma.sort_values()
    assert match_df.txt_overlap_token.between(0,1).all(), match_df.txt_overlap_token.sort_values()

    # set general params
    match_df['session_id'] = session_id
    match_df['page_id'] = page_id
    match_df['muse_id'] = muse_id
    match_df['keep_stopwords'] = keep_stopwords

    # clear page before insertion
    try:
        sql = "delete from {} where session_id = '{}' and page_id = {} and keep_stopwords = {};".format('analytics.text_indic_ann_matches', session_id, page_id, keep_stopwords)
        c = db_conn.cursor()
        c.execute(sql)
        db_conn.commit()
        logger.debug('delete from text_indic_ann_matches ok.')
    except:
        logger.warning('delete from text_indic_ann_matches failed.')
    
    # insert match results into SQL table
    logger.debug(sw.tick('match'))
    match_df.to_sql('text_indic_ann_matches', db_engine, schema='analytics', index=False, if_exists='append', method='multi')
    logger.debug(sw.tick('to_sql n={}'.format(len(match_df))))
    return df

    
def _OLD_match_musetext_vs_indicator_example(txt_df, annot_df):
    """ 
    Match single text sentence with single annotation example
    """
    #print("_match_musetext_vs_indicator_example")
    assert len(annot_df) > 0, annot_df
    # get txt sentence id
    sentence_id = txt_df['sentence_id'].tolist()[0]
    # get annotation example id and code
    example_id = annot_df['example_id'].tolist()[0]
    indicator_code = annot_df['indicator_code'].tolist()[0]
    #print("_match_musetext_vs_indicator_example example_id", example_id, indicator_code)

    d = {}
    for cs in [False]: # True, 
        for sw in [True, False]:
            match_vars_d = None
            #match_vars_d = _match_tokens(txt_df, annot_df, cs, sw)
            if match_vars_d is not None:
                d.update(match_vars_d)
    
    # sum all values
    all_nums = [x for x in d.values() if _is_number(x)]
    d['all_sum_n'] = sum(all_nums)
    
    # base dictionary
    res_d = {'muse_text_sentence_id': [sentence_id], 
             'annotation_example_id': [example_id],
             'annotation_example_code': [indicator_code],
             'muse_text_sentence_len': len(txt_df), 
             'annotation_example_len': len(annot_df),
             'muse_text_sentence_full': ' '.join(txt_df.token.tolist()),
             'muse_ann_sentence_full': ' '.join(annot_df.token.tolist()) 
             }
    res_d.update(d)
    
    res_df = pd.DataFrame(data=res_d)
    assert len(res_df)>0
    return res_df
