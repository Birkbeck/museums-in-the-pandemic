# -*- coding: utf-8 -*-

"""
Linguistic models for text analytics

"""
import logging
logger = logging.getLogger(__name__)

from db.db import open_sqlite, run_select_sql
import pandas as pd
import pickle
import os
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
from bs4.element import Comment
import re
from utils import remove_empty_elem_from_list, remove_multiple_spaces_tabs
import matplotlib.pyplot as plt


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

def get_indicator_annotations():
    """  """
    in_fn = "data/annotations/indicators_and_annotations-v3.xlsx"
    indic_df = pd.read_excel(in_fn,0)
    ann_df = pd.read_excel(in_fn,1)
    # select first 4 columns
    ann_df = ann_df.iloc[:, : 4]
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
    
    #train_ds, dicts = load_ds(os.path.join(data_dir,'atis.train.pkl'))
    #test_ds, dicts  = load_ds(os.path.join(data_dir,'atis.test.pkl'))
  
    #bert_model()
    logger.debug("end of NLP")

def bert_model():
    """
    - https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03
    - https://towardsdatascience.com/first-time-using-and-fine-tuning-the-bert-framework-for-classification-799def68a5e4
    """
    pass
