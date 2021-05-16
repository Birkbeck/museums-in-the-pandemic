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


def setup_ling_model():
    """ 
    - https://towardsdatascience.com/deep-learning-for-specific-information-extraction-from-unstructured-texts-12c5b9dceada
    - https://towardsdatascience.com/text-classification-in-python-dd95d264c802
    - supervised machine learning classification model
    - HuggingFace package: 
        https://github.com/huggingface/transformers#Migrating-from-pytorch-pretrained-bert-to-pytorch-transformers
    """
    logger.debug('setup_ling_model')
    indic_df, annot_df = get_indicator_annotations()
    
    #train_ds, dicts = load_ds(os.path.join(data_dir,'atis.train.pkl'))
    #test_ds, dicts  = load_ds(os.path.join(data_dir,'atis.test.pkl'))
    
    logger.debug("end of NLP")

def bert_model():
    """
    - https://towardsdatascience.com/bert-for-dummies-step-by-step-tutorial-fb90890ffe03
    - https://towardsdatascience.com/first-time-using-and-fine-tuning-the-bert-framework-for-classification-799def68a5e4
    """
    pass

