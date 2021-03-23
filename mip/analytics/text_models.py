# -*- coding: utf-8 -*-

"""
Linguistic models for text analytics


"""

from db.db import open_sqlite, run_select_sql
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Comment
import re
from utils import remove_empty_elem_from_list, remove_multiple_spaces_tabs
import logging
logger = logging.getLogger(__name__)


def get_indicator_annotations():
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
    """
    logger.debug('setup_ling_model')
    indic_df, annot_df = get_indicator_annotations()


def bert_model():
    """
    - https://towardsdatascience.com/first-time-using-and-fine-tuning-the-bert-framework-for-classification-799def68a5e4
    """
    pass

