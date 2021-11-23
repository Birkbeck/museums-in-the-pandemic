# -*- coding: utf-8 -*-

import pandas as pd
import pickle
import itertools
import re
import sys
from termcolor import colored
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
import sqlite3
from ipywidgets import widgets
from ipywidgets import interact, interactive, fixed, interact_manual

def open_local_db():
    conn = sqlite3.connect('mip_corpus_search_test.db')
    return conn

def run_search(text):
    sql = 'select count(*) from social_media_msg;'
    df = pd.read_sql(sql, db_conn)
    return df
    
# MAIN
db_conn = open_local_db()
assert db_conn, 'db not connected'
print('ok')