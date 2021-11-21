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
from datetime import datetime
import nltk
from ipywidgets import widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.core.display import display, HTML

# set up NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = list(stopwords.words('english'))
stop_words.extend([".",",",";","'",'//',"'s"])
stop_words = set(stop_words)

def open_local_db():
  conn = sqlite3.connect('mip_corpus_search-test2.db')
  return conn

def filter_search_string_for_sql(text):
  text = text.replace('*','%')
  return text

def filter_search_string_for_regex(text, case_sensitive):
  text = text.replace('*','.*')
  if not case_sensitive:
    text = "(?i)" + text 
  return text

def run_search(text, case_sensitive, search_facebook, search_twitter,
  search_websites):
  assert len(text) > 3, 'search string too short!'
  assert search_facebook or search_twitter or search_websites, 'select at least one platform'
  where = ''
  platforms = []
  if search_websites: platforms.append('website')
  if search_facebook or search_twitter:
    # search social media
    if search_facebook: platforms.append('facebook')
    if search_twitter: platforms.append('twitter')
    where = ','.join(["'"+x+"'" for x in platforms])
    sql = "select * from social_media_msg where platform in ({}) and msg_text like '%{}%';".format(where, 
      filter_search_string_for_sql(text))

    #print(sql)
    df = pd.read_sql(sql, db_conn)
    print('N results:',len(df))
  
  return df

def get_before_after_strings(s, regex, context_size_words):
  #print('get_before_after_strings:', s)
  mm = re.finditer(regex, s)
  sep = ' '
  for m in mm:
      beg, end = m.span()
      bef = s[0:beg]
      aft = s[end+1:-1]
      bef_words = bef.split(sep)
      #print(bef_words)
      if len(bef_words) > context_size_words:
        bef_words = bef_words[-context_size_words:]
      aft_words = aft.split(sep)
      if len(aft_words) > context_size_words:
        aft_words = aft_words[0:context_size_words]
      assert len(bef_words) <= context_size_words
      assert len(aft_words) <= context_size_words
      #print(bef_words, aft_words)
      return bef_words, m.group(), aft_words

def get_now_string():
  now = datetime.now()
  current_time = now.isoformat()
  current_time = now.strftime("%Y%m%d-%H%M%S")
  return current_time

def generate_html_matches(df, search_string, case_sensitive, context_size_words, max_results):
  """
  <tr>
    <td>Alfreds Futterkiste</td>
    <td>Maria Anders</td>
    <td>Germany</td>
  </tr>
  <tr>
    <td>Centro comercial Moctezuma</td>
    <td>Francisco Chang</td>
    <td>Mexico</td>
  </tr>
  """
  css = """<style>
  table, tr, th, td {
    margin: 3px;
    border: solid 1px gray;
    font-size: .9em;
    border-collapse: collapse;
    font-family: sans-serif;
    vertical-align: top;
    }
  .before_col { text-align: right; }
  strong { background-color: blue; color: white; }
  </style>"""
  j = 0
  results_page_d = []
  search_regex = filter_search_string_for_regex(search_string, case_sensitive)
  print("search_regex:",search_regex)
  
  for i, r in df.iterrows():
    j += 1
    bef_words, match_text, aft_words = get_before_after_strings(r['msg_text'], search_regex, context_size_words)
    results_page_d.append({'res':j, 'museum_id':r['museum_id'], 
      'before':' '.join(bef_words), 'match': match_text,
      'after':' '.join(aft_words), 'platform':r['platform'] })
  results_page_df = pd.DataFrame(results_page_d)

  header = "<tr>" + ''.join(["<th>{}</th>".format(x) for x in results_page_df.columns]) + "</tr>"
  table_rows_h = ''
  j = 0
  for idx, row in results_page_df.iterrows():
    j += 1
    if j > max_results: break
    #for m in mm:
    #repl_str = '<strong>{}</strong>'.format(match_text)
    #html_m = html_m.replace(match_text, repl_str)
    row_h = ''
    for c in results_page_df.columns:
      css_class = ''
      if 'before' in c:
        css_class='before_col'
      row_h += '<td class="{}">{}</td>'.format(css_class, row[c])
    table_rows_h += "<tr>{}</tr>".format(row_h)
  h = css + """<table>""" + header + table_rows_h + "</table>"
  return h, results_page_df

def filter_tokens(tokens):
  filt_tokens = [w.strip() for w in tokens]
  filt_tokens = [w for w in filt_tokens if not w.lower() in stop_words]
  filt_tokens = [w for w in filt_tokens if len(w)>1]
  return filt_tokens

def clean_text(txt):
  """Remove links"""
  link_regex = r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)"
  clean_txt = re.sub(link_regex, ' ', txt)
  #print("1",txt)
  #print("2",clean_txt)
  return clean_txt

def an_results(df, search_string, case_sensitive, context_size):
  assert context_size > 0 and context_size <= 10, "context_size is too big/small" 
  assert len(search_string) > 1, search_string
  print('N results: ',len(df))
  print('N museums: ',df.museum_id.nunique())
  print('Platforms:\n',df.platform.value_counts())
  
  search_regex = filter_search_string_for_regex(search_string, case_sensitive)
  before_tokens = []
  after_tokens = []
  # extract context
  for i, r in df.iterrows():
    # TODO find context_size
    txt = clean_text(r['msg_text'].strip())
    match = re.search(search_regex, txt, re.MULTILINE)
    if match:
      beg, end = match.span()
      before = txt[0:beg].lower()
      before_tokens.extend(word_tokenize(before))
      after = txt[end:-1].lower()
      after_tokens.extend(word_tokenize(after))
    #print(after, "MATCH", before)
  
  before_tokens = filter_tokens(before_tokens)
  after_tokens = filter_tokens(after_tokens)
  print('before\n\n')
  print(pd.Series(before_tokens).value_counts())
  print('after\n\n')
  print(pd.Series(after_tokens).value_counts())
  
# MAIN
db_conn = open_local_db()
assert db_conn, 'db not connected'
print('ok')