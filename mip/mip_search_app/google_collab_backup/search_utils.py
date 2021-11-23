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
  #conn = sqlite3.connect('mip_corpus_search-test2.db')
  conn = sqlite3.connect('mip_corpus_search.db')
  return conn

def filter_search_string_for_sql(text):
  text = text.replace('*','%')
  return text

def filter_search_string_for_regex(text, case_sensitive):
  text = text.replace('*','.*')
  text = r"\b{}\b".format(text)
  if not case_sensitive:
    text = r"(?i)" + text
  return text

def run_search(text, case_sensitive, search_facebook, search_twitter,
  search_websites, search_website_sentences):
  assert len(text) > 3, 'search string too short!'
  assert search_facebook or search_twitter or search_websites or search_website_sentences, 'select at least one platform'
  if search_websites and search_website_sentences:
    raise Exception('select either search_websites or search_website_sentences, not both')
  where = ''
  platforms = []
  web_df = pd.DataFrame()
  soc_df = pd.DataFrame()
  # search websites
  if search_websites:
    # unused option
    sql = "select * from websites_text where page_text like '%{}%';".format(filter_search_string_for_sql(text))
    web_df = pd.read_sql(sql, db_conn)
    if len(web_df) > 0:
      n_u_museums = web_df.museum_id.nunique()
      web_df['platform'] = 'website'
      web_df['session_time'] = web_df['session_id'].apply(sessionid_to_time)
      print('Websites: {} matches found. Unique museums: {} - N sessions: {}'.format(len(web_df), n_u_museums, web_df.session_id.nunique()))
    else: 
      print("Websites: no matches found.") 
  
  if search_website_sentences:
    sql = "select * from websites_sentences_text where sentence_text like '%{}%';".format(filter_search_string_for_sql(text))
    web_df = pd.read_sql(sql, db_conn)
    if len(web_df) > 0:
      n_u_museums = web_df.museum_id.nunique()
      web_df['platform'] = 'website_sentences'
      web_df['session_time'] = web_df['session_id'].apply(sessionid_to_time)
      print('WEBSITES: {} matches found. N sessions: {}. Unique museums: {}'.format(len(web_df), web_df.session_id.nunique(), n_u_museums))
    else:
      print('WEBSITES: no matches found.')
  
  # search social media
  if search_facebook or search_twitter:
    if search_facebook: platforms.append('facebook')
    if search_twitter: platforms.append('twitter')
    where = ','.join(["'"+x+"'" for x in platforms])
    sql = "select * from social_media_msg where platform in ({}) and msg_text like '%{}%';".format(where, 
      filter_search_string_for_sql(text))
    
    soc_df = pd.read_sql(sql, db_conn)
    if len(soc_df) > 0:
      n_u_museums = soc_df.museum_id.nunique()
      for pname, subdf in soc_df.groupby('platform'):
        print('{}: {} matches found. Unique museums: {}'.format(pname.upper(), len(subdf), subdf.museum_id.nunique()))
    else: 
      print("TWITTER/FACEBOOK: no matches found.")

  df = merge_results(web_df, soc_df)
  return df

def get_before_after_strings(s, regex, context_size_words):
  #print('get_before_after_strings:', regex)
  assert len(regex) > 1
  assert len(s) > 0
  sep = ' '
  assert context_size_words > 0 and context_size_words < 1000
  for m in re.finditer(regex, s):
      beg, end = m.start(), m.end()
      #print(m, beg, end)
      assert beg >= 0
      assert end >= 0
      bef = s[0:beg]
      aft = s[end:-1]
      
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
      assert len(bef_words) >= 0
      assert len(aft_words) >= 0
      match_str = m.group()#.strip()
      #print('match_str',match_str)
      assert match_str
      #print(bef_words, match_str, aft_words)
      return bef_words, match_str, aft_words
  # nothing found
  return None, None, None

def get_now_string():
  now = datetime.now()
  current_time = now.isoformat()
  current_time = now.strftime("%Y%m%d-%H%M%S")
  return current_time

def sessionid_to_time(session_id):
  dd = datetime.strptime(session_id, '%Y%m%d')
  return dd

def merge_results(web_df, soc_df):
  #print('merge_results')
  #print(soc_df.columns)
  #print(web_df.columns)
  # Index(['museum_id', 'account', 'msg_text', 'msg_time', 'platform'], dtype='object')
  # Index(['museum_id', 'museum_name', 'session_id', 'page_id', 'url', 'sentence_id', 'sentence_text']
  if len(web_df)>0:
    #web_df = web_df.rename(columns={'msg_text':'sentence_text','msg_time':'session_id'})
    if 'sentence_text' in web_df.columns: 
      web_df['msg_text'] = web_df['sentence_text']
    if 'page_text' in web_df.columns:
      web_df['msg_text'] = web_df['page_text']
    if 'session_id' in web_df.columns:
      web_df['msg_time'] = web_df['session_time']
  
  df = pd.concat([web_df,soc_df], axis=0, ignore_index=True)
  return df

def msg_time_to_string(t):
  if not isinstance(t, str):
    t = str(t)
  res = t[0:10]
  return res

def generate_html_matches(res_df, search_string, case_sensitive, context_size_words, max_results):
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
  # ==== generate results ====
  results_page_d = []
  search_regex = filter_search_string_for_regex(search_string, case_sensitive)
  print("search_regex: '{}'".format(search_regex))
  #df = merge_results(soc_df, web_df)
  for nn, subdf in res_df.groupby('platform'):
    j = 0
    for i, r in subdf.iterrows():
      j += 1
      msg_txt = r['msg_text'].strip()
      assert len(msg_txt) > 0
      bef_words, match_text, aft_words = get_before_after_strings(msg_txt, search_regex, context_size_words)
      if match_text is None: continue 
      if 'account' not in r:
        r['account'] = ''
      results_page_d.append({'res':j, 'museum_id':r['museum_id'], 'account': r['account'],
        'before':' '.join(bef_words), 'match': match_text, 'msg_time':msg_time_to_string(r.msg_time), #.str.slice(0,10)
        'after':' '.join(aft_words), 'platform':r['platform'] })
  results_page_df = pd.DataFrame(results_page_d)
  # sort results
  results_page_df = results_page_df.sort_values(['msg_time','museum_id'],ascending=False)

  # ==== generate HTML from results ====
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
    .match_col { font-weight: bold; color: blue; }
    strong { background-color: blue; color: white; }
    </style>"""
  
  columns_table_html_web = ['res','museum_id','before','match','after','msg_time']
  columns_table_html_social = ['res','museum_id','account','before','match','after','msg_time']
  
  # loop over website, facebook, twitter
  h = css
  for plat_name, subdf in results_page_df.groupby('platform'):
    j = 0
    # select columns
    if plat_name in ["twitter",'facebook']: 
      columns_table_html = columns_table_html_social
    else: columns_table_html = columns_table_html_web
    # header
    header = "<tr>" + ''.join(["<th>{}</th>".format(x) for x in columns_table_html]) + "</tr>"
    # rows
    table_rows_h = ''
    for idx, row in subdf.iterrows():
      j += 1
      if j > max_results: break
      row_h = ''
      for c in columns_table_html:
        css_class = ''
        if 'before' in c:
          css_class='before_col'
        if 'match' in c:
          css_class='match_col'
        row_h += '<td class="{}">{}</td>'.format(css_class, row[c])
      table_rows_h += "<tr>{}</tr>".format(row_h)
    h += "<h3>{} (first {})</h3><table>{}{}</table>".format(plat_name, max_results, header, table_rows_h)
  return h, results_page_df

def filter_tokens(tokens):
  filt_tokens = [w.strip() for w in tokens]
  filt_tokens = [w for w in filt_tokens if not w.lower() in stop_words]
  filt_tokens = [w for w in filt_tokens if len(w)>1]
  return filt_tokens

def generate_derived_attributes_muse_df(df):
    print("generate_derived_attributes_muse_df")
    df['governance_simpl'] = df['governance'].str.split(':').str[0].str.lower()
    df['subject_matter_simpl'] = df['subject_matter'].str.split(':').str[0]
    df['country'] = df['admin_area'].str.split('/').str[1]
    df['region'] = df['admin_area'].str.split('/').str[2]
    df['region'] = np.where(df['country'] == 'England', df['region'], df['country'])
    df['region'] = df['region'].str.replace('\(English Region\)','')
    return df

def clean_text(txt):
  """Remove links"""
  link_regex = r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)"
  clean_txt = re.sub(link_regex, ' ', txt)
  #print("1",txt)
  #print("2",clean_txt)
  return clean_txt

def load_museum_attr():
  fn = 'museums_wattributes-2020-02-23.tsv'
  df = pd.read_csv(fn, sep='\t')
  #print(df.columns)
    # remove closed museums
  df = df.rename(columns={'muse_id':'museum_id'})
  df = df[df.closing_date.str.lower() == 'still open']
  df = generate_derived_attributes_muse_df(df)
  #print(df.columns)
  return df

def an_results(df, search_string, case_sensitive, context_size):
  assert context_size > 0 and context_size <= 10, "context_size is too big/small" 
  assert len(search_string) > 1, search_string
  print('N results: {} • N unique museums: {}'.format(len(df),df.museum_id.nunique()))
  print(df.platform.value_counts().to_frame('n_results'))
  
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
  
  display(HTML("<h3>Tokens before '{}'</h3>".format(search_string)))
  limit_tokens = 20
  before_df = pd.Series(before_tokens).value_counts().to_frame('occurrences')
  j = 0
  for i, row in before_df.head(limit_tokens).iterrows():
    j += 1
    print(i, "({})".format(row['occurrences']), end=' ')
    if j % 7 == 0: print()

  display(HTML("<h3>Tokens after '{}'</h3>".format(search_string)))
  after_df = pd.Series(after_tokens).value_counts().to_frame('occurrences')
  j = 0
  for i, row in after_df.head(limit_tokens).iterrows():
    j += 1
    print(i, "({})".format(row['occurrences']), end=' ') 
    if j % 7 == 0: print()
  print('\n')

  # ==== analyse result attributes ====
  res_attr_df = df.merge(mus_attr_df, on='museum_id', how='left')
  res_attr_df = res_attr_df[['museum_id','museum_name','platform','governance','country','region','size','governance_simpl','subject_matter_simpl']]
  res_attr_df = res_attr_df.drop_duplicates()
  res_stats_df = []
  print("Unique museum results:",len(res_attr_df))
  for attr in ['governance','region','size','subject_matter_simpl']:
    mus_counts_df = res_attr_df.groupby(attr).size().to_frame('n_museums').reset_index()
    mus_counts_df['attribute'] = attr
    tot_mus_df = mus_attr_df.groupby(attr).size().to_frame('n_tot_museums').reset_index()
    mus_counts_df = mus_counts_df.merge(tot_mus_df, on=attr, how='left')
    mus_counts_df['museum_attribute_pc'] = round(mus_counts_df['n_museums'] / mus_counts_df['n_tot_museums'] * 100,1)
    mus_counts_df['museum_result_pc'] = round(mus_counts_df['n_museums'] / len(res_attr_df) * 100,1)
    mus_counts_df = mus_counts_df.rename(columns={attr:'attribute_value'})
    mus_counts_df = mus_counts_df.sort_values('museum_result_pc', ascending=False)
    
    res_stats_df.append(mus_counts_df)
  # combine results
  res_stats_df = pd.concat(res_stats_df, ignore_index=True)
  res_stats_df = res_stats_df[['attribute','attribute_value','n_museums','museum_result_pc',
    'n_tot_museums','museum_attribute_pc']]
  for nm, df in res_stats_df.groupby('attribute'):
    display(HTML("<h3>By {}</h3>".format(nm)))
    display(df.drop(columns=['attribute']))
  return res_stats_df
  
# MAIN
db_conn = open_local_db()
assert db_conn, 'db not connected'

mus_attr_df = load_museum_attr()

print('ok')
