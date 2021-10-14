# -*- coding: utf-8 -*-
import pandas as pd
import logging
import ast
import json
import datetime
import time
from db.db import open_sqlite, run_select_sql
from db.db import connect_to_postgresql_db, create_alchemy_engine_posgresql
from pandas.io.json._normalize import nested_to_record
from vpn import vpn_random_region
import requests
from utils import flatten_dict
from facebook_scraper import get_posts
logger = logging.getLogger(__name__)

"""
Facebook scraper based on CrowdTangle
"""

API_PAUSE_SECS = 12.1

# load API configuration
with open('.secrets.json') as f:
    config = json.load(f)
    crowdtangle_api_key = config['ct_key']

def get_facebook_pages_from_col(x):
    """ extract facebook pages from string """

    def remove_category(s):
        import re
        # pages/category/College---University/
        clean_s = re.sub(r"pages/category/[^\/]+/", '', s)
        clean_s = re.sub(r"category/[^\/]+/", '', clean_s)
        return clean_s

    if x == 'no_resource' or 'search?q=' in x or 'redirect.html' in x:
        return []
    x1 = x.replace("['",'').replace("']",'').replace("', '",',')
    accounts = x1.split(',')
    assert type(accounts) is list
    assert len(accounts) > 0
    accounts = [s.lower() for s in accounts]
    for expr in ['www.facebook.com/events/','www.facebook.com/pages/','en-gb.facebook.com/events/',
                'www.facebook.com/groups/','www.facebook.com/pg/',
                'en-gb.facebook.com/pages/','en-gb.facebook.com/',
                'www.facebook.com/','facebook.com/']:
        accounts = [s.strip().replace(expr.lower(),'') for s in accounts]    
    
    accounts = [remove_category(s) for s in accounts]
    # remove /status/ and parameters
    accounts = [s.split('/')[0] for s in accounts]
    accounts = [s.split('?')[0] for s in accounts]
    accounts = [x for x in accounts if x]
    # remove duplicates
    accounts = list(set(accounts))
    #print('\n',x,'\n\t',accounts)
    for a in accounts:
        assert not '/' in a, a
        assert not a in ['pages','photos','reviews','posts','about','category','pg','groups','events'], a
    return accounts


def scrape_facebook(museums_df):
    print("scrape_facebook")
    db_con = connect_to_postgresql_db()
    db_engine = create_alchemy_engine_posgresql()
    create_fb_dump(db_con)

    scraped_posts = 0
    i = 0
    #museums_df = museums_df.sample(len(museums_df)) # shuffle
    for idx, mus in museums_df.iterrows():
        i+=1
        print(">", i, 'of', len(museums_df), ' -- ', mus['museum_id'])
        pages = get_facebook_pages_from_col(mus['facebook_pages'])
        for p in pages:
            n = scrape_facebook_page(p, mus['museum_id'], db_con, db_engine)
            scraped_posts += n

    print("scraped_pages =", scraped_posts)


def get_earliest_date(fbdata):
    times = [p['time'] for p in fbdata]
    min_t = min(times)
    return min_t


def scrape_facebook_page(page_name, muse_id, db_conn, db_engine):
    """
    Docs: https://github.com/CrowdTangle/API/wiki/Posts
    """

    if fb_page_exists_in_db(muse_id, page_name, db_conn):
        print(page_name,'already in local DB.')
        return 0

    date_blocks = ['2019-01-01','2020-01-01', '2021-01-01','2022-01-01']
    posts = []
    for i in range(len(date_blocks)-1):
        start_date = date_blocks[i]
        end_date = date_blocks[i+1]
        posts.extend(query_crowdtangle(page_name, start_date, end_date, db_engine))
        #if len(posts)==0:
        #    break
    
    if len(posts) == 0:
        print('warning: no posts found for ',page_name)
        pd.DataFrame({'museum_id':[muse_id], 'page_name':[page_name]}).to_sql('facebook_pages_not_found', db_engine, 
            schema='facebook', index=False, if_exists='append', method='multi')
        return 0
    # build data frame
    flat_posts = []
    for p in posts:
        flatp = flatten_dict(p)
        flatp['museum_id'] = muse_id
        flatp['query_account'] = page_name
        flat_posts.append(flatp)
    
    insert_fb_data(flat_posts, db_conn)
    #posts_df.to_sql('facebook_posts', db_engine, schema='facebook', index=False, if_exists='append', method='multi')
    return len(flat_posts)


def query_crowdtangle(account, start_date, end_date, db_engine):
    print('\tquery_crowdtangle',account, start_date, end_date)
    assert account
    assert db_engine
    base_url = 'https://api.crowdtangle.com/posts'
    do_next_token = True
    next_url = None
    all_posts = []
    while do_next_token:
        headers = {}
        params = {'token': crowdtangle_api_key, 'accounts': [account], 'count': 100,
            'startDate': start_date, 
            'endDate': end_date}
        # query crowdtangle
        # 6 queries per minute
        time.sleep(API_PAUSE_SECS)
        if not next_url:
            response = requests.request("GET", base_url, headers=headers, params=params)
        else:
            response = requests.request("GET", next_url)
            next_url = None
        
        if response.status_code == 200 and response.ok:
            res = json.loads(response.text)
            if 'code' in res and res['code'] == 40:
                # account not found
                print('warning: account "',account,'" not found')
                return all_posts
            res = json.loads(response.content)
            for p in res['result']['posts']:
                all_posts.append(p)
            
            print('\t\tposts =',len(res['result']['posts']), ' tot =',len(all_posts))
            if 'pagination' in res['result'] and 'nextPage' in res['result']['pagination']:
                next_url = res['result']['pagination']['nextPage']
                do_next_token = True
            else:
                # end of cycle
                do_next_token = False
        else: 
            raise RuntimeError(response.text)
    
    return all_posts


def scrape_facebook_page_OLD(page_name, muse_id, date_limit, db_conn, db_engine):
    """ 
    Scrape posts from facebook page in @url, going back at least to @date_limit
    @returns True if page was scraped or False if the page was already present in the DB
    """
    assert False
    assert page_name
    assert muse_id

    #if page_exists_in_db(page_name, db_conn):
        # skip page
        #return False

    limit = page_limit
    while True:
        time.sleep(.1)
        try:
            # scrape fb (slow)
            print('> scrape_facebook_page',page_name,'...')
            posts_iter = get_posts(page_name, pages=page_limit, timeout=10, options={"posts_per_page": 200},
                credentials=('',''))
            posts = []
            for p in posts_iter:
                print('.', end='')
                #print(p)
                posts.append(p)
            print('\tposts',len(posts))
            min_date = get_earliest_date(posts)
            
            if min_date > date_limit:
                msg = "Too few posts, increasing limit - earliest date="+ str(min_date)
                logger.warn(msg)
                print(msg)
                # too few posts, double limit
                limit = limit * 2
                if limit > max_limit:
                    # save posts to DB and move on
                    logger.warn("avoid infinite loop, saving and skipping: "+page_name)
                    print(page_name, "posts n", len(posts))
                    insert_fb_data(page_name, posts, muse_id, db_conn, db_engine)
                    return True

                logger.debug("too few posts, increase limit to "+str(limit))
                time.sleep(1)
                continue
            
            # all good, save posts to DB and move on
            msg = page_name + " done: posts n " + str(len(posts)) + '; earliest_date_found=' + str(min_date)
            print(msg)
            logger.debug(msg)
            insert_fb_data(page_name, posts, muse_id, db_conn)
            return True
        
        except Exception as e:
            logger.warning("error while scraping Facebook, changing VPN")
            raise e
            continue_scraping = True
            # TODO change VPN
            time.sleep(2)


def fb_page_exists_in_db(museum_id, page_name, db_con):
    """ True if page already exists in Facebook dump table """
    try:
        sql = "select count(*) as page_posts_n from facebook.facebook_posts_dump where museum_id = '{}' and query_account = '{}';".format(museum_id, page_name)
        df = run_select_sql(sql, db_con)
    except Exception as e:
        print('warning',str(e))
        return False
    
    val =  df.page_posts_n.tolist()[0]
    if val > 0: 
        return True

    # check missing pages
    try:
        sql = "select count(*) as page_posts_n from facebook.facebook_pages_not_found where museum_id = '{}' and page_name = '{}';".format(museum_id, page_name)
        df = run_select_sql(sql, db_con)
    except Exception as e:
        print('warning',str(e))
        return False
    val =  df.page_posts_n.tolist()[0]
    if val > 0:
        # page found but no data
        return True
    return False


def create_fb_dump(db_conn):
    """ create table for Facebook dump """
    c = db_conn.cursor()
    # Create table
    c.execute('''CREATE SCHEMA IF NOT EXISTS facebook;
        CREATE TABLE IF NOT EXISTS facebook.facebook_posts_dump 
            (
            museum_id text NOT NULL,
            query_account text NOT NULL,
            page_name text,
            post_id text NOT NULL,
            post_text text,
            user_id text,
            post_ts timestamptz NOT NULL,
            facebook_data_json JSON NOT NULL,
            collection_ts timestamptz DEFAULT CURRENT_TIMESTAMP)
            ;
        ''')
    db_conn.commit()
    print('create_facebook_dump')


def insert_fb_data(posts, db_conn):
    """ Insert posts from @fbdata list into Facebook dump table """
    '''
    page_name
    post_id
    museum_id
    user_id
    post_text
    post_ts
    facebook_data_json
    '''
    cur = db_conn.cursor()
    for x in posts:
        # get attributes
        ts = datetime.datetime.fromisoformat(x['date'].replace("Z", "+00:00"))
        json_attr = json.dumps(x)
        msg = None
        account_handle = None
        if 'message' in x:
            msg = x['message']
        if 'account_handle' in x:
            account = x['account_handle']
        else: 
            account = None
        # insert sql
        sql = '''INSERT INTO facebook.facebook_posts_dump(page_name, query_account, post_id, museum_id, user_id, post_text, post_ts, facebook_data_json)
              VALUES(%s,%s,%s,%s,%s,%s,%s,%s);'''
        cur.execute(sql, [account, x['query_account'], x['id'], x['museum_id'], x['account_id'], msg, ts, json_attr])
    
    db_conn.commit()
