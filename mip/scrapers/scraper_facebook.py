# -*- coding: utf-8 -*-
import pandas as pd
import logging
import ast
import json
import datetime
import time
from db.db import open_sqlite, run_select_sql
from db.db import connect_to_postgresql_db, create_alchemy_engine_posgresql
from vpn import vpn_random_region
from facebook_scraper import get_posts
logger = logging.getLogger(__name__)

"""
Facebook scraper based on facebook_scraper
"""

# initial page limit for facebook_scraper.get_posts
page_limit = 100
# to avoid infinite loop
max_limit = 500

def scrape_facebook(museums_df):
    # TODO: implement for all museum data
    print("scrape_facebook","page_limit =",page_limit)
    db_con = connect_to_postgresql_db()
    db_engine = create_alchemy_engine_posgresql()
    
    
    date_limit = datetime.datetime(2019, 1, 1)
    pagedf=pd.read_excel('tmp/fb_urls_final.xlsx')


    
    scraped_pages = 0
    for row in pagedf.iterrows():
        print(row[1].url)
        if row[1].url[0]=='[':
            listbug=ast.literal_eval(row[1].url)
            for item in listbug:
                b = scrape_facebook_page(item, row[1].museum_id, date_limit, db_con, db_engine)
                if b:
                    scraped_pages += 1 
        else:
            b = scrape_facebook_page(row[1].url, row[1].museum_id, date_limit, db_con, db_engine)
            if b:
                scraped_pages += 1

    print("scraped_pages =", scraped_pages)


def get_earliest_date(fbdata):
    i = 0
    times = [p['time'] for p in fbdata]
    min_t = min(times)
    return min_t


def scrape_facebook_page(url, muse_id, date_limit, db_conn, db_engine):
    """ 
    Scrape posts from facebook page in @url, going back at least to @date_limit
    @returns True if page was scraped or False if the page was already present in the DB
    """
    urllist = url.split("/")
    page_name = urllist[1]
    assert page_name
    assert muse_id

    #if page_exists_in_db(page_name, db_conn):
        # skip page
        #return False

    limit = page_limit
    while True:
        time.sleep(.1)
        try:
            msg = page_name+' limit='+str(limit)+' ...'
            logger.debug(msg)
            print('\t'+msg)
            # scrape fb (slow)
            posts = get_posts(page_name, pages=page_limit)
            posts = [p for p in posts]
            
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


def page_exists_in_db(page_name, db_con):
    """ True if page already exists in Facebook dump table """
    sql = "select count(page_name) as page_posts_n from facebook_dump where page_name = '{}';".format(page_name)
    df = run_select_sql(sql, db_con)
    val =  df.page_posts_n.tolist()[0]
    if val > 0: 
        return True
    return False


def create_fb_dump(db_conn):
    """ create table for Facebook dump """
    c = db_conn.cursor()

    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS facebook_dump 
            (page_name text NOT NULL,
            post_id text PRIMARY KEY,
            muse_id text NOT NULL,
            user_id text,
            post_ts DATETIME NOT NULL,
            facebook_data_json text NOT NULL,
            collection_ts DATETIME DEFAULT CURRENT_TIMESTAMP)
            ;
            ''')
    db_conn.commit()
    print('create_facebook_dump')


def insert_fb_data(page_name, fbdata, muse_id, db_con, db_engine):
    """ Insert posts from @fbdata list into Facebook dump table """
    assert muse_id
    
    assert db_con
    if len(fbdata) == 0: 
        return

    #cur = db_con.cursor()
    done_ids = []
    for x in fbdata:
        # extract fields
        post_id_str = x['post_id']
        if post_id_str in done_ids: 
            # repeated post, skip it
            logger.debug("repeated post, skip it")
            continue

        user_id = x['user_id']
        ts = x['time']
        x['time'] = ts.isoformat()
        json_attr = json.dumps(x)
        # insert sql
        data = {'page_name': [page_name],'post_id':[post_id_str],'user_id':[user_id],'muse_id':[muse_id],'post_ts':[ts],'facebook_data_json':[json_attr]}
 
        # Create the pandas DataFrame
        fb_df = pd.DataFrame(data, columns = ['page_name', 'post_id','user_id','muse_id','post_ts','facebook_data_json' ])
        
        fb_df.to_sql('facebook_messages', db_engine, schema='facebook', index=False, if_exists='append', method='multi')
        done_ids.append(post_id_str)
    #db_con.commit()
