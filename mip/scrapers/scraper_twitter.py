# -*- coding: utf-8 -*-

import logging
from re import L
#import twint
logger = logging.getLogger(__name__)


import datetime
#from twython import Twython
#from searchtweets import ResultStream, gen_rule_payload, load_credentials, collect_results
import json
import time
import pandas as pd
import requests
from db.db import open_sqlite, connect_to_postgresql_db, create_alchemy_engine_posgresql
    

"""
Twitter scraper
"""

# load twitter API configuration
with open('.secrets.json') as f:
    twitter_config = json.load(f)

twitter_db_fn = 'tmp/tweets.db'


def create_tweet_dump(db_conn):
    """ create table for Twitter dump """
    c = db_conn.cursor()

    # Create table
    c.execute('''
        CREATE SCHEMA IF NOT EXISTS twitter;
        CREATE EXTENSION IF NOT EXISTS hstore;
        CREATE TABLE IF NOT EXISTS twitter.tweets_dump 
            (tw_id text NOT NULL,
            account text NOT NULL,
            author_id text NOT NULL,
            tweet_text text NOT NULL,
            muse_id text NOT NULL,
            tw_ts timestamptz NOT NULL,
            tweet_data_json json NOT NULL,
            collection_ts timestamptz DEFAULT CURRENT_TIMESTAMP
            );''')
            # , PRIMARY KEY(tw_id, muse_id)) # removed to avoid insert exception
    db_conn.commit()
    print('create_tweet_dump')


def create_twitter_api_headers():
    bearer_token = twitter_config['tw_bearer']
    assert bearer_token
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def query_twitter_api_endpoint(headers, params):
    search_url = "https://api.twitter.com/2/tweets/search/all"
    response = requests.request("GET", search_url, headers=headers, params=params)
    #print(response.status_code)
    if response.status_code != 200:
        raise RuntimeError(response.status_code, response.text)
    return response.json()


def scrape_twitter_accounts(museums_df):
    """ Scrape twitter accounts for all museums """

    def get_twitter_accounts_from_col(x):
        """ extract twitter accounts from string """
        if x == 'no_resource' or 'search?q=' in x:
            return []
        x1 = x.replace("['",'').replace("']",'').replace("', '",',')
        accounts = x1.split(',')
        assert type(accounts) is list
        assert len(accounts) > 0
        accounts = [s.strip().replace('www.twitter.com/','') for s in accounts]
        accounts = [s.strip().replace('twitter.com/','') for s in accounts]
        # remove status
        accounts = [s.split('/')[0] for s in accounts]
        # remove duplicates
        accounts = list(set(accounts))
        return accounts

    no_twitter_mus = pd.DataFrame()
    # open connection to db
    db_engine = create_alchemy_engine_posgresql()
    db_con = connect_to_postgresql_db() #open_sqlite(twitter_db_fn)
    # init table
    create_tweet_dump(db_con)
    min_date = datetime.datetime(2019, 1, 1, 0, 0, 0)
    i = 0
    for idx, mus in museums_df.iterrows():
        i += 1
        print('> museum ', i, 'of', len(museums_df))
        mus_id = mus['museum_id']
        tw_accounts = get_twitter_accounts_from_col(mus['twitter_id'])
        
        # scan museums
        for acc in tw_accounts:
            if has_db_museum_tweets(mus_id, acc, db_con):
                continue
            scrape_twitter_account(mus_id, acc, min_date, db_con)
        if len(tw_accounts) == 0:
            no_twitter_mus = no_twitter_mus.append(mus)
    db_con.close()
    # insert no_twitter_mus into DB
    no_twitter_mus.to_sql('museums_no_twitter', db_engine, schema='twitter', index=False, if_exists='replace', method='multi')


def has_db_museum_tweets(muse_id, user_name, db_con):
    sql = '''select count(*) as cnt from twitter.tweets_dump where muse_id = '{}' and account = '{}';'''.format(muse_id, user_name)
    df = pd.read_sql(sql, db_con)
    cnt = df.cnt[0]
    return cnt > 0


def scrape_twitter_account(muse_id, user_name, min_date, db_con):
    """
    API code based on
    https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/master/Full-Archive-Search/full-archive-search.py
    and 
    https://developer.twitter.com/en/docs/twitter-api/v1/tweets/timelines/api-reference/get-statuses-user_timeline
    """
    assert min_date < datetime.datetime.now()
    # Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
    # expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
    # query_params = {'query': '(from:twitterdev -is:retweet) OR #twitterdev','tweet.fields': 'author_id'}

    # date format: YYYY-MM-DDTHH:mm:ssZ
    # '2019-01-01T00:00:00Z'
    start_time_iso = min_date.astimezone().isoformat()

    keep_querying = True
    next_token = None
    query_params = {'query': 'from:'+user_name+' OR to:'+user_name, 
        'tweet.fields': 'attachments, author_id, conversation_id, created_at, entities, geo, id, in_reply_to_user_id, lang, public_metrics, possibly_sensitive, referenced_tweets, reply_settings, source, text, withheld'.replace(' ',''),
        'expansions': 'attachments.poll_ids, attachments.media_keys, author_id, entities.mentions.username, geo.place_id, in_reply_to_user_id, referenced_tweets.id, referenced_tweets.id.author_id'.replace(' ',''),
        'start_time': start_time_iso,
        'max_results': 450
        #'user.fields': 'created_at, description, entities, id, location, name, pinned_tweet_id, profile_image_url, protected, public_metrics, url, username, verified, withheld'.replace(' ',''),
        #'place.fields': 'contained_within, country, country_code, full_name, geo, id, name, place_type'.replace(' ',''),
    }
    # NOTE 'context_annotations' was removed because of the low tweet limit.
    
    headers = create_twitter_api_headers()
    found_tweets = 0
    PAUSE_SECS = 2.0

    json_results = []
    while keep_querying:
        if next_token:
            query_params['next_token'] = next_token
        
        json_response = query_twitter_api_endpoint(headers, query_params)
        time.sleep(PAUSE_SECS)
        if not 'data' in json_response:
            print('warning: no data found for', muse_id, user_name, str(json_response))
            break
        n_tweets = len(json_response['data'])
        found_tweets += n_tweets
        print('\tn_tweets',n_tweets,'; found_tweets',found_tweets)
        if 'meta' in json_response and 'next_token' in json_response['meta']:
            # next token found
            next_token = json_response['meta']['next_token']
        else:
            # next token not found, stop
            keep_querying = False
        
        json_results.append(json_response['data'])
    
    # save data
    for j in json_results:
        insert_tweets_into_db(j, muse_id, user_name, db_con)


def insert_tweets_into_db(tweets, muse_id, tw_account, db_con):
    """ Insert twitter data into DB """
    assert muse_id
    assert db_con

    if len(tweets) == 0: 
        return
    cur = db_con.cursor()

    for x in tweets:
        # extract fields
        tw_id_str = x['id']
        user_id = x['author_id']
        ts = datetime.datetime.fromisoformat(x['created_at'].replace("Z", "+00:00"))
        json_attr = json.dumps(x)
        # insert sql
        sql = '''INSERT INTO twitter.tweets_dump(tw_id, tw_ts, account, author_id, tweet_text, muse_id, tweet_data_json)
              VALUES(%s,%s,%s,%s,%s,%s,%s);'''
        cur.execute(sql, [tw_id_str, ts, tw_account, user_id, x['text'], muse_id, json_attr])
    
    db_con.commit()