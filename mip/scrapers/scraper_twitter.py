# -*- coding: utf-8 -*-

import logging
#import twint
logger = logging.getLogger(__name__)


import datetime
#from twython import Twython
#from searchtweets import ResultStream, gen_rule_payload, load_credentials, collect_results
import json
import time
import requests
from db.db import open_sqlite, connect_to_postgresql_db

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
            (tw_id text PRIMARY KEY,
            author_id text NOT NULL,
            tweet_text text NOT NULL,
            muse_id text NOT NULL,
            tw_ts timestamptz NOT NULL,
            tweet_data_json json NOT NULL,
            collection_ts timestamptz DEFAULT CURRENT_TIMESTAMP)
            ;
            ''')
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
    db_con = connect_to_postgresql_db() #open_sqlite(twitter_db_fn)
    create_tweet_dump(db_con)
    accounts = ['adlingtonhall']
    # set date threshold
    min_date = datetime.datetime(2019, 1, 1, 0, 0, 0)
    scrape_twitter_account('TODO_my_museum_id', accounts[0], min_date, db_con)
    db_con.close()

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
        'tweet.fields': 'attachments, author_id, context_annotations, conversation_id, created_at, entities, geo, id, in_reply_to_user_id, lang, public_metrics, possibly_sensitive, referenced_tweets, reply_settings, source, text, withheld'.replace(' ',''),
        'expansions': 'attachments.poll_ids, attachments.media_keys, author_id, entities.mentions.username, geo.place_id, in_reply_to_user_id, referenced_tweets.id, referenced_tweets.id.author_id'.replace(' ',''),
        'start_time': start_time_iso,
        'max_results': 500
        #'user.fields': 'created_at, description, entities, id, location, name, pinned_tweet_id, profile_image_url, protected, public_metrics, url, username, verified, withheld'.replace(' ',''),
        #'place.fields': 'contained_within, country, country_code, full_name, geo, id, name, place_type'.replace(' ',''),
        }
    
    headers = create_twitter_api_headers()
    found_tweets = 0

    while keep_querying:
        if next_token:
            query_params['next_token'] = next_token
        
        json_response = query_twitter_api_endpoint(headers, query_params)
        n_tweets = len(json_response['data'])
        found_tweets += n_tweets
        print('n_tweets',n_tweets,'; found_tweets',found_tweets)
        if 'meta' in json_response and 'next_token' in json_response['meta']:
            # next token found
            next_token = json_response['meta']['next_token']
        else:
            # next token not found, stop
            keep_querying = False
        
        # save data
        insert_tweets_into_db(json_response['data'], muse_id, db_con)
        time.sleep(.2)


def insert_tweets_into_db(tweets, muse_id, db_con):
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
        sql = '''INSERT INTO twitter.tweets_dump(tw_id, tw_ts, author_id, tweet_text, muse_id, tweet_data_json)
              VALUES(%s,%s,%s,%s,%s,%s);'''
        cur.execute(sql, [tw_id_str, ts, user_id, x['text'], muse_id, json_attr])
    
    db_con.commit()