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
try:
    with open('.secrets.json') as f:
        twitter_config = json.load(f)
except:
    print('Warning: .secrets.json file not found to use Twitter API.')

twitter_db_fn = 'tmp/tweets.db'
MAX_TWEETS_PER_ACCOUNT = 80000


def create_tweet_dump(db_conn):
    """ create table for Twitter dump """
    c = db_conn.cursor()

    # Create table
    c.execute('''
        CREATE SCHEMA IF NOT EXISTS twitter;
        CREATE EXTENSION IF NOT EXISTS hstore;
        CREATE TABLE IF NOT EXISTS twitter.tweets_dump 
            (tw_id text NOT NULL,
            author_id text NOT NULL,
            author_account text NOT NULL,
            tweet_text text NOT NULL,
            is_reply boolean NOT NULL,
            museum_account text NOT NULL,
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
    pause_secs = 2.0
    while True:
        time.sleep(pause_secs)
        response = requests.request("GET", search_url, headers=headers, params=params)
        #print(response.status_code)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503 or response.status_code == 429:
            pause_secs *= 2
            print('Error',response.status_code,response.text, '\n\tPausing secs=', pause_secs)
        else:
            raise RuntimeError(response.status_code, response.text)


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
        accounts = [s.strip().replace('https://','') for s in accounts]
        accounts = [s.strip().replace('http://','') for s in accounts]
        accounts = [s.strip().replace('mobile.twitter.com/','') for s in accounts]
        accounts = [s.strip().replace('www.twitter.com/','') for s in accounts]
        accounts = [s.strip().replace('twitter.com/','') for s in accounts]
        
        # remove /status/ and parameters
        accounts = [s.split('/')[0] for s in accounts]
        accounts = [s.split('?')[0] for s in accounts]
        accounts = [x for x in accounts if x]
        accounts = [x for x in accounts if len(x)>2]
        # remove duplicates
        accounts = list(set(accounts))
        for a in accounts:
            assert len(a) > 0
            assert not a.lower() in ['twitter','photo','status','com','www','#','facebook']
        return accounts

    no_twitter_mus = pd.DataFrame()

    # open connection to db
    db_engine = create_alchemy_engine_posgresql()
    db_con = connect_to_postgresql_db() #open_sqlite(twitter_db_fn)
    # init table
    min_date = datetime.datetime(2019, 1, 1, 0, 0, 0)
    create_tweet_dump(db_con)
    i = 0
    # museums_df = museums_df.sample(len(museums_df)) # SHUFFLE
    #museums_df = museums_df.sample(2) # DEBUG
    for idx, mus in museums_df.iterrows():
        i += 1
        mus_id = mus['museum_id']
        tw_accounts = get_twitter_accounts_from_col(mus['twitter_id'])
        print('> museum ', mus_id, '-', i, 'of', len(museums_df), tw_accounts)
        #continue # DEBUG
        # scan accounts
        found_tweets = 0
        
        for acc in tw_accounts:
            assert acc
            # apply updates from social_media_urls_corrected.tsv (6 Dec 2021)
            #if mus['twitter_action'] == 'drop' or mus['twitter_action'] == 'update':
            #    # delete old accounts, if any
            #    old_tw_accounts = get_twitter_accounts_from_col(mus['twitter_id_old'])
            #    assert len(old_tw_accounts) >= 0
            #    for old_acc in old_tw_accounts:
            #        delete_twitter_account_from_db(mus_id, old_acc, db_con)
            
            if has_db_museum_tweets(mus_id, acc, db_con):
                continue
            # scrape twitter account
            found_tweets += scrape_twitter_account(mus_id, acc, min_date, db_con, db_engine)
        
        if found_tweets == 0:
            no_twitter_mus = no_twitter_mus.append(mus)
    db_con.close()
    # insert no_twitter_mus into DB
    no_twitter_mus.to_sql('museums_no_twitter', db_engine, schema='twitter', index=False, if_exists='replace', method='multi')


def delete_twitter_account_from_db(muse_id, user_name, db_con):
    sql = '''delete from twitter.tweets_dump where muse_id = '{}' and museum_account = '{}';'''.format(muse_id, user_name)
    print('deleting twitter account: ',user_name)
    cur = db_con.cursor()
    ret = cur.execute(sql)
    i = 0


def has_db_museum_tweets(muse_id, user_name, db_con):
    '''@returns True if Twitter account exists in DB'''
    sql = '''select count(*) as cnt from twitter.tweets_dump where muse_id = '{}' and museum_account = '{}';'''.format(muse_id, user_name)
    df = pd.read_sql(sql, db_con)
    cnt = df.cnt[0]
    found = cnt > 0
    if found: return True
    
    try:
        sql = '''select * from twitter.twitter_accounts_not_found where museum_id = '{}' and user_name = '{}';'''.format(muse_id, user_name)
        df = pd.read_sql(sql, db_con)
        found = len(df) > 0
        return found
    except Exception as e:
        print('warning:', str(e))
        return False


def scrape_twitter_account(muse_id, user_name, min_date, db_con, db_engine):
    """
    API code based on
    https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/master/Full-Archive-Search/full-archive-search.py
    and 
    https://developer.twitter.com/en/docs/twitter-api/v1/tweets/timelines/api-reference/get-statuses-user_timeline
    """
    assert min_date < datetime.datetime.now()
    assert len(user_name.strip()) > 0
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
        #'user.fields': 'created_at, description, entities, id, location, name, pinned_tweet_id, profile_image_url, protected, public_metrics, url, username, verified, withheld'.replace(' ',''),
        'user.fields': 'created_at, description, id, location, name, profile_image_url, public_metrics, url, username, verified'.replace(' ',''),
        'expansions': 'attachments.poll_ids, attachments.media_keys, author_id, entities.mentions.username, geo.place_id, in_reply_to_user_id, referenced_tweets.id, referenced_tweets.id.author_id'.replace(' ',''),
        'start_time': start_time_iso,
        'max_results': 450
        #'place.fields': 'contained_within, country, country_code, full_name, geo, id, name, place_type'.replace(' ',''),
    }
    # NOTE 'context_annotations' was removed because of the low tweet limit.
    
    headers = create_twitter_api_headers()
    found_tweets = 0

    json_results = []
    while keep_querying:
        if next_token:
            query_params['next_token'] = next_token
        
        json_response = query_twitter_api_endpoint(headers, query_params)
        
        if not 'data' in json_response:
            print('warning: no data found for', muse_id, user_name, str(json_response))
            break
        n_tweets = len(json_response['data'])
        found_tweets += n_tweets
        print('\tn_tweets',n_tweets,'; found_tweets',found_tweets)
        
        if found_tweets > MAX_TWEETS_PER_ACCOUNT:
            print('warning: ',user_name+' seems to have too many tweets for a museum. Skipping')
            return 0
        
        if 'meta' in json_response and 'next_token' in json_response['meta']:
            # next token found
            next_token = json_response['meta']['next_token']
        else:
            # next token not found, stop
            keep_querying = False
        
        json_results.append(json_response)
    
    # save data
    if len(json_results)==0:
        pd.DataFrame({'museum_id':[muse_id], 'user_name':[user_name]}).to_sql('twitter_accounts_not_found', db_engine, 
            schema='twitter', index=False, if_exists='append', method='multi')

    for j in json_results:
        insert_tweets_into_db(j, muse_id, user_name, db_con)
    return found_tweets


def get_tweets_from_db(museum_id, db_conn):
    ''' get tweets' text to find indicators '''
    assert museum_id
    sql = '''select muse_id as museum_id, tw_id as msg_id, author_account as account, tweet_text as msg, tw_ts as ts, 'twitter' as platform from twitter.tweets_dump where muse_id = '{}' and not is_reply '''.format(museum_id)
    df = pd.read_sql(sql, db_conn)
    return df


def insert_tweets_into_db(tweets_json, muse_id, tw_account, db_con):
    """ Insert twitter data into DB """
    assert muse_id
    assert db_con
    # extract tweets and users
    tweets = tweets_json['data']
    assert 'includes' in tweets_json
    users_j = tweets_json['includes']['users']
    places_d = {}
    if 'places' in tweets_json['includes']:
        for p in tweets_json['includes']['places']:
            places_d[p['id']] = p
    users_d = {}
    for u in users_j:
        users_d[u['id']] = u

    if len(tweets) == 0: 
        return
    cur = db_con.cursor()

    for x in tweets:
        # extract fields
        tw_id_str = x['id']
        user_id = x['author_id']
        place_info = {}
        if 'geo' in x and 'place_id' in x['geo']:
            # add place info
            place_info = places_d[x['geo']['place_id']]
        assert user_id in users_d, 'user not found'
        user_info = users_d[user_id]
        user_name = user_info['username']
        ts = datetime.datetime.fromisoformat(x['created_at'].replace("Z", "+00:00"))
        x['author_info'] = user_info
        x['place_info'] = place_info
        is_reply = False
        if 'in_reply_to_user_id' in x:
            is_reply = True
        json_attr = json.dumps(x)
        b_museum_user = user_name == tw_account
        # insert sql
        sql = '''INSERT INTO twitter.tweets_dump(tw_id, tw_ts, museum_account, author_account, 
            author_id, tweet_text, is_reply, muse_id, tweet_data_json)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s);'''
        cur.execute(sql, [tw_id_str, ts, tw_account, user_name, user_id, x['text'], is_reply, muse_id, json_attr])
    
    db_con.commit()