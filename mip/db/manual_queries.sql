# -*- coding: utf-8 -*-
-- Manual queries

-- Note: DO NOT EXECUTE AS A SCRIPT


------------------------------------------------
-- Queries
------------------------------------------------
select count(td.tw_id) as tweets_n from twitter.tweets_dump td;
select muse_id, author_id, count(td.tw_id) as tweets_n from twitter.tweets_dump td group by muse_id, author_id;
-- access JSON fields
select td.tweet_data_json->'author_id' from twitter.tweets_dump td;


-
------------------------------------------------
-- Clear DB
------------------------------------------------

drop table twitter.tweets_dump;
