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

-- Website scraping

-- list of dump tables
SELECT table_name FROM information_schema.tables WHERE table_schema='websites';

select page_id, root_url, url, referer_url, depth from websites.web_pages_dump_20210303 limit 100;
select count(page_id) from websites.web_pages_dump_20210303;
select muse_id, count(page_id) as page_n from websites.web_pages_dump_20210303 group by muse_id;
select count(distinct muse_id) as muse_n from websites.web_pages_dump_20210303;

-
------------------------------------------------
-- Clear DB
------------------------------------------------

drop table twitter.tweets_dump;
