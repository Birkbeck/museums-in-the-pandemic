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

-- stats from dump tables
select * from websites.web_pages_dump_20210303 limit 100;
select page_id, url, referer_url, depth from websites.web_pages_dump_20210303 limit 100;

-- number of scraped pages in a single session
select count(page_id) from websites.web_pages_dump_20210304;

-- how many museums are in a scraping session
select count(distinct muse_id) as muse_n from websites.web_pages_dump_20210304;

-- count rows in URL redirection
select count(url) from websites.url_redirections;

select * from websites.url_redirections where url='http://www.americanairmuseum.com/place/52';

CREATE INDEX IF NOT EXISTS idx1 ON websites.web_pages_dump_20210304 USING btree(muse_id);
CREATE INDEX IF NOT EXISTS idx2 ON websites.web_pages_dump_20210304 USING btree(url);


--delete from websites.url_redirections;

-- test speed of URL/session query
select page_id from websites.web_pages_dump_20210304 wpd where session_id = '20210304' and url = 'https://www.britishmuseum.org/';

select muse_id, count(page_id) as page_n from websites.web_pages_dump_20210304 group by muse_id;

select muse_id, url_domain, count(page_id) as page_n from websites.web_pages_dump_20210304 group by url_domain, muse_id;


select * from websites.web_pages_dump_20210304 wpd where is_start_url;

select muse_id, url,is_start_url,a.* from websites.web_pages_dump_20210304 p, websites.web_pages_dump_20210304_attr a where p.page_id = a.page_id and p.is_start_url;

https://marblebar.org.au/company/st-peters-heritage-centre-hall-1460398/
select * from websites.web_pages_dump_20210303 where muse_id = 'mm.domus.SW005';



-
------------------------------------------------
-- Clear DB
------------------------------------------------

--drop table twitter.tweets_dump;
