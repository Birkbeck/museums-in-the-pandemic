-- Manual queries

-- Note: DO NOT EXECUTE AS A SCRIPT

-

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


select session_id, muse_id, count(page_id) as page_n, sum(page_content_length) as data_size from websites.url_redirections group by session_id, muse_id;        

-- how many extracted pages attr
select count(distinct page_id) from websites.web_pages_dump_20210304 union
   select count(distinct page_id) from websites.web_pages_dump_20210304_attr;

select count(distinct page_id) from websites.web_pages_dump_20210304;


select * from websites.web_pages_dump_20210304_attr wpda offset 10000 limit 100;

select * from websites.web_pages_dump_20210304_attr wpda limit 100;

select * from websites.web_pages_dump_20210324 wpd where prev_session_page_id is not null;

select d.page_id, d.url, d.session_id, a.attrib_name, a.attrib_val from websites.web_pages_dump_20210304 d left join websites.web_pages_dump_20210304_attr a 
on d.page_id = a.page_id 
where d.page_id = 181901;
where url = 'https://www.broughtonhouse.com/';

select * from websites.web_pages_dump_20210304 wpda where page_id = 290427;

select d.page_id, d.url, d.session_id, a.attrib_name, a.attrib_val from websites.web_pages_dump_20210304 d left join websites.web_pages_dump_20210304_attr a 
        on d.page_id = a.page_id 
        where url = 'https://www.thisisdurham.com/northernsaints/see-and-do/activities/cycling';

CREATE TABLE IF NOT EXISTS websites.test12 (
            page_id serial PRIMARY KEY,
            url text NOT NULL,
            referer_url text,
            session_id text NOT NULL,
            is_start_url boolean NOT NULL,
            url_domain text NOT NULL,
            muse_id text NOT NULL,
            page_content text NOT NULL,
            page_content_length numeric NOT NULL,
            depth numeric NOT NULL,
            ts timestamp DEFAULT CURRENT_TIMESTAMP,
            google_rank numeric,
            prev_session_diff json,
            prev_session_id text,
            prev_session_page_id numeric,
            UNIQUE(url, session_id));

           
select d.page_id, d.url, d.session_id, a.attrib_name, a.attrib_val from websites.web_pages_dump_20210304 d left join websites.web_pages_dump_20210304_attr a 
        on d.page_id = a.page_id 
        where url = 'https://www.ducksters.com/';
       
select d.page_id, d.url, d.session_id, a.attrib_name, a.attrib_val from websites.web_pages_dump_20210304 d left join websites.web_pages_dump_20210304_attr a 
        on d.page_id = a.page_id 
        where url = 'https://www.timeout.com/london/museums/alexander-fleming-laboratory-museum';
       
select count(page_id) as attr_n, count(distinct page_id) as page_n from websites.web_pages_dump_20210304_attr wpda;

select count(*) as pages, count(distinct muse_id) n_museums from websites.web_pages_dump_20210325 wpda;

select distinct muse_id from websites.web_pages_dump_20210325 wpd order by muse_id;

-- bug fix
select * from websites.web_pages_dump_20210304 wpd where page_id = 950935;


SELECT
  pgClass.relname, pgClass.reltuples AS n_rows
FROM
  pg_class pgClass
LEFT JOIN
  pg_namespace pgNamespace ON (pgNamespace.oid = pgClass.relnamespace)
WHERE
  pgNamespace.nspname NOT IN ('pg_catalog', 'information_schema') AND
  pgClass.relkind='r'
order by pgClass.relname;


CREATE TABLE IF NOT EXISTS analytics.sentence_tokens
    (token_id SERIAL PRIMARY KEY,
    sentence_id integer NOT NULL,
    page_id integer NOT NULL,
    session_id text NOT NULL,
    token text NOT NULL,
    lemma text,
    pos_tag text,
    is_stop boolean);
    CREATE INDEX IF NOT EXISTS analytics.sentence_tokens_sent_idx ON analytics.sentence_tokens USING btree(sentence_id);
    CREATE INDEX IF NOT EXISTS analytics.sentence_tokens_session_idx ON analytics.sentence_tokens USING btree(session_id);
    CREATE INDEX IF NOT EXISTS analytics.sentence_tokens_page_idx ON analytics.sentence_tokens USING btree(page_id);
           
select * from websites.web_pages_dump_20210503
	where url = 'https://www.100bgmus.org.uk/';
select d.url, d.page_content from websites.web_pages_dump_20210521 as d 
	where d.url = 'https://www.100bgmus.org.uk/';


select url, a.page_id, attrib_name, attrib_val from websites.web_pages_dump_20210304 p, websites.web_pages_dump_20210304_attr a 
	where a.page_id = p.page_id and p.page_id = 422811 and a.attrib_name = 'all_text';

-- ALTER TABLE websites.web_pages_dump_20210304
  -- ADD new_page_b boolean;
-- update websites.web_pages_dump_20210304 set new_page_b = true; 

--ALTER TABLE websites.web_pages_dump_20210304
  --ADD prev_session_diff_b boolean;
--update websites.web_pages_dump_20210304 set prev_session_diff_b = true;

------------------------------------------------
-- Text analytics
------------------------------------------------

-- drop table analytics.text_indic_ann_matches;

select count(*) from analytics.text_indic_ann_matches t;
select count(distinct muse_id) from analytics.text_indic_ann_matches t;
select * from analytics.text_indic_ann_matches t order by t.ann_overlap_lemma desc limit 100;

select * from analytics.text_indic_ann_matches t limit 200;

select * from analytics.text_indic_ann_matches t where t.keep_stopwords limit 100; 

select count(*) from analytics.indicator_annotation_tokens k;

select * from analytics.indicator_annotation_tokens t where t.example_id = 'ann_ex_00120';



-- quartiles of lemmas
select
  percentile_disc(0.0) within group (order by t.lemma_n),
  percentile_disc(0.25) within group (order by t.lemma_n),
  percentile_disc(0.5) within group (order by t.lemma_n),
  percentile_disc(0.75) within group (order by t.lemma_n),
  percentile_disc(1.0) within group (order by t.lemma_n)
from analytics.text_indic_ann_matches t;

-
------------------------------------------------
-- Clear DB
------------------------------------------------

--drop table twitter.tweets_dump;


