
-- Manual SQL queries for MIP

-- analytics
select * from analytics.text_indic_ann_matches_20210304 limit 100;
select count(*) from analytics.text_indic_ann_matches_20210304;
select count(distinct muse_id) from analytics.text_indic_ann_matches_20210304;

select * from websites.web_pages_dump_20210304_attr where page_id in (149714, 406382);
select * from websites.web_pages_dump_20210304 where page_id in (149714, 406382);

-- twitter 
select muse_id, count(*) from twitter.tweets_dump group by muse_id ;
select count(distinct muse_id) from twitter.tweets_dump;
select count(*) from twitter.twitter_accounts_not_found;
select count(*) from twitter.tweets_dump td;
select count(*) from twitter.museums_no_twitter td;
select * from twitter.twitter_accounts_not_found tanf;

select count(*) from twitter.tweets_dump td where muse_id = 'mm.ace.685';

select muse_id as museum_id, , author_account as account, museum_account as museum_account, tweet_text as msg_text, tw_ts as msg_time from twitter.tweets_dump limit 100;

CREATE INDEX tweets_muse_id_idx ON twitter.tweets_dump(muse_id);


-- facebook
select count(*) from facebook.facebook_posts_dump;
select museum_id, page_name, query_account, count(*) from facebook.facebook_posts_dump group by museum_id, page_name, query_account;
select count(distinct museum_id) from facebook.facebook_posts_dump;
select count(distinct page_name) from facebook.facebook_posts_dump;
select count(distinct page_name) from facebook.facebook_pages_not_found;

CREATE INDEX facebook_muse_id_idx ON facebook.facebook_posts_dump(museum_id);

select * from facebook.facebook_posts_dump fpd where query_account = 'ntbaddesleyclinton' limit 100;
select * from facebook.facebook_posts_dump fpd where page_name = 'NTBaddesleyClinton' limit 100;

select facebook_data_json -> 'account_name' as n from facebook.facebook_posts_dump fpd where query_account = 'ntbaddesleyclinton' limit 1000;

select muse_id,page_id,sentence_id,example_id,indicator_code,session_id,ann_ex_tokens,page_tokens,sem_similarity,token_n,lemma_n,ann_overlap_lemma,ann_overlap_token,example_len,txt_overlap_lemma,txt_overlap_token,ann_overlap_criticwords from analytics.text_indic_ann_matches_20210303 t 
        where keep_stopwords and ann_overlap_criticwords > 0;


-- websites
select * from websites.web_pages_dump_20210303 wpda where url ilike '%english-heritage.org%';

select * from websites.web_pages_dump_20210404 wpda where url ilike '%alfordmanor%';
select * from websites.web_pages_dump_20210914 wpda where url ilike '%alfordmanor%';

select * from websites.web_pages_dump_20210914 wpda where is_start_url and url ilike '%trewithengardens.co.uk%';

select * from websites.web_pages_dump_20210629 wpd where url ilike '%www.upperheyfordheritage.co.uk/home-page/tours/';
select * from websites.web_pages_dump_20210914 wpd where url ilike '%www.upperheyfordheritage.co.uk/home-page/tours/'

select count(*) from websites.url_redirections ur;
select count(*) from analytics.website_sizes ws;

select * from websites.web_pages_dump_20210914 p where p.referer_url in ('https://www.upperheyfordheritage.co.uk/home-page/tours/index.html','https://www.upperheyfordheritage.co.uk/home-page/tours/','http://www.upperheyfordheritage.co.uk/home-page/tours','http://www.upperheyfordheritage.co.uk/home-page/tours/index.html','https://www.upperheyfordheritage.co.uk/home-page/tours/index.htm','https://www.upperheyfordheritage.co.uk/home-page/tours/index.php','http://www.upperheyfordheritage.co.uk/home-page/tours/index.php','http://www.upperheyfordheritage.co.uk/home-page/tours/index.htm','http://www.upperheyfordheritage.co.uk/home-page/tours/','https://www.upperheyfordheritage.co.uk/home-page/tours');

select * from websites.web_pages_dump_20210404 where url = 'https://www.shetlandheritageassociation.com/members/south-mainland/george-waterston-memorial-museum';

select * from websites.web_pages_dump_20210404 where muse_id = 'mm.misc.137';

-- debugging of the page logic     
--   20210420 29599;
--   20210914 197;
--   20210304 745284;
select * from websites.web_pages_dump_20210914 wpda where is_start_url and url ilike '%trewithengardens.co.uk%';
select * from websites.web_pages_dump_20210420 wpda where is_start_url and url ilike '%trewithengardens.co.uk%';
select * from websites.web_pages_dump_20210304 wpda where is_start_url and url ilike '%trewithengardens.co.uk%';


select * from websites.web_pages_dump_20210914 wpd where url ilike '%www.nationaltrust.org.uk/hinton-ampner';

select * from websites.web_pages_dump_20211011 wpd where url ilike '%spodemuseumtrust.org%';
select * from websites.web_pages_dump_20210521 wpd where url ilike '%www.jjfox.co.uk/heritage';


select * from websites.web_pages_dump_20210304 wpd where url ilike 'https://www.lincolnshirelife.co.uk/posts/view/st-katherines%';

-- https://www.visitherefordshire.co.uk/thedms.aspx?dms=3&venue=1400410
-- mm.ace.1186
select * from websites.web_pages_dump_20210521 wpd where url ilike 'https://www.visitherefordshire.co.uk%';

-- https://www.dunollie.org/1745-house
select * from websites.web_pages_dump_20210521 wpd where url ilike 'https://www.dunollie.org/';

select * from websites.web_pages_dump_20210521 wpd where url ilike 'https://www.sheppyscider.com/';


select * from websites.web_pages_dump_20210521 wpd where muse_id = 'mm.ace.1186';

select * from websites.web_pages_dump_20210304_attr wpda where page_id = 745284;

select count(*) from websites.web_pages_dump_20210303 wpda;

select count(prev_session_diff_b, new_page_b) from websites.web_pages_dump_20210304 wpda;

select * from websites.web_pages_dump_20210304_attr wpda where page_id = 251480;

select * from websites.web_pages_dump_20210304_attr wpda where page_id = 251480;

select * from websites.web_pages_dump_20210901 wpda where page_id = 23250;

SELECT pgClass.relname, pgClass.reltuples AS n_rows, pgClass.relnamespace
    FROM
        pg_class pgClass
    LEFT JOIN
        pg_namespace pgNamespace ON (pgNamespace.oid = pgClass.relnamespace)
    WHERE
        pgNamespace.nspname NOT IN ('pg_catalog', 'information_schema') 
        AND pgClass.relkind='r'
    order by pgClass.relname;

  SELECT * FROM information_schema.tables; 
 
select schemaname, relname, n_live_tup, n_dead_tup
from pg_stat_user_tables
order by schemaname, relname desc;

select * from pg_catalog.pg_stat_user_tables 

-- ----------------------------------------
-- Social media indicators
-- ----------------------------------------

select count(*) from analytics.indicators_social_media_matches;

select count(*) as match_n, count(distinct muse_id) as museum_n from analytics.indicators_social_media_matches;

select count(*) as match_n, count(distinct muse_id) as museum_n from analytics.indicators_social_media_matches where ann_overlap_criticwords  > 0;
-- EOF