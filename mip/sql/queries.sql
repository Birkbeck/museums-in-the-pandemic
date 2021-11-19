
-- Manual SQL queries for MIP

-- analytics
select * from analytics.text_indic_ann_matches_20210304 limit 100;
select count(*) from analytics.text_indic_ann_matches_20210304;
select count(distinct muse_id) from analytics.text_indic_ann_matches_20210304;

select * from websites.web_pages_dump_20210304_attr where page_id in (149714, 406382);
select * from websites.web_pages_dump_20210304 where page_id in (149714, 406382);

-- twitter 
select muse_id,count(*) from twitter.tweets_dump group by muse_id ;
select count(distinct muse_id) from twitter.tweets_dump;
select count(*) from twitter.twitter_accounts_not_found;
select count(*) from twitter.tweets_dump td;
select count(*) from twitter.museums_no_twitter td;
select * from twitter.twitter_accounts_not_found tanf;

select count(*) from twitter.tweets_dump td where muse_id = 'mm.ace.685';


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


-- page debugging    
--   20210420 29599;
--   20210914 197;
--   20210304 745284;
select * from websites.web_pages_dump_20210914 wpda where is_start_url and url ilike '%trewithengardens.co.uk%';
select * from websites.web_pages_dump_20210420 wpda where is_start_url and url ilike '%trewithengardens.co.uk%';
select * from websites.web_pages_dump_20210304 wpda where is_start_url and url ilike '%trewithengardens.co.uk%';


select * from websites.web_pages_dump_20210914 wpd where url ilike '%www.nationaltrust.org.uk/hinton-ampner';

select * from websites.web_pages_dump_20211011 wpd where url ilike '%spodemuseumtrust.org%';

select * from websites.web_pages_dump_20210304_attr wpda where page_id = 745284;

select count(*) from websites.web_pages_dump_20210303 wpda;

select count(prev_session_diff_b, new_page_b) from websites.web_pages_dump_20210304 wpda;

select * from websites.web_pages_dump_20210304_attr wpda where page_id = 251480;

select * from websites.web_pages_dump_20210304_attr wpda where page_id = 251480;

select * from websites.web_pages_dump_20210901 wpda where page_id = 23250;

-- EOF