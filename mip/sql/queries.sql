
-- Manual SQL queries for MIP

-- analytics
select * from analytics.text_indic_ann_matches_20210304 limit 100;
select count(*) from analytics.text_indic_ann_matches_20210304;
select count(distinct muse_id) from analytics.text_indic_ann_matches_20210304;

-- twitter 
select muse_id,count(*) from twitter.tweets_dump group by muse_id ;
select count(distinct muse_id) from twitter.tweets_dump;
select count(*) from twitter.twitter_accounts_not_found;
select count(*) from twitter.tweets_dump td;
select count(*) from twitter.museums_no_twitter td;
select * from twitter.twitter_accounts_not_found tanf;

-- facebook
select count(*) from facebook.facebook_posts_dump;
select museum_id, page_name, query_account, count(*) from facebook.facebook_posts_dump group by museum_id, page_name, query_account;
select count(distinct museum_id) from facebook.facebook_posts_dump;
select count(distinct page_name) from facebook.facebook_posts_dump;
select count(distinct page_name) from facebook.facebook_pages_not_found;

select * from facebook.facebook_posts_dump fpd where query_account = 'ntbaddesleyclinton' limit 100;
select * from facebook.facebook_posts_dump fpd where page_name = 'NTBaddesleyClinton' limit 100;

select facebook_data_json -> 'account_name' as n from facebook.facebook_posts_dump fpd where query_account = 'ntbaddesleyclinton' limit 1000;