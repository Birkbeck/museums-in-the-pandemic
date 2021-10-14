
-- Manual SQL queries for MIP

select muse_id,count(*) from twitter.tweets_dump group by muse_id ;

select count(distinct muse_id) from twitter.tweets_dump;

select count(*) from twitter.twitter_accounts_not_found;
select count(*) from twitter.tweets_dump td ;

select * from twitter.twitter_accounts_not_found tanf;

select count(*) from facebook.facebook_posts_dump;

select count(distinct museum_id) from facebook.facebook_posts_dump;
select count(distinct page_name) from facebook.facebook_pages_not_found;