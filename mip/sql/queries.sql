
-- Manual SQL queries for MIP

select muse_id,count(*) from twitter.tweets_dump group by muse_id ;

select count(distinct muse_id) from twitter.tweets_dump;

select count(*) from twitter.twitter_accounts_not_found;

