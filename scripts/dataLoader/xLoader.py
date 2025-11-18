import tweepy

client = tweepy.Client(bearer_token="YOUR_BEARER_TOKEN")
query = "TSLA OR Tesla -is:retweet"

tweets = client.search_recent_tweets(query=query, max_results=20, tweet_fields=["created_at","author_id","text"])
for t in tweets.data:
    print(t.created_at, t.text)
