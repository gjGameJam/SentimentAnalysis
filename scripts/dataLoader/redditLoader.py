import praw
import time
import random
from prawcore.exceptions import RequestException, ResponseException, ServerError
import pandas as pd
from datetime import datetime

class RedditLoader:
    def __init__(self, client_id, client_secret, user_agent, max_retries=3, cooldown=60):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        self.max_retries = max_retries
        self.cooldown = cooldown  # seconds to sleep after repeated fails

    def _safe_request(self, func, *args, **kwargs):
        """Retry wrapper for API calls."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (RequestException, ResponseException, ServerError) as e:
                wait = 2 ** attempt + random.uniform(0, 1)
                print(f"[WARN] Reddit API error: {e}. Retrying in {wait:.2f}s...")
                time.sleep(wait)
        print("[ERROR] Max retries exceeded. Cooling down...")
        time.sleep(self.cooldown)
        return None

    def fetch_posts(self, query, limit=100, subreddits=None):
        """Fetch posts mentioning an asset from given subreddits."""
        if subreddits is None:
            subreddits = ["wallstreetbets", "stocks", "investing"]

        all_posts = []

        for sub in subreddits:
            print(f"[INFO] Fetching from r/{sub} ...")
            subreddit = self.reddit.subreddit(sub)
            results = self._safe_request(subreddit.search, query, sort="new", limit=limit)
            if results is None:
                continue
            for post in results:
                all_posts.append({
                    "timestamp": datetime.utcfromtimestamp(post.created_utc),
                    "subreddit": sub,
                    "title": post.title,
                    "text": post.selftext,
                    "score": post.score,
                    "url": post.url,
                    "num_comments": post.num_comments,
                })
            # Randomized short sleep between subreddit queries to avoid rate hits
            time.sleep(random.uniform(1, 3))

        df = pd.DataFrame(all_posts)
        if not df.empty:
            df.drop_duplicates(subset=["title", "timestamp"], inplace=True)
        return df

if __name__ == "__main__":
    loader = RedditLoader(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="SentimentModel/1.0"
    )

    df = loader.fetch_posts("TSLA", limit=50)
    print(df.head())