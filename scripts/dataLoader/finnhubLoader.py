import os
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timedelta

class FinnhubNewsLoader:
    def __init__(self):
        # Load .env two folders up
        env_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(dotenv_path=env_path)
        self.api_key = os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY not found in .env")

    def fetch_company_news(self, symbol, days=30):
        """Fetch company news for the last `days` days."""
        to_date = datetime.utcnow()
        from_date = to_date - timedelta(days=days)
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": symbol,
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
            "token": self.api_key
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise ConnectionError(f"Finnhub API returned {response.status_code}: {response.text}")

        data = response.json()
        if not data:
            return pd.DataFrame()  # no news

        # Convert to DataFrame
        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["datetime"], unit="s")  # convert Unix timestamp
        return df[["datetime", "headline", "source", "url", "summary"]]

if __name__ == "__main__":
    loader = FinnhubNewsLoader()
    df = loader.fetch_company_news("TSLA", days=7)
    #print(df.head())
    # print all summaries
    # for i, row in df.iterrows():
    #     print(row["summary"])
