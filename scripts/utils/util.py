from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from trafilatura import fetch_url, extract
import numpy as np


def safe_extract(url: str) -> str:
    try:
        html = fetch_url(url)
        if not html:
            return None

        text = extract(html)
        return text or None

    except Exception as e:
        print(f"[WARN] Extraction failed for {url}: {e}")
        return None

def aggregate_sentiment(probabilities, timestamps=None):
    if timestamps is None:
        weights = np.ones(len(probabilities))
    else:
        # recency weighting: newer = heavier 
        # TODO: tune decay rate if needed
        now = max(timestamps)
        weights = np.exp(-0.1 * ((now - timestamps) / 86400))  # per-day decay

    weights = weights / weights.sum()
    return float((probabilities * weights).sum())


# class to help ensure data is formatted/populated/not duplicate when preprocessing
@dataclass(frozen=True)
class MarketTextRecord:
    symbol: str
    text_raw: str
    timestamp: datetime
    source: str
    url: Optional[str] = None

    def __post_init__(self):
        # type + content validation
        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"timestamp must be datetime, got {type(self.timestamp)}")

        if not isinstance(self.text_raw, str) or not self.text_raw.strip():
            raise ValueError("text_raw must be a non-empty string")

        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")

        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")

        # url is optional, but force string type
        if not isinstance(self.url, str):
            raise TypeError(f"url must be a string, got {type(self.url)}")

    # to print/view data as text
    def as_dict(self):
        return {
            "symbol": self.symbol,
            "text_raw": self.text_raw,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "url": self.url,
        }

    # hash to avoid duplicates
    def __hash__(self):
        # text_raw is often long; hashing full text is fine but may be slow at scale.
        # consider hashing a stable digest instead if throughput matters.
        return hash((self.symbol, self.text_raw, self.timestamp, self.source))

