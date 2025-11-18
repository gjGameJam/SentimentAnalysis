# text cleaning, tokenization, normalization
# preprocessor.py
from typing import List
from dataclasses import asdict
from datetime import datetime
import re
from .utils.util import MarketTextRecord
from scripts.dataLoader.finnhubLoader import FinnhubNewsLoader

class MarketTextPreprocessor:
    #Preprocess raw market text data into normalized MarketTextRecord objects

    REQUIRED_COLUMNS = {
    "datetime": "timestamp as datetime",
    "text": "main semantic content",
    "url": "permalink or reference",
    }

    OPTIONAL_COLUMNS = ["headline", "summary"]


    def __init__(self, min_text_length=20, allowed_symbols=None):
        
        #param min_text_length: discard texts shorter than this
        #param allowed_symbols: optional whitelist of symbols
        
        self.min_text_length = min_text_length
        self.allowed_symbols = allowed_symbols or []

    def normalize_text(self, text: str) -> str:
        if not text:
            return ""

        # Lowercase
        text = text.lower().strip()

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove boilerplate source prefixes ("reuters – ...", "marketwatch - ...")
        text = re.sub(r"^(reuters|marketwatch|bloomberg|yahoo finance)[\s\-–:]+", "", text)

        # Remove URLs (embedded or appended)
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove ticker markup: $TSLA, (TSLA), [TSLA]
        text = re.sub(r"\$[A-Za-z]{1,5}\b", "", text)
        text = re.sub(r"\([A-Za-z]{1,5}\)", "", text)
        text = re.sub(r"\[[A-Za-z]{1,5}\]", "", text)

        # Remove common disclaimers
        text = re.sub(
            r"(all rights reserved\.?|this article is .*? purposes only\.?|© ?\d{4} .*?)$",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Normalize unicode punctuation (smart quotes, long dashes → ASCII)
        text = (
            text.replace("\u201c", '"')
                .replace("\u201d", '"')
                .replace("\u2018", "'")
                .replace("\u2019", "'")
                .replace("\u2013", "-")
                .replace("\u2014", "-")
        )

        # Final trim + collapse whitespace again
        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        return text

    def normalize_df_schema(self, df):
        # Ensure datetime exists
        if "datetime" not in df.columns:
            raise ValueError("DataFrame missing required column: 'datetime'")

        # Resolve text column (summary > text > headline)
        text_col = None
        for candidate in ["summary", "text", "headline"]:
            if candidate in df.columns:
                text_col = candidate
                break

        if text_col is None:
            raise ValueError("DataFrame missing any text column: ['summary', 'text', 'headline']")

        # Create unified 'text' column
        df = df.copy()
        df["text"] = df[text_col]

        # Ensure url exists
        if "url" not in df.columns:
            df["url"] = ""

        return df[["datetime", "text", "url"]]



    def validate_record(self, record: MarketTextRecord) -> bool:
        #Return True if record passes validation checks
        if len(record.text_raw) < self.min_text_length:
            return False
        if self.allowed_symbols and record.symbol not in self.allowed_symbols:
            return False
        if not isinstance(record.timestamp, datetime):
            return False
        return True

    def dedupe_records(self, records: List[MarketTextRecord]) -> List[MarketTextRecord]:
        #Remove duplicates based on hash/equality
        return list(set(records))  # dataclass hash handles deduplication

    def convert_df(self, df, symbol: str, source: str) -> List[MarketTextRecord]:
        df = self.normalize_df_schema(df)

        records = []
        for _, row in df.iterrows():
            raw_text = row["text"]
            text = self.normalize_text(raw_text)

            # Skip rows with no real content
            if not text.strip():
                continue

            timestamp = row["datetime"]
            url = row.get("url", "")

            record = MarketTextRecord(
                symbol=symbol,
                text_raw=text,
                timestamp=timestamp,
                source=source,
                url=url
            )

            if self.validate_record(record):
                records.append(record)

        return self.dedupe_records(records)




#function just for testing/output
#going through loaders and making/using preprocessor will be done in main
if __name__ == "__main__":
    
    loader = FinnhubNewsLoader() # example usage with Finnhub loader
    df = loader.fetch_company_news("TSLA", days=7)

    preprocessor = MarketTextPreprocessor()
    records = preprocessor.convert_df(df, symbol="TSLA", source="finnhub")
    for r in records[:5]:
        print(asdict(r))
        print() #empty line for easy viewing

    # assert all(len(r.text_raw) >= preprocessor.min_text_length for r in records)
    # assert all(isinstance(r.timestamp, datetime) for r in records)
    # assert len(records) == len(set(records))  # dedupe verified
