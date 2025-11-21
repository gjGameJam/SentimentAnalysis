# entry point; ties everything together

from scripts.dataLoader.finnhubLoader import FinnhubNewsLoader
from scripts.preprocess import MarketTextPreprocessor
from scripts.embedder import TextEmbedder
from dataclasses import asdict
import sys


if __name__ == "__main__":

    # --- arg parse ---
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.main [ticker]")
        sys.exit(1)

    ticker = sys.argv[1]

    # --- load raw data ---
    loader = FinnhubNewsLoader()
    df = loader.fetch_company_news(ticker, days=7)

    # --- normalize + structure ---
    preprocessor = MarketTextPreprocessor()
    records = preprocessor.convert_df(df, symbol=ticker, source="finnhub")

    # quick inspection
    for r in records[:5]:
        print(asdict(r))
        print()

    # --- embed text ---
    embedder = TextEmbedder("sentence-transformers/all-mpnet-base-v2")

    #sanity check from embedding below
    texts = [r.text_raw for r in records]  # or r.text_clean depending on your naming
    embeddings = embedder.embed(texts)

    print(f"Generated {len(embeddings)} embeddings.")
    print(embeddings[0].shape) #ensure 768 dimensions
    print(len(records), len(embeddings)) #ensure all got used (except for empties)

    for r in records[:10]: #Look for extremely short or boilerplate-heavy inputs slipping through
        print(len(r.text_raw), r.text_raw[:120])



