# entry point; ties everything together

from scripts.dataLoader.finnhubLoader import FinnhubNewsLoader
from scripts.preprocess import MarketTextPreprocessor
from scripts.embedder import TextEmbedder
from scripts.classifier import ZeroShotFinancialSentiment
from dataclasses import asdict
import numpy as np
from collections import Counter
import sys
import matplotlib.pyplot as plt


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
    # for r in records[:2]:
    #     print(asdict(r))
    #     print()

    classifier = ZeroShotFinancialSentiment()

    # =====================================================================
    # NEW SENTIMENT LOOP (correct handling of label + confidence + polarity)
    # =====================================================================

    signed_sentiments = []
    detailed = []

    for r in records:

        raw = classifier.classify(r.text_raw)

        # Normalize HF classifier outputs
        if isinstance(raw, list):
            raw = raw[0]

        if isinstance(raw, tuple):
            raw = {"label": raw[0], "score": float(raw[1])}

        label = raw["label"]
        conf = float(raw["score"])

        # Convert to signed sentiment contribution
        if label == "positive":
            signed = conf
        elif label == "negative":
            signed = -conf
        else:
            signed = 0.0     # neutral does not shift the score

        signed_sentiments.append(signed)

        detailed.append({
            "timestamp": float(r.timestamp.timestamp()),
            "symbol": r.symbol,
            "url": r.url,
            "text": r.text_raw,
            "label": label,
            "confidence": conf,
            "signed": signed,
        })

    # =====================================================================
    # AGGREGATION (simple, correct mean of signed sentiment)
    # =====================================================================

    if len(signed_sentiments) == 0:
        final_score = 0.0
    else:
        final_score = sum(signed_sentiments) / len(signed_sentiments)

    print("\n========== SENTIMENT SUMMARY ==========")
    print(f"Articles analyzed: {len(signed_sentiments)}")
    print(f"Final aggregated sentiment for {ticker}: {final_score:.4f}")

    # Optional debugging:
    labels_only = [d["label"] for d in detailed]
    print("Label counts:", Counter(labels_only))
    print("Signed sentiment stats ->",
          f"min={min(signed_sentiments):.3f}",
          f"max={max(signed_sentiments):.3f}",
          f"mean={np.mean(signed_sentiments):.3f}")

    # --- embedder code remains disabled ---
    # --- embed text --- #embedder = TextEmbedder("sentence-transformers/all-mpnet-base-v2")
    #sanity check from embedding below #texts = [r.text_raw for r in records]
    # gte all texts in records # embeddings = embedder.embed(texts) # print(f"Generated {len(embeddings)} embeddings.")
    # print(embeddings[0].shape) #ensure 768 dimensions # print(len(records), len(embeddings)) #ensure all got used (except for empties)
    # for r in records[:10]:
        #Look for extremely short or boilerplate-heavy inputs slipping through
        # print(len(r.text_raw), r.text_raw[:120])
