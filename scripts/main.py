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
from .utils.util import visualize, sanitize_for_matplotlib


if __name__ == "__main__":

    # --- arg parse ---
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.main [ticker]")
        sys.exit(1)

    ticker = sys.argv[1]

    # --- load raw data ---
    loader = FinnhubNewsLoader()
    df = loader.fetch_company_news(ticker, days=7)

    if df.empty:
        print(f"No articles found for {ticker}. Exiting.")
        sys.exit(0)

    # --- normalize + structure ---
    preprocessor = MarketTextPreprocessor()
    records = preprocessor.convert_df(df, symbol=ticker, source="finnhub")

    # sanity ceck
    # print("First 5 raw texts fetched:")
    # for r in records[:5]:
    #     print(r.text_raw[:500], "\n---\n")

    classifier = ZeroShotFinancialSentiment()

    # =====================================================================
    # SENTIMENT LOOP
    # =====================================================================

    signed_sentiments = []
    detailed = []

    label2id = classifier.model.config.label2id

    for r in records:

        raw_output = classifier.classify(r.text_raw)

        # Normalize output
        if isinstance(raw_output, list) and len(raw_output) == 1 and isinstance(raw_output[0], dict):
            raw = raw_output[0]
        elif isinstance(raw_output, dict):
            raw = raw_output
        else:
            raise RuntimeError(f"Unexpected output from classifier: {raw_output}")

        label = raw["label"]
        conf  = raw["score"]
        probs = np.array(raw["probs"])

        # Dynamic indexing
        p_neg = probs[label2id["negative"]]
        p_neu = probs[label2id["neutral"]]
        p_pos = probs[label2id["positive"]]

        # Expectation-based sentiment
        signed = np.tanh(3 * (p_pos - p_neg))

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
    # AGGREGATION
    # =====================================================================

    if len(signed_sentiments) == 0:
        print(f"No valid sentiment data found for {ticker}. Exiting.")
        sys.exit(0)

    final_score = float(np.mean(signed_sentiments))

    print("\n========== SENTIMENT SUMMARY ==========")
    print(f"Articles analyzed: {len(signed_sentiments)}")
    print(f"Final aggregated sentiment for {ticker}: {final_score:.4f}")

    # Optional debugging
    labels_only = [d["label"] for d in detailed]
    label_counts = Counter(labels_only)
    print("Label counts:", label_counts)
    print("Signed sentiment stats ->",
          f"min={min(signed_sentiments):.3f}",
          f"max={max(signed_sentiments):.3f}",
          f"mean={np.mean(signed_sentiments):.3f}")

    category_names = ['Negative', 'Neutral', 'Positive']

    negativeNews = label_counts.get("negative", 0)
    neutralNews  = label_counts.get("neutral", 0)
    positiveNews = label_counts.get("positive", 0)

    ticker_and_score = f"{ticker} ({final_score:.4f})"
    results = {ticker_and_score: [negativeNews, neutralNews, positiveNews]}

    most_negative = sanitize_for_matplotlib(min(detailed, key=lambda d: d["signed"])["text"][:120])
    most_positive = sanitize_for_matplotlib(max(detailed, key=lambda d: d["signed"])["text"][:120])

    visualize(results, category_names, most_negative, most_positive)
    plt.show()



# --- embedder code remains disabled --- 
# --- embed text --- #embedder = TextEmbedder("sentence-transformers/all-mpnet-base-v2") 
#sanity check from embedding below #texts = [r.text_raw for r in records] 
# gte all texts in records # embeddings = embedder.embed(texts) 
# print(f"Generated {len(embeddings)} embeddings.") 
# print(embeddings[0].shape) #ensure 768 dimensions # print(len(records), len(embeddings)) 
#ensure all got used (except for empties) 
# for r in records[:10]: 
#Look for extremely short or boilerplate-heavy inputs slipping through 
# print(len(r.text_raw), r.text_raw[:120])
