from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from trafilatura import fetch_url, extract
import numpy as np
import matplotlib.pyplot as plt

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


def sanitize_for_matplotlib(text: str) -> str:
    """
    Escapes characters that would trigger Matplotlib mathtext parsing.
    Prevents crashes when displaying arbitrary news headlines.
    """
    if text is None:
        return ""

    # Mathtext-sensitive characters
    replacements = {
        "$":  r"\$",
        #"%":  r"\%",
        "&":  r"\&",
        "_":  r"\_",
        "#":  r"\#",
        "{":  r"\{",
        "}":  r"\}",
        "^":  r"\^{}",
        "~":  r"\~{}",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


#visualization function
def visualize(results, category_names, most_negative_title, most_positive_title):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 1.4))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncols=len(category_names), bbox_to_anchor=(.25, 1),
              loc='lower left', fontsize='small')

    # plt.text(0.5, 1.05, 'Sentiment Distribution', ha='left', va='center',
    #          transform=ax.transAxes, fontsize='large')

    plt.text(0.0, -0.1, f'Most Negative News: {most_negative_title}',
             transform=ax.transAxes, fontsize='medium', color='black')

    plt.text(0.0, -0.2, f'Most Positive News: {most_positive_title}',
             transform=ax.transAxes, fontsize='medium', color='black')

    #fig.subplots_adjust(top=0.35)
    fig.subplots_adjust(bottom=0.35)

    return fig, ax


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

