from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np

class ZeroShotFinancialSentiment:
    def __init__(self, model_name="ProsusAI/finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    @torch.no_grad()
    def classify(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        labels = [self.id2label[np.argmax(p)] for p in probs]
        scores = [float(np.max(p)) for p in probs]

        return list(zip(labels, scores))

    def prob_positive(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=384
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(0)  # shape (C,)
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        # find index of "positive"
        id2label = self.model.config.id2label
        pos_idx = None
        for idx, lbl in id2label.items():
            if lbl.lower().startswith("pos"):
                pos_idx = int(idx)
                break
        if pos_idx is None:
            raise RuntimeError(f"Positive label not found in id2label: {id2label}")

        return float(probs[pos_idx])
