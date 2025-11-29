from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np

class ZeroShotFinancialSentiment:
    def __init__(self, model_name="ProsusAI/finbert"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Trust model metadata instead of hardcoding
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def classify(self, text):
        # Accept either a string or list of strings
        single = isinstance(text, str)
        texts = [text] if single else text

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        logits = self.model(**enc).logits            # (batch, 3)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        out = []
        for p in probs:
            idx = int(np.argmax(p))
            out.append({
                "label": self.id2label[idx],
                "score": float(p[idx]),
                "probs": p.tolist()
            })

        # For single-text input, return a dict instead of list
        return out[0] if single else out

    @torch.no_grad()
    def prob_positive(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=384
        ).to(self.device)

        logits = self.model(**enc).logits.squeeze(0)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        # Find correct index from id2label
        pos_idx = None
        for i, lbl in self.id2label.items():
            if lbl.lower().startswith("pos"):
                pos_idx = int(i)
                break

        if pos_idx is None:
            raise RuntimeError("Positive label not found")

        return float(probs[pos_idx])
