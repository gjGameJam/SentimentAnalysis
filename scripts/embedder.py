# feature extraction (FinBERT, embeddings)

"""
TextEmbedder — wraps any embedding backend (HF, OpenAI, local transformer).
Input: list of already-normalized strings
Output: list/array of embedding vectors

Design goals:
- Model loads once on init
- Supports batching
- Handles empty or invalid inputs gracefully
- Returns consistent embedding shapes
- Isolates ALL embedding logic from the pipeline
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import torch


class TextEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32):
        # ---------------------------------------------------------------
        # Load embedding model here.
        # - Could be sentence-transformers, HuggingFace, or OpenAI API.
        # - Detect GPU automatically if not provided.
        # - Initialize tokenizer + model if using HF.
        # - Store batch_size for later use.
        #
        # Example responsibilities:
        #   self.model = ...
        #   self.tokenizer = ...
        # ---------------------------------------------------------------

        self.model_name = model_name
        self.batch_size = batch_size

        # Load SentenceTransformer model: https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/README.md
        self.model = SentenceTransformer(model_name)
        self.model.eval()

    def _embed_batch(self, batch):
        # ---------------------------------------------------------------
        # Internal helper: embed one batch of text.
        # - Convert text -> model inputs
        # - Move tensors to correct device
        # - Forward pass through the model
        # - Extract the embedding (CLS, pooled output, etc.)
        # - Return embeddings as numpy arrays or lists
        # ---------------------------------------------------------------

        if not batch:
            return []

        # SentenceTransformers does batching internally but batching manually
        # gives you more control and consistent GPU memory use.
        with torch.no_grad():
            vectors = self.model.encode(
                batch,
                batch_size=len(batch),  # ST handles micro-batching
                convert_to_numpy=True,
                normalize_embeddings=False,
            )

        return vectors

    def embed(self, texts):
        # ---------------------------------------------------------------
        # Public API method.
        #
        # Inputs:
        #   texts : list[str]
        #     Already preprocessed/clean text from MarketTextRecord.
        #
        # Workflow:
        #   1. Filter out empty strings (optional)
        #   2. Split into batches of size `self.batch_size`
        #   3. Call _embed_batch for each batch
        #   4. Concatenate batch results into a single list/array
        #   5. Return embeddings in exact input order
        #
        # Error handling:
        #   - Ensure empty inputs return empty outputs
        #   - Log or skip problematic entries
        #
        # Output:
        #   list[np.ndarray] or 2D np.ndarray
        # ---------------------------------------------------------------

        if not texts:
            return []

        # Filter but keep index alignment
        cleaned_texts = []
        valid_idx = []
        for i, t in enumerate(texts):
            if t and isinstance(t, str) and t.strip():
                cleaned_texts.append(t.strip())
                valid_idx.append(i)

        if not cleaned_texts:
            # Return an empty vector list
            return []

        # Batch processing
        all_embeds = []
        for i in range(0, len(cleaned_texts), self.batch_size):
            batch = cleaned_texts[i : i + self.batch_size]
            batch_vecs = self._embed_batch(batch)
            all_embeds.extend(batch_vecs)

        # Convert to numpy array
        all_embeds = np.vstack(all_embeds)

        # Re-insert padded entries for any skipped text inputs (empty strings)
        final_output = [None] * len(texts)
        ptr = 0
        for i in valid_idx:
            final_output[i] = all_embeds[ptr]
            ptr += 1

        return final_output

    def embed_record_batch(self, records):
        # ---------------------------------------------------------------
        # Convenience method if you want:
        # Input: list[MarketTextRecord]
        # Extract: r.text_raw for each record
        # Pass to embed()
        # Return: embeddings
        #
        # Does not mutate records - only reads text.
        # ---------------------------------------------------------------

        texts = [r.text_raw for r in records]
        return self.embed(texts)
