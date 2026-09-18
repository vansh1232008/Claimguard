"""Embedding provider with a dependency-free fallback.

Order of preference:
1. sentence-transformers (real dense embeddings) when the package is installed.
2. A deterministic hashed bag-of-words vector, which is enough for the policy
   corpus to retrieve sensibly with zero downloads.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

import numpy as np

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")
FALLBACK_DIM = 384

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "on", "by",
    "with", "that", "this", "it", "as", "be", "at", "from", "any", "all", "not",
}


def _tokenise(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class HashingEmbedder:
    """Signed hashing trick + sublinear term frequency, L2 normalised."""

    name = "hashing-tfidf"
    dim = FALLBACK_DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[int, float] = {}
            tokens = _tokenise(text)
            for token in tokens:
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                counts[idx] = counts.get(idx, 0.0) + sign
            for idx, val in counts.items():
                if val == 0:  # signed hashing can cancel a bucket to zero
                    continue
                out[i, idx] = math.copysign(1.0 + math.log1p(abs(val)), val)
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder():
    try:
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        logger.info("Embeddings: sentence-transformers (%s)", settings.embedding_model)
        return embedder
    except Exception as exc:
        logger.info("Embeddings: hashing fallback (sentence-transformers unavailable: %s)", exc)
        return HashingEmbedder()
