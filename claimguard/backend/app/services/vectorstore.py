"""Policy RAG vector store.

Qdrant when QDRANT_URL is set; otherwise an in-process cosine-similarity index
over the same chunks. Both expose ``search(query, top_k)`` returning the same
chunk dicts, so the policy agent never knows which one it is talking to.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.config import settings
from app.core.logging import get_logger
from app.services.embeddings import get_embedder

logger = get_logger(__name__)


@dataclass
class PolicyChunk:
    chunk_id: str
    clause_id: str
    title: str
    text: str
    source: str
    product: str = "motor_comprehensive"

    def to_payload(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------- loading
_HEADING_RE = re.compile(r"^##\s+(?P<clause>[A-Z0-9.\-]+)\s+(?P<title>.+)$", re.MULTILINE)


def load_policy_chunks(corpus_dir: Path | None = None) -> list[PolicyChunk]:
    """Parse the markdown policy corpus into clause-level chunks.

    Each `## <clause-id> <title>` heading starts a new chunk. Clause-level
    chunking matters: retrieval has to cite the exact clause the decision rests
    on, and paragraph-level chunks lose that anchor.
    """
    corpus_dir = corpus_dir or settings.policy_corpus_dir
    chunks: list[PolicyChunk] = []
    if not corpus_dir.exists():
        logger.warning("Policy corpus directory %s does not exist", corpus_dir)
        return chunks

    for path in sorted(corpus_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        product = "motor_comprehensive"
        first_line = raw.splitlines()[0] if raw.splitlines() else ""
        if first_line.startswith("# "):
            product = first_line[2:].strip().lower().replace(" ", "_")

        matches = list(_HEADING_RE.finditer(raw))
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            body = raw[start:end].strip()
            if not body:
                continue
            clause_id = match.group("clause")
            title = match.group("title").strip()
            chunks.append(
                PolicyChunk(
                    chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{path.name}:{clause_id}").hex,
                    clause_id=clause_id,
                    title=title,
                    text=f"{clause_id} {title}\n{body}",
                    source=path.name,
                    product=product,
                )
            )
    logger.info("Loaded %d policy clauses from %s", len(chunks), corpus_dir)
    return chunks


# ---------------------------------------------------------------- in-memory
class InMemoryVectorStore:
    name = "in-memory"

    def __init__(self) -> None:
        self._chunks: list[PolicyChunk] = []
        self._matrix: np.ndarray | None = None
        self._embedder = get_embedder()

    def index(self, chunks: list[PolicyChunk]) -> int:
        self._chunks = chunks
        if not chunks:
            self._matrix = None
            return 0
        self._matrix = self._embedder.encode([c.text for c in chunks])
        return len(chunks)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if self._matrix is None or not self._chunks:
            return []
        q = self._embedder.encode([query])[0]
        scores = self._matrix @ q
        order = np.argsort(-scores)[:top_k]
        results = []
        for idx in order:
            chunk = self._chunks[int(idx)]
            payload = chunk.to_payload()
            payload["score"] = round(float(scores[int(idx)]), 4)
            results.append(payload)
        return results

    def count(self) -> int:
        return len(self._chunks)


# -------------------------------------------------------------------- qdrant
class QdrantVectorStore:
    name = "qdrant"

    def __init__(self) -> None:
        from qdrant_client import QdrantClient

        self._embedder = get_embedder()
        self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self._collection = settings.qdrant_collection

    def index(self, chunks: list[PolicyChunk]) -> int:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        if not chunks:
            return 0
        vectors = self._embedder.encode([c.text for c in chunks])
        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=i, vector=vectors[i].tolist(), payload=chunks[i].to_payload())
            for i in range(len(chunks))
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        vector = self._embedder.encode([query])[0].tolist()
        hits = self._client.search(
            collection_name=self._collection, query_vector=vector, limit=top_k
        )
        results = []
        for hit in hits:
            payload = dict(hit.payload or {})
            payload["score"] = round(float(hit.score), 4)
            results.append(payload)
        return results

    def count(self) -> int:
        try:
            return int(self._client.count(self._collection).count)
        except Exception:
            return 0


# --------------------------------------------------------------------- façade
_store = None


def get_vector_store():
    global _store
    if _store is not None:
        return _store
    if settings.qdrant_url:
        try:
            _store = QdrantVectorStore()
            logger.info("Vector store: Qdrant at %s", settings.qdrant_url)
        except Exception as exc:
            logger.warning("Qdrant unavailable (%s); using in-memory store", exc)
            _store = InMemoryVectorStore()
    else:
        _store = InMemoryVectorStore()
        logger.info("Vector store: in-memory")
    if _store.count() == 0:
        _store.index(load_policy_chunks())
    return _store


def reindex_policies() -> dict:
    store = get_vector_store()
    chunks = load_policy_chunks()
    n = store.index(chunks)
    return {"backend": store.name, "indexed_clauses": n}


def dump_chunks(path: Path) -> None:
    """Handy for debugging retrieval quality."""
    chunks = [c.to_payload() for c in load_policy_chunks()]
    path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
