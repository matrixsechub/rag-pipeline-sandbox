"""End-to-end RAGPipeline test against a real in-memory ChromaDB.

sentence-transformers pulls in torch, so when it is not installed a
deterministic bag-of-words embedder stands in for it. Skipped entirely when
chromadb is unavailable.
"""

import hashlib
import os
import sys
import types
from pathlib import Path

import pytest

if os.environ.get("RAG_REQUIRE_CHROMADB") == "1":
    # CI sets this so a missing dependency fails instead of silently skipping.
    import chromadb  # noqa: F401
    import numpy as np
else:
    pytest.importorskip("chromadb")
    np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]


class _HashingEncoder:
    """Bag-of-words hashed into 256 dims; shared words => similar vectors."""

    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=True, **_):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                word = word.strip(".,?!()\"'")
                if word:
                    out[row, int(hashlib.sha256(word.encode()).hexdigest(), 16) % 256] += 1.0
        if normalize_embeddings:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out


try:
    import sentence_transformers  # noqa: F401
except ImportError:
    sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=_HashingEncoder)

from src.rag_pipeline import RAGPipeline  # noqa: E402


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic"])
def test_build_and_query_every_strategy(strategy):
    pipeline = RAGPipeline(
        docs_dir=str(ROOT / "data" / "sample_docs"),
        chunk_strategy=strategy,
        collection_name=f"pytest_{strategy}",
        top_k=2,
    )
    pipeline.store.reset()
    pipeline.build()

    assert pipeline.store.count() > 0
    results = pipeline.store.query("What is retrieval-augmented generation?", top_k=2)
    assert [r["rank"] for r in results] == [1, 2]
    assert {r["metadata"]["source"] for r in results} <= {"doc1.txt", "doc2.txt"}

    answer = pipeline.query("What is retrieval-augmented generation?")
    assert "[Source 1]" in answer


def test_query_before_build_is_rejected():
    pipeline = RAGPipeline(docs_dir=str(ROOT / "data" / "sample_docs"), collection_name="pytest_unbuilt")
    with pytest.raises(RuntimeError, match="not built"):
        pipeline.query("anything")
