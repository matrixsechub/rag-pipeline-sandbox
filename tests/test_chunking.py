"""Unit tests for src/chunking.py (stdlib only)."""

import pytest

from src import chunking
from src.chunking import chunk_text, fixed_chunk, semantic_chunk

TEXT = (
    "RAG retrieves documents at inference time. It grounds answers in sources. "
    "Chunking splits documents. Embeddings map text to vectors. "
    "Vector stores search by similarity."
)


def test_fixed_chunk_respects_size_and_overlap():
    text = "abcdefghij" * 10  # 100 chars, no whitespace to strip
    chunks = fixed_chunk(text, chunk_size=30, overlap=10)
    assert all(len(c) <= 30 for c in chunks)
    # Each chunk starts 20 chars after the previous one.
    assert chunks[1] == text[20:50]
    assert "".join(c[:20] for c in chunks[:-1]) + chunks[-1] == text


@pytest.mark.parametrize("size,overlap", [(10, 10), (10, 15), (0, 0), (10, -1)])
def test_fixed_chunk_rejects_parameters_that_cannot_advance(size, overlap):
    with pytest.raises(ValueError):
        fixed_chunk("some text", chunk_size=size, overlap=overlap)


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic"])
def test_empty_text_yields_no_chunks(strategy):
    assert chunk_text("", strategy=strategy) == []


def test_recursive_fallback_keeps_sentences_whole(monkeypatch):
    monkeypatch.setattr(chunking, "_LANGCHAIN_AVAILABLE", False)
    chunks = chunk_text(TEXT, strategy="recursive", chunk_size=80, chunk_overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)
    assert all(c.endswith(".") for c in chunks)


def test_semantic_chunk_merges_short_paragraphs():
    text = "Short one.\n\nShort two.\n\n" + ("Long paragraph. " * 10).strip()
    chunks = semantic_chunk(text, min_chunk_size=50)
    assert chunks[0].startswith("Short one. Short two.")
    assert all(c.strip() for c in chunks)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        chunk_text(TEXT, strategy="nope")
