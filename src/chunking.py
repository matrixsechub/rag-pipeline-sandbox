
"""
chunking.py
-----------
Provides three chunking strategies for splitting documents into
smaller pieces suitable for embedding and retrieval.

Strategies:
  1. Fixed-size chunking     — Simple, predictable, ignores semantics
  2. Recursive chunking      — Respects sentence/paragraph boundaries
  3. Semantic chunking       — Groups sentences by embedding similarity (placeholder)
"""

from typing import List
import re

# Optional: langchain splitters for recursive strategy
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. FIXED-SIZE CHUNKING
# ---------------------------------------------------------------------------

def fixed_chunk(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into fixed-size character chunks with optional overlap.

    This is the simplest strategy. It does not respect word or sentence
    boundaries unless the split happens to land on whitespace.

    Args:
        text:       Input document string.
        chunk_size: Maximum number of characters per chunk.
        overlap:    Number of characters to overlap between consecutive chunks.

    Returns:
        List of text chunks.

    Pros:
        - Simple and deterministic
        - Fast, no dependencies
    Cons:
        - May split mid-sentence or mid-word
        - Ignores semantic structure
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap

    # Remove any empty chunks
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# 2. RECURSIVE CHUNKING
# ---------------------------------------------------------------------------

def recursive_chunk(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Split text recursively using a hierarchy of separators.

    Tries to split on: paragraphs → sentences → words → characters.
    This preserves natural language structure much better than fixed chunking.

    Uses LangChain's RecursiveCharacterTextSplitter if available,
    otherwise falls back to a simple sentence-aware splitter.

    Args:
        text:          Input document string.
        chunk_size:    Target maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.

    Pros:
        - Respects sentence/paragraph boundaries
        - More semantically coherent chunks
    Cons:
        - Slightly more complex
        - Chunk sizes may vary
    """
    if not text:
        return []

    if _LANGCHAIN_AVAILABLE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )
        return splitter.split_text(text)

    # Fallback: split on sentences using regex
    return _fallback_sentence_chunk(text, chunk_size, chunk_overlap)


def _fallback_sentence_chunk(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Simple sentence-aware chunker used when LangChain is not installed.
    Splits on punctuation boundaries and groups sentences into chunks.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk, optionally seeding with overlap
            if overlap > 0 and current_chunk:
                seed = current_chunk[-overlap:]
                current_chunk = seed + " " + sentence
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# 3. SEMANTIC CHUNKING (Placeholder / Simplified)
# ---------------------------------------------------------------------------

def semantic_chunk(
    text: str,
    similarity_threshold: float = 0.75,
    min_chunk_size: int = 100,
) -> List[str]:
    """
    Group sentences into chunks based on semantic similarity.

    PLACEHOLDER IMPLEMENTATION:
    In a full implementation, this would:
      1. Split text into sentences
      2. Embed each sentence
      3. Compute cosine similarity between adjacent sentences
      4. Start a new chunk when similarity drops below a threshold

    This demo version uses paragraph breaks and heuristic sentence
    grouping to simulate semantic boundaries without requiring
    embeddings at chunk time (avoids circular dependency).

    Args:
        text:                 Input document string.
        similarity_threshold: (Placeholder) Threshold for boundary detection.
        min_chunk_size:       Minimum characters before a new chunk is allowed.

    Returns:
        List of semantically coherent text chunks.

    Pros:
        - Produces the most coherent chunks
        - Boundaries align with topic shifts
    Cons:
        - Requires embeddings at chunk time (expensive)
        - More complex to implement and tune
    """
    if not text:
        return []

    # Split on paragraph breaks first (strong semantic boundaries)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    # Merge very short paragraphs with their successor (simulate grouping)
    merged: List[str] = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) < min_chunk_size:
            buffer += (" " if buffer else "") + para
        else:
            merged.append(buffer.strip())
            buffer = para

    if buffer:
        merged.append(buffer.strip())

    # NOTE: A production version would replace the above with:
    #   embeddings = get_embeddings(sentences)
    #   similarities = cosine_similarity(embeddings[i], embeddings[i+1])
    #   split where similarities[i] < similarity_threshold

    return [c for c in merged if c]


# ---------------------------------------------------------------------------
# CHUNKING DISPATCHER
# ---------------------------------------------------------------------------

STRATEGIES = {
    "fixed": fixed_chunk,
    "recursive": recursive_chunk,
    "semantic": semantic_chunk,
}


def chunk_text(
    text: str,
    strategy: str = "recursive",
    **kwargs,
) -> List[str]:
    """
    Dispatch to a named chunking strategy.

    Args:
        text:     Input document string.
        strategy: One of 'fixed', 'recursive', 'semantic'.
        **kwargs: Passed through to the selected strategy function.

    Returns:
        List of text chunks.

    Raises:
        ValueError: If strategy name is not recognized.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES.keys())}"
        )
    return STRATEGIES[strategy](text, **kwargs)


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = """
Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval
with large language model generation. Instead of relying solely on parametric knowledge
baked into model weights, RAG fetches relevant documents at inference time.

The retrieval component typically uses a vector database to store embedded text chunks.
When a query arrives, it is also embedded, and the nearest chunks are retrieved via
cosine similarity or approximate nearest-neighbor search.

The retrieved context is then injected into the LLM prompt, grounding the response
in factual, up-to-date documents. This significantly reduces hallucination compared
to pure generation.

Chunking is a critical preprocessing step. Poor chunking leads to poor retrieval,
which leads to poor answers. Fixed chunking is simple but naive. Recursive chunking
respects sentence boundaries. Semantic chunking groups by topic.
""".strip()

    for strategy in ["fixed", "recursive", "semantic"]:
        chunks = chunk_text(sample, strategy=strategy)
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy.upper()}  ({len(chunks)} chunks)")
        print('='*50)
        for i, c in enumerate(chunks):
            print(f"[{i+1}] {c[:120]}{'...' if len(c) > 120 else ''}")
