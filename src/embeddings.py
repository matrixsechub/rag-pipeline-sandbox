
"""
embeddings.py
-------------
Loads a local SentenceTransformer model and provides a utility
function to generate dense vector embeddings from a list of strings.

No API keys required. Model is downloaded once and cached locally.
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Default model: small, fast, and good for semantic similarity tasks.
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Module-level model cache to avoid reloading on every call.
_model: SentenceTransformer | None = None


def _load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load (or return cached) SentenceTransformer model."""
    global _model
    if _model is None:
        print(f"[embeddings] Loading model: {model_name}")
        _model = SentenceTransformer(model_name)
    return _model


def get_embeddings(
    text_list: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    normalize: bool = True,
) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        text_list:   List of strings to embed.
        model_name:  SentenceTransformer model identifier.
        batch_size:  Number of texts to encode per batch.
        normalize:   Whether to L2-normalize the output vectors.

    Returns:
        List of embedding vectors (each a list of floats).

    Example:
        >>> from src.embeddings import get_embeddings
        >>> vecs = get_embeddings(["Hello world", "RAG is useful"])
        >>> len(vecs)
        2
        >>> len(vecs[0])  # 384 for all-MiniLM-L6-v2
        384
    """
    if not text_list:
        return []

    model = _load_model(model_name)

    embeddings: np.ndarray = model.encode(
        text_list,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=len(text_list) > 100,
    )

    # Return as plain Python lists (ChromaDB-compatible)
    return embeddings.tolist()


def get_single_embedding(
    text: str,
    model_name: str = DEFAULT_MODEL,
) -> List[float]:
    """
    Convenience wrapper to embed a single string.

    Args:
        text:       The input string.
        model_name: SentenceTransformer model identifier.

    Returns:
        A single embedding vector as a list of floats.
    """
    return get_embeddings([text], model_name=model_name)[0]


if __name__ == "__main__":
    samples = [
        "Retrieval-Augmented Generation combines search with generation.",
        "Embeddings map text into a continuous vector space.",
        "ChromaDB is a lightweight vector database.",
    ]

    print("Generating embeddings for sample texts...\n")
    vecs = get_embeddings(samples)

    for text, vec in zip(samples, vecs):
        print(f"Text : {text}")
        print(f"Shape: {len(vec)} dims  |  First 5 values: {[round(v, 4) for v in vec[:5]]}")
        print()
