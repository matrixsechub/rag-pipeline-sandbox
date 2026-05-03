
"""
retrieval.py
------------
Manages a ChromaDB vector store for storing and querying
embedded document chunks.

Provides:
  - VectorStore class wrapping a ChromaDB collection
  - add_documents()  : embed + store chunks with metadata
  - query()          : retrieve top-K similar chunks for a query
  - reset()          : clear the collection
"""

import uuid
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from src.embeddings import get_embeddings, get_single_embedding


# ---------------------------------------------------------------------------
# VECTOR STORE
# ---------------------------------------------------------------------------

class VectorStore:
    """
    A thin wrapper around a ChromaDB collection for RAG retrieval.

    Uses an in-memory ChromaDB client by default (no disk persistence
    required for the sandbox). Pass `persist_dir` to enable persistence.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_dir:     Optional path for persistent storage.
                         If None, data lives in memory only.

    Example:
        >>> store = VectorStore("rag_demo")
        >>> store.add_documents(["RAG is useful.", "Chunking matters."])
        >>> results = store.query("What is RAG?", top_k=1)
        >>> print(results[0]["document"])
    """

    def __init__(
        self,
        collection_name: str = "rag_sandbox",
        persist_dir: Optional[str] = None,
    ):
        self.collection_name = collection_name

        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.EphemeralClient()

        # get_or_create ensures idempotent setup
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

        print(f"[retrieval] Collection '{collection_name}' ready. "
              f"Documents in store: {self.collection.count()}")

    # ------------------------------------------------------------------
    # INGEST
    # ------------------------------------------------------------------

    def add_documents(
        self,
        chunks: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Embed a list of text chunks and add them to the collection.

        Args:
            chunks:    List of text strings to store.
            metadatas: Optional list of metadata dicts (one per chunk).
                       Useful for storing source filename, chunk index, etc.
            ids:       Optional list of unique string IDs.
                       Auto-generated (UUID) if not provided.

        Returns:
            None
        """
        if not chunks:
            print("[retrieval] No chunks provided — skipping.")
            return

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in chunks]

        # Default empty metadata if not provided
        if metadatas is None:
            metadatas = [{} for _ in chunks]

        print(f"[retrieval] Embedding {len(chunks)} chunks...")
        embeddings = get_embeddings(chunks)

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        print(f"[retrieval] Added {len(chunks)} chunks. "
              f"Total in store: {self.collection.count()}")

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-K most similar chunks for a given query.

        Embeds the query text and performs approximate nearest-neighbor
        search in the ChromaDB collection using cosine similarity.

        Args:
            query_text: The user's question or search query.
            top_k:      Number of results to return.

        Returns:
            List of dicts, each containing:
              - "document"  : the raw chunk text
              - "metadata"  : associated metadata dict
              - "id"        : the chunk's unique ID
              - "distance"  : cosine distance (lower = more similar)
              - "rank"      : 1-indexed rank in results

        Example:
            >>> results = store.query("What is retrieval?", top_k=2)
            >>> for r in results:
            ...     print(r["rank"], r["distance"], r["document"][:60])
        """
        if not query_text.strip():
            return []

        query_embedding = get_single_embedding(query_text)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Flatten ChromaDB's nested result structure
        formatted = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for rank, (doc, meta, dist, doc_id) in enumerate(
            zip(documents, metadatas, distances, ids), start=1
        ):
            formatted.append({
                "rank": rank,
                "document": doc,
                "metadata": meta,
                "id": doc_id,
                "distance": round(dist, 4),
            })

        return formatted

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (clears all data)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[retrieval] Collection '{self.collection_name}' reset.")


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    store = VectorStore("demo_collection")

    sample_chunks = [
        "Retrieval-Augmented Generation (RAG) fetches external documents to ground LLM responses.",
        "Chunking splits documents into smaller pieces for more precise retrieval.",
        "Embeddings convert text into vectors that capture semantic meaning.",
        "ChromaDB is an open-source vector database optimized for similarity search.",
        "Cosine similarity measures the angle between two vectors in embedding space.",
        "Hallucination occurs when a language model generates plausible but incorrect information.",
    ]

    metadatas = [{"source": "intro_rag.txt", "chunk_index": i}
                 for i in range(len(sample_chunks))]

    store.add_documents(sample_chunks, metadatas=metadatas)

    print("\n" + "="*60)
    query = "How does RAG reduce hallucinations?"
    print(f"Query: {query}\n")
    results = store.query(query, top_k=3)

    for r in results:
        print(f"Rank {r['rank']} | Distance: {r['distance']}")
        print(f"  {r['document']}")
        print(f"  Metadata: {r['metadata']}")
        print()
