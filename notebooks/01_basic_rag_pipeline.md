
# Notebook 01 — Basic RAG Pipeline Walkthrough

A step-by-step walkthrough of the complete RAG pipeline from raw documents to grounded answers.

---

## Overview

```
Load Docs → Chunk → Embed → Store → Retrieve → Generate Response
```

Each step is explained with code and output examples.

---

## Step 1: Load Documents

Documents are plain `.txt` files stored in `data/sample_docs/`.

```python
from pathlib import Path

docs_dir = Path("data/sample_docs")

documents = []
for filepath in sorted(docs_dir.glob("*.txt")):
    content = filepath.read_text(encoding="utf-8").strip()
    documents.append({"filename": filepath.name, "content": content})
    print(f"Loaded: {filepath.name} ({len(content)} chars)")
```

**Expected Output:**
```
Loaded: doc1.txt (2341 chars)
Loaded: doc2.txt (2187 chars)
```

---

## Step 2: Chunk Documents

We split each document into smaller overlapping segments.

```python
from src.chunking import chunk_text

all_chunks = []

for doc in documents:
    chunks = chunk_text(doc["content"], strategy="recursive", chunk_size=300, chunk_overlap=50)
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "text": chunk,
            "source": doc["filename"],
            "chunk_index": i
        })

print(f"Total chunks: {len(all_chunks)}")
print(f"\nFirst chunk preview:\n{all_chunks[0]['text'][:200]}")
```

**Expected Output:**
```
Total chunks: 18

First chunk preview:
Retrieval-Augmented Generation (RAG) is a machine learning technique that combines
the strengths of information retrieval systems with large language model generation.
```

---

## Step 3: Generate Embeddings

Each chunk is converted into a dense 384-dimensional vector.

```python
from src.embeddings import get_embeddings

texts = [c["text"] for c in all_chunks]
embeddings = get_embeddings(texts)

print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimensions: {len(embeddings[0])}")
print(f"First 5 values of embedding[0]: {[round(v, 4) for v in embeddings[0][:5]]}")
```

**Expected Output:**
```
Number of embeddings: 18
Embedding dimensions: 384
First 5 values of embedding[0]: [0.0521, -0.0342, 0.1183, -0.0891, 0.0674]
```

> **Why normalize?** L2-normalizing embeddings converts cosine similarity to a simple dot product,
> which is computationally cheaper and numerically stable.

---

## Step 4: Store in ChromaDB

Embeddings and their corresponding texts are stored in a vector collection.

```python
from src.retrieval import VectorStore

store = VectorStore("rag_demo")

metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in all_chunks]
ids = [f"{c['source']}_chunk_{c['chunk_index']}" for c in all_chunks]

store.add_documents(texts, metadatas=metadatas, ids=ids)

print(f"Chunks in store: {store.count()}")
```

**Expected Output:**
```
[retrieval] Collection 'rag_demo' ready. Documents in store: 0
[retrieval] Embedding 18 chunks...
[retrieval] Added 18 chunks. Total in store: 18
Chunks in store: 18
```

---

## Step 5: Retrieve Relevant Chunks

At query time, we embed the question and find the most similar chunks.

```python
query = "What is retrieval-augmented generation?"

results = store.query(query, top_k=3)

for r in results:
    print(f"Rank {r['rank']} | Distance: {r['distance']} | Source: {r['metadata']['source']}")
    print(f"  {r['document'][:120]}...\n")
```

**Expected Output:**
```
Rank 1 | Distance: 0.0412 | Source: doc1.txt
  Retrieval-Augmented Generation (RAG) is a machine learning technique that combines the strengths ...

Rank 2 | Distance: 0.1038 | Source: doc1.txt
  RAG addresses these limitations by augmenting the generation process with a dynamic retrieval step...

Rank 3 | Distance: 0.1891 | Source: doc2.txt
  Chunking is the process of splitting a document into smaller, overlapping text segments...
```

> **What is distance?** Lower cosine distance = higher similarity.
> A distance of 0.04 means the query and chunk are almost identical in meaning.

---

## Step 6: Generate a Grounded Response

The retrieved chunks are injected into a prompt for the LLM.

```python
from src.rag_pipeline import _placeholder_llm

context = [r["document"] for r in results]
response = _placeholder_llm(query, context)

print(response)
```

**Expected Output:**
```
QUERY: What is retrieval-augmented generation?

============================================================
RETRIEVED CONTEXT (3 chunks):
============================================================
[Source 1]: Retrieval-Augmented Generation (RAG) is a machine learning technique...
----------------------------------------
[Source 2]: RAG addresses these limitations by augmenting the generation process...
----------------------------------------
[Source 3]: Chunking is the process of splitting a document...

============================================================
GROUNDED RESPONSE (placeholder):
============================================================
Based on the retrieved documents, here is a summary:

  • Retrieval-Augmented Generation (RAG) is a machine learning technique.
  • RAG addresses these limitations by augmenting the generation process.
  • Chunking is the process of splitting a document into smaller segments.
```

---

## Full Pipeline (One-liner)

All of the above steps are encapsulated in `RAGPipeline`:

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(docs_dir="data/sample_docs", chunk_strategy="recursive", top_k=3)
pipeline.build()

answer = pipeline.query("What is retrieval-augmented generation?")
print(answer)
```

---

## Key Takeaways

| Step | Purpose | Key Component |
|---|---|---|
| Load | Get raw text into memory | `pathlib.Path.read_text()` |
| Chunk | Split into retrievable units | `src/chunking.py` |
| Embed | Convert text to vectors | `src/embeddings.py` |
| Store | Index vectors for fast search | `ChromaDB` |
| Retrieve | Find semantically similar chunks | Cosine similarity |
| Generate | Produce a grounded answer | LLM + context prompt |
