
# Notebook 02 — Chunking Strategies Comparison

A side-by-side comparison of Fixed, Recursive, and Semantic chunking strategies,
with pros, cons, and visual examples.

---

## Why Chunking Matters

> **Garbage in, garbage out.**
> If your chunks are incoherent or too large, retrieval quality degrades.
> The goal is chunks that are: (1) small enough to be precise, (2) large enough to be meaningful.

Chunking affects:
- **Retrieval precision** — smaller chunks = more targeted retrieval
- **Context coherence** — chunks should contain complete thoughts
- **Embedding quality** — very short chunks lose semantic signal
- **Latency** — more chunks = more embeddings to compute and store

---

## Setup

```python
from src.chunking import fixed_chunk, recursive_chunk, semantic_chunk

sample_text = """
Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval
with large language model generation. Instead of relying solely on parametric knowledge
baked into model weights, RAG fetches relevant documents at inference time.

The retrieval component typically uses a vector database to store embedded text chunks.
When a query arrives, it is also embedded, and the nearest chunks are retrieved via
cosine similarity or approximate nearest-neighbor search.

The retrieved context is then injected into the LLM prompt, grounding the response
in factual documents. This significantly reduces hallucination compared to pure generation.

Chunking is a critical preprocessing step. Poor chunking leads to poor retrieval,
which leads to poor answers. Fixed chunking is simple but naive. Recursive chunking
respects sentence boundaries. Semantic chunking groups by topic.
""".strip()

print(f"Sample text length: {len(sample_text)} characters")
```

---

## Strategy 1: Fixed-Size Chunking

```python
fixed_chunks = fixed_chunk(sample_text, chunk_size=200, overlap=30)

print(f"Fixed chunking → {len(fixed_chunks)} chunks\n")
for i, chunk in enumerate(fixed_chunks):
    print(f"[{i+1}] ({len(chunk)} chars)")
    print(f"     {chunk[:100]}...")
    print()
```

**Example Output:**
```
Fixed chunking → 6 chunks

[1] (200 chars)
     Retrieval-Augmented Generation (RAG) is a technique that combines information
     retrieval with large language model gen...

[2] (200 chars)
     age model generation. Instead of relying solely on parametric knowledge
     baked into model weights, RAG fetches...

[3] (200 chars)
     into model weights, RAG fetches relevant documents at inference time.
     The retrieval component typically uses...
```

> ⚠️ Notice: Chunk 2 starts mid-sentence ("age model generation.") — a classic
> fixed chunking failure. The split at character 200 falls inside a word.

### Fixed Chunking: Pros & Cons

| ✅ Pros | ❌ Cons |
|---|---|
| Simple to implement | May split mid-sentence or mid-word |
| Deterministic and predictable | Ignores semantic structure |
| No dependencies | Poor retrieval at chunk boundaries |
| Very fast | Overlap is a workaround, not a solution |

**Best for:** Quick prototyping, structured data with predictable formats.

---

## Strategy 2: Recursive Chunking

```python
recursive_chunks = recursive_chunk(sample_text, chunk_size=200, chunk_overlap=30)

print(f"Recursive chunking → {len(recursive_chunks)} chunks\n")
for i, chunk in enumerate(recursive_chunks):
    print(f"[{i+1}] ({len(chunk)} chars)")
    print(f"     {chunk[:100]}...")
    print()
```

**Example Output:**
```
Recursive chunking → 5 chunks

[1] (198 chars)
     Retrieval-Augmented Generation (RAG) is a technique that combines information
     retrieval with large language model generation.

[2] (195 chars)
     Instead of relying solely on parametric knowledge baked into model weights,
     RAG fetches relevant documents at inference time.

[3] (189 chars)
     The retrieval component typically uses a vector database to store embedded
     text chunks.
```

> ✅ Notice: Each chunk ends at a natural sentence boundary. Recursive chunking
> tries paragraph breaks first, then sentence breaks, before resorting to character splits.

### Recursive Chunking: Pros & Cons

| ✅ Pros | ❌ Cons |
|---|---|
| Respects sentence/paragraph boundaries | Slightly more complex |
| More coherent chunks | Chunk sizes vary |
| Better retrieval quality than fixed | Requires a text splitter library |
| Industry standard approach | Still doesn't understand semantics |

**Best for:** Most RAG use cases. Start here.

---

## Strategy 3: Semantic Chunking

```python
semantic_chunks = semantic_chunk(sample_text, min_chunk_size=150)

print(f"Semantic chunking → {len(semantic_chunks)} chunks\n")
for i, chunk in enumerate(semantic_chunks):
    print(f"[{i+1}] ({len(chunk)} chars)")
    print(f"     {chunk[:120]}...")
    print()
```

**Example Output:**
```
Semantic chunking → 4 chunks

[1] (312 chars)
     Retrieval-Augmented Generation (RAG) is a technique that combines information
     retrieval with large language model generation. Instead of relying...

[2] (289 chars)
     The retrieval component typically uses a vector database to store embedded
     text chunks. When a query arrives...

[3] (254 chars)
     The retrieved context is then injected into the LLM prompt, grounding the
     response in factual documents...
```

> ✅ Each chunk corresponds to a conceptual unit: (1) RAG overview,
> (2) retrieval mechanics, (3) generation & grounding, (4) chunking importance.

### Full Semantic Chunking (Production Implementation)

```python
# This is how a production semantic chunker works.
# Not implemented here to avoid circular embedding dependency at chunk time,
# but this is the correct approach:

from src.embeddings import get_embeddings
import numpy as np

def production_semantic_chunk(text, threshold=0.75):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    embeddings = get_embeddings(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Cosine similarity between adjacent sentence embeddings
        e1 = np.array(embeddings[i-1])
        e2 = np.array(embeddings[i])
        similarity = np.dot(e1, e2)  # Already normalized
        
        if similarity >= threshold:
            current_chunk.append(sentences[i])  # Same topic → keep together
        else:
            chunks.append(" ".join(current_chunk))  # Topic shift → new chunk
            current_chunk = [sentences[i]]
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
```

### Semantic Chunking: Pros & Cons

| ✅ Pros | ❌ Cons |
|---|---|
| Most semantically coherent chunks | Requires embedding each sentence |
| Boundaries align with topic shifts | Computationally expensive |
| Best retrieval quality in theory | Threshold tuning required |
| Works well for dense documents | Harder to debug and explain |

**Best for:** High-quality production RAG where retrieval precision is critical.

---

## Side-by-Side Comparison

Given the same 600-character input:

| Strategy | Chunks | Avg Size | Boundary Type | Compute Cost |
|---|---|---|---|---|
| Fixed | 6 | 200 chars | Character count | Very Low |
| Recursive | 5 | 190 chars | Sentence/paragraph | Low |
| Semantic | 4 | 235 chars | Topic shift | High |

---

## Choosing a Strategy

```
Is this a prototype?
  └─ Yes → Fixed chunking is fine
  └─ No  → Is compute cost a concern?
              └─ Yes → Recursive chunking (recommended default)
              └─ No  → Semantic chunking for best quality
```

---

## Key Takeaway

> There is no universally "best" chunking strategy.
> The right choice depends on document type, retrieval quality requirements,
> and available compute. **Recursive chunking is the best default.**
