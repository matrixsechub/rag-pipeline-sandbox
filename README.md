> 📡 **GitHub SOP** — All repo operations governed by the [MSH-OPS GitHub SOP](https://app.clickup.com/9017787639/docs/8cr117q-3237). Branch strategy, PR protocol, and merge rules apply.

---

# RAG Pipeline Sandbox

> A minimal sandbox demonstrating embeddings, chunking, retrieval, and grounded LLM responses.  
> Designed for recruiters evaluating RAG fundamentals.

---

## Features

- **Document Chunking** — Fixed-size, recursive, and semantic chunking strategies
- **Embeddings** — Local sentence-transformer embeddings (no API key required)
- **Vector Storage & Retrieval** — ChromaDB for storing and querying embedded chunks
- **Grounded Response Generation** — Placeholder LLM that uses retrieved context to answer queries
- **Notebooks** — Step-by-step walkthroughs of each pipeline stage
- **Evaluation Examples** — Faithfulness, relevance, and hallucination detection demos
- **100% Local** — No paid APIs, no external services, runs entirely on your machine

---

## Tech Stack

| Component | Library |
|---|---|
| Embeddings | `sentence-transformers` |
| Vector Store | `chromadb` |
| LLM (placeholder) | Pure Python (no API) |
| Chunking | Custom + `langchain_text_splitters` |

---

*Built for learning and recruiter demos. Not production code.*