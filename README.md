
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

## Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE                            │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Load    │───▶│  Chunk   │───▶│  Embed   │              │
│  │  Docs    │    │  Text    │    │  Chunks  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                        │                    │
│                                        ▼                    │
│                                  ┌──────────┐              │
│                                  │  Store   │              │
│                                  │ ChromaDB │              │
│                                  └──────────┘              │
│                                        │                    │
│  ┌──────────┐    ┌──────────┐         │                    │
│  │ Generate │◀───│ Retrieve │◀────────┘                    │
│  │ Response │    │ Top-K    │    (query embedding)         │
│  └──────────┘    └──────────┘                              │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────┐                                              │
│  │  Answer  │                                              │
│  └──────────┘                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## How It Works

1. **Load** — Raw `.txt` documents are read from `/data/sample_docs/`
2. **Chunk** — Documents are split into smaller overlapping pieces to preserve context
3. **Embed** — Each chunk is converted into a dense vector using a local SentenceTransformer model
4. **Store** — Vectors + metadata are persisted in a local ChromaDB collection
5. **Retrieve** — At query time, the query is embedded and the top-K most similar chunks are fetched
6. **Generate** — A (placeholder) LLM combines the retrieved context with the query to produce a grounded answer

---

## Notebooks Overview

| Notebook | Description |
|---|---|
| `01_basic_rag_pipeline.md` | End-to-end walkthrough: load → chunk → embed → retrieve → respond |
| `02_chunking_strategies.md` | Compare fixed, recursive, and semantic chunking with pros/cons |
| `03_evaluation_examples.md` | Faithfulness checks, relevance scoring, hallucination detection |

---

## Example Queries

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(docs_dir="data/sample_docs")
pipeline.build()

answer = pipeline.query("What is retrieval-augmented generation?")
print(answer)

answer = pipeline.query("How does chunking affect retrieval quality?")
print(answer)
```

---

## Safety Notes

- No proprietary or company-internal content is included
- No paid or external API calls are made
- No sensitive or personal data is stored
- All models and vector stores run locally
- This is **demonstration code only** — not production-ready

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/rag-pipeline-sandbox.git
cd rag-pipeline-sandbox

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
# Run the full end-to-end pipeline
python src/rag_pipeline.py

# Or use it interactively in Python
python -c "
from src.rag_pipeline import RAGPipeline
p = RAGPipeline('data/sample_docs')
p.build()
print(p.query('What is RAG?'))
"
```

---

## Future Extensions

- [ ] Swap placeholder LLM with a real local model (e.g., Ollama + Llama 3)
- [ ] Add hybrid retrieval (BM25 + dense vectors)
- [ ] Implement re-ranking (cross-encoder)
- [ ] Add streaming response output
- [ ] Build a simple Gradio or Streamlit UI
- [ ] Add RAGAS-based automated evaluation
- [ ] Support PDF, HTML, and Markdown ingestion
- [ ] Add parent-document retrieval strategy

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
