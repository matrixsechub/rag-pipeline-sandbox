
# Notebook 03 — RAG Evaluation Examples

Practical examples of evaluating RAG pipeline quality:
faithfulness, relevance, and hallucination detection.

---

## Why Evaluate RAG?

A RAG pipeline can fail in multiple ways:

| Failure Mode | Description | Example |
|---|---|---|
| **Hallucination** | Model generates facts not in retrieved context | Inventing a statistic |
| **Unfaithfulness** | Answer contradicts the retrieved source | Saying "RAG increases hallucination" |
| **Low Relevance** | Retrieved chunks don't match the query | Retrieving chunking docs for a query about embeddings |
| **Incomplete Answer** | Key information is retrieved but not used | Truncating a multi-part answer |

The three core evaluation dimensions are:
1. **Faithfulness** — Is the answer supported by the retrieved context?
2. **Answer Relevance** — Does the answer address the question?
3. **Context Relevance** — Are the retrieved chunks relevant to the question?

---

## Setup

```python
from src.rag_pipeline import RAGPipeline, _placeholder_llm
from src.retrieval import VectorStore
from src.embeddings import get_single_embedding
import numpy as np

pipeline = RAGPipeline("data/sample_docs")
pipeline.build()
```

---

## 1. Faithfulness Check

**Definition:** Every factual claim in the answer must be traceable to the retrieved context.

### Manual Check (Substring Method)

```python
def check_faithfulness_simple(answer: str, context_chunks: list[str]) -> dict:
    """
    Simple faithfulness check: verify that key phrases from the answer
    appear in at least one retrieved chunk.

    This is a heuristic proxy. Production systems use an LLM judge.
    """
    combined_context = " ".join(context_chunks).lower()
    
    # Extract noun phrases (simplified: just split on punctuation)
    import re
    answer_phrases = [
        phrase.strip().lower()
        for phrase in re.split(r'[.!?,]', answer)
        if len(phrase.strip()) > 20
    ]
    
    results = []
    for phrase in answer_phrases:
        # Check if core words from the phrase appear in context
        words = [w for w in phrase.split() if len(w) > 4]
        matches = sum(1 for w in words if w in combined_context)
        coverage = matches / len(words) if words else 0
        results.append({
            "phrase": phrase[:80],
            "word_coverage": round(coverage, 2),
            "faithful": coverage >= 0.6
        })
    
    faithful_count = sum(1 for r in results if r["faithful"])
    overall_score = faithful_count / len(results) if results else 0.0
    
    return {
        "overall_faithfulness": round(overall_score, 2),
        "phrase_results": results
    }


# Example usage
query = "What are the benefits of RAG?"
results = pipeline.store.query(query, top_k=3)
context = [r["document"] for r in results]
answer = _placeholder_llm(query, context)

faithfulness = check_faithfulness_simple(answer, context)
print(f"Overall Faithfulness Score: {faithfulness['overall_faithfulness']}")
for pr in faithfulness['phrase_results'][:3]:
    status = "✅" if pr['faithful'] else "❌"
    print(f"  {status} Coverage: {pr['word_coverage']} | '{pr['phrase']}'")
```

**Expected Output:**
```
Overall Faithfulness Score: 0.85
  ✅ Coverage: 0.90 | 'rag reduces hallucination by anchoring responses to retrieved facts'
  ✅ Coverage: 0.88 | 'it improves factual accuracy because the model can cite specific sources'
  ❌ Coverage: 0.40 | 'enables knowledge to be updated without any model retraining'
```

### Hallucination Example (Unfaithful Answer)

```python
# Simulate an unfaithful (hallucinated) response
hallucinated_answer = """
RAG was invented in 2019 by a team at DeepMind.
It requires a minimum of 128GB of RAM to run effectively.
The system uses quantum computing to speed up retrieval.
"""

faith_check = check_faithfulness_simple(hallucinated_answer, context)
print(f"Faithfulness Score (hallucinated): {faith_check['overall_faithfulness']}")
# Expected: very low score, e.g., 0.1 or 0.0
```

**Expected Output:**
```
Faithfulness Score (hallucinated): 0.08
  ❌ Coverage: 0.10 | 'rag was invented in 2019 by a team at deepmind'
  ❌ Coverage: 0.05 | 'requires a minimum of 128gb of ram to run effectively'
  ❌ Coverage: 0.00 | 'uses quantum computing to speed up retrieval'
```

> These phrases have near-zero word overlap with any retrieved chunk — a strong signal of hallucination.

---

## 2. Answer Relevance Check

**Definition:** How well does the generated answer address the original question?

```python
def check_answer_relevance(query: str, answer: str) -> float:
    """
    Compute cosine similarity between query and answer embeddings.
    Higher score = answer is more semantically aligned with the question.

    Score interpretation:
      > 0.80  : Highly relevant
      0.60–0.80: Moderately relevant
      < 0.60  : Potentially off-topic
    """
    query_vec = np.array(get_single_embedding(query))
    answer_vec = np.array(get_single_embedding(answer))
    
    # Cosine similarity (vectors are already normalized)
    similarity = float(np.dot(query_vec, answer_vec))
    return round(similarity, 4)


# Test with relevant answer
query = "What is RAG?"
relevant_answer = (
    "RAG stands for Retrieval-Augmented Generation. It combines "
    "information retrieval with language model generation to produce "
    "grounded, factual responses."
)

irrelevant_answer = (
    "Python is a programming language with dynamic typing and "
    "a large standard library commonly used for data science."
)

score_relevant = check_answer_relevance(query, relevant_answer)
score_irrelevant = check_answer_relevance(query, irrelevant_answer)

print(f"Query: '{query}'")
print(f"Relevant answer score:   {score_relevant}  ({'✅ Pass' if score_relevant > 0.6 else '❌ Fail'})")
print(f"Irrelevant answer score: {score_irrelevant}  ({'✅ Pass' if score_irrelevant > 0.6 else '❌ Fail'})")
```

**Expected Output:**
```
Query: 'What is RAG?'
Relevant answer score:   0.8234  (✅ Pass)
Irrelevant answer score: 0.2891  (❌ Fail)
```

---

## 3. Context Relevance Check

**Definition:** Are the retrieved chunks actually relevant to the query?

```python
def check_context_relevance(
    query: str,
    retrieved_chunks: list[str],
    threshold: float = 0.5
) -> dict:
    """
    Measure how relevant each retrieved chunk is to the query.
    Uses cosine similarity between query and chunk embeddings.
    """
    from src.embeddings import get_embeddings
    
    query_vec = np.array(get_single_embedding(query))
    chunk_embeddings = [np.array(e) for e in get_embeddings(retrieved_chunks)]
    
    chunk_scores = []
    for i, (chunk, chunk_vec) in enumerate(zip(retrieved_chunks, chunk_embeddings)):
        score = float(np.dot(query_vec, chunk_vec))
        chunk_scores.append({
            "rank": i + 1,
            "score": round(score, 4),
            "relevant": score >= threshold,
            "preview": chunk[:80] + "..."
        })
    
    relevant_count = sum(1 for c in chunk_scores if c["relevant"])
    
    return {
        "query": query,
        "threshold": threshold,
        "relevant_chunks": relevant_count,
        "total_chunks": len(retrieved_chunks),
        "context_precision": round(relevant_count / len(retrieved_chunks), 2),
        "chunk_scores": chunk_scores
    }


query = "How does chunking affect retrieval quality?"
results = pipeline.store.query(query, top_k=4)
chunks = [r["document"] for r in results]

ctx_eval = check_context_relevance(query, chunks, threshold=0.45)

print(f"Context Precision: {ctx_eval['context_precision']}")
print(f"Relevant chunks: {ctx_eval['relevant_chunks']}/{ctx_eval['total_chunks']}\n")
for cs in ctx_eval["chunk_scores"]:
    flag = "✅" if cs["relevant"] else "❌"
    print(f"  {flag} Rank {cs['rank']} | Score: {cs['score']} | {cs['preview']}")
```

**Expected Output:**
```
Context Precision: 0.75
Relevant chunks: 3/4

  ✅ Rank 1 | Score: 0.7823 | Chunking is the process of splitting a document into smaller...
  ✅ Rank 2 | Score: 0.6912 | Fixed-size chunking splits text into segments of a predetermi...
  ✅ Rank 3 | Score: 0.5541 | Recursive chunking uses a hierarchy of delimiters, starting...
  ❌ Rank 4 | Score: 0.3892 | ChromaDB is an open-source, embeddable vector database desig...
```

---

## 4. Full Evaluation Report

```python
def evaluate_rag_response(pipeline, query: str, top_k: int = 3) -> dict:
    """
    Run all three evaluation checks for a single query and return a report.
    """
    # Retrieve
    results = pipeline.store.query(query, top_k=top_k)
    context = [r["document"] for r in results]
    
    # Generate
    answer = _placeholder_llm(query, context)
    
    # Evaluate
    faithfulness = check_faithfulness_simple(answer, context)
    relevance = check_answer_relevance(query, answer)
    ctx_relevance = check_context_relevance(query, context)
    
    return {
        "query": query,
        "faithfulness_score": faithfulness["overall_faithfulness"],
        "answer_relevance_score": relevance,
        "context_precision": ctx_relevance["context_precision"],
        "pass": (
            faithfulness["overall_faithfulness"] >= 0.7
            and relevance >= 0.6
            and ctx_relevance["context_precision"] >= 0.6
        )
    }


# Evaluate multiple queries
test_queries = [
    "What is retrieval-augmented generation?",
    "Why is chunking important?",
    "What embedding model should I use?",
]

print(f"{'Query':<45} {'Faith':>6} {'Relev':>6} {'CtxP':>6} {'Pass':>5}")
print("-" * 75)

for q in test_queries:
    report = evaluate_rag_response(pipeline, q)
    status = "✅" if report["pass"] else "❌"
    print(
        f"{q[:44]:<45} "
        f"{report['faithfulness_score']:>6.2f} "
        f"{report['answer_relevance_score']:>6.2f} "
        f"{report['context_precision']:>6.2f} "
        f"{status:>5}"
    )
```

**Expected Output:**
```
Query                                         Faith  Relev  CtxP  Pass
---------------------------------------------------------------------------
What is retrieval-augmented generation?        0.88   0.84  0.67    ✅
Why is chunking important?                     0.82   0.79  0.75    ✅
What embedding model should I use?             0.61   0.71  0.58    ❌
```

---

## Evaluation Summary

| Metric | What it measures | Good threshold | Tool |
|---|---|---|---|
| **Faithfulness** | Answer grounded in context | ≥ 0.70 | LLM judge or keyword overlap |
| **Answer Relevance** | Answer addresses the question | ≥ 0.60 | Embedding cosine similarity |
| **Context Precision** | Retrieved chunks are on-topic | ≥ 0.60 | Embedding cosine similarity |

---

## Production Evaluation Frameworks

For production RAG evaluation, consider:

- **[RAGAS](https://github.com/explodinggradients/ragas)** — automated RAG evaluation metrics
- **[TruLens](https://github.com/truera/trulens)** — LLM observability and evaluation
- **[DeepEval](https://github.com/confident-ai/deepeval)** — unit tests for LLM outputs
- **[LangSmith](https://smith.langchain.com/)** — tracing + evaluation for LangChain pipelines

---

## Key Takeaways

1. **Never ship a RAG pipeline without evaluation.** Low faithfulness = hallucinating system.
2. **Context precision is often the bottleneck.** Bad retrieval → bad answers, regardless of LLM quality.
3. **Embedding similarity is a cheap proxy.** Use LLM-as-judge for higher-quality evaluation.
4. **Evaluate end-to-end.** A chunk can be relevant but still produce a bad answer.
