# notebooks/evaluation.py
"""
Simple evaluation script to compute Retrieval Recall@k and generate benchmark report.
Assumes index already built at ./data/faiss_demo
"""
from src.vectorstore import FaissIndex
from src.embedder import embed_texts
import json
import os

def load_index():
    idx = FaissIndex.load(384, "./data/faiss_demo")
    return idx

# Example benchmark queries with gold page(s) from the Qatar doc.
BENCHMARKS = [
    {"query": "What is Qatar's real GDP growth projection for 2024?", "gold_pages": [3,4]},
    {"query": "What policy recommendations are made about VAT?", "gold_pages": [12]},
    {"query": "Show central government finance table", "gold_pages": [20]},
    # Add more based on manual inspection of qatar_test_doc.pdf
]

def recall_at_k(idx, query, gold_pages, k=5):
    q_emb = embed_texts([query])
    res = idx.search(q_emb, k=k)[0]
    retrieved_pages = [hit['meta']['page'] for hit in res]
    hit = any([p in gold_pages for p in retrieved_pages])
    return 1 if hit else 0, retrieved_pages

def run_benchmarks():
    idx = load_index()
    results = []
    for b in BENCHMARKS:
        r, pages = recall_at_k(idx, b["query"], b["gold_pages"], k=5)
        results.append({"query": b["query"], "gold": b["gold_pages"], "retrieved": pages, "hit": r})
    print("Benchmark results:")
    print(json.dumps(results, indent=2))
    # compute recall
    recall = sum([r["hit"] for r in results]) / len(results)
    print(f"Recall@5: {recall:.2f}")

if __name__ == "__main__":
    run_benchmarks()
