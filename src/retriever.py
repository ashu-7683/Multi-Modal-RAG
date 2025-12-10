# src/retriever.py
from .embedder import embed_texts
from .vectorstore import FaissIndex

class Retriever:
    def __init__(self, faiss_index: FaissIndex):
        self.index = faiss_index

    def retrieve(self, query: str, k=5):
        q_emb = embed_texts([query])
        results = self.index.search(q_emb, k=k)
        return results[0]
