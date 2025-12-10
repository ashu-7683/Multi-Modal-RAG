# src/vectorstore.py
import faiss
import numpy as np
import pickle

class FaissIndex:
    def __init__(self, dim:int, index_path=None):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas = []
        self.index_path = index_path

    def add(self, embeddings, metadatas):
        import faiss
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadatas.extend(metadatas)

    def search(self, q_emb, k=5):
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        results = []
        for i_list, d_list in zip(I, D):
            hits = []
            for idx, score in zip(i_list, d_list):
                if idx == -1:
                    continue
                hits.append({"score": float(score), "meta": self.metadatas[idx], "idx": int(idx)})
            results.append(hits)
        return results

    def save(self, path):
        faiss.write_index(self.index, path + ".index")
        with open(path + ".meta.pkl", "wb") as f:
            pickle.dump(self.metadatas, f)

    @classmethod
    def load(cls, dim, path):
        obj = cls(dim, index_path=path)
        obj.index = faiss.read_index(path + ".index")
        with open(path + ".meta.pkl", "rb") as f:
            obj.metadatas = pickle.load(f)
        return obj
