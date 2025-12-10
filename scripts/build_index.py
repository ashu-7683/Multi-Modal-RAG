# scripts/build_index.py
import argparse
from src.ingestion import ingest_document
from src.chunking import create_chunks_from_pages
from src.embedder import embed_texts
from src.vectorstore import FaissIndex
import os
import numpy as np

def main(pdf_path, out_path="./data/faiss_demo"):
    pages = ingest_document(pdf_path)
    chunks = create_chunks_from_pages(pages)
    texts = [c["text"] for c in chunks]
    embs = embed_texts(texts)
    dim = embs.shape[1]
    idx = FaissIndex(dim, index_path=out_path)
    metas = []
    for c in chunks:
        metas.append({"id": c["id"], "page": c["page"], "text": c["text"]})
    idx.add(embs, metas)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    idx.save(out_path)
    print("Index saved to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", default="./data/faiss_demo")
    args = parser.parse_args()
    main(args.pdf, args.out)
