# app/streamlit_app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import os
from src.ingestion import ingest_document
from src.chunking import create_chunks_from_pages
from src.embedder import embed_texts
from src.vectorstore import FaissIndex
from src.retriever import Retriever
from src.rag_qa import answer_query, initialize_llm
import numpy as np


st.set_page_config(page_title="Multi-Modal RAG QA", layout="wide")
st.title("Multi-Modal RAG QA")

if "index_created" not in st.session_state:
    st.session_state["index_created"] = False

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded:
    save_path = "./data/uploaded.pdf"
    os.makedirs("./data", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success("Saved uploaded PDF to data/uploaded.pdf")

if st.button("Ingest & Index") and not os.path.exists("./data/faiss_demo.index"):
#if st.button("Ingest & Index"):
    pdf_path = "./data/uploaded.pdf"
    if not os.path.exists(pdf_path):
        st.error("Upload a PDF first.")
    else:
        with st.spinner("Ingesting document..."):
            pages = ingest_document(pdf_path)
            chunks = create_chunks_from_pages(pages)
            texts = [c["text"] for c in chunks]
            embs = embed_texts(texts)
            dim = embs.shape[1]
            idx = FaissIndex(dim, index_path="./data/faiss_demo")
            metas = []
            for c in chunks:
                metas.append({"id": c["id"], "page": c["page"], "text": c["text"]})
            idx.add(embs, metas)
            idx.save("./data/faiss_demo")
            st.session_state["index_created"] = True
            st.success("Indexed document. Ready to query.")

st.markdown("---")
q = st.text_input("Ask a question about the document:")

if q and (st.session_state.get("index_created") or os.path.exists("./data/faiss_demo.index")):
    # init llm wrapper (OpenAI or local)
    initialize_llm()
    idx = FaissIndex.load(384, "./data/faiss_demo")
    retr = Retriever(idx)
    with st.spinner("Retrieving and answering..."):
        answer, hits = answer_query(q, retr, k=5)
    st.subheader("Answer")
    st.write(answer)
    st.subheader("Top retrieved chunks")
    for h in hits:
        st.markdown(f"**page {h['meta']['page']}** — score: {h['score']:.3f}")
        st.write(h['meta']['text'][:400].replace("\n", " ") + "...")
else:
    if q:
        st.info("No index found. Upload PDF and click 'Ingest & Index' first.")
