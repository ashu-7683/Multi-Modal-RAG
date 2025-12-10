<<<<<<< HEAD
# Multi-Modal-RAG
=======
# Multi-Modal RAG QA — Demo

This repository implements an end-to-end multi-modal Retrieval-Augmented Generation (RAG)
prototype: ingestion (PDF, tables, images+OCR) → chunking → embeddings → FAISS retrieval →
RAG answer generation → Streamlit demo.

Test documents:
- Assignment spec: multi-modal_rag_qa_assignment.pdf. :contentReference[oaicite:2]{index=2}
- Qatar test doc: qatar_test_doc.pdf. :contentReference[oaicite:3]{index=3}

Quickstart:
1. Create venv and install: `pip install -r requirements.txt`
2. Ensure system Tesseract is installed & accessible.
3. Put `qatar_test_doc.pdf` in `data/` or upload through UI.
4. Set environment variables: `OPENAI_API_KEY` (optional) or `USE_LOCAL_LLM=1` and `LOCAL_LLM_MODEL`.
5. Run: `streamlit run app/streamlit_app.py`
>>>>>>> 0fef8f9 (first commit)
