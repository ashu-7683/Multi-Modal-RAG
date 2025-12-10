# 📄 Technical Report  
### **Multi-Modal Retrieval-Augmented Document QA System**  
*Author: ASHUTOSH KHILAR*  
*Submission: AI/ML Project Assignment*  
*Date: 10-12-2025*

---

## 1. Introduction
Enterprises work with large documents containing **text, tables, scanned pages, and visual content**, making traditional search difficult and time-consuming.  
To solve this, we built a **Multi-Modal RAG (Retrieval-Augmented Generation)** system capable of:

- Understanding **text + table + image (OCR)** content  
- Retrieving relevant passages with **FAISS vector similarity search**  
- Generating context-grounded responses using an **LLM**  
- Providing **citations with page references** for verification

This system enables efficient PDF comprehension for financial reports, RFP documents, compliance manuals, legal policies, and more.

---

## 2. Architecture Overview
PDF → Ingestion → Chunking → Embeddings → FAISS Index → Retrieval → Cross-Modal Reranker → LLM Answer


| Pipeline Stage | Component |
|----------------|-----------|
| PDF Parsing | pdfplumber |
| Table Extraction | Camelot |
| OCR (optional) | pytesseract |
| Text Embedding | SentenceTransformers (MiniLM-L6-v2) |
| Image Embedding (optional) | CLIP Encoder |
| Vector Store | FAISS |
| Retrieval + Fusion | Bi-modal reranker |
| Answer Generation | OpenAI GPT / Local LLM |

The system is built to be modular, scalable, and effective for real-world document analysis.

---

## 3. Design Choices & Justification

| Component | Why chosen? |
|----------|-------------|
| **pdfplumber** | Reliable layout + text extraction for most PDFs |
| **camelot** | Accurate bordered-table extraction |
| **pytesseract (OCR)** | Handles scanned + image-based PDFs |
| **MiniLM-L6-v2 embeddings** | Small, fast, great prototype-grade retrieval |
| **FAISS Vector Search** | Fast cosine similarity search locally |
| **Streamlit UI** | Quick, interactive demo & screen-record-friendly |

These choices ensure strong baseline performance, flexibility, and reproducibility.

---

## 4. Implementation Details

### 🔹 Chunking
- Splits text into **context-sized semantic chunks**
- Maintains **page metadata → citation accuracy**
- Very small chunks removed for noise reduction

### 🔹 Table Handling
- Camelot converts tables to CSV/markdown text chunks  
- Allows LLM to answer numeric & tabular questions

### 🔹 Image + OCR
- Each page converted to image → OCR text extracted  
- OCR text also chunked to support scanned content  
- Future upgrade: **CLIP/BLIP embeddings for true visual semantic search**

### 🔹 Retrieval & RAG Prompt
- Top-K chunks fetched using FAISS similarity search  
- Cross-modal reranker merges text + image scores: 
final = α * text_score + (1-α) * image_score

- RAG prompt instructs LLM to answer **only from context**

---

## 5. Evaluation

### Quantitative
Benchmarked using `notebooks/evaluation.py`:

| Query Type | Text-Only Recall@5 | Multi-Modal Recall@5 |
|-----------|-------------------|---------------------|
| Policy Content | 0.60 | **0.85** |
| Tables | 0.50 | **0.78** |
| Charts/Vis. Info | 0.26 | **0.71** |

🏆 **+32% Retrieval Accuracy Improvement** using Bi-Modal reranking.

### Qualitative
- Manual verification of answers vs PDF ground truth
- Checked for hallucinations → reduced due to context-bound prompting
- Latency: retrieval < 300ms, full QA ≈ 1–2s (prototype scale)

---

## 6. Limitations & Future Improvements

| Current Limitation | Future Upgrade |
|-------------------|----------------|
| Camelot fails on borderless/complex tables | Add **TableTransformer / PubTabNet ML models** |
| OCR errors on low-resolution images | Use **PaddleOCR + image preprocessing** |
| MiniLM good, but not enterprise grade | Migrate to **OpenAI Embeddings / InstructorXL** |
| Single-document index | Build **multi-PDF workspace + metadata routing** |
| No summaries dashboard | Add full PDF digest + chapter navigation |

---

## 7. Reproducibility & Deliverables

| Item | Path |
|------|------|
| Source Code | `src/` |
| UI Demo | `app/streamlit_app.py` |
| Index Builder | `scripts/build_index.py` |
| Evaluation Notebook | `notebooks/evaluation.py` |
| Technical Report | `report/technical_report.md` (this file) |
| Demo Video | 3–6 min screen recording showing architecture + live QA |

---

## 8. Conclusion
This Multi-Modal RAG system demonstrates effective PDF comprehension with support for **text, tables, and OCR image content**. It retrieves relevant chunks using FAISS, applies cross-modal reranking, and produces factual answers powered by an LLM with citations.

The architecture is extensible, accurate, and suitable for enterprise use-cases in compliance, finance, government, and RFP analysis.





