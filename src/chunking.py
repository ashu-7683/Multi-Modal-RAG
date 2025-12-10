# src/chunking.py
from typing import List, Dict

def split_text_semantic(text, max_chars=2500):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 < max_chars:
            cur = (cur + "\n\n" + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks

def create_chunks_from_pages(pages):
    chunks = []
    for p in pages:
        page_num = p["page_num"]
        text = (p.get("text","") + "\n\n" + p.get("ocr_text","")).strip()
        text_chunks = split_text_semantic(text)
        for i, c in enumerate(text_chunks):
            chunks.append({
                "id": f"page{page_num}_chunk{i}",
                "page": page_num,
                "text": c,
                "table": False,
                "source": "pdf_page",
            })
        for t_i, table in enumerate(p.get("tables", [])):
            table_text = table.to_csv(index=False)
            chunks.append({
                "id": f"page{page_num}_table{t_i}",
                "page": page_num,
                "text": table_text,
                "table": True,
                "source": "pdf_table"
            })
    return chunks
