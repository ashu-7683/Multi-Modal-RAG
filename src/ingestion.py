# src/ingestion.py
import pdfplumber
from PIL import Image
import camelot
from typing import List, Dict, Any
from .ocr_utils import run_ocr_on_image
import io, os

def extract_from_pdf(path: str) -> List[Dict[str, Any]]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pil_page = page.to_image(resolution=200).original
            pages.append({"page_num": i, "text": text, "image": pil_page})
    try:
        tables = camelot.read_pdf(path, flavor='stream')
        for t in tables:
            p = int(t.page)
            pages[p-1].setdefault("tables", []).append(t.df)
    except Exception as e:
        print("Camelot error:", e)
    return pages

def ingest_document(path: str, use_ocr=False):
    pages = extract_from_pdf(path)
    if use_ocr:
        for p in pages:
            p["ocr_text"] = run_ocr_on_image(p["image"])
    else:
        for p in pages:
            p["ocr_text"] = ""              # skip OCR for speed
    return pages