# src/embedder.py
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from PIL import Image
import os



import torch
# Load text embedding model
_text_model = SentenceTransformer("all-MiniLM-L12-v2")

# Load CLIP model for image->embedding support
try:
    from transformers import CLIPProcessor, CLIPModel
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    CLIP_AVAILABLE = True
except:
    CLIP_AVAILABLE = False


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of text strings."""
    return _text_model.encode(texts, normalize_embeddings=True)


def embed_image_clip(query: str | Image.Image) -> np.ndarray:
    """
    Create an image embedding using CLIP.
    If input is string -> treat as text query to match image vectors
    If input is PIL image -> embed visual content
    """
    if not CLIP_AVAILABLE:
        raise RuntimeError("CLIP model not installed. Run: pip install transformers")

    if isinstance(query, str):  # text → semantic representation for image lookup
        inputs = _clip_processor(text=[query], images=None, return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = _clip_model.get_text_features(**inputs)
    else:                       # PIL Image → extract vision embedding
        inputs = _clip_processor(images=query, return_tensors="pt")
        with torch.no_grad():
            emb = _clip_model.get_image_features(**inputs)

    emb = emb / emb.norm()      # normalize for cosine similarity
    return emb.cpu().numpy()[0]
