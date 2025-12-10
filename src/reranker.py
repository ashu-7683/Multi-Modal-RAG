# src/reranker.py
from sklearn.preprocessing import normalize
import numpy as np

def rrf(scores_list, k=60):
    """
    Reciprocal Rank Fusion on a list of ranked lists (each is list of docs with 'id')
    scores_list: list of lists of (doc_id, rank)
    returns dict doc_id -> rrf_score
    """
    rrf_scores = {}
    for lst in scores_list:
        for rank, (doc_id, _) in enumerate(lst, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
    return rrf_scores

def cross_modal_score(text_sim, image_sim, alpha=0.8):
    """
    Combine text_sim and image_sim into single score: alpha * text + (1-alpha) * image
    Both sims are floats in [0,1].
    """
    return alpha * text_sim + (1.0 - alpha) * image_sim

def rerank_by_cross_modal(hits_text, hits_image, alpha=0.8):
    """
    hits_text: list of dicts {'id','score','meta'}
    hits_image: same
    returns merged sorted list by combined score
    """
    # map id->score
    text_map = {h['meta']['id']: h['score'] for h in hits_text}
    image_map = {h['meta']['id']: h['score'] for h in hits_image}

    ids = set(list(text_map.keys()) + list(image_map.keys()))
    merged = []
    for id_ in ids:
        t = text_map.get(id_, 0.0)
        im = image_map.get(id_, 0.0)
        combined = cross_modal_score(t, im, alpha=alpha)
        # fetch meta from whichever exists
        meta = None
        for h in hits_text + hits_image:
            if h['meta']['id'] == id_:
                meta = h['meta']
                break
        merged.append({'meta': meta, 'score': combined})
    merged_sorted = sorted(merged, key=lambda x: x['score'], reverse=True)
    return merged_sorted
