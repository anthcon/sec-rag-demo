# reranker.py
"""
Second-stage cross-encoder reranker with CE score floor and balanced selection.

Design decisions (walkthrough prep):
  - Model: BAAI/bge-reranker-v2-m3 — lightweight, no GPU needed for <60 candidates.
  - Lazy-loaded singleton: model loads once per process.
  - CE score floor (0.05): prevents good similarity matches from being
    zeroed out by an overly literal cross-encoder. Chunks already passed
    a similarity gate to reach this stage.
  - Fused score = 0.7 * norm_ce + 0.3 * norm_adj.
  - Balanced selection guarantees each target company gets fair representation.
"""

from sentence_transformers import CrossEncoder

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_model = None

CE_SCORE_FLOOR = 0.05  # minimum normalized CE score for any retrieved chunk


def _get_model():
    global _model
    if _model is None:
        print(f"  Loading cross-encoder: {_MODEL_NAME} …")
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def _normalize(values):
    """Min-max normalize a list of floats to [0, 1]."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def cross_encoder_rerank(
    query: str,
    reranked_results: list,       # list of (doc, sim_score, adj_score)
    top_k: int = 20,
    ce_weight: float = 0.7,
    target_companies: list = None,
):
    """
    Score all candidates with the cross-encoder, apply CE floor,
    fuse with recency, then balanced-select across companies.

    Args:
        query:              search query for CE scoring
        reranked_results:   output of rerank_results() — 3-tuples of
                            (Document, similarity_score, adjusted_score)
        top_k:              how many to return
        ce_weight:          blend weight for cross-encoder vs recency
        target_companies:   list of company names for balanced selection

    Returns:
        list of (doc, sim_score, adj_score, raw_ce_score, fused_score)
        sorted descending by fused_score, length <= top_k
    """
    if not reranked_results:
        return []

    model = _get_model()

    # ---- Score all (query, chunk) pairs ----
    pairs = [(query, doc.page_content) for doc, _sim, _adj in reranked_results]
    raw_ce_scores = model.predict(pairs).tolist()

    # ---- Normalize CE scores, then apply floor ----
    norm_ce = _normalize(raw_ce_scores)
    norm_ce_floored = [max(score, CE_SCORE_FLOOR) for score in norm_ce]

    # ---- Normalize recency-adjusted scores ----
    adj_scores = [adj for _doc, _sim, adj in reranked_results]
    norm_adj = _normalize(adj_scores)

    # ---- Fuse ----
    fused = []
    for i, (doc, sim, adj) in enumerate(reranked_results):
        fused_score = ce_weight * norm_ce_floored[i] + (1 - ce_weight) * norm_adj[i]
        fused.append((doc, sim, adj, raw_ce_scores[i], fused_score))

    # ---- Balanced selection or global top-k ----
    if target_companies and len(target_companies) > 1:
        return _balanced_select(fused, target_companies, top_k)
    else:
        fused.sort(key=lambda x: x[4], reverse=True)
        return fused[:top_k]


def _balanced_select(fused, target_companies, top_k):
    """
    Guarantee each target company gets at least per_company slots,
    then fill remaining by best fused score across all companies.
    """
    # Bucket by company, sort each bucket by fused score
    buckets = {}
    for item in fused:
        company = item[0].metadata.get("company", "Unknown")
        buckets.setdefault(company, []).append(item)
    for pool in buckets.values():
        pool.sort(key=lambda x: x[4], reverse=True)

    per_company = max(1, top_k // len(target_companies))
    remainder = top_k - (per_company * len(target_companies))

    selected = []
    extras = []

    for company in target_companies:
        pool = buckets.get(company, [])
        selected.extend(pool[:per_company])
        extras.extend(pool[per_company:])

    # Non-target company chunks go into extras too
    for c, pool in buckets.items():
        if c not in target_companies:
            extras.extend(pool)

    # Fill remaining slots by fused score
    if remainder > 0:
        extras.sort(key=lambda x: x[4], reverse=True)
        selected.extend(extras[:remainder])

    selected.sort(key=lambda x: x[4], reverse=True)
    return selected