# reranker.py
"""
Second-stage cross-encoder reranker.
Scores (query, chunk) pairs with a cross-encoder and returns
the top-k results sorted by fused score (cross-encoder + recency).

Design decisions (be ready to explain in walkthrough):
  - Model: BAAI/bge-reranker-v2-m3 — lightweight, strong on financial text,
    no GPU required for <60 candidates.
  - Lazy-loaded singleton so the model is only downloaded/loaded once per
    process, not per query.
  - Fused score = 0.7 * normalized_ce_score + 0.3 * normalized_adj_score.
    Keeps recency signal without letting it dominate semantic relevance.
  - Balanced selection guarantees each target company gets fair
    representation before filling remaining slots by fused score.
"""

from sentence_transformers import CrossEncoder

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_model = None


def _get_model():
    global _model
    if _model is None:
        print(f"  Loading cross-encoder: {_MODEL_NAME} …")
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def _normalize(values):
    """Min-max normalize a list of floats to [0, 1]."""
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
    Args:
        query:              the user's original question
        reranked_results:   output of rerank_results() — tuples of
                            (Document, similarity_score, adjusted_score)
        top_k:              how many to return
        ce_weight:          blend weight for cross-encoder vs recency
        target_companies:   list of company names for balanced selection

    Returns:
        list of (doc, sim_score, adj_score, ce_score, fused_score)
        sorted descending by fused_score, length <= top_k
    """
    if not reranked_results:
        return []

    model = _get_model()

    # ---- Build pairs — FIXED: unpack all 3 values ----
    pairs = [(query, doc.page_content) for doc, _sim, _adj in reranked_results]
    ce_scores = model.predict(pairs).tolist()

    # ---- Normalize both signals ----
    norm_ce = _normalize(ce_scores)
    adj_scores = [adj for _doc, _sim, adj in reranked_results]
    norm_adj = _normalize(adj_scores)

    # ---- Fuse scores ----
    fused = []
    for i, (doc, sim, adj) in enumerate(reranked_results):
        fused_score = ce_weight * norm_ce[i] + (1 - ce_weight) * norm_adj[i]
        fused.append((doc, sim, adj, ce_scores[i], fused_score))

    # ---- Balanced selection across target companies ----
    if target_companies and len(target_companies) > 1:
        fused = _balanced_select(fused, target_companies, top_k)
    else:
        fused.sort(key=lambda x: x[4], reverse=True)
        fused = fused[:top_k]

    return fused


def _balanced_select(fused, target_companies, top_k):
    """
    Guarantee each target company gets at least per_company slots,
    then fill remaining by best fused score.
    """
    # Sort each company's pool by fused score
    buckets = {}
    for item in fused:
        company = item[0].metadata.get("company", "Unknown")
        buckets.setdefault(company, []).append(item)
    for pool in buckets.values():
        pool.sort(key=lambda x: x[4], reverse=True)

    per_company = max(1, top_k // len(target_companies))
    overflow = top_k - (per_company * len(target_companies))

    selected = []
    extras = []

    for company in target_companies:
        pool = buckets.get(company, [])
        selected.extend(pool[:per_company])
        extras.extend(pool[per_company:])

    # Any non-target company chunks go into extras
    for c, pool in buckets.items():
        if c not in target_companies:
            extras.extend(pool)

    extras.sort(key=lambda x: x[4], reverse=True)
    selected.extend(extras[:overflow])
    selected.sort(key=lambda x: x[4], reverse=True)

    return selected