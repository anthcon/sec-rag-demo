# reranker.py
"""
Second-stage cross-encoder reranker.
Scores (query, chunk) pairs with a cross-encoder and returns
the top-k results sorted by fused score (cross-encoder + recency).

Design decisions to be ready to explain:
  - Model: BAAI/bge-reranker-v2-m3 — lightweight, strong on financial text,
    no GPU required for <60 candidates.
  - Lazy-loaded singleton so the model is only downloaded/loaded once per
    process, not per query.
  - Fused score = 0.7 * normalized_ce_score + 0.3 * normalized_adj_score.
    Keeps recency signal without letting it dominate semantic relevance.
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
):
    """
    Args:
        query:            the user's original question
        reranked_results: tuples of (Document, score)
        top_k:            how many to return
        ce_weight:        blend weight for cross-encoder vs recency
    Returns:
        list of (doc, score, ce_score, fused_score)
        sorted descending by fused_score, length <= top_k
    """
    if not reranked_results:
        return []

    model = _get_model()

    # Build pairs for scoring
    pairs = [(query, doc.page_content) for doc, _, _ in reranked_results]
    ce_scores = model.predict(pairs).tolist()

    # Normalize both signals
    norm_ce = _normalize(ce_scores)
    adj_scores = [adj for _, _, adj in reranked_results]
    norm_adj = _normalize(adj_scores)

    # Fuse
    fused = []
    for i, (doc, score) in enumerate(reranked_results):
        fused_score = ce_weight * norm_ce[i] + (1 - ce_weight) * norm_adj[i]
        fused.append((doc, score, ce_scores[i], fused_score))

    fused.sort(key=lambda x: x[4], reverse=True)
    return fused[:top_k]