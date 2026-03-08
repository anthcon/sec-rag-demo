"""
Cross-encoder reranker with recency fusion.

Scores (query, chunk) pairs with BAAI/bge-reranker-v2-m3,
blends with filing recency via exponential decay, returns top-k.

Design decisions:
  - bge-reranker-v2-m3: lightweight, strong on financial text, CPU-friendly.
  - Lazy singleton: model loaded once per process.
  - Fused score = 0.7 * cross-encoder + 0.3 * recency.
    Cross-encoder dominates; recency breaks ties toward newer filings.
"""

from datetime import datetime
from sentence_transformers import CrossEncoder

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_model = None

HALF_LIFE_DAYS = 365
CE_WEIGHT = 0.7


def _get_model():
    global _model
    if _model is None:
        print(f"  Loading cross-encoder: {_MODEL_NAME} …")
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def _normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _recency_score(filing_date_str: str) -> float:
    """Exponential decay: 1.0 for today, 0.5 at HALF_LIFE_DAYS ago."""
    try:
        filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return 0.5
    age_days = max((datetime.now() - filing_date).days, 0)
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def cross_encoder_rerank(
    query: str,
    raw_results: list[tuple],
    top_k: int = 40,
    ce_weight: float = CE_WEIGHT,
) -> list[tuple]:
    """
    Args:
        query:       user's original question
        raw_results: list of (Document, sim_score) from ChromaDB
        top_k:       max results to return (set higher than final TOP_K
                     so balanced_select has room to work)
        ce_weight:   blend weight for cross-encoder vs recency

    Returns:
        list of (Document, sim_score, fused_score) sorted by fused_score desc
    """
    if not raw_results:
        return []

    model = _get_model()

    pairs = [(query, doc.page_content) for doc, _, _ in raw_results]
    ce_scores = model.predict(pairs).tolist()

    recency_scores = [
        _recency_score(doc.metadata.get("filing_date"))
        for doc, _, _ in raw_results
    ]

    norm_ce = _normalize(ce_scores)
    norm_recency = _normalize(recency_scores)

    results = []
    for i, (doc, sim, adj) in enumerate(raw_results):
        fused = ce_weight * norm_ce[i] + (1 - ce_weight) * norm_recency[i]
        results.append((doc, sim, adj, ce_scores[i], fused))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]