# reranker.py
"""
Second-stage scoring, reranking, and selection.

Owns ALL scoring logic:
  1. recency_weighted_score()  — decay-adjusted similarity
  2. rerank_by_recency()       — sort raw results by adjusted score
  3. select_top_k()            — balanced per-company selection
  4. cross_encoder_rerank()    — cross-encoder fusion with recency

Design decisions to be ready to explain:
  - Model: BAAI/bge-reranker-v2-m3 — lightweight, strong on financial text,
    no GPU required for <60 candidates.
  - Lazy-loaded singleton so the model is only downloaded/loaded once per
    process, not per query.
  - Fused score = ce_weight * normalized_ce_score + (1 - ce_weight) * normalized_adj_score.
    Keeps recency signal without letting it dominate semantic relevance.
  - select_top_k guarantees balanced company representation for multi-entity
    comparison queries before filling remaining slots by best score.
"""

from datetime import datetime
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Config defaults (can be overridden by caller)
# ---------------------------------------------------------------------------
HALF_LIFE_DAYS = 365
RECENCY_WEIGHT = 0.4          # blend weight: similarity vs recency decay

_CE_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_ce_model = None


# ---------------------------------------------------------------------------
# Cross-encoder singleton
# ---------------------------------------------------------------------------
def _get_model():
    global _ce_model
    if _ce_model is None:
        print(f"  Loading cross-encoder: {_CE_MODEL_NAME} …")
        _ce_model = CrossEncoder(_CE_MODEL_NAME)
    return _ce_model


def _normalize(values):
    """Min-max normalize a list of floats to [0, 1]."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


# ---------------------------------------------------------------------------
# 1. Recency-weighted scoring
# ---------------------------------------------------------------------------
def recency_weighted_score(
    similarity_score: float,
    filing_date_str: str | None,
    half_life_days: int = HALF_LIFE_DAYS,
    recency_weight: float = RECENCY_WEIGHT,
) -> float:
    """Return similarity blended with an exponential-decay recency signal."""
    try:
        filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return similarity_score * (1 - recency_weight * 0.5)

    age_days = max((datetime.now() - filing_date).days, 0)
    decay = 0.5 ** (age_days / half_life_days)
    return similarity_score * ((1 - recency_weight) + recency_weight * decay)


# ---------------------------------------------------------------------------
# 2. Rerank raw retrieval results by adjusted score
# ---------------------------------------------------------------------------
def rerank_by_recency(raw_results):
    """
    Takes raw (doc, similarity_score) pairs from ChromaDB.
    Returns list of (doc, sim_score, adj_score) sorted by adj_score desc.
    """
    reranked = []
    for doc, sim_score in raw_results:
        filing_date = doc.metadata.get("filing_date", None)
        adj = recency_weighted_score(sim_score, filing_date)
        reranked.append((doc, sim_score, adj))
    reranked.sort(key=lambda x: x[2], reverse=True)
    return reranked


# ---------------------------------------------------------------------------
# 3. Balanced per-company selection
# ---------------------------------------------------------------------------
def select_top_k(reranked, target_companies, k):
    """
    Guarantee each target company gets fair representation,
    then fill remaining slots by best adjusted score.

    Args:
        reranked:          list of (doc, sim_score, adj_score[, ...])
                           — works with 3-tuples or 5-tuples
        target_companies:  list[str] from company_resolver
        k:                 total slots to return
    """
    if not target_companies or len(target_companies) <= 1:
        return reranked[:k]

    # Score accessor: works whether tuple has 3 or 5 elements
    def _sort_key(item):
        return item[-1]   # last element is always the "best" score

    buckets = {}
    for item in reranked:
        company = item[0].metadata.get("company", "Unknown")
        buckets.setdefault(company, []).append(item)

    per_company = k // len(target_companies)
    overflow = k - per_company * len(target_companies)

    top_results = []
    extras = []

    for company in target_companies:
        pool = buckets.get(company, [])
        top_results.extend(pool[:per_company])
        extras.extend(pool[per_company:])

    # Defensive: include chunks from unexpected companies
    for c, docs in buckets.items():
        if c not in target_companies:
            extras.extend(docs)

    extras.sort(key=_sort_key, reverse=True)
    top_results.extend(extras[:overflow])
    top_results.sort(key=_sort_key, reverse=True)

    return top_results


# ---------------------------------------------------------------------------
# 4. Cross-encoder reranking (optional second pass)
# ---------------------------------------------------------------------------
def cross_encoder_rerank(
    query: str,
    reranked_results: list,       # list of (doc, sim_score, adj_score)
    top_k: int = 20,
    ce_weight: float = 0.7,
):
    """
    Scores each (query, chunk) pair with a cross-encoder, fuses with
    the existing adjusted score, and returns the top-k.

    Returns:
        list of (doc, sim_score, adj_score, ce_score, fused_score)
        sorted descending by fused_score, length <= top_k
    """
    if not reranked_results:
        return []

    model = _get_model()

    pairs = [(query, doc.page_content) for doc, _, _ in reranked_results]
    ce_scores = model.predict(pairs).tolist()

    norm_ce = _normalize(ce_scores)
    adj_scores = [adj for _, _, adj in reranked_results]
    norm_adj = _normalize(adj_scores)

    fused = []
    for i, (doc, sim, adj) in enumerate(reranked_results):
        fused_score = ce_weight * norm_ce[i] + (1 - ce_weight) * norm_adj[i]
        fused.append((doc, sim, adj, ce_scores[i], fused_score))

    fused.sort(key=lambda x: x[4], reverse=True)
    return fused[:top_k]