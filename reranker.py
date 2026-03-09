# reranker.py
"""
SEC Filing RAG Pipeline — Reranker Module

All scoring and reranking logic lives here:
  - Stage 1: recency-weighted scoring
  - Stage 2: cross-encoder reranking (standard + per-company)
  - Balanced selection across target companies

Design decisions (walkthrough prep):
  - Model: BAAI/bge-reranker-v2-m3 — lightweight, no GPU needed for <60 candidates.
  - Lazy-loaded singleton: model loads once per process.
  - CE score floor (0.05): prevents good similarity matches from being
    zeroed out by an overly literal cross-encoder. Chunks already passed
    a similarity gate to reach this stage.
  - Fused score = CE_WEIGHT * norm_ce + (1 - CE_WEIGHT) * norm_adj.
  - Balanced selection guarantees each target company gets fair representation.
  - Per-company rewriting fixes vocabulary mismatch (Amazon="net sales",
    Google="revenues") by scoring each company's chunks with a
    company-specific CE query.
"""

from datetime import datetime
from sentence_transformers import CrossEncoder

# ─── Constants (single source of truth) ──────────────────────────────
CE_SCORE_FLOOR = 0.05       # minimum normalized CE score for any chunk
CE_WEIGHT = 0.7             # blend: CE_WEIGHT * ce + (1 - CE_WEIGHT) * recency
HALF_LIFE_DAYS = 365        # recency decay half-life
RECENCY_WEIGHT = 0.4        # how much recency affects similarity score

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_model = None


# ─── Model Loading ───────────────────────────────────────────────────
def _get_model():
    global _model
    if _model is None:
        print(f"  Loading cross-encoder: {_MODEL_NAME} …")
        _model = CrossEncoder(_MODEL_NAME)
    return _model


# ─── Helpers ─────────────────────────────────────────────────────────
def _normalize(values):
    """Min-max normalize a list of floats to [0, 1]."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _balanced_select(fused, target_companies, top_k):
    """
    Guarantee each target company gets at least per_company slots,
    then fill remaining by best fused score across all companies.
    """
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

    for c, pool in buckets.items():
        if c not in target_companies:
            extras.extend(pool)

    if remainder > 0:
        extras.sort(key=lambda x: x[4], reverse=True)
        selected.extend(extras[:remainder])

    selected.sort(key=lambda x: x[4], reverse=True)
    return selected


def _fuse_and_select(items, raw_ce_scores, adj_scores, top_k,
                     target_companies=None):
    """
    Shared fusion logic: normalize, floor, fuse, select.
    Used by both standard and per-company paths.
    """
    norm_ce = _normalize(raw_ce_scores)
    norm_ce_floored = [max(s, CE_SCORE_FLOOR) for s in norm_ce]
    norm_adj = _normalize(adj_scores)

    fused = []
    for i, (doc, sim, adj) in enumerate(items):
        fused_score = CE_WEIGHT * norm_ce_floored[i] + (1 - CE_WEIGHT) * norm_adj[i]
        fused.append((doc, sim, adj, raw_ce_scores[i], fused_score))

    if target_companies and len(target_companies) > 1:
        return _balanced_select(fused, target_companies, top_k)
    else:
        fused.sort(key=lambda x: x[4], reverse=True)
        return fused[:top_k]


# ─── Stage 1: Recency Scoring ───────────────────────────────────────
def recency_weighted_score(similarity_score, filing_date_str):
    """Apply exponential time-decay to a similarity score."""
    try:
        filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return similarity_score * (1 - RECENCY_WEIGHT * 0.5)
    age_days = max(0, (datetime.now() - filing_date).days)
    decay = 0.5 ** (age_days / HALF_LIFE_DAYS)
    return similarity_score * ((1 - RECENCY_WEIGHT) + RECENCY_WEIGHT * decay)


def stage1_rerank(raw_results):
    """
    Stage 1: apply recency weighting to raw similarity scores.

    Args:
        raw_results: list of (Document, similarity_score) from ChromaDB

    Returns:
        list of (Document, similarity_score, adjusted_score)
        sorted descending by adjusted_score
    """
    reranked = []
    for doc, sim_score in raw_results:
        filing_date = doc.metadata.get("filing_date", None)
        adjusted_score = recency_weighted_score(sim_score, filing_date)
        reranked.append((doc, sim_score, adjusted_score))
    reranked.sort(key=lambda x: x[2], reverse=True)
    return reranked


# ─── Stage 2: Cross-Encoder Reranking ───────────────────────────────
def cross_encoder_rerank(query, reranked_results, top_k=20,
                         target_companies=None):
    """
    Standard Stage-2: score ALL candidates with a single query,
    fuse with recency, then balanced-select or global top-k.

    Args:
        query:             search query for CE scoring
        reranked_results:  output of stage1_rerank() — 3-tuples
        top_k:             how many to return
        target_companies:  list of company names for balanced selection

    Returns:
        list of (doc, sim, adj, raw_ce, fused) sorted by fused desc
    """
    if not reranked_results:
        return []

    model = _get_model()
    pairs = [(query, doc.page_content) for doc, _s, _a in reranked_results]
    raw_ce_scores = model.predict(pairs).tolist()
    adj_scores = [adj for _, _, adj in reranked_results]

    return _fuse_and_select(
        items=reranked_results,
        raw_ce_scores=raw_ce_scores,
        adj_scores=adj_scores,
        top_k=top_k,
        target_companies=target_companies,
    )


def cross_encoder_rerank_per_company(question, reranked_results,
                                      target_companies, rewritten_queries,
                                      top_k=20):
    """
    Per-company Stage-2: score each company's chunks with its own
    rewritten query, normalize within each batch, then merge and
    balanced-select.

    Args:
        question:           original user question (fallback)
        reranked_results:   output of stage1_rerank()
        target_companies:   list of company names
        rewritten_queries:  dict {company: rewritten_query_string}
        top_k:              how many to return

    Returns:
        list of (doc, sim, adj, raw_ce, fused) sorted by fused desc
    """
    if not reranked_results:
        return []

    # Bucket candidates by company
    by_company = {}
    for item in reranked_results:
        co = item[0].metadata.get("company", "Unknown")
        by_company.setdefault(co, []).append(item)

    model = _get_model()
    all_scored = []

    for company in target_companies:
        items = by_company.get(company, [])
        if not items:
            continue

        ce_query = rewritten_queries.get(company, question)
        pairs = [(ce_query, doc.page_content) for doc, _s, _a in items]
        raw_ce = model.predict(pairs).tolist()

        # Normalize within this company's batch
        norm_ce = _normalize(raw_ce)
        norm_ce_floored = [max(s, CE_SCORE_FLOOR) for s in norm_ce]
        adj_scores = [adj for _, _, adj in items]
        norm_adj = _normalize(adj_scores)

        for i, (doc, sim, adj) in enumerate(items):
            fused = CE_WEIGHT * norm_ce_floored[i] + (1 - CE_WEIGHT) * norm_adj[i]
            all_scored.append((doc, sim, adj, raw_ce[i], fused))

    # Handle non-target company chunks (safety net)
    for co, items in by_company.items():
        if co not in target_companies:
            for doc, sim, adj in items:
                all_scored.append((doc, sim, adj, 0.0, CE_SCORE_FLOOR))

    return _balanced_select(all_scored, target_companies, top_k)