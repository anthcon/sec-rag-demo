# reranker.py
"""
SEC Filing RAG Pipeline — Reranker Module

Recency-weighted reranking: applies exponential time-decay to similarity
scores so recent filings rank higher without completely burying older ones.
Uses a 365-day half-life.
"""
from datetime import datetime

# ─── Constants ───────────────────────────────────────────────────────
HALF_LIFE_DAYS = 365
RECENCY_WEIGHT = 0.4


def recency_weighted_score(similarity_score, filing_date_str):
    """Apply exponential time-decay to a similarity score."""
    try:
        filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return similarity_score * (1 - RECENCY_WEIGHT * 0.5)

    age_days = max(0, (datetime.now() - filing_date).days)
    decay = 0.5 ** (age_days / HALF_LIFE_DAYS)
    return similarity_score * ((1 - RECENCY_WEIGHT) + RECENCY_WEIGHT * decay)


def recency_rerank(raw_results):
    """
    Apply recency weighting to raw similarity scores.

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
    print(f"Recency rerank: {len(reranked)} chunks, top adjusted score: {reranked[0][2]:.4f}" if reranked else "Recency rerank: 0 chunks")
    return reranked