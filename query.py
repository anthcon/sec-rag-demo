# query.py
"""
Retrieval + LLM generation for SEC filing questions.

Responsibility: retrieve chunks, delegate scoring/reranking to reranker.py,
assemble the prompt, and make the single LLM call.
"""

import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from prompt_template import PROMPT_TEMPLATE
from company_resolver import resolve_companies
from reranker import (
    rerank_by_recency,
    select_top_k,
    cross_encoder_rerank,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"

K = 20
CHUNKS_PER_COMPANY = 10
MAX_TOKENS = 2000
TEMPERATURE = 0.2
FALLBACK_RETRIEVAL_POOL = K * 3
MIN_RELEVANCE = 0.3              # drop query if best score below this
USE_CROSS_ENCODER = True         # toggle cross-encoder reranking


# ---------------------------------------------------------------------------
# Retrieval helpers (still in query.py — they talk to ChromaDB)
# ---------------------------------------------------------------------------
def retrieve_filtered(db, question, target_companies):
    """Per-company filtered retrieval, merged into one list."""
    all_raw = []
    for company in target_companies:
        try:
            results = db.similarity_search_with_relevance_scores(
                question,
                k=CHUNKS_PER_COMPANY,
                filter={"company": company},
            )
            print(f"  {company}: {len(results)} chunks retrieved")
            all_raw.extend(results)
        except Exception as e:
            print(f"  {company}: retrieval failed — {e}")
    return all_raw


def retrieve_unfiltered(db, question):
    """Fallback: no company filter."""
    print("  No companies detected — falling back to unfiltered retrieval")
    return db.similarity_search_with_relevance_scores(
        question, k=FALLBACK_RETRIEVAL_POOL
    )


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------
def build_context(top_results):
    """Format ranked chunks into a single context string for the prompt."""
    parts = []
    for item in top_results:
        doc = item[0]
        meta = doc.metadata
        header = (
            f"[{meta.get('company', '?')} | {meta.get('filing_type', '?')} | "
            f"{meta.get('quarter', '?')} | Filed: {meta.get('filing_date', '?')}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def print_sources(top_results):
    """Print source metadata for transparency."""
    print(f"\nSources used ({len(top_results)}):")
    for item in top_results:
        doc = item[0]
        meta = doc.metadata
        sim = item[1]
        # Adapted score label depends on tuple length
        if len(item) == 5:
            label = f"sim: {sim:.3f} | ce: {item[3]:.3f} | fused: {item[4]:.3f}"
        else:
            label = f"sim: {sim:.3f} | adj: {item[2]:.3f}"
        print(
            f"  - {meta.get('company')} | {meta.get('filing_type')} | "
            f"{meta.get('filing_date')} | {label}"
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def query(question: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 1. Resolve companies
    target_companies = resolve_companies(question)
    print(f"\nResolved companies: {target_companies if target_companies else 'None (unfiltered)'}")

    # 2. Retrieve
    if target_companies:
        raw_results = retrieve_filtered(db, question, target_companies)
    else:
        raw_results = retrieve_unfiltered(db, question)

    if not raw_results or max(score for _, score in raw_results) < MIN_RELEVANCE:
        print("No sufficiently relevant results found.")
        return

    # 3. Score & rerank (all in reranker.py)
    recency_ranked = rerank_by_recency(raw_results)

    if USE_CROSS_ENCODER:
        reranked = cross_encoder_rerank(question, recency_ranked, top_k=len(recency_ranked))
        top_results = select_top_k(reranked, target_companies, K)
    else:
        top_results = select_top_k(recency_ranked, target_companies, K)

    # 4. Assemble prompt & call LLM
    context = build_context(top_results)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    llm = ChatOpenAI(model=LLM_MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    response = llm.invoke(prompt)

    print(f"\nAnswer:\n{response.content}")
    print_sources(top_results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Enter question: ")
    else:
        question = " ".join(sys.argv[1:])
    query(question)