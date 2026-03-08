"""
query.py – SEC RAG query engine (v2)

Enhancements over v1:
  - Cross-encoder reranking after initial similarity retrieval
  - Long-context split/fuse when retrieved context exceeds token limit
  - Larger initial retrieval pool to feed the reranker
"""

import sys
from datetime import datetime
from dotenv import load_dotenv
import tiktoken
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from prompt_template import PROMPT_TEMPLATE
from company_resolver import resolve_companies
from reranker import cross_encoder_rerank

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────
CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"

K = 20                           # final chunks sent to LLM
CHUNKS_PER_COMPANY = 15          # per-company retrieval pool (feeds reranker)
MAX_TOKENS = 2000
TEMPERATURE = 0.2
HALF_LIFE_DAYS = 365
RECENCY_WEIGHT = 0.4
FALLBACK_RETRIEVAL_POOL = K * 3
MAX_CONTEXT_TOKENS = 32_000      # split/fuse threshold


# ── Helpers ─────────────────────────────────────────────────────────

def recency_weighted_score(similarity_score, filing_date_str,
                           half_life_days=HALF_LIFE_DAYS):
    """Blend similarity with an exponential-decay recency bonus."""
    try:
        filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return similarity_score * (1 - RECENCY_WEIGHT * 0.5)

    age_days = (datetime.now() - filing_date).days
    if age_days < 0:
        age_days = 0

    decay = 0.5 ** (age_days / half_life_days)
    return similarity_score * ((1 - RECENCY_WEIGHT) + RECENCY_WEIGHT * decay)


# ── Retrieval ───────────────────────────────────────────────────────

def retrieve_filtered(db, question, target_companies):
    """Per-company filtered retrieval → merged list."""
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


# ── Selection ───────────────────────────────────────────────────────

def select_top_k(reranked, target_companies, k):
    """
    Balanced selection across target companies.
    Expects tuples of (doc, sim_score, adj_score, ce_score, fused_score).
    Sorts by fused_score (index 4).
    """
    if not target_companies or len(target_companies) <= 1:
        return reranked[:k]

    buckets = {}
    for item in reranked:
        company = item[0].metadata.get("company", "Unknown")
        buckets.setdefault(company, []).append(item)

    per_company = k // len(target_companies)
    overflow = k - (per_company * len(target_companies))

    top_results = []
    extras = []

    for company in target_companies:
        pool = buckets.get(company, [])
        top_results.extend(pool[:per_company])
        extras.extend(pool[per_company:])

    for c, docs in buckets.items():
        if c not in target_companies:
            extras.extend(docs)

    extras.sort(key=lambda x: x[4], reverse=True)
    top_results.extend(extras[:overflow])
    top_results.sort(key=lambda x: x[4], reverse=True)
    return top_results


# ── Long-context handling ───────────────────────────────────────────

def _count_tokens(text, model=LLM_MODEL):
    """Return token count for text using tiktoken."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def _split_chunks(chunks):
    """Split a list into two roughly equal halves."""
    mid = len(chunks) // 2
    return chunks[:mid], chunks[mid:]


def _fuse_answers(answer_a, answer_b, question, llm):
    """Ask the LLM to merge two partial answers into one cohesive reply."""
    fuse_prompt = (
        "You received two partial answers to the same question. "
        "Combine them into a single, cohesive, well-organized response. "
        "Remove redundancy but preserve all unique facts and citations.\n\n"
        f"Question: {question}\n\n"
        f"--- Answer A ---\n{answer_a}\n\n"
        f"--- Answer B ---\n{answer_b}\n\n"
        "Combined answer:"
    )
    return llm.invoke(fuse_prompt).content


def handle_long_context(context, question, llm):
    """
    If context fits within MAX_CONTEXT_TOKENS, generate normally.
    Otherwise split into two halves, generate partial answers, then fuse.
    """
    token_count = _count_tokens(context)
    print(f"  Context tokens: {token_count:,}")

    if token_count <= MAX_CONTEXT_TOKENS:
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        return llm.invoke(prompt).content

    # Split
    print("  ⚠ Context exceeds token limit — using split/fuse strategy")
    parts = context.split("\n\n---\n\n")
    half_a, half_b = _split_chunks(parts)

    ctx_a = "\n\n---\n\n".join(half_a)
    ctx_b = "\n\n---\n\n".join(half_b)

    prompt_a = PROMPT_TEMPLATE.format(context=ctx_a, question=question)
    prompt_b = PROMPT_TEMPLATE.format(context=ctx_b, question=question)

    answer_a = llm.invoke(prompt_a).content
    answer_b = llm.invoke(prompt_b).content

    return _fuse_answers(answer_a, answer_b, question, llm)


# ── Main query pipeline ────────────────────────────────────────────

def query(question: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Step 1: Resolve companies
    target_companies = resolve_companies(question)
    print(f"\nResolved companies: {target_companies if target_companies else 'None (unfiltered)'}")

    # Step 2: Retrieve
    if target_companies:
        raw_results = retrieve_filtered(db, question, target_companies)
    else:
        raw_results = retrieve_unfiltered(db, question)

    if not raw_results or max(score for _, score in raw_results) < 0.3:
        print("No sufficiently relevant results found.")
        return

    # Step 3: Cross-encoder rerank
    print(f"  Reranking {len(raw_results)} candidates with cross-encoder …")
    reranked = cross_encoder_rerank(question, raw_results, top_k=K)

    # Step 4: Balanced selection
    top_results = select_top_k(reranked, target_companies, K)

    # Step 5: Build context
    context_parts = []
    for doc, sim_score, adj_score, ce_score, fused_score in top_results:
        meta = doc.metadata
        header = (
            f"[{meta.get('company', '?')} | {meta.get('filing_type', '?')} | "
            f"{meta.get('quarter', '?')} | Filed: {meta.get('filing_date', '?')}]"
        )
        context_parts.append(f"{header}\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    # Step 6: Generate (with long-context safety net)
    llm = ChatOpenAI(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    answer = handle_long_context(context, question, llm)

    # Output
    print(f"\nAnswer:\n{answer}")
    print(f"\nSources used ({len(top_results)}):")
    for doc, sim_score, adj_score, ce_score, fused_score in top_results:
        meta = doc.metadata
        print(
            f"  - {meta.get('company')} | {meta.get('filing_type')} | "
            f"{meta.get('filing_date')} | sim: {sim_score:.3f} | "
            f"adj: {adj_score:.3f} | ce: {ce_score:.3f} | fused: {fused_score:.3f}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Enter question: ")
    else:
        question = " ".join(sys.argv[1:])

    query(question)