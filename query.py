# query.py
"""
SEC Filing RAG Pipeline — Query Module (Orchestrator)

Pipeline stages:
  1. Resolve target companies from user question
  2. Expand query with financial synonyms (for embedding retrieval)
  3. Retrieve chunks per company from ChromaDB
  4. Stage-1 rerank: recency weighting              → reranker.stage1_rerank
  5. Per-company query rewriting (for cross-encoder scoring)
  6. Stage-2 rerank: cross-encoder + balanced select → reranker.cross_encoder_rerank*
  7. Generate answer:
       - Single company  → direct generation
       - Multi-company   → split & fuse (per-company answers → synthesis)

Design decisions (walkthrough):
  - Query expansion bridges vocab gap for EMBEDDING retrieval
    ("revenue" → "net sales, total revenue, net revenues")
  - Per-company rewriting bridges vocab gap for CROSS-ENCODER scoring
    ("Compare Amazon revenue" → "What were Amazon's net sales…")
  - Split & fuse prevents one company's data from crowding out another
    in the LLM's attention window (inspired by FinanceRAG paper)
  - All scoring constants and reranking logic live in reranker.py
"""

import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from prompt_template import PROMPT_TEMPLATE, SYNTHESIS_TEMPLATE
from company_resolver import resolve_companies
from reranker import (
    stage1_rerank,
    cross_encoder_rerank,
    cross_encoder_rerank_per_company,
    CE_WEIGHT,
)

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────
CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"

K = 20                        # total chunks to feed into context
CHUNKS_PER_COMPANY = 15       # how many to pull from ChromaDB per company
MAX_TOKENS = 2000
TEMPERATURE = 0.2
FALLBACK_RETRIEVAL_POOL = K * 3

USE_QUERY_EXPANSION = True
USE_PER_COMPANY_REWRITE = True
USE_SPLIT_AND_FUSE = True


# ─── Query Expansion (for embedding retrieval) ──────────────────────
def expand_query(question: str, llm: ChatOpenAI) -> str:
    """
    Append financial synonyms/keywords so the embedding search
    matches SEC filing vocabulary.  One cheap LLM call (~50 tokens).
    """
    prompt = (
        "You are a financial search assistant. Given the user question below, "
        "produce a SHORT comma-separated list of 5-8 alternative keywords or "
        "phrases that SEC 10-K/10-Q filings might use to discuss the same topic. "
        "Include synonyms, line-item names, and common financial terms. "
        "Return ONLY the comma-separated keywords, nothing else.\n\n"
        f"Question: {question}"
    )
    try:
        resp = llm.invoke(prompt)
        keywords = resp.content.strip()
        expanded = f"{question} {keywords}"
        print(f"  Expanded query: {expanded[:140]}…")
        return expanded
    except Exception as e:
        print(f"  Query expansion failed ({e}), using original query")
        return question


# ─── Per-Company Query Rewriting (for cross-encoder) ────────────────
def rewrite_query_for_company(question: str, company: str,
                              llm: ChatOpenAI) -> str:
    """
    Rewrite the user question using the exact financial terminology
    that a specific company uses in its SEC filings.

    Why: Cross-encoders are vocabulary-sensitive.  Amazon says "net sales",
    Google says "revenues".  A generic question about "revenue growth"
    scores ~0.0 against Amazon chunks but ~0.3 against Google chunks.
    Per-company rewriting fixes this asymmetry.
    """
    prompt = (
        f"Rewrite this question using the exact financial terminology "
        f"found in {company}'s SEC 10-K/10-Q filings. "
        f"For example, Amazon uses 'net sales', Alphabet uses 'revenues', "
        f"Apple uses 'net sales' and 'total net revenues'.\n\n"
        f"Original: {question}\n"
        f"Rewritten for {company} (one sentence, no explanation):"
    )
    try:
        resp = llm.invoke(prompt)
        rewritten = resp.content.strip()
        print(f"    CE query [{company}]: {rewritten[:100]}…")
        return rewritten
    except Exception as e:
        print(f"    Rewrite failed for {company} ({e}), using original")
        return question


# ─── Retrieval ───────────────────────────────────────────────────────
def retrieve_filtered(db, question, target_companies):
    """Per-company filtered retrieval."""
    all_raw = []
    for company in target_companies:
        try:
            results = db.similarity_search_with_relevance_scores(
                question,
                k=CHUNKS_PER_COMPANY,
                filter={"company": company},
            )
            if results:
                best = max(s for _, s in results)
                print(f"  {company}: {len(results)} chunks (best sim: {best:.3f})")
            else:
                print(f"  {company}: 0 chunks")
            all_raw.extend(results)
        except Exception as e:
            print(f"  {company}: retrieval failed — {e}")
    return all_raw


def retrieve_unfiltered(db, question):
    """Fallback: no company filter."""
    print("  No companies detected — unfiltered retrieval")
    return db.similarity_search_with_relevance_scores(
        question, k=FALLBACK_RETRIEVAL_POOL
    )


# ─── Context Building ───────────────────────────────────────────────
def build_context(results):
    """Turn a list of reranker output tuples into a context string."""
    parts = []
    for doc, sim, adj, ce, fused in results:
        meta = doc.metadata
        header = (
            f"[{meta.get('company', '?')} | {meta.get('filing_type', '?')} | "
            f"{meta.get('quarter', '?')} | Filed: {meta.get('filing_date', '?')}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ─── Generation ─────────────────────────────────────────────────────
def generate_single(question, context, llm):
    """Standard single-shot generation."""
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return llm.invoke(prompt).content


def generate_split_and_fuse(question, top_results, target_companies, llm):
    """
    For multi-company comparison queries:
      1. Generate a per-company answer using only that company's chunks
      2. Synthesize into a final comparison

    Why: Prevents one company's rich data from crowding out the other
    in the LLM's attention window.  Also lets each sub-answer focus
    on extracting numbers without cross-company confusion.
    (Inspired by FinanceRAG paper's long-context management.)
    """
    by_company = {}
    for item in top_results:
        co = item[0].metadata.get("company", "Unknown")
        by_company.setdefault(co, []).append(item)

    per_company_answers = {}
    for company in target_companies:
        company_results = by_company.get(company, [])
        if not company_results:
            per_company_answers[company] = (
                f"No relevant SEC filing data was found for {company}."
            )
            continue

        context = build_context(company_results)
        company_question = f"Regarding {company} only: {question}"
        answer = generate_single(company_question, context, llm)
        per_company_answers[company] = answer
        print(f"  ✓ Generated sub-answer for {company} "
              f"({len(company_results)} chunks)")

    analyses_text = ""
    for company, answer in per_company_answers.items():
        analyses_text += f"\n### {company}:\n{answer}\n"

    synthesis_prompt = SYNTHESIS_TEMPLATE.format(
        per_company_analyses=analyses_text,
        question=question,
    )
    print("  ✓ Synthesizing comparison…")
    return llm.invoke(synthesis_prompt).content


# ─── Main Query Pipeline ────────────────────────────────────────────
def query(question: str):
    # Init shared resources
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    llm = ChatOpenAI(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    # --- Step 1: Resolve target companies ---
    target_companies = resolve_companies(question)
    print(f"Resolved companies: "
          f"{target_companies if target_companies else 'None (unfiltered)'}")

    is_comparison = len(target_companies) > 1 if target_companies else False

    # --- Step 2: Query expansion (for embedding retrieval) ---
    search_query = question
    if USE_QUERY_EXPANSION:
        search_query = expand_query(question, llm)

    # --- Step 3: Retrieve from ChromaDB ---
    if target_companies:
        raw_results = retrieve_filtered(db, search_query, target_companies)
    else:
        raw_results = retrieve_unfiltered(db, search_query)

    if not raw_results:
        print("No results found.")
        return

    best_sim = max(score for _, score in raw_results)
    if best_sim < 0.3:
        print(f"Best similarity {best_sim:.3f} is below threshold. Aborting.")
        return

    # --- Step 4: Stage-1 rerank (recency weighting) ---
    reranked = stage1_rerank(raw_results)

    # --- Step 5: Stage-2 rerank (cross-encoder) ---
    print(f"\n  Reranking {len(reranked)} candidates with cross-encoder…")

    if USE_PER_COMPANY_REWRITE and target_companies:
        # Build per-company rewritten queries
        rewritten_queries = {
            company: rewrite_query_for_company(question, company, llm)
            for company in target_companies
        }
        top_results = cross_encoder_rerank_per_company(
            question=question,
            reranked_results=reranked,
            target_companies=target_companies,
            rewritten_queries=rewritten_queries,
            top_k=K,
        )
    else:
        top_results = cross_encoder_rerank(
            query=question,
            reranked_results=reranked,
            top_k=K,
            target_companies=target_companies,
        )

    # --- Step 6: Generate answer ---
    if is_comparison and USE_SPLIT_AND_FUSE:
        print(f"\n  Comparison detected — using split & fuse generation…")
        answer = generate_split_and_fuse(
            question, top_results, target_companies, llm
        )
    else:
        context = build_context(top_results)
        answer = generate_single(question, context, llm)

    # --- Output ---
    print(f"\nAnswer:\n{answer}")

    print(f"\nSources ({len(top_results)}):")
    for doc, sim, adj, ce, fused in top_results:
        meta = doc.metadata
        print(
            f"  - {meta.get('company')} | {meta.get('filing_type')} | "
            f"{meta.get('filing_date')} | sim: {sim:.3f} | adj: {adj:.3f} | "
            f"ce: {ce:.3f} | fused: {fused:.3f}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Enter question: ")
    else:
        question = " ".join(sys.argv[1:])

    query(question)