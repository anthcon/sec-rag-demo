
# query.py
"""
SEC Filing RAG Pipeline — Query Module

Pipeline stages:
  1. Resolve target companies from user question
  2. Expand query with financial synonyms (for embedding retrieval)
  3. Retrieve chunks per company from ChromaDB
  4. Stage-1 rerank: recency weighting
  5. Per-company query rewriting (for cross-encoder scoring)
  6. Stage-2 rerank: cross-encoder with CE floor + balanced selection
  7. Generate answer:
       - Single company  -> direct generation
       - Multi-company   -> split & fuse (per-company answers -> synthesis)

Design decisions (walkthrough):
  - Query expansion bridges vocab gap for EMBEDDING retrieval
    ("revenue" -> "net sales, total revenue, net revenues")
  - Per-company rewriting bridges vocab gap for CROSS-ENCODER scoring
    ("Compare Amazon revenue" -> "What were Amazon's net sales...")
  - Split & fuse prevents one company's data from crowding out another
    in the LLM's attention window (inspired by FinanceRAG paper)
  - CE score floor (0.05) in reranker prevents good sim-matches from
    being zeroed out by an overly literal cross-encoder
  - rewritten_queries are threaded from rerank step into split & fuse
    so each sub-answer uses the company-specific question, NOT the
    original multi-company comparison question
"""

import sys
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from prompt_template import PROMPT_TEMPLATE, SYNTHESIS_TEMPLATE
from company_resolver import resolve_companies
from reranker import cross_encoder_rerank

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────
CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"

K = 30                        # total chunks to feed into context
CHUNKS_PER_COMPANY = 15       # how many to pull from ChromaDB per company
MAX_TOKENS = 2000
TEMPERATURE = 0.2
HALF_LIFE_DAYS = 365
RECENCY_WEIGHT = 0.4
CE_WEIGHT = 0.7
FALLBACK_RETRIEVAL_POOL = K * 3

USE_QUERY_EXPANSION = True
USE_PER_COMPANY_REWRITE = True
USE_SPLIT_AND_FUSE = True


# ─── LLM Logging Helper ─────────────────────────────────────────────
def log_llm(tag: str, prompt: str, response: str):
    """
    Print a formatted log block for every LLM call so you can verify
    exactly what was sent and what came back.

    Args:
        tag:      short label for the call (e.g. "QUERY_EXPANSION")
        prompt:   the full prompt string sent to the LLM
        response: the full response text received from the LLM
    """
    divider = "=" * 72
    print(f"\n{divider}")
    print(f"  LLM CALL: {tag}")
    print(divider)
    print(f"  PROMPT:\n{prompt}")
    print(f"\n  RESPONSE:\n{response}")
    print(f"{divider}\n")


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
        response_text = resp.content.strip()
        log_llm("QUERY_EXPANSION", prompt, response_text)
        expanded = f"{question} {response_text}"
        print(f"  Expanded query: {expanded[:140]}...")
        return expanded
    except Exception as e:
        print(f"  Query expansion failed ({e}), using original query")
        return question


# ─── Per-Company Query Rewriting (for cross-encoder) ────────────────
def rewrite_query_for_company(question: str, company: str,
                              llm: ChatOpenAI) -> str:
    """
    Rewrite the user question so it asks ONLY about the specified company,
    using the exact financial terminology that company uses in its SEC filings.

    Why: Cross-encoders are vocabulary-sensitive.  Amazon says "net sales",
    Google says "revenues".  A generic question about "revenue growth"
    scores ~0.0 against Amazon chunks but ~0.3 against Google chunks.
    Per-company rewriting fixes this asymmetry.

    Critical: The rewritten question must mention ONLY the target company.
    For a comparison question like "Compare Amazon and Google revenue",
    the rewrite for Amazon should be "What were Amazon's net sales..."
    with NO mention of Google, and vice versa.
    """
    prompt = (
        f"You are a financial query rewriter.\n\n"
        f"TASK: Rewrite the question below so it asks ONLY about {company}. "
        f"Do NOT mention any other company. Use the exact financial terminology "
        f"found in {company}'s SEC 10-K/10-Q filings.\n\n"
        f"EXAMPLES of company-specific terms:\n"
        f"  - Amazon uses 'net sales', 'operating income', 'AWS'\n"
        f"  - Alphabet/Google uses 'revenues', 'operating income', 'Google Cloud'\n"
        f"  - Apple uses 'net sales', 'total net revenues'\n\n"
        f"RULES:\n"
        f"  1. The rewritten question must ONLY be about {company}.\n"
        f"  2. Remove all references to other companies.\n"
        f"  3. Remove comparison language ('compare', 'vs', 'versus', 'relative to').\n"
        f"  4. Use {company}'s SEC filing vocabulary for financial terms.\n"
        f"  5. Return ONE sentence, no explanation.\n\n"
        f"Original question: {question}\n"
        f"Rewritten for {company}:"
    )
    try:
        resp = llm.invoke(prompt)
        rewritten = resp.content.strip()
        log_llm(f"REWRITE_QUERY [{company}]", prompt, rewritten)
        print(f"    CE query [{company}]: {rewritten[:100]}...")
        return rewritten
    except Exception as e:
        print(f"    Rewrite failed for {company} ({e}), using original")
        return question


# ─── Recency Scoring ────────────────────────────────────────────────
def recency_weighted_score(similarity_score, filing_date_str,
                           half_life_days=HALF_LIFE_DAYS):
    try:
        filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return similarity_score * (1 - RECENCY_WEIGHT * 0.5)
    age_days = max(0, (datetime.now() - filing_date).days)
    decay = 0.5 ** (age_days / half_life_days)
    return similarity_score * ((1 - RECENCY_WEIGHT) + RECENCY_WEIGHT * decay)


def rerank_results(raw_results):
    """Stage 1: apply recency weighting to similarity scores."""
    reranked = []
    for doc, sim_score in raw_results:
        filing_date = doc.metadata.get("filing_date", None)
        adjusted_score = recency_weighted_score(sim_score, filing_date)
        reranked.append((doc, sim_score, adjusted_score))
    reranked.sort(key=lambda x: x[2], reverse=True)
    return reranked


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


# ─── Split & Fuse Generation ────────────────────────────────────────
def generate_single(question, context, llm):
    """Standard single-shot generation. Logs the full prompt and response."""
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    resp = llm.invoke(prompt)
    response_text = resp.content
    log_llm("GENERATE_SINGLE", prompt, response_text)
    return response_text


def generate_split_and_fuse(question, top_results, target_companies, llm,
                            rewritten_queries=None):
    """
    For multi-company comparison queries:
      1. Generate a per-company answer using ONLY that company's chunks
         and that company's REWRITTEN query (not the original multi-company
         comparison question)
      2. Synthesize per-company answers into a final comparison

    Why: Prevents one company's rich data from crowding out the other
    in the LLM's attention window.  Also lets each sub-answer focus
    on extracting numbers without cross-company confusion.
    (Inspired by FinanceRAG paper's long-context management.)

    KEY FIX: Uses rewritten_queries[company] for each sub-answer so the
    LLM sees "What were Amazon's net sales..." instead of
    "Compare Amazon and Google revenue growth...". This prevents the LLM
    from hallucinating data for the other company.

    Args:
        question:           original user question (used for synthesis)
        top_results:        reranked results (5-tuples)
        target_companies:   list of company names
        llm:                ChatOpenAI instance
        rewritten_queries:  dict mapping company name -> rewritten question
                            (from rerank_with_per_company_queries)
    """
    if rewritten_queries is None:
        rewritten_queries = {}

    # Bucket results by company
    by_company = {}
    for item in top_results:
        co = item[0].metadata.get("company", "Unknown")
        by_company.setdefault(co, []).append(item)

    # Generate per-company answers
    per_company_answers = {}
    for company in target_companies:
        company_results = by_company.get(company, [])
        if not company_results:
            per_company_answers[company] = (
                f"No relevant SEC filing data was found for {company}."
            )
            print(f"  [MISS] No chunks for {company} — skipping generation")
            continue

        context = build_context(company_results)

        # KEY FIX: use the rewritten, company-specific query
        company_question = rewritten_queries.get(
            company,
            f"Regarding {company} only: {question}"
        )
        print(f"  -> Sub-question for {company}: {company_question[:120]}...")

        answer = generate_single(company_question, context, llm)
        per_company_answers[company] = answer
        print(f"  [OK] Generated sub-answer for {company} "
              f"({len(company_results)} chunks)")

    # Synthesize
    analyses_text = ""
    for company, answer in per_company_answers.items():
        analyses_text += f"\n### {company}:\n{answer}\n"

    synthesis_prompt = SYNTHESIS_TEMPLATE.format(
        per_company_analyses=analyses_text,
        question=question,
    )
    print("  [OK] Synthesizing comparison...")
    resp = llm.invoke(synthesis_prompt)
    synthesis_response = resp.content
    log_llm("SYNTHESIS", synthesis_prompt, synthesis_response)
    return synthesis_response


# ─── Cross-Encoder with Per-Company Rewriting ───────────────────────
def rerank_with_per_company_queries(question, reranked, target_companies,
                                     llm, top_k):
    """
    Rewrite the query per company, then score each company's chunks
    with its own rewritten query.  Merge and balanced-select.

    This is the key fix for the vocabulary mismatch problem:
    Amazon says "net sales", Google says "revenues".

    Returns:
        tuple of (top_results, rewritten_queries)
        - top_results:       list of 5-tuples (doc, sim, adj, ce, fused)
        - rewritten_queries: dict mapping company name -> rewritten question
                             (threaded downstream to generate_split_and_fuse)
    """
    from reranker import _get_model, _normalize, CE_SCORE_FLOOR, _balanced_select

    # Generate per-company rewritten queries
    rewritten = {}
    for company in target_companies:
        rewritten[company] = rewrite_query_for_company(question, company, llm)

    # Bucket candidates by company
    by_company = {}
    for item in reranked:
        co = item[0].metadata.get("company", "Unknown")
        by_company.setdefault(co, []).append(item)

    model = _get_model()

    # Score each company's chunks with its rewritten query
    all_scored = []
    for company in target_companies:
        items = by_company.get(company, [])
        if not items:
            continue

        ce_query = rewritten.get(company, question)
        pairs = [(ce_query, doc.page_content) for doc, _s, _a in items]
        raw_ce = model.predict(pairs).tolist()

        # Normalize within this company's batch
        norm_ce = _normalize(raw_ce)
        norm_ce_floored = [max(s, CE_SCORE_FLOOR) for s in norm_ce]
        adj_scores = [adj for _, _, adj in items]
        norm_adj = _normalize(adj_scores)

        ce_weight = CE_WEIGHT
        for i, (doc, sim, adj) in enumerate(items):
            fused = ce_weight * norm_ce_floored[i] + (1 - ce_weight) * norm_adj[i]
            all_scored.append((doc, sim, adj, raw_ce[i], fused))

    # Handle any non-target company chunks (shouldn't happen with filtered
    # retrieval, but just in case)
    for co, items in by_company.items():
        if co not in target_companies:
            for doc, sim, adj in items:
                all_scored.append((doc, sim, adj, 0.0, CE_SCORE_FLOOR))

    top_results = _balanced_select(all_scored, target_companies, top_k)
    return top_results, rewritten


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

    # --- Step 1: Resolve target companies (from original question) ---
    target_companies = resolve_companies(question)
    print(f"Resolved companies: "
          f"{target_companies if target_companies else 'None (unfiltered)'}")

    is_comparison = target_companies and len(target_companies) > 1

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
    reranked = rerank_results(raw_results)

    # --- Step 5: Stage-2 rerank (cross-encoder) ---
    print(f"\n  Reranking {len(reranked)} candidates with cross-encoder...")

    rewritten_queries = {}  # will be populated if per-company rewrite is used

    if USE_PER_COMPANY_REWRITE and target_companies:
        # Per-company rewritten queries -> per-company CE scoring
        # KEY FIX: capture rewritten_queries to thread into split & fuse
        top_results, rewritten_queries = rerank_with_per_company_queries(
            question, reranked, target_companies, llm, K
        )
    else:
        # Standard: single query for all chunks
        top_results = cross_encoder_rerank(
            query=question,
            reranked_results=reranked,
            top_k=K,
            ce_weight=CE_WEIGHT,
            target_companies=target_companies,
        )

    # --- Step 6: Generate answer ---
    if is_comparison and USE_SPLIT_AND_FUSE:
        print(f"\n  Comparison detected — using split & fuse generation...")
        # KEY FIX: pass rewritten_queries so each sub-answer uses
        # the company-specific question, not the original comparison
        answer = generate_split_and_fuse(
            question, top_results, target_companies, llm,
            rewritten_queries=rewritten_queries,
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
