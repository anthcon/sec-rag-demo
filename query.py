# query.py
import sys
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from prompt_template import PROMPT_TEMPLATE
from company_resolver import resolve_companies
from reranker import cross_encoder_rerank

load_dotenv()

CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"

K = 20
CHUNKS_PER_COMPANY = 10
MAX_TOKENS = 2000
TEMPERATURE = 0.2
HALF_LIFE_DAYS = 365
RECENCY_WEIGHT = 0.4
FALLBACK_RETRIEVAL_POOL = K * 3
CE_WEIGHT = 0.7
USE_QUERY_EXPANSION = True


# ─── Query Expansion ────────────────────────────────────────────────
def expand_query(question: str, llm: ChatOpenAI) -> str:
    """
    Use the LLM to rewrite the user question into a short list of
    finance-specific search keywords / synonyms, then append them
    to the original query for better embedding + CE matching.

    Design decision (walkthrough):
      - Cheap single LLM call (gpt-4o-mini) adds ~0.3s latency.
      - Bridges vocabulary gap: user says "revenue", filing says
        "net sales" or "net revenues".  Embedding model alone
        often misses these.
    """
    expansion_prompt = (
        "You are a financial search assistant. Given the user question below, "
        "produce a SHORT comma-separated list of 5-8 alternative keywords or "
        "phrases that SEC 10-K/10-Q filings might use to discuss the same topic. "
        "Include synonyms, line-item names, and common financial abbreviations. "
        "Return ONLY the comma-separated keywords, nothing else.\n\n"
        f"Question: {question}"
    )
    try:
        resp = llm.invoke(expansion_prompt)
        keywords = resp.content.strip()
        expanded = f"{question} {keywords}"
        print(f"  Expanded query: {expanded[:120]}…")
        return expanded
    except Exception as e:
        print(f"  Query expansion failed ({e}), using original query")
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
    print(f"\nResolved companies: "
          f"{target_companies if target_companies else 'None (unfiltered)'}")

    # --- Step 2: Query expansion ---
    search_query = question
    if USE_QUERY_EXPANSION:
        search_query = expand_query(question, llm)

    # --- Step 3: Retrieve ---
    if target_companies:
        raw_results = retrieve_filtered(db, search_query, target_companies)
    else:
        raw_results = retrieve_unfiltered(db, search_query)

    if not raw_results or max(score for _, score in raw_results) < 0.3:
        print("No sufficiently relevant results found.")
        return

    # --- Step 4: Stage-1 rerank (recency weighting) ---
    reranked = rerank_results(raw_results)

    # --- Step 5: Stage-2 rerank (cross-encoder + balanced selection) ---
    print(f"\n  Running cross-encoder on {len(reranked)} candidates …")
    top_results = cross_encoder_rerank(
        query=question,           # original question for CE (not expanded)
        reranked_results=reranked,
        top_k=K,
        ce_weight=CE_WEIGHT,
        target_companies=target_companies,
    )

    # --- Step 6: Build context ---
    context_parts = []
    for doc, sim_score, adj_score, ce_score, fused_score in top_results:
        meta = doc.metadata
        header = (
            f"[{meta.get('company', '?')} | {meta.get('filing_type', '?')} | "
            f"{meta.get('quarter', '?')} | Filed: {meta.get('filing_date', '?')}]"
        )
        context_parts.append(f"{header}\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # --- Step 7: Generate answer ---
    response = llm.invoke(prompt)

    # --- Output ---
    print(f"\nAnswer:\n{response.content}")
    print(f"\nSources used ({len(top_results)}):")
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