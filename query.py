"""
SEC Filing RAG — Query Pipeline
Retrieves relevant chunks per company, reranks with cross-encoder,
and generates an answer via gpt-4o-mini in a single API call.
"""

import sys
import tiktoken
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from prompt_template import PROMPT_TEMPLATE
from company_resolver import resolve_companies
from reranker import cross_encoder_rerank

load_dotenv()

# --- Configuration ---
CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"

CHUNKS_PER_COMPANY = 15
TOP_K = 20
MAX_CONTEXT_TOKENS = 10000
MAX_RESPONSE_TOKENS = 2000
TEMPERATURE = 0.2
MIN_SCORE_WARNING = 0.15
FALLBACK_K = 40


def count_tokens(text: str, model: str = LLM_MODEL) -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def retrieve_for_companies(db, question: str, companies: list[str]) -> list[tuple]:
    """
    Per-company focused retrieval.
    Prepends company name to query so the bi-encoder targets that
    company's content instead of diluting across all companies.
    """
    all_results = []
    for company in companies:
        focused_query = f"{company} {question}"
        try:
            results = db.similarity_search_with_relevance_scores(
                focused_query,
                k=CHUNKS_PER_COMPANY,
                filter={"company": company},
            )
            if results:
                print(f"  {company}: {len(results)} chunks (best sim: {results[0][1]:.3f})")
            else:
                print(f"  {company}: 0 chunks")
            all_results.extend(results)
        except Exception as e:
            print(f"  {company}: retrieval failed — {e}")
    return all_results


def retrieve_unfiltered(db, question: str) -> list[tuple]:
    print("  No companies detected — unfiltered retrieval")
    return db.similarity_search_with_relevance_scores(question, k=FALLBACK_K)


def balanced_select(scored_results, companies: list[str], k: int) -> list:
    """
    Guarantee each target company gets fair representation.
    Remaining slots filled by best fused score.
    """
    if not companies or len(companies) <= 1:
        return scored_results[:k]

    buckets = {}
    for item in scored_results:
        company = item[0].metadata.get("company", "Unknown")
        buckets.setdefault(company, []).append(item)

    per_company = max(k // len(companies), 1)
    selected = []
    overflow = []

    for company in companies:
        pool = buckets.get(company, [])
        selected.extend(pool[:per_company])
        overflow.extend(pool[per_company:])

    for c, pool in buckets.items():
        if c not in companies:
            overflow.extend(pool)

    overflow.sort(key=lambda x: x[-1], reverse=True)
    remaining = k - len(selected)
    if remaining > 0:
        selected.extend(overflow[:remaining])

    selected.sort(key=lambda x: x[-1], reverse=True)
    return selected[:k]


def build_context(top_results, max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """Assemble context string from ranked chunks, respecting token budget."""
    parts = []
    token_count = 0

    for item in top_results:
        doc = item[0]
        meta = doc.metadata
        header = (
            f"[{meta.get('company', '?')} | {meta.get('filing_type', '?')} | "
            f"{meta.get('quarter', '?')} | Filed: {meta.get('filing_date', '?')}]"
        )
        block = f"{header}\n{doc.page_content}"
        block_tokens = count_tokens(block)

        if token_count + block_tokens > max_tokens:
            break

        parts.append(block)
        token_count += block_tokens

    print(f"  Context: {token_count} tokens from {len(parts)} chunks")
    return "\n\n---\n\n".join(parts)


def query(question: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # --- Step 1: Resolve companies from query ---
    target_companies = resolve_companies(question)
    print(f"\nResolved companies: {target_companies or 'None (unfiltered)'}")

    # --- Step 2: Per-company focused retrieval ---
    if target_companies:
        raw_results = retrieve_for_companies(db, question, target_companies)
    else:
        raw_results = retrieve_unfiltered(db, question)

    if not raw_results:
        print("No results found.")
        return

    best_score = max(score for _, score in raw_results)
    if best_score < MIN_SCORE_WARNING:
        print(f"  WARNING: Best similarity = {best_score:.3f} — results may be weak")

    # --- Step 3: Cross-encoder rerank ---
    print(f"  Reranking {len(raw_results)} candidates…")
    reranked = cross_encoder_rerank(question, raw_results)

    # --- Step 4: Balanced selection ---
    top_results = balanced_select(reranked, target_companies, TOP_K)

    # --- Step 5: Build context + LLM call ---
    context = build_context(top_results)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        max_tokens=MAX_RESPONSE_TOKENS,
        temperature=TEMPERATURE,
    )
    response = llm.invoke(prompt)

    print(f"\nAnswer:\n{response.content}")
    print(f"\nSources ({len(top_results)}):")
    for doc, sim, fused in top_results:
        meta = doc.metadata
        print(
            f"  - {meta.get('company')} | {meta.get('filing_type')} | "
            f"{meta.get('filing_date')} | sim: {sim:.3f} | fused: {fused:.3f}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Enter question: ")
    else:
        question = " ".join(sys.argv[1:])
    query(question)