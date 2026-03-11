# query.py
"""
SEC Filing RAG Pipeline — Query Module

Pipeline stages:
  1. Resolve target companies from user question
  2. Per-company query rewriting (financial terminology alignment)
  3. Retrieve chunks per company from ChromaDB
  4. Stage-1 rerank: recency weighting
  5. Generate answer via single-shot LLM call

Design decisions:
  - Per-company rewriting bridges vocab gap between user language and
    SEC filing terminology (e.g. "revenue" → Amazon "net sales",
    Google "revenues"). Concatenated rewrites become the embedding query.
  - Recency weighting in stage-1 rerank favors newer filings.
"""

import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from prompt_template import PROMPT_TEMPLATE
from company_resolver import resolve_companies
from reranker import stage1_rerank

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────
CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"
K = 20
CHUNKS_PER_COMPANY = 10
MAX_TOKENS = 2000
TEMPERATURE = 0.2
FALLBACK_RETRIEVAL_POOL = K * 3


# ─── LLM Logging Helper ─────────────────────────────────────────────
def log_llm(tag: str, prompt: str, response: str):
    """Print a formatted log block for every LLM call."""
    divider = "=" * 72
    print(f"\n{divider}")
    print(f"  LLM CALL: {tag}")
    print(divider)
    print(f"  PROMPT:\n{prompt}")
    print(f"\n  RESPONSE:\n{response}")
    print(f"{divider}\n")


# ─── Per-Company Query Rewriting ─────────────────────────────────────
def rewrite_query_for_company(question: str, company: str,
                              llm: ChatOpenAI) -> str:
    """
    Rewrite the user question so it targets ONLY the specified company
    using that company's SEC filing vocabulary.

    Cross-encoders and embeddings are vocabulary-sensitive. Amazon says
    "net sales", Google says "revenues". Per-company rewriting fixes
    this asymmetry.
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
        f"  5. If the original question targets specific numbers: anchor the rewrite "
        f"toward 'consolidated statements of operations', 'net sales', 'operating income', "
        f"'segment revenue' — NOT risk factors or general descriptions.\n"
        f"  6. Return ONE sentence, no explanation.\n\n"
        f"Original question: {question}\n"
        f"Rewritten for {company}:"
    )
    try:
        resp = llm.invoke(prompt)
        rewritten = resp.content.strip()
        log_llm(f"REWRITE_QUERY [{company}]", prompt, rewritten)
        print(f"    CE query [{company}]: {rewritten[:150]}...")
        return rewritten
    except Exception as e:
        print(f"    Rewrite failed for {company} ({e}), using original")
        return question


# ─── Retrieval ───────────────────────────────────────────────────────
def retrieve_filtered(db, question, target_companies):
    """Per-company filtered retrieval from ChromaDB."""
    all_raw = []
    for company in target_companies:
        try:
            results = db.similarity_search_with_relevance_scores(
                question, k=CHUNKS_PER_COMPANY, filter={"company": company},
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
    """Turn reranker output tuples into a formatted context string."""
    parts = []
    for doc, sim, adj in results:
        meta = doc.metadata
        header = (
            f"[{meta.get('company', '?')} | {meta.get('filing_type', '?')} | "
            f"{meta.get('quarter', '?')} | Filed: {meta.get('filing_date', '?')}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ─── Generation ──────────────────────────────────────────────────────
def generate_single(question, context, llm):
    """Single-shot generation. Logs the full prompt and response."""
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    resp = llm.invoke(prompt)
    response_text = resp.content
    log_llm("GENERATE_SINGLE", prompt, response_text)
    return response_text


# ─── Main Query Pipeline ────────────────────────────────────────────
def query(question: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    llm = ChatOpenAI(
        model=LLM_MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
    )

    # Step 1: Resolve target companies
    target_companies = resolve_companies(question)
    print(f"Resolved companies: "
          f"{target_companies if target_companies else 'None (unfiltered)'}")

    # Step 2: Rewrite query per company for terminology alignment
    rewritten = {
        company: rewrite_query_for_company(question, company, llm)
        for company in target_companies
    }
    search_query = " ".join(rewritten.values()) if rewritten else question
    print(f"Final search query for retrieval: {search_query[:250]}...")

    # Step 3: Retrieve from ChromaDB
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

    # Step 4: Stage-1 rerank (recency weighting)
    reranked = stage1_rerank(raw_results)

    # Step 5: Generate answer
    context = build_context(reranked)
    answer = generate_single(question, context, llm)

    # Output
    print(f"\nAnswer:\n{answer}")
    print(f"\nSources ({len(reranked)}):")
    for doc, sim, adj in reranked:
        meta = doc.metadata
        print(
            f"  - {meta.get('company')} | {meta.get('filing_type')} | "
            f"{meta.get('filing_date')}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Enter question: ")
    else:
        question = " ".join(sys.argv[1:])
    query(question)