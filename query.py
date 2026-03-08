import sys
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from prompt_template import PROMPT_TEMPLATE
from company_resolver import resolve_companies

load_dotenv()

CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"

K = 20
CHUNKS_PER_COMPANY = 10        # per-company retrieval pool before reranking
MAX_TOKENS = 2000
TEMPERATURE = 0.2
HALF_LIFE_DAYS = 365
RECENCY_WEIGHT = 0.4
FALLBACK_RETRIEVAL_POOL = K * 3  # only used when no companies detected


def recency_weighted_score(similarity_score, filing_date_str, half_life_days=HALF_LIFE_DAYS):
    try:
        filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return similarity_score * (1 - RECENCY_WEIGHT * 0.5)

    age_days = (datetime.now() - filing_date).days
    if age_days < 0:
        age_days = 0

    decay = 0.5 ** (age_days / half_life_days)
    return similarity_score * ((1 - RECENCY_WEIGHT) + RECENCY_WEIGHT * decay)


def rerank_results(raw_results):
    """Apply recency weighting and sort by adjusted score."""
    reranked = []
    for doc, sim_score in raw_results:
        filing_date = doc.metadata.get("filing_date", None)
        adjusted_score = recency_weighted_score(sim_score, filing_date)
        reranked.append((doc, sim_score, adjusted_score))
    reranked.sort(key=lambda x: x[2], reverse=True)
    return reranked


def retrieve_filtered(db, question, target_companies):
    """
    Per-company filtered retrieval.
    Queries the vector store once per company with a metadata filter,
    then merges and reranks all results together.
    """
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
    """Fallback: no company filter, original behavior."""
    print("  No companies detected — falling back to unfiltered retrieval")
    return db.similarity_search_with_relevance_scores(
        question, k=FALLBACK_RETRIEVAL_POOL
    )


def select_top_k(reranked, target_companies, k):
    """
    Balanced selection: guarantee each target company gets fair representation,
    then fill remaining slots by best adjusted score.
    """
    if not target_companies or len(target_companies) <= 1:
        return reranked[:k]

    buckets = {}
    for item in reranked:
        company = item[0].metadata.get("company", "Unknown")
        if company not in buckets:
            buckets[company] = []
        buckets[company].append(item)

    per_company = k // len(target_companies)
    overflow = k - (per_company * len(target_companies))

    top_results = []
    extras = []

    for company in target_companies:
        pool = buckets.get(company, [])
        top_results.extend(pool[:per_company])
        extras.extend(pool[per_company:])

    # Include any chunks from companies not in target list (shouldn't happen
    # with filtered retrieval, but defensive)
    for c, docs in buckets.items():
        if c not in target_companies:
            extras.extend(docs)

    extras.sort(key=lambda x: x[2], reverse=True)
    top_results.extend(extras[:overflow])
    top_results.sort(key=lambda x: x[2], reverse=True)

    return top_results


def query(question: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    # --- Step 1: Resolve companies from query ---
    target_companies = resolve_companies(question)
    print(f"\nResolved companies: {target_companies if target_companies else 'None (unfiltered)'}")

    # --- Step 2: Retrieve ---
    if target_companies:
        raw_results = retrieve_filtered(db, question, target_companies)
    else:
        raw_results = retrieve_unfiltered(db, question)

    if not raw_results or max(score for _, score in raw_results) < 0.3:
        print("No sufficiently relevant results found.")
        return

    # --- Step 3: Rerank with recency weighting ---
    reranked = rerank_results(raw_results)

    # --- Step 4: Balanced selection ---
    top_results = select_top_k(reranked, target_companies, K)

    # --- Step 5: Build context and query LLM ---
    context_parts = []
    for doc, sim_score, adj_score in top_results:
        meta = doc.metadata
        header = (
            f"[{meta.get('company', '?')} | {meta.get('filing_type', '?')} | "
            f"{meta.get('quarter', '?')} | Filed: {meta.get('filing_date', '?')}]"
        )
        context_parts.append(f"{header}\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    response = llm.invoke(prompt)

    print(f"\nAnswer:\n{response.content}")
    print(f"\nSources used ({len(top_results)}):")
    for doc, sim_score, adj_score in top_results:
        meta = doc.metadata
        print(
            f"  - {meta.get('company')} | {meta.get('filing_type')} | "
            f"{meta.get('filing_date')} | sim: {sim_score:.3f} | adj: {adj_score:.3f}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Enter question: ")
    else:
        question = " ".join(sys.argv[1:])

    query(question)