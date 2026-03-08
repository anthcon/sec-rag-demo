import sys
import tiktoken
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from prompt_template import PROMPT_TEMPLATE, FUSION_PROMPT_TEMPLATE
from company_resolver import resolve_companies
from reranker import cross_encoder_rerank

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

# --- Long-context management (per FinanceRAG paper) ---
CONTEXT_TOKEN_LIMIT = 12000  # conservative for gpt-4o-mini's 128k window
ENCODING_NAME = "cl100k_base"  # tiktoken encoding for gpt-4o family


# ──────────────────────────────────────────────
#  Query Expansion — keyword extraction via LLM
#  (FinanceRAG §3.1: original query + extracted keywords)
# ──────────────────────────────────────────────

_KEYWORD_SYSTEM = (
    "You are a financial search assistant. Given a user question about SEC filings, "
    "extract 5-10 precise search keywords or short phrases that would help retrieve "
    "the most relevant filing excerpts. Include: company names, financial metrics, "
    "time periods, filing types (10-K, 10-Q), and domain-specific terms. "
    "Return ONLY a comma-separated list of keywords, nothing else."
)

def expand_query(question: str, llm: ChatOpenAI) -> str:
    """
    Combine the original query with LLM-extracted keywords.
    This is the 'Original + Keywords' strategy from the FinanceRAG
    ablation study (Table 1), which scored highest at NDCG@10 = 0.58102.
    """
    try:
        response = llm.invoke([
            {"role": "system", "content": _KEYWORD_SYSTEM},
            {"role": "user", "content": question},
        ])
        keywords = response.content.strip()
        expanded = f"{question} {keywords}"
        print(f"  Expanded query: {expanded[:120]}…")
        return expanded
    except Exception as e:
        print(f"  Query expansion failed ({e}), using original query")
        return question


# ──────────────────────────────────────────────
#  Token counting for long-context management
# ──────────────────────────────────────────────

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding(ENCODING_NAME)
    return len(enc.encode(text))


# ──────────────────────────────────────────────
#  Recency-weighted scoring (unchanged)
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
#  Retrieval helpers (unchanged)
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
#  Balanced top-k selection (unchanged)
# ──────────────────────────────────────────────

def select_top_k(reranked, target_companies, k):
    """
    Guarantee each target company gets fair representation,
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

    for c, docs in buckets.items():
        if c not in target_companies:
            extras.extend(docs)

    extras.sort(key=lambda x: x[2], reverse=True)
    top_results.extend(extras[:overflow])
    top_results.sort(key=lambda x: x[2], reverse=True)

    return top_results


# ──────────────────────────────────────────────
#  Context builder
# ──────────────────────────────────────────────

def build_context(results):
    """Build formatted context string from ranked results.
    Each result is a tuple whose first element is the doc and which
    may be length 3 (sim reranked) or 5 (cross-encoder reranked).
    """
    parts = []
    for item in results:
        doc = item[0]
        meta = doc.metadata
        header = (
            f"[{meta.get('company', '?')} | {meta.get('filing_type', '?')} | "
            f"{meta.get('quarter', '?')} | Filed: {meta.get('filing_date', '?')}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ──────────────────────────────────────────────
#  Long-context split / fuse
#  (FinanceRAG §3.3, Algorithm 1 lines 7-13)
# ──────────────────────────────────────────────

def generate_answer(question: str, top_results: list, llm: ChatOpenAI) -> str:
    """
    If total context fits under CONTEXT_TOKEN_LIMIT, generate in one shot.
    Otherwise, split the context in half, generate two partial answers,
    then fuse them with a second LLM call.
    """
    context = build_context(top_results)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    token_count = count_tokens(prompt)

    if token_count <= CONTEXT_TOKEN_LIMIT:
        print(f"  Context tokens: {token_count} (single-pass)")
        response = llm.invoke(prompt)
        return response.content

    # --- Split / Fuse ---
    mid = len(top_results) // 2
    first_half = top_results[:mid]
    second_half = top_results[mid:]

    print(f"  Context tokens: {token_count} > {CONTEXT_TOKEN_LIMIT} — using split/fuse")

    ctx_a = build_context(first_half)
    prompt_a = PROMPT_TEMPLATE.format(context=ctx_a, question=question)
    resp_a = llm.invoke(prompt_a).content

    ctx_b = build_context(second_half)
    prompt_b = PROMPT_TEMPLATE.format(context=ctx_b, question=question)
    resp_b = llm.invoke(prompt_b).content

    fusion_prompt = FUSION_PROMPT_TEMPLATE.format(
        question=question,
        answer_a=resp_a,
        answer_b=resp_b,
    )
    fused = llm.invoke(fusion_prompt)
    return fused.content


# ──────────────────────────────────────────────
#  Main query pipeline
# ──────────────────────────────────────────────

def query(question: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    # --- Step 1: Resolve companies from query ---
    target_companies = resolve_companies(question)
    print(f"\nResolved companies: {target_companies if target_companies else 'None (unfiltered)'}")

    # --- Step 2: Query expansion (FinanceRAG §3.1) ---
    expanded_question = expand_query(question, llm)

    # --- Step 3: Retrieve using expanded query ---
    if target_companies:
        raw_results = retrieve_filtered(db, expanded_question, target_companies)
    else:
        raw_results = retrieve_unfiltered(db, expanded_question)

    if not raw_results or max(score for _, score in raw_results) < 0.3:
        print("No sufficiently relevant results found.")
        return

    # --- Step 4: First-stage rerank with recency weighting ---
    reranked = rerank_results(raw_results)

    # --- Step 5: Second-stage cross-encoder rerank (FinanceRAG §3.2) ---
    #     Uses the ORIGINAL question (not expanded) for CE scoring,
    #     because the CE model scores semantic match directly.
    print("  Running cross-encoder rerank…")
    ce_reranked = cross_encoder_rerank(
        query=question,
        reranked_results=reranked,
        top_k=K,
    )

    # --- Step 6: Balanced selection (post CE) ---
    #     ce_reranked items are 5-tuples: (doc, sim, adj, ce_score, fused_score)
    #     select_top_k expects 3-tuples with score at index 2, so we adapt:
    adapted = [(doc, sim, fused) for doc, sim, adj, ce_score, fused in ce_reranked]
    top_results = select_top_k(adapted, target_companies, K)

    # --- Step 7: Generate answer with long-context management ---
    answer = generate_answer(question, top_results, llm)

    print(f"\nAnswer:\n{answer}")
    print(f"\nSources used ({len(top_results)}):")
    for item in top_results:
        doc = item[0]
        meta = doc.metadata
        scores = f"fused: {item[2]:.3f}" if len(item) >= 3 else ""
        print(f"  - {meta.get('company')} | {meta.get('filing_type')} | {meta.get('filing_date')} | {scores}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Enter question: ")
    else:
        question = " ".join(sys.argv[1:])

    query(question)