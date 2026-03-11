# query.py
"""
SEC Filing RAG Pipeline — Query Module

Pipeline:
  1. Resolve target companies from user question
  2. Retrieve chunks per company from ChromaDB
  3. Recency rerank
  4. Generate answer via single-shot LLM call
"""
import sys
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from prompt_template import PROMPT_TEMPLATE
from company_resolver import resolve_companies
from reranker import recency_rerank

load_dotenv()

# ─── Configuration ───────────────────────────────────────────────────
CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"
RETRIEVAL_K = 20
CHUNKS_PER_COMPANY = 10
MAX_TOKENS = 2000
TEMPERATURE = 0.2
SIMILARITY_THRESHOLD = 0.3

# ─── LLM Logging ────────────────────────────────────────────────────
LOG_FILE = "llm_log.txt"

def log_llm_call(tag: str, prompt: str, response: str):
    """Append every LLM call to a plaintext log file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 72}\n")
        f.write(f"[{datetime.now().isoformat()}] {tag}\n")
        f.write(f"{'=' * 72}\n")
        f.write(f"PROMPT:\n{prompt}\n\n")
        f.write(f"RESPONSE:\n{response}\n")


# ─── Retrieval ───────────────────────────────────────────────────────
def retrieve_per_company(db, question, target_companies):
    """Per-company filtered retrieval from ChromaDB."""
    all_chunks = []
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
            all_chunks.extend(results)
        except Exception as e:
            print(f"  {company}: retrieval failed — {e}")
    return all_chunks


def retrieve_unfiltered(db, question):
    """Fallback when no companies detected."""
    print("  No companies detected — unfiltered retrieval")
    return db.similarity_search_with_relevance_scores(
        question, k=RETRIEVAL_K
    )


# ─── Context Building ───────────────────────────────────────────────
def build_context(scored_chunks):
    """Format reranker output into a context string for the LLM."""
    sections = []
    for doc, sim, adjusted in scored_chunks:
        metadata = doc.metadata
        header = (
            f"[{metadata.get('company', '?')} | {metadata.get('filing_type', '?')} | "
            f"{metadata.get('quarter', '?')} | Filed: {metadata.get('filing_date', '?')}]"
        )
        sections.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


# ─── Generation ──────────────────────────────────────────────────────
def generate_answer(question, context, llm):
    """Single-shot LLM generation from retrieved context."""
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
    log_llm_call("GENERATE", prompt, response.content)
    return response.content


# ─── Main Query Pipeline ────────────────────────────────────────────
def query(question: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    llm = ChatOpenAI(
        model=LLM_MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
    )

    # 1: Resolve target companies
    target_companies = resolve_companies(question)
    print(f"Resolved companies: "
          f"{target_companies if target_companies else 'None (unfiltered)'}")

    # 2: Retrieve from ChromaDB
    if target_companies:
        retrieved_chunks = retrieve_per_company(db, question, target_companies)
    else:
        retrieved_chunks = retrieve_unfiltered(db, question)

    if not retrieved_chunks:
        print("No results found.")
        return

    best_sim = max(score for _, score in retrieved_chunks)
    if best_sim < SIMILARITY_THRESHOLD:
        print(f"Best similarity {best_sim:.3f} below threshold. Aborting.")
        return

    # 3: Recency rerank
    scored_chunks = recency_rerank(retrieved_chunks)

    # 4: Generate answer
    context = build_context(scored_chunks)
    answer = generate_answer(question, context, llm)

    print(f"\nAnswer:\n{answer}")
    print(f"\nSources ({len(scored_chunks)}):")
    for doc, sim, adjusted in scored_chunks:
        metadata = doc.metadata
        print(
            f"  - {metadata.get('company')} | {metadata.get('filing_type')} | "
            f"{metadata.get('filing_date')}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Enter question: ")
    else:
        question = " ".join(sys.argv[1:])
    query(question)