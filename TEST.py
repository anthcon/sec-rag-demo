import sys
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from prompt_template import PROMPT_TEMPLATE, SYNTHESIS_TEMPLATE
from company_resolver import resolve_companies
from reranker import cross_encoder_rerank

CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gpt-4o-mini"

K = 20                        # total chunks to feed into context
CHUNKS_PER_COMPANY = 10       # how many to pull from ChromaDB per company
MAX_TOKENS = 2000
TEMPERATURE = 0.2
HALF_LIFE_DAYS = 365
RECENCY_WEIGHT = 0.4
CE_WEIGHT = 0.7
FALLBACK_RETRIEVAL_POOL = K * 3

USE_QUERY_EXPANSION = True
USE_PER_COMPANY_REWRITE = True
USE_SPLIT_AND_FUSE = True

load_dotenv()

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

def generate_single(question, context, llm):
    """Standard single-shot generation. Logs the full prompt and response."""
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    resp = llm.invoke(prompt)
    response_text = resp.content
    log_llm("GENERATE_SINGLE", prompt, response_text)
    return response_text

def query(question: str):
    # Init shared resources
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    llm = ChatOpenAI(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    answer = generate_single(question, "no context", llm)

    # --- Output ---
    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Enter question: ")
    else:
        question = " ".join(sys.argv[1:])

    query(question)
