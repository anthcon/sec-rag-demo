# SEC Filing RAG Demo

A retrieval-augmented generation system that answers business questions about SEC financial filings (10-K and 10-Q, 2023–2025) using a local vector index, company-aware retrieval, recency-weighted reranking, and a single LLM API call.

## Architecture

The pipeline has four stages, all executed at query time:

- **Company Resolver** — match company names/tickers from the question
- **Filtered Retrieval** — Per-company similarity search against ChromaDB
- **Recency Rerank** — Exponential time-decay (365-day half-life)
- **LLM Generation** — Single gpt-4o-mini call with structured prompt

### Stage 1 — Company Resolver (`company_resolver.py`)

Extracts company names from the user's natural-language question:

- **Pass 1:** Word-boundary substring matching against an alias index (tickers, full names, common shorthand like "Google" → Alphabet/Google, "JPMorgan" → JPMorgan Chase).

This drives per-company filtered retrieval, ensuring multi-company comparison questions pull chunks from each relevant company.

### Stage 2 — Filtered Retrieval (`query.py`)

If companies are resolved, ChromaDB similarity search runs per company using a metadata filter (`{"company": "<name>"}`), returning up to 10 chunks each. If no companies are detected, an unfiltered top-20 search runs as fallback. A similarity threshold of 0.3 gates low-confidence results.

### Stage 3 — Recency Rerank (`reranker.py`)

Applies exponential time-decay to similarity scores so recent filings rank higher without burying older ones:

```
adjusted_score = similarity * ((1 - 0.4) + 0.4 × 0.5^(age_days / 365))
```

- **Half-life:** 365 days — a filing one year old retains ~80% of its recency bonus.
- **Recency weight:** 0.4 — similarity still dominates; recency is a tiebreaker.

### Stage 4 — LLM Generation (`prompt_template.py`)

A single gpt-4o-mini call (temperature 0.2) with a structured prompt that enforces:

- **Source fidelity** — never attribute data from one company's filing to another.
- **Verbatim numbers** — exact figures, no rounding.
- **Explicit gap reporting** — "data not available" rather than hallucinating.
- **Table parsing** — extract data from pipe-delimited or irregularly formatted tables.


## Corpus

50 S&P 500 companies × ~5 filings each = 246 SEC filings (10-K and 10-Q, 2023–2025). Companies include Apple, Amazon, Tesla, JPMorgan Chase, NVIDIA, Microsoft, Meta, and 43 others. Full ticker list is in `create_index.py`.

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
git clone https://github.com/anthcon/sec-rag-demo.git
cd sec-rag-demo
git checkout main
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
tar -xzvf chroma_index.tar.gz
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

### Index

The ChromaDB vector index is pre-built and committed to the repo in `chroma_index.tar.gz`. No indexing step is required to run queries. Just need to unzip it.

If you ever need to rebuild it from scratch (~2 hours on CPU):

```bash
python create_index.py
```

This chunks all 246 filings (1500 chars / 200 overlap), embeds them with bge-small-en-v1.5, and writes to `./chroma/`.

## Usage

### Run a Query

```bash
python query.py "What are the primary risk factors facing Apple, Tesla, and JPMorgan?"
```

Interactive mode (prompts for input):

```bash
python query.py
```

### What Happens

1. Company resolver extracts Apple, Tesla, JPMorgan Chase from the question.
2. ChromaDB returns up to 10 chunks per company, filtered by metadata.
3. Recency reranker re-sorts chunks favoring recent filings.
4. Context is assembled and sent to gpt-4o-mini in a single API call.
5. Answer prints to stdout with source citations.

### Example Questions

```bash
"How did NVIDIA's revenue change between 2023 and 2024?"
"Compare Apple and Microsoft's R&D spending."
"What liquidity risks does Goldman Sachs report?"
"Summarize Tesla's capital expenditure plans."
```

## Project Structure

```
sec-rag-demo/
├── README.md
├── requirements.txt
├── .env                       # API key (not committed)
├── manifest.json              # Corpus metadata (inside edgar_corpus folder)
├── edgar_corpus/              # SEC filing .txt files
├── chroma/                    # Pre-built vector index
├── create_index.py            # Indexing pipeline (run once, already done)
├── company_resolver.py        # Alias index
├── query.py                   # Full query pipeline (retrieve → rerank → generate)
├── reranker.py                # Recency-weighted reranking
├── prompt_template.py         # Prompt template
├── prompt_iterations.md       # Log of prompt changes and reasoning
└── llm_log.txt                # Auto-generated log of every LLM call
```

## Design Decisions

- **Local embeddings (bge-small-en-v1.5)** — Eliminates API rate limits and cost during indexing. Reserves the API budget for the generation call where model quality matters most.

- **Company-aware retrieval** — Multi-company questions (e.g., "Compare Apple and Tesla") need chunks from each company. The resolver + per-company filtered search guarantees balanced coverage instead of letting one company dominate by similarity score alone.

- **Recency reranking over semantic reranking** — A cross-encoder reranker (e.g., ms-marco) would improve relevance but adds ~2s latency and a model dependency. Recency weighting is zero-cost, deterministic, and directly addresses the most common failure mode: returning stale filings when newer data exists.

- **Chunk size 1500 chars / 200 overlap** — Balances retrieval granularity with coherent context.

- **K=10 per company (vs. flat K=20)** — Per-company retrieval with 10 chunks each scales with the number of companies in the question, giving better coverage for comparisons.

- **Single LLM call** — Per the assignment constraint. All retrieval, reranking, and context assembly happens before the call.

- **gpt-4o-mini at temperature 0.2** — Cost-efficient for synthesis. Low temperature keeps answers factual and grounded in the retrieved context.

- **LLM call logging** — Every prompt and response is appended to llm_log.txt for debugging and evaluation.

## Evaluation

Quality was assessed manually across several dimensions:

- **Factual accuracy** — For each test question, I verified key numbers (revenue, R&D spend, risk factors) against the original SEC filing text. I checked that the system consistently returned verbatim figures rather than hallucinated or rounded numbers, which was the primary quality bar.

- **Source attribution** — Checked that cited filing metadata (company, form type, period) matched the claims in the generated answer. Multi-company comparison questions were the hardest case here — early prompt iterations would sometimes blend data across companies, which the current prompt's source-fidelity instructions resolved.

- **Retrieval relevance** — Reviewed the retrieved chunks (logged in llm_log.txt) to confirm they were topically on-target. The company-aware filtered retrieval eliminated the main failure mode of irrelevant cross-company chunks showing up. The similarity threshold of 0.3 caught the remaining low-confidence retrievals.

- **Recency correctness** — Tested questions where both old and new filings existed (e.g., "What is Apple's latest revenue?") and confirmed the reranker surfaced the most recent filing without burying relevant older context.

- **Failure modes observed** — Questions about companies not in the corpus return a clean "data not available" message rather than hallucinating. Very broad questions ("Tell me everything about Apple") sometimes hit the context window limit with marginal chunks; the per-company K=10 cap keeps this bounded.

- **Prompt iteration tracking** — Each prompt change and its impact on answer quality is logged in prompt_iterations.md. This includes what broke, what improved, and why the change was made.

## Configuration

Key parameters at the top of each script:

| Script | Parameter | Default | Purpose |
|--------|-----------|---------|---------|
| `create_index.py` | `CHUNK_SIZE` | 1500 | Characters per chunk |
| `create_index.py` | `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `query.py` | `CHUNKS_PER_COMPANY` | 10 | Chunks retrieved per company |
| `query.py` | `RETRIEVAL_K` | 20 | Chunks for unfiltered fallback |
| `query.py` | `SIMILARITY_THRESHOLD` | 0.3 | Minimum similarity to proceed |
| `query.py` | `TEMPERATURE` | 0.2 | LLM generation temperature |
| `reranker.py` | `HALF_LIFE_DAYS` | 365 | Recency decay half-life |
| `reranker.py` | `RECENCY_WEIGHT` | 0.4 | Weight of recency vs. similarity |
