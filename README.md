
# SEC Filing RAG Demo

A retrieval-augmented generation (RAG) system that answers business questions about SEC financial filings using a local vector index and a single LLM API call.

## Architecture

1. **Indexing** — SEC filings are chunked, embedded locally using `bge-small-en-v1.5`, and stored in a ChromaDB vector database.
2. **Retrieval** — User questions are embedded with the same model and matched against the index via similarity search.
3. **Generation** — Retrieved chunks (with metadata) are injected into a prompt template and sent to `gpt-4o-mini` in a single API call.

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
git clone <repo-url>
cd sec-rag-demo
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

### Build the Index

One-time step, approximately 2 hours on CPU.

```bash
python create_index.py
```

This downloads the `bge-small-en-v1.5` embedding model on first run (~133MB), chunks all 246 filings, and writes the vector index to `./chroma/`.

### Run a Query

```bash
python query.py "What are the primary risk factors facing Apple, Tesla, and JPMorgan?"
```

Or run interactively:

```bash
python query.py
```

## Project Structure

```
sec-rag-demo/
├── README.md
├── requirements.txt
├── .env                       # API key (not committed)
├── manifest.json              # Corpus metadata
├── edgar_corpus/              # SEC filing .txt files
├── create_index.py            # Indexing pipeline
├── query.py                   # Retrieval + LLM generation
├── prompt_template.py         # Final prompt template
├── prompt_iterations.md       # Log of prompt changes and reasoning
├── evaluation_notes.md        # Quality assessment notes
└── chroma/                    # Vector store (auto-generated)
```

## Design Decisions

- **Local embeddings (bge-small-en-v1.5)** — Eliminates API rate limits and cost during indexing. Reserves API budget for the generation call where model quality matters most.
- **Chunk size 1500 chars / 200 overlap** — Balances retrieval granularity with coherent context. Fits within the model's 512-token window.
- **Metadata extracted from filenames** — Each chunk carries company name, ticker, filing type, quarter, and date — enables richer citations and potential filtered retrieval.
- **K=20 retrieval results** — Multi-company comparison questions need breadth across filings from different companies.
- **gpt-4o-mini at temperature 0.2** — Cost-efficient for synthesis tasks. Low temperature keeps answers factual and grounded.
- **Single LLM call** — Per the assignment constraint. All retrieval and context assembly happens before the call.

## Configuration

Key parameters are defined as constants at the top of each script:

- **create_index.py** — `CHUNK_SIZE`, `CHUNK_OVERLAP`, `BATCH_SIZE`, `EMBEDDING_MODEL`
- **query.py** — `K`, `MAX_TOKENS`, `TEMPERATURE`, `LLM_MODEL`
