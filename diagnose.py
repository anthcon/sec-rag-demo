from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from collections import defaultdict
import sys
import io

# Force stdout to use utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
collection = db._collection

count = collection.count()
print(f"Total documents in index: {count}")

# --- Batch fetch metadata (ChromaDB can't handle 71K in one call) ---
BATCH = 5000
companies = defaultdict(int)
filing_types = defaultdict(int)
company_filings = defaultdict(set)   # company -> set of (filing_type, filing_date)
dates = []

for offset in range(0, count, BATCH):
    batch = collection.get(
        include=["metadatas"],
        limit=BATCH,
        offset=offset
    )
    for m in batch["metadatas"]:
        c = m.get("company", "MISSING")
        ft = m.get("filing_type", "MISSING")
        fd = m.get("filing_date", "")
        q = m.get("quarter", "")
        companies[c] += 1
        filing_types[ft] += 1
        if fd:
            dates.append(fd)
            company_filings[c].add((ft, fd, q))

# --- Company coverage ---
print(f"\nUnique companies in index ({len(companies)}):")
for c, cnt in sorted(companies.items(), key=lambda x: -x[1]):
    print(f"  {c}: {cnt} chunks")

# --- Filing type coverage ---
print(f"\nFiling types:")
for ft, cnt in sorted(filing_types.items(), key=lambda x: -x[1]):
    print(f"  {ft}: {cnt} chunks")

# --- Date range ---
if dates:
    print(f"\nDate range: {min(dates)} → {max(dates)}")

# --- Per-company filing inventory (the key diagnostic) ---
print("\n--- Per-Company Filing Inventory ---")
for c in sorted(company_filings.keys()):
    filings = sorted(company_filings[c])
    print(f"\n{c} ({companies[c]} chunks):")
    for ft, fd, q in filings:
        print(f"  {ft} | filed: {fd} | quarter: {q}")

# --- Targeted test: Amazon + Alphabet revenue chunks ---
print("\n--- Retrieval Test: Amazon revenue ---")
results = db.similarity_search_with_relevance_scores(
    "Amazon total revenue net sales year over year growth", k=5
)
for doc, score in results:
    m = doc.metadata
    print(f"  score={score:.3f} | {m.get('company')} | {m.get('filing_type')} | {m.get('filing_date')} | {doc.page_content[:100]}...")

print("\n--- Retrieval Test: Alphabet revenue ---")
results = db.similarity_search_with_relevance_scores(
    "Alphabet Google total revenue year over year growth", k=5
)
for doc, score in results:
    m = doc.metadata
    print(f"  score={score:.3f} | {m.get('company')} | {m.get('filing_type')} | {m.get('filing_date')} | {doc.page_content[:100]}...")