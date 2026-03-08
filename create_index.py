from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import re
import shutil

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "edgar_corpus"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
BATCH_SIZE = 5000

TICKER_MAP = {
    "AAPL": "Apple", "ABBV": "AbbVie", "ADBE": "Adobe", "AMD": "AMD",
    "AMZN": "Amazon", "AXP": "American Express", "BAC": "Bank of America",
    "BA": "Boeing", "BLK": "BlackRock", "BRK": "Berkshire Hathaway",
    "CAT": "Caterpillar", "CMCSA": "Comcast", "COST": "Costco",
    "CRM": "Salesforce", "CSCO": "Cisco", "CVX": "Chevron",
    "DE": "John Deere", "DIS": "Disney", "GE": "General Electric",
    "GOOG": "Alphabet/Google", "GS": "Goldman Sachs", "HD": "Home Depot",
    "IBM": "IBM", "INTC": "Intel", "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase", "KO": "Coca-Cola", "LLY": "Eli Lilly",
    "LMT": "Lockheed Martin", "MA": "Mastercard", "MCD": "McDonald's",
    "META": "Meta", "MRK": "Merck", "MSFT": "Microsoft", "MS": "Morgan Stanley",
    "NFLX": "Netflix", "NKE": "Nike", "NVDA": "NVIDIA", "ORCL": "Oracle",
    "PEP": "PepsiCo", "PFE": "Pfizer", "PG": "Procter & Gamble",
    "RTX": "RTX/Raytheon", "SBUX": "Starbucks", "TGT": "Target",
    "TMO": "Thermo Fisher", "TSLA": "Tesla", "T": "AT&T",
    "UNH": "UnitedHealth", "UPS": "UPS", "VZ": "Verizon",
    "V": "Visa", "WMT": "Walmart", "XOM": "ExxonMobil",
}


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def parse_filename_metadata(filename: str) -> dict:
    """Extract ticker, filing type, quarter, and date from filename."""
    base = os.path.basename(filename).replace("_full.txt", "")
    parts = base.split("_")

    ticker = parts[0] if parts else "UNKNOWN"
    filing_type = parts[1] if len(parts) > 1 else "UNKNOWN"

    date = "UNKNOWN"
    quarter = "UNKNOWN"
    for part in parts[2:]:
        if re.match(r"\d{4}-\d{2}-\d{2}", part):
            date = part
        elif re.match(r"\d{4}Q\d", part):
            quarter = part

    return {
        "ticker": ticker,
        "company": TICKER_MAP.get(ticker, ticker),
        "filing_type": filing_type,
        "quarter": quarter,
        "filing_date": date,
        "source": os.path.basename(filename),
    }


def load_documents():
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    for doc in documents:
        meta = parse_filename_metadata(doc.metadata.get("source", ""))
        doc.metadata.update(meta)

    print(f"Loaded {len(documents)} documents.")
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    db = Chroma.from_documents(
        chunks[:BATCH_SIZE],
        embeddings,
        persist_directory=CHROMA_PATH,
    )
    print(f"Added batch 1 — {min(BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        db.add_documents(batch)
        print(f"Added batch {i // BATCH_SIZE + 1} — {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()