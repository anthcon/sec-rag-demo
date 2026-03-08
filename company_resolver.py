# company_resolver.py
import re
from rapidfuzz import fuzz, process

# Build from your existing TICKER_MAP — one source of truth
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

def _build_alias_index(ticker_map: dict) -> dict[str, str]:
    """
    Creates a lowercase alias -> canonical company name mapping.
    Handles tickers, full names, and common shorthand.
    """
    aliases = {}
    # Manual additions for names people actually type
    EXTRA_ALIASES = {
        "google": "Alphabet/Google",
        "alphabet": "Alphabet/Google",
        "amazon": "Amazon",
        "aws": "Amazon",
        "facebook": "Meta",
        "fb": "Meta",
        "jpmorgan": "JPMorgan Chase",
        "jp morgan": "JPMorgan Chase",
        "j&j": "Johnson & Johnson",
        "jnj": "Johnson & Johnson",
        "berkshire": "Berkshire Hathaway",
        "unitedhealth": "UnitedHealth",
        "exxon": "ExxonMobil",
        "raytheon": "RTX/Raytheon",
        "thermo fisher": "Thermo Fisher",
        "coca cola": "Coca-Cola",
        "coke": "Coca-Cola",
        "pepsi": "PepsiCo",
        "mcdonalds": "McDonald's",
        "nike": "Nike",
        "netflix": "Netflix",
    }

    for ticker, name in ticker_map.items():
        aliases[ticker.lower()] = name        # "amzn" -> "Amazon"
        aliases[name.lower()] = name           # "amazon" -> "Amazon"
        # Split compound names: "Alphabet/Google" -> ["alphabet", "google"]
        for part in re.split(r"[/\s]+", name):
            if len(part) > 2:
                aliases[part.lower()] = name

    aliases.update({k.lower(): v for k, v in EXTRA_ALIASES.items()})
    return aliases

ALIAS_INDEX = _build_alias_index(TICKER_MAP)
# All known aliases for fuzzy matching
ALL_ALIAS_KEYS = list(ALIAS_INDEX.keys())


def resolve_companies(query: str, score_cutoff: int = 75) -> list[str]:
    """
    Extract and resolve company names from a natural language query.
    Returns list of canonical company names found in the query.
    Pass 1: exact word-boundary match for aliases >= 3 chars
    Pass 2: fuzzy match on remaining tokens >= 4 chars
    """
    query_lower = query.lower()
    found = set()
    consumed_spans = []
    # --- Pass 1: word-boundary substring match (longer aliases first) ---
    for alias in sorted(ALL_ALIAS_KEYS, key=len, reverse=True):
        if len(alias) < 3:
            continue  # skip 1-2 char tickers like "v", "t", "ma"
        # Use word boundaries to avoid matching inside other words
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, query_lower):
            found.add(ALIAS_INDEX[alias])
            query_lower = re.sub(pattern, ' ', query_lower)
    # --- Pass 2: fuzzy match remaining tokens ---
    remaining_tokens = [t for t in query_lower.split() if len(t) >= 4]
    # Only fuzzy-match against aliases that are >= 4 chars
    long_aliases = [a for a in ALL_ALIAS_KEYS if len(a) >= 4]
    for token in remaining_tokens:
        match = process.extractOne(
            token,
            long_aliases,
            scorer=fuzz.ratio,
            score_cutoff=score_cutoff,
        )
        if match:
            found.add(ALIAS_INDEX[match[0]])
    return list(found)