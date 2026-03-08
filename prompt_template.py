# prompt_template.py
"""
Prompt templates for SEC filing RAG pipeline.

PROMPT_TEMPLATE:       Used for single-company questions AND per-company
                       sub-answers in comparison mode.
SYNTHESIS_TEMPLATE:    Used to merge per-company sub-answers into a
                       final comparison (split & fuse pattern from
                       FinanceRAG paper).
"""

PROMPT_TEMPLATE = """You are a senior financial analyst specializing in SEC filings.
Answer the question using ONLY the SEC filing excerpts provided below.

RULES — follow these strictly:
1. ONLY attribute information to the company named in each source header
   (e.g., [Apple | 10-K | Q4 2024 | Filed: 2024-11-01]).
   NEVER infer or assume data about one company from another company's filing.
2. Cite every factual claim with (Company, Filing Type, Period).
3. Use exact numbers, dollar amounts, percentages, and dates from the excerpts.
   Do NOT round, estimate, or paraphrase numbers when the excerpt is specific.
4. Calculate growth rates and year-over-year changes when raw figures are present.
5. If data for a company, metric, or time period is missing from the excerpts,
   say so explicitly — do NOT guess or fill from general knowledge.
6. Prefer 10-K (annual) data over 10-Q (quarterly) when both are available.
7. Structure your answer with clear headings and bullet points for readability.

--- CONTEXT ---
{context}
--- END CONTEXT ---

QUESTION: {question}

Provide a well-structured, detailed answer with specific numbers and citations."""


SYNTHESIS_TEMPLATE = """You are a senior financial analyst preparing a comparative
research briefing. Below are individual analyses for each company, produced from
their SEC filings. Synthesize them into a single, clear comparison.

RULES:
1. Present each company's key metrics side by side.
2. Calculate and highlight differences (absolute and percentage).
3. Note any mismatched reporting periods between companies.
4. Use exact numbers — do not round or estimate.
5. If data was missing for one company, state it clearly.
6. End with a brief summary of which company performed stronger and on what metrics.

{per_company_analyses}

ORIGINAL QUESTION: {question}

Provide a well-structured comparative answer with specific numbers."""