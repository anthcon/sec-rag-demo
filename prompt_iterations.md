# Prompt Iterations

## V1 — Baseline

PROMPT_TEMPLATE = """You are a senior financial analyst. Answer the user's question
using ONLY the SEC filing excerpts provided below.
Rules:
- Cite every claim with the source filing in parentheses (company, filing type, period).
- Compare across companies when the question asks for it.
- If the context is insufficient to fully answer, state what is missing.
- Be specific — use numbers, dates, and direct references from the filings.
--- CONTEXT ---
{context}
--- END CONTEXT ---
QUESTION: {question}
Provide a well-structured, detailed answer."""

**What it did:** Basic "answer using only context" instruction with simple source citation by filename.

**Failure observed:** Cross-company hallucination — the model attributed NVIDIA disclosure data to Apple:

> **Apple**: Similar to JPMorgan, Apple faces reputational risks related to its handling of personal information and compliance with data privacy laws, which could deter customers and lead to increased regulatory scrutiny **(NVIDIA, 10K, 2024Q1, Filed: 2024-02-21)**.

---

## V2 — Anti-Hallucination Directive

**Change:** Added explicit "do not borrow from another company's disclosures" directive.

**Motivation:** Directly targeted the V1 cross-company hallucination where NVIDIA data bled into the Apple section.

**Result:** Eliminated the NVIDIA→Apple bleed. Model now says "insufficient data" instead of transplanting facts between companies.

**Weakness found:** Model still occasionally paraphrased or rounded dollar amounts. The soft "Be specific" instruction was not strong enough to enforce verbatim numbers. Also lacked guidance on what to do when data is genuinely missing — phrasing was vague.

---

## V3 — Enforced Source Fidelity + Verbatim Numbers + Explicit "No Data" Protocol

STRICT RULES:
1. SOURCE FIDELITY: Only attribute information to the company named in that excerpt's
   header. Never infer, guess, or transplant data from one company's filing to another.
2. CITE EVERY CLAIM: After each factual statement, include a parenthetical citation.
3. USE EXACT NUMBERS: Quote figures, dollar amounts, percentages, and dates verbatim.
   Do not round or approximate.
4. MISSING DATA: If context is insufficient, state:
   "The provided filings do not contain data on [topic] for [company/period]."
   Do NOT fabricate or fill gaps with general knowledge.
5. COMPARISONS: Organize per-company with clear data gap notes.
6. RECENCY: Prefer most recently filed data unless the question specifies a period.
7. STRUCTURE: Lead with a direct answer, then supporting detail.

**Changes from V2:**

- Numbered rules with labels — easier for the model to reference internally and for us to debug which rule was violated.
- Rule 3 (USE EXACT NUMBERS): Replaced vague "Be specific" with a hard constraint to quote verbatim. Targets number-rounding hallucination.
- Rule 4 (MISSING DATA): Prescriptive phrasing template. Removes ambiguity about when and how to declare gaps.
- Rule 6 (RECENCY): Aligns prompt behavior with the recency-weighted reranker in `query.py`.
- Rule 7 (STRUCTURE): "Lead with direct answer" — optimized for live demo where the interviewer wants the answer fast.
- Tightened system identity line to "answering questions about SEC filings" instead of generic "senior financial analyst."
- Added standalone sentence: "Do NOT use any outside knowledge."

**Motivation:** Research (FinanceRAG paper, RAG failure-point literature) confirms that explicit, enumerated constraints reduce hallucination more reliably than prose instructions. Verbatim-quoting directives are especially effective for financial data where precision matters.

**Result:** Pending evaluation during live demo.

---

## V4 — Table Parsing Directive

**Change:** Added to the rules:

Extract data even from imperfectly formatted tables. SEC excerpts may contain
pipe-delimited tables, irregular spacing, or split rows — parse them carefully.
Only state data is unavailable if you genuinely cannot find ANY relevant figures
in the excerpts after thorough review.

**Motivation:** The model was returning "data not available" for questions whose answers existed in the corpus — likely because SEC filing tables arrive as pipe-delimited or irregularly spaced text after chunking. The model appeared to skip these rather than parse them.

**Result:** Pending evaluation.