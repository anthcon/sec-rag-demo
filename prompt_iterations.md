# Prompt Iterations

## V1 — Initial
```bash
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
```
- Basic "answer using only context" instruction
- Simple source citation by filename
- Result: hallucination attributing NVIDIA data to Apple
    ```bash
    **Apple**: Similar to JPMorgan, Apple faces reputational risks related to its handling of personal information and compliance with data privacy laws, which could deter customers and lead to increased regulatory scrutiny (NVIDIA, 10K, 2024Q1, Filed: 2024-02-21).
    ```

## V2 — [what you changed]
- Why: hallucination attributing NVIDIA data to Apple
    ```bash
    **Apple**: Similar to JPMorgan, Apple faces reputational risks related to its handling of personal information and compliance with data privacy laws, which could deter customers and lead to increased regulatory scrutiny (NVIDIA, 10K, 2024Q1, Filed: 2024-02-21).
    ```
- Result: [what improved]

## V3 — ...