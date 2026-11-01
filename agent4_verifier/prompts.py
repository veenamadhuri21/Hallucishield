"""
prompts.py
----------
All prompt text for Agent-4 lives here, separate from the code logic.
NOTE: no curly braces are used inside the prompt text, because these are
filled with plain str.format()/f-strings, not LangChain templates.
"""

MAKE_QUERIES_PROMPT = """\
Write 2 short search queries to fact-check this claim. \
Query 1: key entities/keywords. Query 2: the opposite/negated \
version of the claim. Return a JSON list of strings only.
Claim: {claim}"""

RETRY_QUERIES_PROMPT = """\
Previous searches found no decisive evidence.
Claim: {claim}
Previous queries: {old_queries}
Write 2 NEW different search queries (different wording/entities). \
Return a JSON list of strings only."""

LLM_JUDGE_PROMPT = """\
You are a strict fact-checker. Use ONLY the evidence below, not your own knowledge.

Claim: {claim}

Evidence:
{context}

Rules:
- SUPPORTED: evidence clearly states the claim is true.
- REFUTED: evidence clearly contradicts the claim (wrong date/number/entity etc).
- NOT_ENOUGH_INFO: evidence is unrelated, partial, or ambiguous. Do not guess.

Return ONLY JSON: {{"verdict": "SUPPORTED|REFUTED|NOT_ENOUGH_INFO", "confidence": 0.0-1.0, \
"evidence_ids": [ints], "reasoning": "one or two sentences quoting the key evidence"}}"""
