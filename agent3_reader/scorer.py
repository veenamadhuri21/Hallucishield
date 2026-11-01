"""
scorer.py
---------
Agent-3 ("the Reader") + verdict formation.

For each claim, reader.read_claim() gathers evidence via web_search/wikipedia_search/
scrape_url into a fresh per-claim Chroma store and returns the most relevant retrieved
passages. A plain (tool-free) LLM call then decides VERDICT/EVIDENCE/SOURCE from only
that retrieved evidence. This verdict step is a placeholder for Agent-4, which will
take over judging the claim from the Reader's stored evidence.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from common.llm_factory import build_llm

from agent3_reader.prompts import VERIFY_FROM_EVIDENCE_HUMAN_PROMPT, VERIFY_FROM_EVIDENCE_SYSTEM_PROMPT
from agent3_reader.reader import read_claim


# ---------------------------------------------------------------------------
# 1. Parsing the plain-text verdict into a dict
# ---------------------------------------------------------------------------
_VERDICT_RE = re.compile(r"VERDICT:\s*(SUPPORTED|CONTRADICTED|UNVERIFIABLE)", re.IGNORECASE)
_EVIDENCE_RE = re.compile(r"EVIDENCE:\s*(.+)")
_SOURCE_RE = re.compile(r"SOURCE:\s*(.+)")


def parse_verdict(raw_text: str) -> dict:
    verdict_match = _VERDICT_RE.search(raw_text)
    evidence_match = _EVIDENCE_RE.search(raw_text)
    source_match = _SOURCE_RE.search(raw_text)
    return {
        "verdict": verdict_match.group(1).upper() if verdict_match else "UNVERIFIABLE",
        "evidence": evidence_match.group(1).strip() if evidence_match else raw_text.strip(),
        "source": source_match.group(1).strip() if source_match else "None",
    }


# ---------------------------------------------------------------------------
# 2. Public helper functions
# ---------------------------------------------------------------------------
def verify_claim(claim: str, llm=None) -> dict:
    """Agent-3 (Reader) gathers evidence for the claim, then a tool-free LLM call
    decides the verdict from only that retrieved evidence. Returns verdict/evidence/source/claim."""
    llm = llm or build_llm(temperature=0)
    read = read_claim(claim, llm)
    human = VERIFY_FROM_EVIDENCE_HUMAN_PROMPT.format(claim=claim, evidence=read["evidence"])
    response = llm.invoke(
        [SystemMessage(content=VERIFY_FROM_EVIDENCE_SYSTEM_PROMPT), HumanMessage(content=human)]
    )
    result = parse_verdict(response.content)
    result["claim"] = claim
    return result


def verify_claims(claims: list[str], llm=None) -> list[dict]:
    """Main entry point: extracted claims (Agent-2's output) -> per-claim verdicts."""
    llm = llm or build_llm(temperature=0)
    return [verify_claim(claim, llm) for claim in claims]


def compute_trust_score(results: list[dict]) -> float:
    """Fraction of claims that are SUPPORTED. CONTRADICTED and UNVERIFIABLE both count against it."""
    if not results:
        return 1.0
    supported = sum(1 for r in results if r["verdict"] == "SUPPORTED")
    return round(supported / len(results), 3)

