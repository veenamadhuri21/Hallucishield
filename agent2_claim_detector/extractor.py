"""
extractor.py
------------
Agent-2: core LangChain logic for claim extraction.

Pipeline (LCEL - LangChain Expression Language):

    ChatPromptTemplate  ->  LLM  ->  StrOutputParser  ->  parse_claims()

An optional second chain re-checks that every extracted claim is grounded
in the original answer (a self-verification step against hallucinated claims).
"""

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.llm_factory import build_llm

from agent2_claim_detector.prompts import (
    EXTRACTION_HUMAN_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    VERIFY_HUMAN_PROMPT,
    VERIFY_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# 1. Chains
# ---------------------------------------------------------------------------
def build_extraction_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACTION_SYSTEM_PROMPT),
            ("human", EXTRACTION_HUMAN_PROMPT),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_verification_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", VERIFY_SYSTEM_PROMPT),
            ("human", VERIFY_HUMAN_PROMPT),
        ]
    )
    return prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# 2. Parsing the plain-text output into Python lists
# ---------------------------------------------------------------------------
_NUMBERED = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")


def parse_claims(raw_text: str) -> dict:
    """Turn the model's CLAIMS / OPINIONS text into {'claims': [...], 'opinions': [...]}."""
    result = {"claims": [], "opinions": []}
    section = None

    for line in raw_text.splitlines():
        stripped = line.strip().upper()
        if stripped.startswith("CLAIMS"):
            section = "claims"
            continue
        if stripped.startswith("OPINIONS"):
            section = "opinions"
            continue

        match = _NUMBERED.match(line)
        if match and section:
            text = match.group(1)
            if text.lower() != "none":
                result[section].append(text)

    return result


# ---------------------------------------------------------------------------
# 3. Public helper functions
# ---------------------------------------------------------------------------
def extract_claims(answer: str, question: str | None = None, llm=None) -> dict:
    """Main entry point: LLM answer (Agent-1's output) -> atomic claims."""
    llm = llm or build_llm(temperature=0)
    chain = build_extraction_chain(llm)
    raw = chain.invoke(
        {
            "question": question or "(not provided)",
            "answer": answer,
        }
    )
    return parse_claims(raw)


def verify_claims(answer: str, claims: list[str], llm=None) -> str:
    """Optional: ask the LLM to confirm every claim is grounded in the answer."""
    llm = llm or build_llm(temperature=0)
    chain = build_verification_chain(llm)
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    return chain.invoke({"answer": answer, "claims": numbered})
