"""
reader.py
---------
Agent-3 ("the Reader"): claim -> gathered evidence, stored in a Chroma RAG store.

Given one claim as the query, a tool-calling agent decides which of web_search
(Tavily), wikipedia_search (free, keyless MediaWiki API), and scrape_url to call.
Every raw tool result is auto-indexed into a per-claim Chroma vector store
(rag.py). The reader then calls retrieve_evidence to pull back the passages
most relevant to the claim and returns them as plain text.

The Reader never judges the claim - deciding SUPPORTED/CONTRADICTED/UNVERIFIABLE
from this retrieved evidence is a separate step (see scorer.py), which will move
into its own Agent-4 module later.
"""

from langchain.agents import create_agent

from common.llm_factory import build_llm

from agent3_reader.prompts import READ_CLAIM_HUMAN_PROMPT, READ_CLAIM_SYSTEM_PROMPT
from agent3_reader.rag import EvidenceStore
from agent3_reader.tools import build_tools


def _build_reader_agent(llm, store: EvidenceStore):
    tools = build_tools(store)
    return create_agent(llm, tools, system_prompt=READ_CLAIM_SYSTEM_PROMPT)


_MAX_ATTEMPTS = 2


def read_claim(claim: str, llm=None) -> dict:
    """Gathers evidence for one claim via web_search/wikipedia_search/scrape_url,
    auto-indexed into a fresh per-claim Chroma store, then returns the most
    relevant retrieved passages. Produces no verdict."""
    llm = llm or build_llm(temperature=0)
    store = EvidenceStore()
    agent = _build_reader_agent(llm, store)
    human = READ_CLAIM_HUMAN_PROMPT.format(claim=claim)

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            agent.invoke({"messages": [{"role": "user", "content": human}]})
            break
        except Exception as e:
            # Some reasoning models (e.g. Groq's gpt-oss-20b) occasionally leak a harmony
            # "<|channel|>..." token into a tool call name, which the provider then rejects.
            # Retry once (it's a sampling-level glitch, not deterministic); if it keeps
            # failing, fall back to whatever evidence (if any) was gathered before the
            # failure instead of crashing the whole pipeline/app.
            print(f"[Agent-3] tool-calling agent failed (attempt {attempt}/{_MAX_ATTEMPTS}): {e}")

    return {"claim": claim, "evidence": store.retrieve(claim), "store": store}


def read_claims(claims: list[str], llm=None) -> list[dict]:
    """Agent-2's extracted claims -> per-claim gathered evidence (no verdicts).
    Each claim gets its own fresh Chroma store, discarded before the next claim starts."""
    llm = llm or build_llm(temperature=0)
    total = len(claims)
    results = []
    for i, claim in enumerate(claims, 1):
        print(f"[{i}/{total}] Agent-3: searching evidence for: {claim}")
        results.append(read_claim(claim, llm))
    return results
