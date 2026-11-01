"""
verifier.py
-----------
Agent-4 ("the Verifier"): claim + Agent-3's Chroma evidence -> verdict.

Flow (per claim):
    query prep -> retrieve from Agent-3's EvidenceStore -> cross-encoder rerank
    + relevance gate -> NLI (+ optional LLM judge) -> fuse -> verdict
    -> if NOT_ENOUGH_INFO: rewrite query / fetch fresh evidence, retry (max_retries)

Verdicts:  SUPPORTED (hallucinated=False) | REFUTED (hallucinated=True)
           NOT_ENOUGH_INFO (hallucinated=None -> "unverifiable")

Ported from a standalone reference implementation, adapted to read from Agent-3's
agent3_reader.rag.EvidenceStore (langchain Chroma wrapper) instead of a raw
chromadb collection, and to drive its LLM judge off this project's common.llm_factory.

NOTE: reranker/NLI models are downloaded from HuggingFace Hub on first use
(cross-encoder/ms-marco-MiniLM-L-6-v2 ~90MB, MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
~700MB) - this can take a while on a slow connection, but only happens once.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from agent3_reader.reader import read_claim
from agent3_reader.rag import EvidenceStore
from agent3_reader.tools import build_tools
from agent4_verifier.prompts import LLM_JUDGE_PROMPT, MAKE_QUERIES_PROMPT, RETRY_QUERIES_PROMPT
from common.llm_factory import build_llm

SUPPORTED, REFUTED, NEI = "SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"


# ---------------------------------------------------------------------------
# 1. Data types
# ---------------------------------------------------------------------------
@dataclass
class Evidence:
    text: str
    source: str = ""          # wikipedia / a URL / "unknown"
    relevance: float = 0.0    # reranker score (0-1)
    entail: float = 0.0       # NLI probabilities vs. the claim
    contradict: float = 0.0
    neutral: float = 0.0


@dataclass
class Verdict:
    claim: str
    verdict: str
    hallucinated: Optional[bool]
    confidence: float
    reasoning: str
    evidence: list = field(default_factory=list)
    attempts: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 2. The agent
# ---------------------------------------------------------------------------
class VerifierAgent:
    """Judges one claim against a single claim's EvidenceStore (built by Agent-3)."""

    def __init__(
        self,
        store: EvidenceStore,
        llm: Optional[Callable[[str], str]] = None,     # fn(prompt) -> text (optional)
        refresh_fn: Optional[Callable[[str], None]] = None,
        # refresh_fn(query): fetch fresh evidence (web_search/wikipedia_search) into the store
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        nli_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        top_k: int = 10,
        keep: int = 4,
        min_relevance: float = 0.10,   # below this, evidence is treated as unrelated (tune!)
        nli_threshold: float = 0.70,
        max_retries: int = 1,
    ):
        self.store, self.llm, self.refresh_fn = store, llm, refresh_fn
        self.reranker_name, self.nli_name = reranker_name, nli_name
        self.top_k, self.keep = top_k, keep
        self.min_relevance, self.nli_threshold = min_relevance, nli_threshold
        self.max_retries = max_retries
        self._reranker = self._nli_tok = self._nli_model = None

    # ------------------------------------------------------------ public API
    def verify(self, claim: str) -> Verdict:
        queries = self._make_queries(claim)
        v = None
        for attempt in range(1, self.max_retries + 2):
            evidence = self._retrieve(queries)
            evidence = self._rerank(claim, evidence)
            v = self._decide(claim, evidence)
            v.attempts = attempt
            if v.verdict != NEI or attempt > self.max_retries:
                break
            # agentic retry: fetch fresh evidence and/or reformulate the query
            queries = self._retry_queries(claim, queries)
            if self.refresh_fn:
                try:
                    self.refresh_fn(queries[0])
                except Exception:
                    pass  # never let a fetch error kill verification
        return v

    def verify_all(self, claims: list[str]) -> list[Verdict]:
        return [self.verify(c) for c in claims]

    # --------------------------------------------------------- step 1: queries
    def _make_queries(self, claim: str) -> list[str]:
        queries = [claim]
        if self.llm:
            try:
                out = self.llm(MAKE_QUERIES_PROMPT.format(claim=claim))
                extra = json.loads(self._extract_json(out, "[", "]"))
                queries += [q for q in extra if isinstance(q, str)][:2]
            except Exception:
                pass
        return queries

    def _retry_queries(self, claim: str, old: list[str]) -> list[str]:
        if self.llm:
            try:
                out = self.llm(RETRY_QUERIES_PROMPT.format(claim=claim, old_queries=old))
                new = json.loads(self._extract_json(out, "[", "]"))
                return [claim] + [q for q in new if isinstance(q, str)][:2]
            except Exception:
                pass
        return old

    # ------------------------------------------------------ step 2: retrieval
    def _retrieve(self, queries: list[str]) -> list[Evidence]:
        seen, out = set(), []
        for q in queries:
            for text, source in self.store.search(q, k=self.top_k):
                key = text.strip()[:200]
                if key in seen:
                    continue
                seen.add(key)
                out.append(Evidence(text=text, source=source))
        return out

    # ------------------------------------------------ step 3: rerank + gate
    def _rerank(self, claim: str, evidence: list[Evidence]) -> list[Evidence]:
        if not evidence:
            return []
        scores = self._rerank_scores(claim, [e.text for e in evidence])
        for e, s in zip(evidence, scores):
            e.relevance = float(s)
        evidence.sort(key=lambda e: e.relevance, reverse=True)
        return [e for e in evidence[: self.keep] if e.relevance >= self.min_relevance]

    def _rerank_scores(self, claim: str, texts: list[str]) -> list[float]:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_name)
        raw = self._reranker.predict([(claim, t) for t in texts])
        return [1 / (1 + math.exp(-x)) for x in raw]  # sigmoid -> 0..1

    # -------------------------------------------------- step 4/5: judge + fuse
    def _decide(self, claim: str, evidence: list[Evidence]) -> Verdict:
        if not evidence:
            return Verdict(claim, NEI, None, 0.0, "No relevant evidence found in the knowledge base.", [])

        for e, (ent, neu, con) in zip(evidence, self._nli_probs(claim, [e.text for e in evidence])):
            e.entail, e.neutral, e.contradict = ent, neu, con
        max_ent = max(e.entail for e in evidence)
        max_con = max(e.contradict for e in evidence)

        if max_con >= self.nli_threshold and max_con > max_ent:
            nli_v, nli_c = REFUTED, max_con
        elif max_ent >= self.nli_threshold and max_ent > max_con:
            nli_v, nli_c = SUPPORTED, max_ent
        else:
            nli_v, nli_c = NEI, 1 - max(max_ent, max_con)

        if not self.llm:
            return Verdict(
                claim,
                nli_v,
                self._halluc(nli_v),
                round(nli_c, 3),
                f"NLI only: max entailment={max_ent:.2f}, max contradiction={max_con:.2f}",
                evidence,
            )

        llm_v, llm_c, why = self._llm_judge(claim, evidence)
        if llm_v == nli_v:
            conf = (llm_c + nli_c) / 2
            final = llm_v
        else:  # disagreement -> trust the LLM (reads full context) but cut confidence
            final, conf = llm_v, min(llm_c, nli_c) * 0.6
            why += f" [NLI disagreed: {nli_v}]"
        return Verdict(claim, final, self._halluc(final), round(conf, 3), why, evidence)

    def _nli_probs(self, claim: str, texts: list[str]):
        import torch

        if self._nli_model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._nli_tok = AutoTokenizer.from_pretrained(self.nli_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_name).eval()
        id2label = {i: label.lower() for i, label in self._nli_model.config.id2label.items()}
        out = []
        for t in texts:
            enc = self._nli_tok(t, claim, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                p = torch.softmax(self._nli_model(**enc).logits[0], dim=-1).tolist()
            d = {id2label[i]: p[i] for i in range(len(p))}
            out.append((d.get("entailment", 0.0), d.get("neutral", 0.0), d.get("contradiction", 0.0)))
        return out

    def _llm_judge(self, claim: str, evidence: list[Evidence]):
        ctx = "\n".join(f"[{i}] ({e.source}) {e.text}" for i, e in enumerate(evidence))
        prompt = LLM_JUDGE_PROMPT.format(claim=claim, context=ctx)
        try:
            data = json.loads(self._extract_json(self.llm(prompt), "{", "}"))
            v = str(data.get("verdict", NEI)).upper().strip()
            v = v if v in (SUPPORTED, REFUTED, NEI) else NEI
            return v, float(data.get("confidence", 0.5)), str(data.get("reasoning", ""))
        except Exception as e:
            return NEI, 0.0, f"LLM judge output could not be parsed ({e})."

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _halluc(verdict: str) -> Optional[bool]:
        return {SUPPORTED: False, REFUTED: True}.get(verdict)

    @staticmethod
    def _extract_json(text: str, open_c: str, close_c: str) -> str:
        text = re.sub(r"```(?:json)?", "", text)
        s, e = text.find(open_c), text.rfind(close_c)
        if s == -1 or e == -1:
            raise ValueError("no JSON found")
        return text[s : e + 1]


# ---------------------------------------------------------------------------
# 3. Bridges to the rest of the project (LLM + refresh)
# ---------------------------------------------------------------------------
def _llm_to_fn(llm) -> Callable[[str], str]:
    """Adapts a LangChain chat model to the plain str->str callable VerifierAgent expects."""
    return lambda prompt: llm.invoke(prompt).content


def make_refresh_fn(store: EvidenceStore) -> Callable[[str], None]:
    """Builds a refresh_fn that fetches fresh web/Wikipedia evidence for a query and adds it
    into this claim's store - web_search/wikipedia_search already call store.add() internally."""
    web_search, wikipedia_search, _scrape_url, _retrieve_evidence = build_tools(store)

    def refresh(query: str) -> None:
        web_search.func(query)
        wikipedia_search.func(query)

    return refresh


# ---------------------------------------------------------------------------
# 4. Public helper functions (mirrors agent3_reader.scorer's shape)
# ---------------------------------------------------------------------------
def verify_claim(claim: str, llm=None) -> dict:
    """Agent-3 gathers evidence for the claim (fresh Chroma store), Agent-4 judges it: reranker
    + NLI + LLM judge fused into a verdict, retrying with fresh evidence if inconclusive."""
    llm = llm or build_llm(temperature=0)
    read = read_claim(claim, llm)  # fresh per-claim EvidenceStore, discarded once this claim is done
    store = read["store"]
    agent = VerifierAgent(store, llm=_llm_to_fn(llm), refresh_fn=make_refresh_fn(store))
    return agent.verify(claim).to_dict()


def verify_claims(claims: list[str], llm=None) -> list[dict]:
    """Main entry point: extracted claims (Agent-2's output) -> per-claim verdicts.
    Each claim gets its own fresh Chroma store - searched by Agent-3, judged by Agent-4,
    then discarded before the next claim starts."""
    llm = llm or build_llm(temperature=0)
    results = []
    total = len(claims)
    for i, claim in enumerate(claims, 1):
        print(f"\n[{i}/{total}] Agent-3: searching evidence for: {claim}")
        result = verify_claim(claim, llm)
        print(f"[{i}/{total}] Agent-4: verdict = {result['verdict']} (confidence={result['confidence']})")
        results.append(result)
    return results


def compute_trust_score(results: list[dict]) -> float:
    """Fraction of claims that are SUPPORTED. REFUTED and NOT_ENOUGH_INFO both count against it."""
    if not results:
        return 1.0
    supported = sum(1 for r in results if r["verdict"] == SUPPORTED)
    return round(supported / len(results), 3)
