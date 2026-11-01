"""
pipeline.py
-----------
Hallucishield end-to-end flow (Agent-1 -> Agent-2 -> Agent-3 -> Agent-4):

    User query -> Agent-1 generates an answer -> Agent-2 extracts claims from that answer
    -> Agent-3 gathers web/Wikipedia/RAG evidence per claim -> Agent-4 judges each claim
    against that evidence (reranker + NLI + LLM judge) and produces a trust score

Examples
    python pipeline.py --question "Who was India's first PM?"
    python pipeline.py --question "Who was India's first PM?" --verify
    python pipeline.py --question "Who was India's first PM?" --read   (Agent-1 -> 2 -> 3 only)
    python pipeline.py --question "Who was India's first PM?" --score  (Agent-1 -> 2 -> 3 -> 4)
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # this network breaks the Xet CDN mid-download - fall back to plain HTTPS

import argparse

from agent1_answer_generator.generator import generate_answer
from agent2_claim_detector.extractor import extract_claims, verify_claims
from agent3_reader.reader import read_claims
from agent4_verifier.verifier import compute_trust_score
from agent4_verifier.verifier import verify_claims as score_claims


def print_numbered(title: str, items: list[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not items:
        print("None")
        return
    for i, item in enumerate(items, 1):
        print(f"{i}. {item}")


def run_pipeline(
    question: str,
    provider: str | None = None,
    model: str | None = None,
    verify: bool = False,
    read: bool = False,
    score: bool = False,
) -> dict:
    """Runs Agent-1 -> Agent-2 (-> Agent-3 if read=True) (-> Agent-4 if score=True). Returns a dict with answer, claims, opinions."""
    answer = generate_answer(question, provider=provider, model=model)
    result = extract_claims(answer, question=question)
    result["answer"] = answer

    if verify and result["claims"]:
        result["verification"] = verify_claims(answer, result["claims"])

    if read and result["claims"]:
        result["reading"] = read_claims(result["claims"])

    if score and result["claims"]:
        result["scoring"] = score_claims(result["claims"])
        result["trust_score"] = compute_trust_score(result["scoring"])

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Hallucishield: Agent-1 (answer) -> Agent-2 (claims) -> Agent-3 (trust score)")
    parser.add_argument("--question", required=True, help="User's question")
    parser.add_argument("--provider", help="ollama | groq | google (all free)")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--verify", action="store_true", help="Run the grounding check on each extracted claim")
    parser.add_argument("--read", action="store_true", help="Run Agent-3 only: gather web/Wikipedia/RAG evidence per claim (no Agent-4 verdict, no heavy ML downloads)")
    parser.add_argument("--score", action="store_true", help="Run Agent-3 + Agent-4: gather evidence, verify each claim, and compute a trust score")
    args = parser.parse_args()

    result = run_pipeline(args.question, provider=args.provider, model=args.model, verify=args.verify, read=args.read, score=args.score)

    print("\nAGENT-1: ANSWER")
    print("---------------")
    print(result["answer"])

    print_numbered("AGENT-2: EXTRACTED CLAIMS", result["claims"])
    print_numbered("AGENT-2: OPINIONS (not treated as factual claims)", result["opinions"])

    if "verification" in result:
        print("\nAGENT-2: GROUNDING CHECK")
        print("------------------------")
        print(result["verification"])

    if "reading" in result:
        print("\nAGENT-3: GATHERED EVIDENCE")
        print("--------------------------")
        for i, r in enumerate(result["reading"], 1):
            print(f"{i}. CLAIM: {r['claim']}")
            print(f"   {r['evidence']}\n")

    if "scoring" in result:
        print("\nAGENT-4: CLAIM VERIFICATION")
        print("---------------------------")
        for i, r in enumerate(result["scoring"], 1):
            print(f"{i}. [{r['verdict']}] {r['claim']} (confidence={r['confidence']})")
            print(f"   Reasoning: {r['reasoning']}")
        print(f"\nTRUST SCORE: {result['trust_score']}")


if __name__ == "__main__":
    main()

