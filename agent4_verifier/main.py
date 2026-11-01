"""
main.py
-------
Standalone CLI for Agent-4, useful for testing claim verification against
Agent-3's gathered evidence on its own, e.g. against a list of claims typed
in or loaded from a file (one claim per line).

Examples
    python -m agent4_verifier.main --claim "Draupadi Murmu assumed office on 25 July 2022."
    python -m agent4_verifier.main --file claims.txt
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # this network breaks the Xet CDN mid-download - fall back to plain HTTPS

import argparse
import sys

from agent4_verifier.verifier import compute_trust_score, verify_claims


def read_claims(path: str | None, single_claim: str | None) -> list[str]:
    if single_claim:
        return [single_claim]
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    print("Paste one claim per line, then press Ctrl+D (Ctrl+Z then Enter on Windows):\n")
    return [line.strip() for line in sys.stdin if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent-4: Chroma/NLI/reranker-grounded claim verification")
    parser.add_argument("--claim", help="A single claim to verify")
    parser.add_argument("--file", help="Text file containing claims, one per line")
    parser.add_argument("--provider", help="ollama | groq | google (all free)")
    parser.add_argument("--model", help="Model name")
    args = parser.parse_args()

    claims = read_claims(args.file, args.claim)
    if not claims:
        sys.exit("No claims provided.")

    from common.llm_factory import build_llm

    llm = build_llm(args.provider, args.model, temperature=0)

    results = verify_claims(claims, llm)

    print("\nAGENT-4: CLAIM VERIFICATION")
    print("---------------------------")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['verdict']}] {r['claim']} (confidence={r['confidence']})")
        print(f"   Reasoning: {r['reasoning']}")

    print(f"\nTRUST SCORE: {compute_trust_score(results)} "
          f"({sum(r['verdict'] == 'SUPPORTED' for r in results)}/{len(results)} claims supported)")


if __name__ == "__main__":
    main()
