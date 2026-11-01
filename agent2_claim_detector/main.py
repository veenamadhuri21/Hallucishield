"""
main.py
-------
Standalone CLI for Agent-2, useful for testing claim extraction on its own
(independent of Agent-1), e.g. against a saved answer file.

Examples
    python -m agent2_claim_detector.main --file sample_answers.txt
    python -m agent2_claim_detector.main --file sample_answers.txt --question "Who was India's first PM?" --verify
    python -m agent2_claim_detector.main                       # paste the answer, then Ctrl+D (Ctrl+Z + Enter on Windows)
"""

import argparse
import sys

from agent2_claim_detector.extractor import build_llm, extract_claims, verify_claims


def read_answer(path: str | None) -> str:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    print("Paste the LLM answer, then press Ctrl+D (Ctrl+Z then Enter on Windows):\n")
    return sys.stdin.read().strip()


def print_numbered(title: str, items: list[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not items:
        print("None")
        return
    for i, item in enumerate(items, 1):
        print(f"{i}. {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact claim extraction for hallucination detection")
    parser.add_argument("--file", help="Text file containing the LLM answer")
    parser.add_argument("--question", help="Original question (context only, never used as a source)")
    parser.add_argument("--provider", help="ollama | groq | google (all free)")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--verify", action="store_true", help="Run the grounding check on each claim")
    args = parser.parse_args()

    answer = read_answer(args.file)
    if not answer:
        sys.exit("No answer text provided.")

    llm = build_llm(args.provider, args.model)

    result = extract_claims(answer, question=args.question, llm=llm)

    print_numbered("EXTRACTED CLAIMS", result["claims"])
    print_numbered("OPINIONS (not treated as factual claims)", result["opinions"])

    if args.verify and result["claims"]:
        print("\nGROUNDING CHECK")
        print("---------------")
        print(verify_claims(answer, result["claims"], llm=llm))


if __name__ == "__main__":
    main()
