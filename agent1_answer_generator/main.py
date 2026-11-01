"""
main.py
-------
Standalone CLI for Agent-1, useful for testing the answer generator on its own.

Examples
    python -m agent1_answer_generator.main --question "Who was India's first PM?"
    python -m agent1_answer_generator.main --question "..." --provider ollama --model llama3.1
"""

import argparse

from agent1_answer_generator.generator import generate_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent-1: LLM answer generator")
    parser.add_argument("--question", required=True, help="User's question")
    parser.add_argument("--provider", help="ollama | groq | google")
    parser.add_argument("--model", help="Model name")
    args = parser.parse_args()

    answer = generate_answer(args.question, provider=args.provider, model=args.model)

    print("\nANSWER")
    print("------")
    print(answer)


if __name__ == "__main__":
    main()
