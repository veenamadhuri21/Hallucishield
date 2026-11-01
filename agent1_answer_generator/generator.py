"""
generator.py
------------
Agent-1: user query -> LLM answer.

The produced answer is handed, unmodified, to Agent-2 (claim_detector) for
claim extraction. This module has no knowledge of Agent-2.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.llm_factory import build_llm

from agent1_answer_generator.prompts import ANSWER_HUMAN_PROMPT, ANSWER_SYSTEM_PROMPT


def build_answer_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            ("human", ANSWER_HUMAN_PROMPT),
        ]
    )
    return prompt | llm | StrOutputParser()


def generate_answer(question: str, llm=None, provider: str | None = None, model: str | None = None) -> str:
    """Main entry point: user question -> LLM answer (Agent-1's output)."""
    llm = llm or build_llm(provider, model, temperature=0.2)
    chain = build_answer_chain(llm)
    return chain.invoke({"question": question}).strip()
