"""
prompts.py
----------
Prompts for Agent-1 (answer generator). Kept separate from logic, same
convention as Agent-2, so they are easy to iterate on independently.
"""

ANSWER_SYSTEM_PROMPT = """\
You are a helpful, knowledgeable assistant.
Answer the user's question as accurately and completely as you can, using your own knowledge.
Give a direct, well-structured answer. Prefer concrete facts (names, dates, numbers, places)
over vague statements when they are relevant to the question.
Do not add disclaimers about being an AI unless the question specifically asks about it.
"""

ANSWER_HUMAN_PROMPT = """\
Question:
{question}
"""
