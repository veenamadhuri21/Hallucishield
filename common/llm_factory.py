"""
llm_factory.py
--------------
Shared LLM factory used by every agent in Hallucishield.
Only FREE options are supported: ollama (local), groq (free tier), google (free tier).
"""

import os

from dotenv import load_dotenv

load_dotenv()


def build_llm(provider: str | None = None, model: str | None = None, temperature: float = 0.0):
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
    model = model or os.getenv("LLM_MODEL")

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model or "llama3.1", temperature=temperature)

    if provider == "groq":
        from langchain_groq import ChatGroq

        model = model or "openai/gpt-oss-20b"
        kwargs = {"model": model, "temperature": temperature, "max_tokens": 4096}
        if "gpt-oss" in model.lower():
            # gpt-oss is a reasoning model: its "thinking" tokens share the max_tokens
            # budget, so without a low reasoning effort it can burn the whole budget
            # on reasoning and return an empty answer for harder/open-ended questions.
            kwargs["reasoning_effort"] = "low"
        return ChatGroq(**kwargs)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model or "gemini-2.5-flash", temperature=temperature)

    raise ValueError(f"Unknown provider '{provider}'. Use ollama, groq or google.")
