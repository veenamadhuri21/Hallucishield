"""
rag.py
------
Per-claim evidence store for Agent-3 ("the Reader"): raw tool output from
web_search / wikipedia_search / scrape_url is chunked, embedded, and added to
a fresh in-memory (never persisted) Chroma collection, so the agent can
retrieve only the most relevant evidence before deciding its verdict instead
of reasoning over potentially huge/noisy raw tool output.

A new EvidenceStore is created for every claim verification, so evidence
gathered for one claim never leaks into another claim's verdict.
"""

import uuid

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


class EvidenceStore:
    """A fresh, in-memory vector store scoped to a single claim verification."""

    def __init__(self):
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # No persist_directory -> ephemeral in-memory Chroma collection, discarded after use.
        self._store = Chroma(
            collection_name=f"evidence-{uuid.uuid4().hex}",
            embedding_function=_get_embeddings(),
        )
        self._has_documents = False

    def add(self, text: str, source: str) -> None:
        """Chunk and embed a piece of raw tool output, tagged with where it came from."""
        if not text or not text.strip():
            return
        docs = self._splitter.create_documents([text], metadatas=[{"source": source}])
        if docs:
            self._store.add_documents(docs)
            self._has_documents = True

    def retrieve(self, query: str, k: int = 4) -> str:
        """Semantic search over everything added so far, most relevant chunks first."""
        if not self._has_documents:
            return "No evidence gathered yet. Call web_search/wikipedia_search/scrape_url first."
        results = self._store.similarity_search(query, k=k)
        if not results:
            return "No relevant evidence found for that query."
        return "\n----\n".join(
            f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in results
        )

    def search(self, query: str, k: int = 10) -> list[tuple[str, str]]:
        """Like retrieve(), but returns structured (text, source) pairs instead of a
        formatted string - used by Agent-4 to rerank/score each passage individually."""
        if not self._has_documents:
            return []
        results = self._store.similarity_search(query, k=k)
        return [(d.page_content, d.metadata.get("source", "unknown")) for d in results]

    @property
    def has_documents(self) -> bool:
        return self._has_documents
