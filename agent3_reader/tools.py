"""
tools.py
--------
Web tools for Agent-3 ("the Reader"): web search (Tavily), Wikipedia search,
and page scraping. build_tools() binds a fresh set of these to a single
EvidenceStore, so every raw tool result is auto-indexed into that claim's
Chroma collection, plus a retrieve_evidence tool to pull back the most
relevant chunks before the agent commits to a verdict.
"""

import ipaddress
import os
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

from agent3_reader.rag import EvidenceStore

load_dotenv()

_tavily_client: TavilyClient | None = None

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


def _get_tavily_client() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Get a free key at tavily.com and add it to .env."
            )
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def _is_safe_url(url: str) -> bool:
    """Blocks non-http(s) schemes and internal/private hosts to avoid SSRF via LLM-chosen URLs."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        host = parsed.hostname.lower()
        if host in ("localhost",):
            return False
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(host))
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        except (socket.gaierror, ValueError):
            pass  # unresolved/non-IP hostname - allow, request itself will fail if bogus
        return True
    except ValueError:
        return False


def _wikipedia_lookup(query: str) -> str:
    """Free, keyless Wikipedia search: find the best matching page, return its intro extract."""
    try:
        search_resp = requests.get(
            WIKIPEDIA_API_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
            timeout=8,
            headers={"User-Agent": "Hallucishield/1.0"},
        )
        search_resp.raise_for_status()
        hits = search_resp.json().get("query", {}).get("search", [])
        if not hits:
            return "No Wikipedia article found for that query."
        title = hits[0]["title"]

        extract_resp = requests.get(
            WIKIPEDIA_API_URL,
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "titles": title,
                "format": "json",
            },
            timeout=8,
            headers={"User-Agent": "Hallucishield/1.0"},
        )
        extract_resp.raise_for_status()
        pages = extract_resp.json().get("query", {}).get("pages", {})
        extract = next(iter(pages.values()), {}).get("extract", "").strip()
        if not extract:
            return f"No extract available for Wikipedia article '{title}'."
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        return f"Title: {title}\nURL: {url}\nExtract: {extract[:2000]}"
    except requests.RequestException as e:
        return f"Wikipedia search failed: {str(e)}"


def build_tools(store: EvidenceStore) -> list:
    """Builds web_search/wikipedia_search/scrape_url/retrieve_evidence bound to this claim's store."""

    @tool
    def web_search(query: str) -> str:
        """Search the web for recent and reliable information on a topic. Returns Titles, URLs and snippets."""
        try:
            results = _get_tavily_client().search(query=query, max_results=5)
        except Exception as e:
            # Returned as text (not raised) so the agent can still conclude UNVERIFIABLE
            # instead of crashing the whole pipeline on a missing key/network error.
            return f"Web search failed: {str(e)}"

        out = []
        for r in results["results"]:
            text = f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:500]}"
            out.append(text)
            store.add(text, source=r["url"])

        return "\n----\n".join(out) if out else "No results found."

    @tool
    def wikipedia_search(query: str) -> str:
        """Search Wikipedia for encyclopedic background facts (names, dates, definitions, history)."""
        text = _wikipedia_lookup(query)
        store.add(text, source="Wikipedia")
        return text

    @tool
    def scrape_url(url: str) -> str:
        """Scrape and return clean text content from a given URL for deeper reading."""
        if not _is_safe_url(url):
            return "Refused to scrape: URL must be a public http(s) address."
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)[:5000]
            store.add(text, source=url)
            return text
        except requests.RequestException as e:
            return f"Could not scrape URL: {str(e)}"

    @tool
    def retrieve_evidence(query: str) -> str:
        """Semantic search over everything web_search/wikipedia_search/scrape_url have found so
        far in this conversation. Always call this with a query close to the claim's wording
        before giving your final verdict."""
        return store.retrieve(query)

    return [web_search, wikipedia_search, scrape_url, retrieve_evidence]

