# Hallucishield

A pipeline that estimates how much an LLM's answer can be trusted.

## Architecture

```
User query
    |
Agent-1 (agent1_answer_generator) --> generates the LLM's answer
    |
Agent-2 (agent2_claim_detector)   --> extracts atomic factual claims from that answer
    |
Agent-3 (agent3_reader)           --> gathers web/Wikipedia/RAG evidence for each claim
    |
Agent-4 (agent4_verifier)         --> judges each claim against that evidence, trust score
```

No frontend yet (Streamlit is planned for later). Right now the project is driven
through [pipeline.py](pipeline.py), which wires Agent-1's output into Agent-2's input,
and optionally into Agent-3 -> Agent-4.

## Project structure

| Path | Purpose |
|---|---|
| `common/llm_factory.py` | Shared LLM factory (ollama / groq / google), used by every agent |
| `agent1_answer_generator/` | Agent-1: user query -> LLM answer |
| `agent2_claim_detector/` | Agent-2: LLM answer -> exact atomic claims + opinions |
| `agent3_reader/` | Agent-3 ("the Reader"): claims -> gathered web/Wikipedia/RAG evidence, stored in Chroma |
| `agent4_verifier/` | Agent-4 ("the Verifier"): Agent-3's evidence -> reranked/NLI/LLM-judged verdict + trust score |
| `pipeline.py` | Orchestrates Agent-1 -> Agent-2 -> Agent-3 -> Agent-4 end to end |

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env           # then add your API key
```

### Free models only

| Provider | Cost | API key | Setup |
|---|---|---|---|
| **Groq** (default) | Free tier, fast | Free key at console.groq.com | Set `LLM_PROVIDER=groq` |
| Ollama | Free, runs locally | None | Install from ollama.com, then `ollama pull llama3.1` |
| Google Gemini | Free tier | Free key at aistudio.google.com | Set `LLM_PROVIDER=google` |

Agent-3 also needs a free `TAVILY_API_KEY` (console at tavily.com) for its `web_search`
tool. Without it, `web_search` fails gracefully and claims fall back to NOT_ENOUGH_INFO
instead of crashing the pipeline. `wikipedia_search` needs no key.

Agent-4 downloads two HuggingFace models on first use (`cross-encoder/ms-marco-MiniLM-L-6-v2`
~90MB, `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` ~700MB) - no API key, but the first
run will be slow while they download and are cached locally.

## Run

Full pipeline (Agent-1 -> Agent-2 -> Agent-3 -> Agent-4):

```bash
python pipeline.py --question "Who was India's first PM?"
python pipeline.py --question "Who was India's first PM?" --verify
python pipeline.py --question "Who was India's first PM?" --score
```

Each agent can also be run standalone for testing:

```bash
# Agent-1 only
python -m agent1_answer_generator.main --question "Who was India's first PM?"

# Agent-2 only, against a saved answer file
python -m agent2_claim_detector.main --file agent2_claim_detector/sample_answers.txt

# Agent-3 only, against a single claim or a file of claims (one per line)
python -m agent3_reader.main --claim "Draupadi Murmu assumed office on 25 July 2022."

# Agent-4 only (gathers its own evidence via Agent-3, then judges it)
python -m agent4_verifier.main --claim "Draupadi Murmu assumed office on 25 July 2022."
```

## How it works

1. **Agent-1** (`agent1_answer_generator/generator.py`): a `ChatPromptTemplate` asks the
   LLM to answer the user's question directly. `temperature=0.2` keeps answers natural
   but reasonably consistent.
2. **Agent-2** (`agent2_claim_detector/extractor.py`): takes Agent-1's answer as its ONLY
   source and extracts independently verifiable factual claims, keeping opinions and
   hallucinated/incorrect statements intact (they are not corrected, only recorded) so a
   later stage can judge them. `temperature=0` keeps extraction deterministic.
3. **Agent-3** (`agent3_reader/`, "the Reader"): for each claim, a fresh LangChain
   tool-calling agent (`langchain.agents.create_agent`, `reader.py`) decides which of
   `web_search` (Tavily), `wikipedia_search` (free, keyless MediaWiki API), and
   `scrape_url` (SSRF-guarded) to call. Every raw tool result is chunked and embedded
   into a per-claim, in-memory Chroma vector store (`agent3_reader/rag.py`, local
   HuggingFace `all-MiniLM-L6-v2` embeddings - no extra API key). The reader then calls
   `retrieve_evidence` to pull back the most relevant grounded passages and returns them,
   without judging the claim - that's Agent-4's job.
4. **Agent-4** (`agent4_verifier/`, "the Verifier"): takes a claim plus Agent-3's Chroma
   store (`verifier.py`) and judges it: a `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker
   scores/filters the retrieved passages, a `DeBERTa`-based NLI model scores
   entailment/contradiction against the claim, and (if an LLM is given) an LLM judge
   reads the same passages and returns its own verdict + reasoning as JSON. The two
   verdicts are fused (LLM wins on disagreement, with reduced confidence) into
   `SUPPORTED` / `REFUTED` / `NOT_ENOUGH_INFO`. If inconclusive, it rewrites the query
   and fetches fresh evidence into the same store, retrying up to `max_retries` times.
   The trust score is the fraction of claims marked `SUPPORTED`.
5. **Pipeline** (`pipeline.py`): calls `generate_answer()`, feeds the result into
   `extract_claims()`, and (with `--score`) runs Agent-3 -> Agent-4 on the extracted
   claims, printing the answer, claims, opinions, per-claim verdicts, and trust score.

## Possible future work

- Streamlit frontend wiring all four agents together.
- Batch evaluation on a dataset with precision/recall of extracted claims and verdicts.
