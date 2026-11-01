"""
app.py
------
Streamlit frontend for Hallucishield.

User enters a question -> Agent-1 generates an answer -> Agent-2 extracts
claims (shown in a collapsible dropdown) -> Agent-4 judges each claim against
Agent-3's gathered evidence, rendered as a tick (SUPPORTED), cross (REFUTED),
or warning sign (NOT_ENOUGH_INFO) -> an overall trust score.

Run:
    streamlit run app.py
"""

import html
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # this network breaks the Xet CDN mid-download - fall back to plain HTTPS

import streamlit as st

from agent1_answer_generator.generator import generate_answer
from agent2_claim_detector.extractor import extract_claims
from agent4_verifier.verifier import compute_trust_score, verify_claim
from common.llm_factory import build_llm

st.set_page_config(page_title="Hallucishield", page_icon="\U0001f6e1\ufe0f", layout="centered")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 840px; }
#MainMenu, footer { visibility: hidden; }

/* ---- Hero header ---- */
.hs-hero {
    background: linear-gradient(135deg, #6C4CF1 0%, #9B6CF5 50%, #5FD0C7 100%);
    border-radius: 18px;
    padding: 2.1rem 2rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 30px rgba(108, 76, 241, 0.25);
}
.hs-hero h1 {
    color: #ffffff;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hs-hero p {
    color: rgba(255, 255, 255, 0.92);
    font-size: 1.02rem;
    margin: 0;
}

/* ---- Answer card ---- */
.answer-card {
    background: #ffffff;
    border: 1px solid #ECE8FB;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 4px 18px rgba(30, 27, 46, 0.06);
    line-height: 1.55;
    margin-bottom: 0.5rem;
}

/* ---- Section headers ---- */
.hs-section-title {
    font-weight: 700;
    font-size: 1.15rem;
    margin: 1.6rem 0 0.7rem 0;
    color: #1E1B2E;
}

/* ---- Stat chips (supported / refuted / unverifiable counts) ---- */
.stat-row { display: flex; gap: 10px; margin-bottom: 1.1rem; flex-wrap: wrap; }
.stat-chip {
    flex: 1;
    min-width: 120px;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 14px;
    border-radius: 12px;
    font-weight: 600;
}
.stat-chip .num { font-size: 1.4rem; font-weight: 800; }
.stat-chip .lbl { font-size: 0.78rem; font-weight: 500; opacity: 0.85; }
.stat-supported { background: rgba(46, 204, 113, 0.12); color: #1e8449; }
.stat-refuted { background: rgba(231, 76, 60, 0.12); color: #c0392b; }
.stat-unverifiable { background: rgba(241, 162, 8, 0.12); color: #b9770e; }

/* ---- Claim verdict cards ---- */
.claim-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border-radius: 14px;
    background: #ffffff;
    border: 1px solid #ECE8FB;
    border-left: 6px solid #999;
    box-shadow: 0 2px 10px rgba(30, 27, 46, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.claim-card:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(30, 27, 46, 0.09); }
.claim-card.supported { border-left-color: #2ecc71; }
.claim-card.refuted { border-left-color: #e74c3c; }
.claim-card.unverifiable { border-left-color: #f1a208; }

.claim-text { flex: 1; font-size: 0.97rem; line-height: 1.45; color: #1E1B2E; }

.icon-badge {
    flex-shrink: 0;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}
.icon-badge.supported { background: rgba(46, 204, 113, 0.15); }
.icon-badge.refuted { background: rgba(231, 76, 60, 0.15); }
.icon-badge.unverifiable { background: rgba(241, 162, 8, 0.15); }

/* ---- Trust score ---- */
.trust-wrap {
    display: flex;
    align-items: center;
    gap: 1.4rem;
    padding: 1.2rem 1.5rem;
    border-radius: 16px;
    margin: 0.3rem 0 1.3rem 0;
}
.trust-high { background: linear-gradient(135deg, rgba(46,204,113,0.14), rgba(95,208,199,0.14)); }
.trust-mid { background: linear-gradient(135deg, rgba(241,162,8,0.14), rgba(255,206,84,0.14)); }
.trust-low { background: linear-gradient(135deg, rgba(231,76,60,0.14), rgba(255,118,117,0.14)); }
.trust-value { font-size: 2.6rem; font-weight: 800; line-height: 1; }
.trust-high .trust-value { color: #1e8449; }
.trust-mid .trust-value { color: #b9770e; }
.trust-low .trust-value { color: #c0392b; }
.trust-meta .trust-label { font-weight: 700; font-size: 0.95rem; color: #1E1B2E; }
.trust-meta .trust-sub { font-size: 0.82rem; color: #6b6478; }
.trust-bar-track { flex: 1; height: 10px; border-radius: 6px; background: rgba(30,27,46,0.08); overflow: hidden; margin-top: 6px; }
.trust-bar-fill { height: 100%; border-radius: 6px; }
.trust-high .trust-bar-fill { background: linear-gradient(90deg, #2ecc71, #5FD0C7); }
.trust-mid .trust-bar-fill { background: linear-gradient(90deg, #f1a208, #ffce54); }
.trust-low .trust-bar-fill { background: linear-gradient(90deg, #e74c3c, #ff7675); }

.hs-footer { text-align: center; color: #9a93ab; font-size: 0.8rem; margin-top: 2.2rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

VERDICT_STYLE = {
    "SUPPORTED": {"icon": "\u2705", "css": "supported", "label": "Supported"},
    "REFUTED": {"icon": "\u274c", "css": "refuted", "label": "Refuted"},
    "NOT_ENOUGH_INFO": {"icon": "\u26a0\ufe0f", "css": "unverifiable", "label": "Not enough info"},
}


def render_claim_card(claim_text: str, verdict: str) -> str:
    style = VERDICT_STYLE.get(verdict, VERDICT_STYLE["NOT_ENOUGH_INFO"])
    return (
        f'<div class="claim-card {style["css"]}">'
        f'<span class="claim-text">{html.escape(claim_text)}</span>'
        f'<span class="icon-badge {style["css"]}" title="{style["label"]}">{style["icon"]}</span>'
        f"</div>"
    )


def render_stat_row(scoring: list[dict]) -> str:
    counts = {"SUPPORTED": 0, "REFUTED": 0, "NOT_ENOUGH_INFO": 0}
    for r in scoring:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    chips = [
        ("stat-supported", "\u2705", counts["SUPPORTED"], "Supported"),
        ("stat-refuted", "\u274c", counts["REFUTED"], "Refuted"),
        ("stat-unverifiable", "\u26a0\ufe0f", counts["NOT_ENOUGH_INFO"], "Not enough info"),
    ]
    cells = "".join(
        f'<div class="stat-chip {css}"><span>{icon}</span>'
        f'<div><div class="num">{num}</div><div class="lbl">{label}</div></div></div>'
        for css, icon, num, label in chips
    )
    return f'<div class="stat-row">{cells}</div>'


def render_trust_score(score: float) -> str:
    tier = "trust-high" if score >= 0.7 else "trust-mid" if score >= 0.4 else "trust-low"
    verdict_word = "Looks trustworthy" if score >= 0.7 else "Mixed signals" if score >= 0.4 else "Low confidence"
    return (
        f'<div class="trust-wrap {tier}">'
        f'<div class="trust-value">{score:.0%}</div>'
        f'<div class="trust-meta" style="flex:1;">'
        f'<div class="trust-label">Overall trust score &middot; {verdict_word}</div>'
        f'<div class="trust-bar-track"><div class="trust-bar-fill" style="width:{score * 100:.0f}%;"></div></div>'
        f"</div></div>"
    )


st.markdown(
    """
    <div class="hs-hero">
        <h1>\U0001f6e1\ufe0f Hallucishield</h1>
        <p>Ask a question, then see which claims in the answer hold up to web &amp; Wikipedia evidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("\u2699\ufe0f Settings")
    provider_choice = st.selectbox("LLM provider", ["Default (.env)", "groq", "ollama", "google"], index=0)
    provider = None if provider_choice == "Default (.env)" else provider_choice
    run_verification = st.checkbox(
        "Run Agent-4 verification", value=True,
        help="Reranker + NLI + LLM judge per claim. Downloads ML models (~800MB total) on first use.",
    )
    st.caption("Agent-3 needs TAVILY_API_KEY in .env for web_search.")

if "result" not in st.session_state:
    st.session_state.result = None

with st.form("question_form"):
    question = st.text_input("Enter your question", placeholder="e.g. Who was India's first PM?")
    submitted = st.form_submit_button("\U0001f50d Analyze", type="primary", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.status("Running Hallucishield pipeline...", expanded=True) as status:
            status.write("Agent-1: generating answer...")
            answer = generate_answer(question, provider=provider)

            status.write("Agent-2: extracting claims...")
            extracted = extract_claims(answer, question=question)
            claims, opinions = extracted["claims"], extracted["opinions"]

            scoring = []
            if run_verification and claims:
                llm = build_llm(provider, temperature=0)
                total = len(claims)
                for i, claim in enumerate(claims, 1):
                    status.write(f"Agent-3/Agent-4: verifying claim {i}/{total}...")
                    scoring.append(verify_claim(claim, llm))

            status.update(label="Done", state="complete", expanded=False)

        st.session_state.result = {
            "question": question,
            "answer": answer,
            "claims": claims,
            "opinions": opinions,
            "scoring": scoring,
        }

result = st.session_state.result
if result:
    st.markdown('<div class="hs-section-title">Answer</div>', unsafe_allow_html=True)
    answer_html = html.escape(result["answer"]).replace("\n", "<br>")
    st.markdown(f'<div class="answer-card">{answer_html}</div>', unsafe_allow_html=True)

    with st.expander(f"\U0001f4cb Extracted claims ({len(result['claims'])})"):
        if result["claims"]:
            for i, c in enumerate(result["claims"], 1):
                st.markdown(f"{i}. {html.escape(c)}")
        else:
            st.write("No independently verifiable factual claims were extracted.")
        if result["opinions"]:
            st.markdown("**Opinions (not fact-checked):**")
            for o in result["opinions"]:
                st.markdown(f"- {html.escape(o)}")

    if result["scoring"]:
        st.markdown('<div class="hs-section-title">Verdicts</div>', unsafe_allow_html=True)
        st.markdown(render_stat_row(result["scoring"]), unsafe_allow_html=True)
        st.markdown(render_trust_score(compute_trust_score(result["scoring"])), unsafe_allow_html=True)
        for r in result["scoring"]:
            st.markdown(render_claim_card(r["claim"], r["verdict"]), unsafe_allow_html=True)
            with st.expander("Reasoning"):
                st.write(r["reasoning"])
                st.caption(f"Confidence: {r['confidence']}")
    elif result["claims"]:
        st.info("Agent-4 verification was skipped (disabled in sidebar).")

st.markdown(
    '<div class="hs-footer">Hallucishield &middot; Agent-1 \u2192 Agent-2 \u2192 Agent-3 \u2192 Agent-4</div>',
    unsafe_allow_html=True,
)
