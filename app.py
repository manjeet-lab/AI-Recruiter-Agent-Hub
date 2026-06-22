"""
AI Recruiter Agent Hub – Main Streamlit Application
====================================================
Google ADK Multi-Agent Architecture:
  • The Streamlit UI communicates ONLY with the Orchestrator Agent.
  • The Orchestrator coordinates sub-agents sequentially:
      Stage 1: Resume Analysis Agent   → structured candidate profile
      Stage 2: Recruiter Scoring Agent  → scores + hiring decision
      Stage 3: Feedback Agent           → strengths / weaknesses / suggestions
      Stage 4: Recommendation Agent     → career growth plan
  • Results are yielded progressively and rendered in real-time.
  • All file I/O (read resume, save report) goes through the MCP Server.
"""

import os
import sys
import time
import json

from dotenv import load_dotenv
# override=False (default) means .env values will NOT overwrite keys that are
# already set in the process environment (e.g. injected by Streamlit Cloud).
# This ensures Streamlit Secrets always win over the local .env file.
load_dotenv(override=False)

import streamlit as st
from tools.pdf_parser import extract_text_from_pdf, PDFParserError

# ── ADK Orchestrator (single entry-point for the entire AI pipeline) ──────────
# app.py does NOT import individual agents directly — all orchestration is
# delegated to run_orchestrator_pipeline(), which internally invokes each
# Google ADK sub-agent in sequence and yields (stage_name, result_dict) tuples.
from agents.orchestrator_agent import run_orchestrator_pipeline

# Ensure the mcp/ directory is importable for MCP client helpers used in sidebar
_MCP_DIR = os.path.join(os.path.dirname(__file__), "mcp")
if _MCP_DIR not in sys.path:
    sys.path.append(_MCP_DIR)
from mcp_client import mcp_save_resume, mcp_read_resume  # noqa: E402


# ---------------------------------------------------------------------------
# Page Config  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Recruiter Agent Hub",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# API Key Helper
# ---------------------------------------------------------------------------

def get_gemini_api_key() -> str:
    """
    Resolve the Gemini API key in priority order:
      1. Streamlit Cloud Secrets  (st.secrets["GEMINI_API_KEY"])
         — set via the Streamlit Community Cloud dashboard; never prompts the user.
      2. Local .env / environment variable  (os.getenv("GEMINI_API_KEY"))
         — loaded by load_dotenv() above; works for local development.
      3. st.session_state["gemini_api_key"]
         — manually entered by the user via the in-browser form (final fallback).
    Returns an empty string when none of the above sources has a value,
    which signals the caller to display the API key entry card.
    """
    # 1. Streamlit Cloud Secrets (highest priority — never shows form to user)
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
        if secret_key:
            print(f"[API KEY] Source: Streamlit Secrets | prefix: {secret_key[:8]}...")
            return secret_key
    except Exception:
        pass  # st.secrets not available in local dev without a secrets.toml

    # 2. Environment variable / .env file
    env_key = os.getenv("GEMINI_API_KEY", "")
    if env_key:
        print(f"[API KEY] Source: os.environ/.env | prefix: {env_key[:8]}...")
        return env_key

    # 3. Manual browser entry stored in session state (final fallback)
    session_key = st.session_state.get("gemini_api_key", "")
    if session_key:
        print(f"[API KEY] Source: session_state (user-entered) | prefix: {session_key[:8]}...")
    return session_key


# ---------------------------------------------------------------------------
# Premium CSS
# ---------------------------------------------------------------------------

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* ── Global ─────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
    }

    /* ── Header Banner ──────────────────────────── */
    .header-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #311042 50%, #0c0a0f 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .header-title {
        font-family: 'Outfit', sans-serif;
        font-size: clamp(1.6rem, 4vw, 2.8rem);
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa 0%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding-bottom: 0.5rem;
    }
    .header-subtitle {
        color: #9ca3af;
        font-size: clamp(0.85rem, 2vw, 1.1rem);
        font-weight: 400;
        margin: 0;
    }

    /* ── Score Cards ────────────────────────────── */
    .score-card {
        background: rgba(31,41,55,0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        transition: transform 0.3s cubic-bezier(0.4,0,0.2,1);
        height: 100%;
    }
    .score-card:hover { transform: translateY(-4px); }

    .card-hire   { border: 2px solid rgba(16,185,129,0.4);  box-shadow: 0 0 25px rgba(16,185,129,0.15);  background: linear-gradient(145deg,rgba(16,185,129,0.05),rgba(15,23,42,0.6)); }
    .card-maybe  { border: 2px solid rgba(245,158,11,0.4);  box-shadow: 0 0 25px rgba(245,158,11,0.15);  background: linear-gradient(145deg,rgba(245,158,11,0.05),rgba(15,23,42,0.6)); }
    .card-reject { border: 2px solid rgba(239,68,68,0.4);   box-shadow: 0 0 25px rgba(239,68,68,0.15);   background: linear-gradient(145deg,rgba(239,68,68,0.05),rgba(15,23,42,0.6)); }
    .card-purple { border: 1px solid rgba(167,139,250,0.25); background: linear-gradient(145deg,rgba(167,139,250,0.04),rgba(15,23,42,0.6)); }

    .card-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .card-value { font-family: 'Outfit', sans-serif; font-size: 2.4rem; font-weight: 800; color: #fff; }
    .val-green  { color: #34d399; text-shadow: 0 0 12px rgba(52,211,153,0.35); }
    .val-amber  { color: #fbbf24; text-shadow: 0 0 12px rgba(251,191,36,0.35); }
    .val-red    { color: #f87171; text-shadow: 0 0 12px rgba(248,113,113,0.35); }
    .val-purple { background: linear-gradient(90deg,#c084fc,#e879f9); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }

    /* ── Progress Bar ───────────────────────────── */
    .prog-row { margin-bottom: 0.9rem; }
    .prog-label { display:flex; justify-content:space-between; font-size:0.85rem; color:#d1d5db; margin-bottom:0.25rem; font-weight:500; }
    .prog-bar-bg { background:rgba(55,65,81,0.6); border-radius:99px; height:8px; overflow:hidden; }
    .prog-bar-fill { height:8px; border-radius:99px; transition:width 0.6s ease; }

    /* ── Info Cards ─────────────────────────────── */
    .info-card {
        background: rgba(17,24,39,0.6);
        border: 1px solid rgba(167,139,250,0.2);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        height: 100%;
        box-sizing: border-box;
    }
    .info-card h4 { color:#a78bfa; font-size:0.9rem; text-transform:uppercase; letter-spacing:0.06em; margin:0 0 0.8rem; font-weight:700; }

    /* ── JSON Container ─────────────────────────── */
    .json-card-container {
        background: rgba(17,24,39,0.6);
        border: 1px solid rgba(167,139,250,0.25);
        box-shadow: 0 8px 32px 0 rgba(167,139,250,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }

    /* ── API Key Card ───────────────────────────── */
    .api-key-card {
        background: rgba(31,41,55,0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        max-width: 480px;
        margin: 3rem auto;
        box-shadow: 0 8px 40px rgba(0,0,0,0.4);
        text-align: center;
    }
    .api-key-card h3 { color:#a78bfa; margin-bottom:0.4rem; font-size:1.4rem; }
    .api-key-card p  { color:#9ca3af; margin-bottom:1.5rem; font-size:0.9rem; }

    /* ── Section Divider ────────────────────────── */
    .section-header {
        font-family:'Outfit',sans-serif;
        font-size:1.3rem;
        font-weight:700;
        color:#e5e7eb;
        padding: 0.3rem 0 0.8rem;
        border-bottom: 1px solid rgba(167,139,250,0.15);
        margin-bottom:1.2rem;
    }

    /* ── Feedback Cards ─────────────────────────── */
    .feedback-card {
        background: rgba(17,24,39,0.65);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        height: 100%;
        box-sizing: border-box;
    }
    .feedback-card.strengths  { border: 1px solid rgba(52,211,153,0.3);  }
    .feedback-card.weaknesses { border: 1px solid rgba(248,113,113,0.3); }
    .feedback-card.comments   { border: 1px solid rgba(251,191,36,0.3);  }
    .feedback-card.suggestions{ border: 1px solid rgba(96,165,250,0.3);  }
    .feedback-card.priority   {
        border: 2px solid rgba(167,139,250,0.45);
        background: linear-gradient(135deg,rgba(167,139,250,0.06),rgba(15,23,42,0.7));
    }
    .feedback-card h4 {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 700;
        margin: 0 0 0.85rem;
    }
    .feedback-card.strengths  h4 { color:#34d399; }
    .feedback-card.weaknesses h4 { color:#f87171; }
    .feedback-card.comments   h4 { color:#fbbf24; }
    .feedback-card.suggestions h4{ color:#60a5fa; }
    .feedback-card.priority   h4 { color:#c084fc; }

    /* Priority improvement badge numbers */
    .priority-badge {
        display:inline-flex;
        align-items:center;
        justify-content:center;
        width:24px;
        height:24px;
        border-radius:50%;
        font-size:0.75rem;
        font-weight:700;
        margin-right:8px;
        flex-shrink:0;
    }
    .badge-1 { background:#7c3aed; color:#fff; }
    .badge-2 { background:#4c1d95; color:#c4b5fd; }
    .badge-3 { background:#2e1065; color:#a78bfa; }
    .priority-item {
        display:flex;
        align-items:flex-start;
        margin-bottom:0.75rem;
        color:#e5e7eb;
        font-size:0.88rem;
        line-height:1.5;
    }

    /* ── Career Growth / Recommendation Cards ───── */
    .rec-card {
        background: rgba(15,23,42,0.7);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        height: 100%;
        box-sizing: border-box;
    }
    .rec-card.skills       { border: 1px solid rgba(129,140,248,0.35); }
    .rec-card.certs        { border: 1px solid rgba(251,191,36,0.35);  }
    .rec-card.projects     { border: 1px solid rgba(52,211,153,0.35);  }
    .rec-card.interview    { border: 1px solid rgba(244,114,182,0.35); }
    .rec-card.courses {
        border: 1px solid rgba(96,165,250,0.3);
        background: rgba(17,24,39,0.65);
    }
    .rec-card h4 {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 700;
        margin: 0 0 1rem;
    }
    .rec-card.skills    h4 { color:#818cf8; }
    .rec-card.certs     h4 { color:#fbbf24; }
    .rec-card.projects  h4 { color:#34d399; }
    .rec-card.interview h4 { color:#f472b6; }
    .rec-card.courses   h4 { color:#60a5fa; }

    /* Course card nested inside the courses rec-card */
    .course-item {
        background: rgba(31,41,55,0.5);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
    }
    .course-title    { font-weight:600; color:#e5e7eb; font-size:0.9rem; }
    .course-provider { font-size:0.75rem; color:#818cf8; font-weight:600;
                       text-transform:uppercase; letter-spacing:0.05em; margin:2px 0 6px; }
    .course-reason   { font-size:0.83rem; color:#9ca3af; line-height:1.45; }

    /* ── 30-Day Roadmap timeline ────────────────── */
    .roadmap-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 0.5rem;
    }
    .roadmap-week {
        background: rgba(31,41,55,0.55);
        border-radius: 12px;
        padding: 1.1rem 1rem;
        border-top: 3px solid;
    }
    .roadmap-week:nth-child(1) { border-color:#818cf8; }
    .roadmap-week:nth-child(2) { border-color:#34d399; }
    .roadmap-week:nth-child(3) { border-color:#f472b6; }
    .roadmap-week:nth-child(4) { border-color:#fbbf24; }
    .roadmap-label {
        font-size:0.7rem;
        text-transform:uppercase;
        letter-spacing:0.1em;
        font-weight:700;
        margin-bottom:0.5rem;
    }
    .roadmap-week:nth-child(1) .roadmap-label { color:#818cf8; }
    .roadmap-week:nth-child(2) .roadmap-label { color:#34d399; }
    .roadmap-week:nth-child(3) .roadmap-label { color:#f472b6; }
    .roadmap-week:nth-child(4) .roadmap-label { color:#fbbf24; }
    .roadmap-text { font-size:0.84rem; color:#d1d5db; line-height:1.55; }

    /* ── Responsive overrides ───────────────────── */
    @media (max-width: 768px) {
        /* Stack Streamlit column layouts on small viewports */
        [data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .score-card { margin-bottom: 0.75rem; }
        .header-container { padding: 1.5rem 1rem; }
        .roadmap-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 480px) {
        .roadmap-grid { grid-template-columns: 1fr; }
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Demo Candidate Loading & Pipeline Simulator (no API key required)
# ---------------------------------------------------------------------------

DEMO_CANDIDATES = {
    "Sarah Chen (Senior AI Engineer)": "sarah_chen",
    "Alex Rivers (Backend Developer)": "alex_rivers",
}


def run_demo_pipeline(candidate_folder: str):
    """
    Simulates the 4-stage recruiter pipeline by loading local pre-generated JSON files.
    Yields (stage_name, stage_result) exactly like the real pipeline runner.
    """
    base_path = os.path.join("demo_data", candidate_folder)

    stages = [
        ("analysis", "analysis.json"),
        ("scoring", "scoring.json"),
        ("feedback", "feedback.json"),
        ("recommendation", "recommendations.json"),
    ]

    for stage_name, filename in stages:
        time.sleep(0.4)  # Simulate network/processing latency for a smooth UI transition
        filepath = os.path.join(base_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        yield stage_name, data



# ---------------------------------------------------------------------------
# Helper: map hiring decision to CSS classes
# ---------------------------------------------------------------------------

def _decision_css(decision: str):
    """Return (card_css_class, value_css_class) for the given hiring decision string."""
    d = decision.lower()
    if "reject" in d:
        return "card-reject", "val-red"
    if "maybe" in d:
        return "card-maybe", "val-amber"
    return "card-hire", "val-green"


# ---------------------------------------------------------------------------
# Renderer: horizontal progress bar
# ---------------------------------------------------------------------------

def render_score_bar(label: str, value: int, color: str = "#a78bfa"):
    """Render a labeled horizontal progress bar clamped to [0, 100]."""
    pct = max(0, min(100, value))
    st.markdown(
        f"""
        <div class="prog-row">
            <div class="prog-label"><span>{label}</span><span>{pct}/100</span></div>
            <div class="prog-bar-bg">
                <div class="prog-bar-fill" style="width:{pct}%;background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Renderer: Recruiter Scoring Dashboard
# ---------------------------------------------------------------------------

def render_scoring_dashboard(scores: dict):
    """
    Render the full recruiter scoring dashboard from a RecruiterScoreSchema dict.
    Sections: KPI cards, category bars, summary, reasoning, highlights, red flags.
    """
    decision   = scores.get("hiring_decision", "N/A")
    overall    = scores.get("overall_recruiter_score", 0)
    ats        = scores.get("ats_score", 0)
    confidence = scores.get("confidence_score", 0)
    cat        = scores.get("category_scores", {})
    summary    = scores.get("recruiter_summary", "")
    reasoning  = scores.get("recruiter_reasoning", [])
    highlights = scores.get("resume_highlights", [])
    red_flags  = scores.get("red_flags", [])

    card_class, val_class = _decision_css(decision)

    # ── Section header ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Recruiter Score Dashboard</div>', unsafe_allow_html=True)

    # ── KPI Cards Row ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f'<div class="score-card card-purple">'
            f'<div class="card-label">Overall Score</div>'
            f'<div class="card-value val-purple">{overall}</div>'
            f'<div style="color:#6b7280;font-size:0.75rem">/ 100</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        ats_color = "val-green" if ats >= 70 else ("val-amber" if ats >= 50 else "val-red")
        st.markdown(
            f'<div class="score-card card-purple">'
            f'<div class="card-label">ATS Score</div>'
            f'<div class="card-value {ats_color}">{ats}</div>'
            f'<div style="color:#6b7280;font-size:0.75rem">/ 100</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="score-card {card_class}">'
            f'<div class="card-label">Hiring Decision</div>'
            f'<div class="card-value {val_class}" style="font-size:1.6rem;margin:4px 0">{decision}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col4:
        conf_color = "val-green" if confidence >= 70 else ("val-amber" if confidence >= 45 else "val-red")
        st.markdown(
            f'<div class="score-card card-purple">'
            f'<div class="card-label">Confidence</div>'
            f'<div class="card-value {conf_color}">{confidence}</div>'
            f'<div style="color:#6b7280;font-size:0.75rem">/ 100</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Category Bars | Recruiter Summary ──────────────────────────────────
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown('<div class="info-card"><h4>📈 Category Scores</h4>', unsafe_allow_html=True)
        cat_colors = {
            "Technical Skills": "#818cf8",
            "Projects":         "#34d399",
            "Experience":       "#f472b6",
            "Communication":    "#fbbf24",
            "Education":        "#60a5fa",
        }
        cat_map = {
            "Technical Skills": cat.get("technical_skills", 0),
            "Projects":         cat.get("projects", 0),
            "Experience":       cat.get("experience", 0),
            "Communication":    cat.get("communication", 0),
            "Education":        cat.get("education", 0),
        }
        for label, val in cat_map.items():
            render_score_bar(label, val, cat_colors[label])
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown(
            f'<div class="info-card"><h4>🧑‍💼 Recruiter Summary</h4>'
            f'<p style="color:#d1d5db;line-height:1.65;font-size:0.9rem">{summary}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Reasoning | Highlights | Red Flags ────────────────────────────────
    r1, r2, r3 = st.columns(3)

    def _bullet_list(items: list, icon: str, color: str) -> str:
        """Build an HTML unordered list with a leading icon per item."""
        rows = "".join(
            f'<li style="margin-bottom:0.4rem;color:#d1d5db">{icon} {item}</li>'
            for item in items
        )
        return f'<ul style="list-style:none;padding:0;margin:0;color:{color}">{rows}</ul>'

    with r1:
        st.markdown(
            f'<div class="info-card"><h4>🔍 Recruiter Reasoning</h4>'
            f'{_bullet_list(reasoning, "▸", "#a78bfa")}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f'<div class="info-card"><h4>✅ Resume Highlights</h4>'
            f'{_bullet_list(highlights, "★", "#34d399")}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with r3:
        flag_html = (
            _bullet_list(red_flags, "⚠", "#f87171")
            if red_flags
            else '<p style="color:#6b7280;font-size:0.85rem">No major red flags identified.</p>'
        )
        st.markdown(
            f'<div class="info-card"><h4>🚩 Red Flags</h4>{flag_html}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Renderer: Recruiter Feedback Section
# ---------------------------------------------------------------------------

def render_feedback_section(feedback: dict):
    """
    Render the Recruiter Feedback dashboard from a FeedbackSchema dict.

    Layout:
      Row 1: Strengths | Weaknesses | Recruiter Comments
      Row 2: Improvement Suggestions (full-width)
      Row 3: Priority Improvements (highlighted numbered badges)
    """
    strengths   = feedback.get("strengths", [])
    weaknesses  = feedback.get("weaknesses", [])
    comments    = feedback.get("recruiter_comments", [])
    suggestions = feedback.get("resume_improvement_suggestions", [])
    priorities  = feedback.get("priority_improvements", [])

    st.markdown('<div class="section-header">💬 Recruiter Feedback Report</div>', unsafe_allow_html=True)

    # ── Row 1: Strengths | Weaknesses | Comments ───────────────────────────
    c1, c2, c3 = st.columns(3)

    def _fb_list(items: list, icon: str) -> str:
        """Build a styled HTML feedback list."""
        rows = "".join(
            f'<li style="margin-bottom:0.55rem;color:#d1d5db;font-size:0.88rem;line-height:1.5">'
            f'{icon}&nbsp;{item}</li>'
            for item in items
        )
        return f'<ul style="list-style:none;padding:0;margin:0">{rows}</ul>'

    with c1:
        st.markdown(
            f'<div class="feedback-card strengths">'
            f'<h4>✅ Strengths</h4>'
            f'{_fb_list(strengths, "★")}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="feedback-card weaknesses">'
            f'<h4>⚠️ Weaknesses</h4>'
            f'{_fb_list(weaknesses, "▸")}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="feedback-card comments">'
            f'<h4>🧑‍💼 Recruiter Comments</h4>'
            f'{_fb_list(comments, "💬")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Improvement Suggestions (full-width) ────────────────────────
    suggestions_html = "".join(
        f'<li style="margin-bottom:0.6rem;color:#93c5fd;font-size:0.88rem;line-height:1.55">'
        f'🔧&nbsp;{s}</li>'
        for s in suggestions
    )
    st.markdown(
        f'<div class="feedback-card suggestions">'
        f'<h4>🛠️ Resume Improvement Suggestions</h4>'
        f'<ul style="list-style:none;padding:0;margin:0">{suggestions_html}</ul>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Row 3: Top Priority Improvements (numbered badges) ─────────────────
    badge_classes = ["badge-1", "badge-2", "badge-3"]
    priority_items_html = "".join(
        f'<div class="priority-item">'
        f'<span class="priority-badge {badge_classes[min(i, 2)]}">{i + 1}</span>'
        f'<span>{item}</span>'
        f'</div>'
        for i, item in enumerate(priorities[:3])
    )
    st.markdown(
        f'<div class="feedback-card priority">'
        f'<h4>🚀 Top Priority Improvements '
        f'<span style="color:#6b7280;font-size:0.72rem;font-weight:400">(Highest impact first)</span></h4>'
        f'{priority_items_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Renderer: Career Growth Plan (Recommendation Agent output)
# ---------------------------------------------------------------------------

def render_career_growth_plan(recs: dict):
    """
    Render the Career Growth Plan dashboard from a RecommendationSchema dict.

    Layout:
      Row 1 : Skills (left) | Certifications (right)
      Row 2 : Recommended Courses (full-width)
      Row 3 : Project Ideas (left) | Interview Prep (right)
      Row 4 : 30-Day Roadmap (responsive grid)
    """
    skills    = recs.get("recommended_skills", [])
    courses   = recs.get("recommended_courses", [])
    certs     = recs.get("recommended_certifications", [])
    projects  = recs.get("recommended_projects", [])
    interview = recs.get("interview_preparation", [])
    roadmap   = recs.get("career_roadmap", {})

    st.markdown('<div class="section-header">🚀 Career Growth Plan</div>', unsafe_allow_html=True)

    # ── Row 1: Skills | Certifications ────────────────────────────────────
    s_col, c_col = st.columns(2)

    def _chip_list(items: list, rgb: str) -> str:
        """Render items as inline coloured pill chips."""
        chips = "".join(
            f'<span style="display:inline-block;background:rgba({rgb},0.12);'
            f'color:rgb({rgb});border:1px solid rgba({rgb},0.3);'
            f'border-radius:999px;padding:3px 12px;font-size:0.8rem;'
            f'font-weight:600;margin:3px 3px 3px 0">{item}</span>'
            for item in items
        )
        return f'<div style="line-height:2.2">{chips}</div>'

    with s_col:
        st.markdown(
            f'<div class="rec-card skills">'
            f'<h4>🧠 Recommended Skills</h4>'
            f'{_chip_list(skills, "129,140,248")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c_col:
        cert_rows = "".join(
            f'<div style="display:flex;align-items:center;margin-bottom:0.5rem">'
            f'<span style="color:#fbbf24;margin-right:8px;font-size:1.1rem">🏆</span>'
            f'<span style="color:#d1d5db;font-size:0.88rem">{c}</span>'
            f'</div>'
            for c in certs
        )
        st.markdown(
            f'<div class="rec-card certs">'
            f'<h4>🏅 Recommended Certifications</h4>'
            f'{cert_rows}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Row 2: Recommended Courses (full-width) ────────────────────────────
    course_cards_html = "".join(
        f'<div class="course-item">'
        f'<div class="course-title">{course.get("title", "")}</div>'
        f'<div class="course-provider">{course.get("provider", "")}</div>'
        f'<div class="course-reason">{course.get("reason", "")}</div>'
        f'</div>'
        for course in courses
    )
    st.markdown(
        f'<div class="rec-card courses">'
        f'<h4>📚 Recommended Courses</h4>'
        f'{course_cards_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Row 3: Project Ideas | Interview Prep ─────────────────────────────
    p_col, i_col = st.columns(2)

    def _numbered_list(items: list, color: str) -> str:
        """Render a numbered list with coloured index numbers."""
        return "".join(
            f'<div style="display:flex;gap:10px;margin-bottom:0.7rem">'
            f'<span style="color:{color};font-weight:700;font-size:0.9rem;'
            f'flex-shrink:0;min-width:18px">{i + 1}.</span>'
            f'<span style="color:#d1d5db;font-size:0.87rem;line-height:1.5">{item}</span>'
            f'</div>'
            for i, item in enumerate(items)
        )

    with p_col:
        st.markdown(
            f'<div class="rec-card projects">'
            f'<h4>🛠️ Recommended Projects</h4>'
            f'{_numbered_list(projects, "#34d399")}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with i_col:
        st.markdown(
            f'<div class="rec-card interview">'
            f'<h4>🎯 Interview Preparation</h4>'
            f'{_numbered_list(interview, "#f472b6")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Row 4: 30-Day Career Roadmap (responsive CSS grid) ─────────────────
    # Uses a CSS grid (auto-fit + minmax) instead of Streamlit columns so the
    # layout gracefully collapses to 2-column and 1-column on smaller screens.
    week_data = [
        ("Week 1", "week_1", "#818cf8"),
        ("Week 2", "week_2", "#34d399"),
        ("Week 3", "week_3", "#f472b6"),
        ("Week 4", "week_4", "#fbbf24"),
    ]
    week_cards_html = "".join(
        f'<div class="roadmap-week">'
        f'<div class="roadmap-label" style="color:{color}">{label}</div>'
        f'<div class="roadmap-text">{roadmap.get(key, "")}</div>'
        f'</div>'
        for label, key, color in week_data
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header" style="font-size:1.1rem">📅 30-Day Career Roadmap</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="roadmap-grid">{week_cards_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🛠️ Recruiter Dashboard Controls")

    app_mode = st.radio(
        "Pipeline Mode",
        ["Instant Demo (Load Sample)", "Upload Resume File"],
        help="Demo mode uses pre-built mock profiles. Upload mode runs the full AI pipeline.",
    )

    selected_candidate_folder = None
    uploaded_file      = None
    parsed_text        = None
    parse_error        = None

    if app_mode == "Instant Demo (Load Sample)":
        st.info("💡 Choose a pre-defined candidate profile to inspect instantly — no API key needed.")
        candidate_choice   = st.selectbox("Select Candidate Profile", list(DEMO_CANDIDATES.keys()))
        selected_candidate_folder = DEMO_CANDIDATES[candidate_choice]

    else:
        st.info("💡 Upload a PDF resume to run the full 4-stage AI pipeline.")
        uploaded_file = st.file_uploader("Upload Resume File", type=["pdf"])

        if uploaded_file is not None:
            try:
                # Save the uploaded bytes to uploads/ via the MCP Server,
                # then read back the parsed text through the same MCP channel.
                file_bytes  = uploaded_file.read()
                mcp_save_resume(uploaded_file.name, file_bytes)
                parsed_text = mcp_read_resume(uploaded_file.name)
                st.success(f"✔️ Saved & parsed via MCP: {uploaded_file.name}")
            except PDFParserError as pe:
                parse_error = str(pe)
                st.error(f"❌ PDF Parser Error: {parse_error}")
            except Exception as e:
                parse_error = str(e)
                st.error(f"❌ Unexpected error during upload: {parse_error}")

    st.markdown("---")
    st.markdown("### 🤖 AI Recruiter Agent")
    st.caption("Powered by Google Gemini & Google ADK · v1.0")


# ---------------------------------------------------------------------------
# Header Banner
# ---------------------------------------------------------------------------

st.markdown("""
<div class="header-container">
    <h1 class="header-title">✨ AI Recruiter Agent Hub</h1>
    <h2 class="header-subtitle">Kaggle AI Agents Capstone &nbsp;•&nbsp; Resume Parser → Analysis → Scoring → Feedback → Recommendations</h2>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Content Area
# ---------------------------------------------------------------------------

if app_mode == "Instant Demo (Load Sample)":
    # ── Instant Demo: run the pre-built mock pipeline and render dashboards ───────────────────
    st.markdown("### 🤖 Demo Pipeline Controls")
    run_demo_btn = st.button(
        "🧠 Run Demo Pipeline",
        type="primary",
        use_container_width=True,
    )

    if run_demo_btn:
        structured_json = scores = feedback = recommendations = None
        elapsed1 = elapsed2 = elapsed3 = elapsed4 = 0.0
        pipeline_ok = True
        t_pipeline_start = time.time()

        # Human-readable labels for each pipeline stage
        STAGE_META = {
            "analysis": (
                "⚡ Stage 1 / 4 — Running Resume Analysis Agent…",
                "✅ Stage 1 complete — Resume Analysis done",
                "🔍 Extracting structured candidate metadata locally…",
            ),
            "scoring": (
                "⚡ Stage 2 / 4 — Running Recruiter Scoring Agent…",
                "✅ Stage 2 complete — Scoring done",
                "🧑‍💼 Evaluating candidate profile locally…",
            ),
            "feedback": (
                "⚡ Stage 3 / 4 — Running Feedback Agent…",
                "✅ Stage 3 complete — Feedback generated",
                "💬 Generating recruiter-style feedback locally…",
            ),
            "recommendation": (
                "⚡ Stage 4 / 4 — Running Recommendation Agent…",
                "✅ Stage 4 complete — Career plan ready",
                "🚀 Building personalised career growth plan locally…",
            ),
        }

        try:
            for stage_name, stage_result in run_demo_pipeline(selected_candidate_folder):
                running_label, done_label, progress_msg = STAGE_META[stage_name]
                t_stage = time.time()

                with st.status(running_label, expanded=True) as status:
                    st.write(progress_msg)

                    if stage_name == "analysis":
                        structured_json = stage_result
                        elapsed1 = time.time() - t_stage
                        status.update(label=f"{done_label} in {elapsed1:.1f}s", state="complete")
                        # Show raw extracted profile in a collapsible expander
                        with st.expander("📋 Extracted Candidate Profile (JSON)", expanded=False):
                            st.markdown('<div class="json-card-container">', unsafe_allow_html=True)
                            st.json(structured_json)
                            st.markdown("</div>", unsafe_allow_html=True)

                    elif stage_name == "scoring":
                        scores   = stage_result
                        elapsed2 = time.time() - t_stage
                        status.update(label=f"{done_label} in {elapsed2:.1f}s", state="complete")

                    elif stage_name == "feedback":
                        feedback = stage_result
                        elapsed3 = time.time() - t_stage
                        status.update(label=f"{done_label} in {elapsed3:.1f}s", state="complete")

                    elif stage_name == "recommendation":
                        recommendations = stage_result
                        elapsed4 = time.time() - t_stage
                        status.update(label=f"{done_label} in {elapsed4:.1f}s", state="complete")

        except Exception as exc:
            st.error(f"❌ Demo pipeline error: {exc}")
            pipeline_ok = False

        # ── Render results when all stages complete successfully ────
        if pipeline_ok and scores:
            total = time.time() - t_pipeline_start
            st.success(
                f"🎉 Full pipeline complete! "
                f"Analysis: {elapsed1:.1f}s | Scoring: {elapsed2:.1f}s | "
                f"Feedback: {elapsed3:.1f}s | Recommendations: {elapsed4:.1f}s | "
                f"Total: {total:.1f}s"
            )
            st.markdown("<br>", unsafe_allow_html=True)

            render_scoring_dashboard(scores)

            if feedback:
                st.markdown("<br>", unsafe_allow_html=True)
                render_feedback_section(feedback)

            if recommendations:
                st.markdown("<br>", unsafe_allow_html=True)
                render_career_growth_plan(recommendations)

            # ── Raw JSON debug view (collapsed by default) ─────────
            with st.expander("🔧 Raw Agent Outputs (Debug View)", expanded=False):
                tab_scores, tab_feedback, tab_recs, tab_profile = st.tabs(
                    ["Scoring JSON", "Feedback JSON", "Recommendations JSON", "Profile JSON"]
                )
                with tab_scores:
                    st.markdown('<div class="json-card-container">', unsafe_allow_html=True)
                    st.json(scores)
                    st.markdown("</div>", unsafe_allow_html=True)
                with tab_feedback:
                    st.markdown('<div class="json-card-container">', unsafe_allow_html=True)
                    st.json(feedback or {})
                    st.markdown("</div>", unsafe_allow_html=True)
                with tab_recs:
                    st.markdown('<div class="json-card-container">', unsafe_allow_html=True)
                    st.json(recommendations or {})
                    st.markdown("</div>", unsafe_allow_html=True)
                with tab_profile:
                    st.markdown('<div class="json-card-container">', unsafe_allow_html=True)
                    st.json(structured_json or {})
                    st.markdown("</div>", unsafe_allow_html=True)

else:
    # ── Upload Resume File mode ───────────────────────────────────────────

    if uploaded_file is None:
        st.info("ℹ️ Please upload a resume PDF file in the sidebar to begin.")

    elif parse_error:
        st.error(f"⚠️ Cannot proceed — parser error: {parse_error}")
        st.info(
            "Make sure the PDF is text-searchable (not a scanned image). "
            "Maximum file size is 5 MB."
        )

    elif parsed_text:
        # ── Extracted text preview (collapsible) ──────────────────────────
        with st.expander("📄 View Extracted Resume Text", expanded=False):
            st.markdown("##### Cleaned & Normalised Content")
            st.text_area(
                label="Extracted Text",
                value=parsed_text,
                height=200,
                disabled=True,
                label_visibility="collapsed",
            )
            st.caption(f"Characters: {len(parsed_text):,} | Words: {len(parsed_text.split()):,}")

        # ── API Key resolution ─────────────────────────────────────────────
        api_key = get_gemini_api_key()

        if not api_key:
            # Show a centred card prompting the user to enter their API key
            st.markdown("""
            <div class="api-key-card">
                <h3>🔑 Gemini API Key Required</h3>
                <p>Your API key is not set in the environment.<br>
                Enter it below to enable AI analysis. It is only used in this session.</p>
            </div>
            """, unsafe_allow_html=True)

            with st.form("api_key_form"):
                api_key_input = st.text_input(
                    "Google Gemini API Key",
                    type="password",
                    placeholder="AIza...",
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button("🚀 Continue with Analysis", use_container_width=True)

            if submitted and api_key_input.strip():
                st.session_state["gemini_api_key"] = api_key_input.strip()
                st.rerun()
            elif submitted:
                st.warning("⚠️ Please paste a valid API key before continuing.")

        if api_key:
            # ── Pipeline controls ──────────────────────────────────────────
            st.markdown("### 🤖 AI Pipeline Controls")
            run_btn = st.button(
                "🧠 Run Full AI Pipeline  (Analysis → Scoring → Feedback → Recommendations)",
                type="primary",
                use_container_width=True,
            )

            if run_btn:
                # ── ADK Orchestrator Pipeline ──────────────────────────────
                # app.py communicates ONLY with run_orchestrator_pipeline().
                # The generator yields one (stage_name, result_dict) tuple per
                # agent stage so the UI can render results progressively.

                structured_json = scores = feedback = recommendations = None
                elapsed1 = elapsed2 = elapsed3 = elapsed4 = 0.0
                pipeline_ok = True
                t_pipeline_start = time.time()

                # Human-readable labels for each pipeline stage
                STAGE_META = {
                    "analysis": (
                        "⚡ Stage 1 / 4 — Running Resume Analysis Agent…",
                        "✅ Stage 1 complete — Resume Analysis done",
                        "🔍 Extracting structured candidate metadata with Gemini…",
                    ),
                    "scoring": (
                        "⚡ Stage 2 / 4 — Running Recruiter Scoring Agent…",
                        "✅ Stage 2 complete — Scoring done",
                        "🧑‍💼 Evaluating candidate profile using recruiter-style AI reasoning…",
                    ),
                    "feedback": (
                        "⚡ Stage 3 / 4 — Running Feedback Agent…",
                        "✅ Stage 3 complete — Feedback generated",
                        "💬 Generating recruiter-style feedback from analysis + scores…",
                    ),
                    "recommendation": (
                        "⚡ Stage 4 / 4 — Running Recommendation Agent…",
                        "✅ Stage 4 complete — Career plan ready",
                        "🚀 Building personalised career growth plan…",
                    ),
                }

                try:
                    for stage_name, stage_result in run_orchestrator_pipeline(
                        resume_filename=uploaded_file.name,
                        api_key=api_key,
                    ):
                        running_label, done_label, progress_msg = STAGE_META[stage_name]
                        t_stage = time.time()

                        with st.status(running_label, expanded=True) as status:
                            st.write(progress_msg)

                            if stage_name == "analysis":
                                structured_json = stage_result
                                elapsed1 = time.time() - t_stage
                                status.update(label=f"{done_label} in {elapsed1:.1f}s", state="complete")
                                # Show raw extracted profile in a collapsible expander
                                with st.expander("📋 Extracted Candidate Profile (JSON)", expanded=False):
                                    st.markdown('<div class="json-card-container">', unsafe_allow_html=True)
                                    st.json(structured_json)
                                    st.markdown("</div>", unsafe_allow_html=True)

                            elif stage_name == "scoring":
                                scores   = stage_result
                                elapsed2 = time.time() - t_stage
                                status.update(label=f"{done_label} in {elapsed2:.1f}s", state="complete")

                            elif stage_name == "feedback":
                                feedback = stage_result
                                elapsed3 = time.time() - t_stage
                                status.update(label=f"{done_label} in {elapsed3:.1f}s", state="complete")

                            elif stage_name == "recommendation":
                                recommendations = stage_result
                                elapsed4 = time.time() - t_stage
                                status.update(label=f"{done_label} in {elapsed4:.1f}s", state="complete")

                except Exception as exc:
                    err_msg = str(exc)
                    pipeline_ok = False

                    # Always print the raw exception to the server logs so the
                    # real cause is never hidden by the friendly UI card.
                    print(f"[PIPELINE ERROR] type={type(exc).__name__} | msg={err_msg}")

                    if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                        # Friendly, actionable card for quota-exceeded errors
                        st.markdown(f"""
                        <div style="background-color:rgba(239,68,68,0.1);border:2px solid #ef4444;
                             border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;">
                            <h3 style="color:#f87171;margin-top:0;font-family:'Outfit',sans-serif;">
                                ⚠️ Gemini API Quota Exceeded (429 Rate Limit)
                            </h3>
                            <p style="color:#e5e7eb;font-size:0.95rem;line-height:1.6;">
                                The Gemini API returned a <code>RESOURCE_EXHAUSTED</code> error.
                                Free-tier keys allow <strong>20 requests per day</strong>.
                                One pipeline run consumes 4 requests (one per agent stage).
                            </p>
                            <details style="margin-bottom:1rem;">
                                <summary style="color:#9ca3af;font-size:0.82rem;cursor:pointer;
                                               margin-bottom:0.4rem;">🔍 Raw error details (for debugging)</summary>
                                <code style="display:block;background:rgba(0,0,0,0.3);padding:0.7rem;
                                             border-radius:8px;font-size:0.78rem;color:#fca5a5;
                                             word-break:break-all;">{err_msg}</code>
                            </details>
                            <h4 style="color:#fbbf24;margin-bottom:0.5rem;font-family:'Outfit',sans-serif;">
                                Recommended Actions:
                            </h4>
                            <ul style="color:#d1d5db;font-size:0.9rem;padding-left:1.2rem;margin-top:0;">
                                <li style="margin-bottom:0.4rem">
                                    <strong>&#9888; New key, same quota?</strong> Generating a new key
                                    from the <em>same Google account / GCP project</em> does <em>not</em>
                                    reset the quota — all keys in one project share the same pool.
                                    Use a <em>different Google account</em> to get fresh free-tier quota.
                                </li>
                                <li style="margin-bottom:0.4rem">
                                    <strong>Switch to Demo Mode:</strong> Set <em>Pipeline Mode</em> to
                                    <em>"Instant Demo (Load Sample)"</em> in the sidebar to explore
                                    pre-loaded dashboards without using the API.
                                </li>
                                <li style="margin-bottom:0.4rem">
                                    <strong>Enable Billing:</strong> Activate a pay-as-you-go key on
                                    Google AI Studio to remove the daily request limit.
                                </li>
                                <li>
                                    <strong>Wait & Retry:</strong> Free-tier quotas reset daily.
                                    Try again after the reset period.
                                </li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"❌ Pipeline error: {exc}")
                        st.info(
                            "Please check your Gemini API key and ensure it is valid. "
                            "If the error persists, verify your network connection."
                        )

                # ── Render results when all stages complete successfully ────
                if pipeline_ok and scores:
                    total = time.time() - t_pipeline_start
                    st.success(
                        f"🎉 Full pipeline complete! "
                        f"Analysis: {elapsed1:.1f}s | Scoring: {elapsed2:.1f}s | "
                        f"Feedback: {elapsed3:.1f}s | Recommendations: {elapsed4:.1f}s | "
                        f"Total: {total:.1f}s"
                    )
                    st.markdown("<br>", unsafe_allow_html=True)

                    render_scoring_dashboard(scores)

                    if feedback:
                        st.markdown("<br>", unsafe_allow_html=True)
                        render_feedback_section(feedback)

                    if recommendations:
                        st.markdown("<br>", unsafe_allow_html=True)
                        render_career_growth_plan(recommendations)

                    # ── Raw JSON debug view (collapsed by default) ─────────
                    with st.expander("🔧 Raw Agent Outputs (Debug View)", expanded=False):
                        tab_scores, tab_feedback, tab_recs, tab_profile = st.tabs(
                            ["Scoring JSON", "Feedback JSON", "Recommendations JSON", "Profile JSON"]
                        )
                        with tab_scores:
                            st.markdown('<div class="json-card-container">', unsafe_allow_html=True)
                            st.json(scores)
                            st.markdown("</div>", unsafe_allow_html=True)
                        with tab_feedback:
                            st.markdown('<div class="json-card-container">', unsafe_allow_html=True)
                            st.json(feedback or {})
                            st.markdown("</div>", unsafe_allow_html=True)
                        with tab_recs:
                            st.markdown('<div class="json-card-container">', unsafe_allow_html=True)
                            st.json(recommendations or {})
                            st.markdown("</div>", unsafe_allow_html=True)
                        with tab_profile:
                            st.markdown('<div class="json-card-container">', unsafe_allow_html=True)
                            st.json(structured_json or {})
                            st.markdown("</div>", unsafe_allow_html=True)
