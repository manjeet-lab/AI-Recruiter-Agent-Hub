"""
Orchestrator Agent
==================
Coordinates the full AI recruiting pipeline as a Google ADK multi-agent hierarchy.

Architecture:
    Streamlit UI  →  run_orchestrator_pipeline()
                         ├─ [MCP] Read resume from uploads/
                         ├─ Resume Analysis Agent
                         ├─ Recruiter Scoring Agent
                         ├─ Feedback Agent
                         ├─ Recommendation Agent
                         └─ [MCP] Save aggregated report to reports/

The UI communicates ONLY with this module — it never imports individual agents
directly, keeping a clean separation between orchestration and presentation.
"""

import os
import sys
import json
from typing import Generator, Tuple, Any

from dotenv import load_dotenv
from google.adk import Agent

# Import individual ADK sub-agents to register them in the Google ADK hierarchy.
# The sub_agents list establishes the coordinator-specialist relationship so ADK
# can reason about the agent tree even though we drive execution manually below.
from agents.resume_analysis_agent import resume_analysis_adk_agent
from agents.scoring_agent import scoring_adk_agent
from agents.feedback_agent import feedback_adk_agent
from agents.recommendation_agent import recommendation_adk_agent

load_dotenv()

# ---------------------------------------------------------------------------
# MCP directory bootstrapping
# ---------------------------------------------------------------------------
# Add the local mcp/ directory to sys.path once at module load time so that
# mcp_client can be imported without path manipulation inside every function.

_WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MCP_DIR = os.path.join(_WORKSPACE_DIR, "mcp")
if _MCP_DIR not in sys.path:
    sys.path.append(_MCP_DIR)

from mcp_client import mcp_read_resume, mcp_save_report  # noqa: E402 — after path setup


# ---------------------------------------------------------------------------
# Google ADK Orchestrator Agent (hierarchy declaration)
# ---------------------------------------------------------------------------
# The orchestrator_agent object registers the four specialists as sub-agents so
# that Google ADK can understand the overall multi-agent structure.  Actual
# sequential execution is driven by run_orchestrator_pipeline() below.

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instruction=(
        "Coordinate the recruiter sub-agents sequentially to parse, score, "
        "evaluate, and provide career recommendations for a candidate."
    ),
    sub_agents=[
        resume_analysis_adk_agent,
        scoring_adk_agent,
        feedback_adk_agent,
        recommendation_adk_agent,
    ],
)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_orchestrator_pipeline(
    resume_filename: str,
    api_key: str,
) -> Generator[Tuple[str, Any], None, None]:
    """
    Execute the multi-agent ADK pipeline sequentially and yield stage results.

    This is the single entry-point for the Streamlit UI.  The UI never imports
    individual agent modules — all coordination happens here.

    File I/O uses the Filesystem MCP Server exclusively:
      - Resume text is fetched via the MCP read_resume tool (uploads/).
      - The consolidated report is saved via the MCP save_report tool (reports/).

    Args:
        resume_filename: Name of the PDF resume stored in the uploads/ directory
                         (previously saved there by the UI via MCP).
        api_key:         Google Gemini API Key.

    Yields:
        (stage_name, stage_result) tuples in order:
          "analysis"       → dict (ResumeSchema)
          "scoring"        → dict (RecruiterScoreSchema)
          "feedback"       → dict (FeedbackSchema)
          "recommendation" → dict (RecommendationSchema)

    Raises:
        ValueError:   If the MCP server cannot load the resume file.
        RuntimeError: Propagated from any agent if it returns an empty response.
    """
    # Inject API key into environment so every ADK agent/runner can pick it up.
    os.environ["GEMINI_API_KEY"] = api_key

    # ── Stage 0: Fetch resume text through the Filesystem MCP Server ──────────
    # The Orchestrator never reads files directly — it delegates to the MCP server
    # which enforces path containment and handles PDF text extraction.
    resume_text = mcp_read_resume(resume_filename)
    if not resume_text or resume_text.startswith("Error"):
        raise ValueError(
            f"Filesystem MCP Server could not load resume '{resume_filename}': {resume_text}"
        )

    # ── Stage 1: Resume Analysis ───────────────────────────────────────────────
    from agents.resume_analysis_agent import analyze_resume_text
    structured_json = analyze_resume_text(resume_text, api_key=api_key)
    yield "analysis", structured_json

    # ── Stage 2: Recruiter Scoring ─────────────────────────────────────────────
    from agents.scoring_agent import score_candidate
    scores = score_candidate(structured_json, api_key=api_key)
    yield "scoring", scores

    # ── Stage 3: Recruiter Feedback ────────────────────────────────────────────
    from agents.feedback_agent import generate_feedback
    feedback = generate_feedback(
        resume_json=structured_json,
        scores_json=scores,
        api_key=api_key,
    )
    yield "feedback", feedback

    # ── Stage 4: Career Recommendations ───────────────────────────────────────
    from agents.recommendation_agent import generate_recommendations
    recommendations = generate_recommendations(
        resume_json=structured_json,
        scores_json=scores,
        feedback_json=feedback,
        api_key=api_key,
    )
    yield "recommendation", recommendations

    # ── Final step: Persist aggregated report via MCP ─────────────────────────
    # Combine all four agent outputs into one JSON document and save it to
    # reports/ through the Filesystem MCP Server (path-safe write).
    report_data = {
        "candidate_profile": structured_json,
        "scoring":           scores,
        "feedback":          feedback,
        "recommendation":    recommendations,
    }
    base_name = os.path.splitext(os.path.basename(resume_filename))[0]
    mcp_save_report(f"report_{base_name}.json", json.dumps(report_data, indent=2))
