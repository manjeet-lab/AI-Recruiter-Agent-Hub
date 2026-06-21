"""
Recruiter Scoring Agent
=======================
Evaluates a candidate's structured resume profile (produced by the Resume
Analysis Agent) exactly as an experienced recruiter would during initial screening.

Responsibilities (this agent ONLY):
  - Score the candidate across independent dimensions.
  - Produce a hiring decision: "Hire", "Maybe Hire", or "Reject".
  - Does NOT re-parse the PDF, generate feedback, or create career roadmaps.

Flow:
  1. Receive the structured dict from analyze_resume_text().
  2. Serialize it to compact JSON and build a detailed prompt.
  3. Call Gemini via Google ADK with a strict output_schema.
  4. Validate and return a dict matching RecruiterScoreSchema.
"""

import os
import json
import asyncio
from typing import List, Optional

import nest_asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from pydantic import BaseModel, Field

load_dotenv()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class CategoryScores(BaseModel):
    """Independent category scores (0–100) evaluated by the recruiter agent."""

    technical_skills: int = Field(..., ge=0, le=100, description="Depth and breadth of hard/technical skills.")
    projects: int = Field(..., ge=0, le=100, description="Quality, complexity, and relevance of listed projects.")
    experience: int = Field(..., ge=0, le=100, description="Relevance, seniority, and impact of prior work experience.")
    communication: int = Field(
        ..., ge=0, le=100,
        description="Clarity and professionalism; presence of measurable achievements and action verbs.",
    )
    education: int = Field(..., ge=0, le=100, description="Relevance and prestige of academic credentials.")


class RecruiterScoreSchema(BaseModel):
    """Complete recruiter evaluation returned by the Scoring Agent."""

    overall_recruiter_score: int = Field(..., ge=0, le=100, description="Holistic quality score (0–100).")
    ats_score: int = Field(
        ..., ge=0, le=100,
        description="ATS-friendliness estimate considering keywords, headers, formatting, and completeness.",
    )
    hiring_decision: str = Field(..., description="Exactly one of: 'Hire', 'Maybe Hire', or 'Reject'.")
    confidence_score: int = Field(..., ge=0, le=100, description="Confidence in the evaluation (0–100).")
    category_scores: CategoryScores = Field(..., description="Independent scores per evaluation category.")
    recruiter_summary: str = Field(..., description="Concise 3–5 sentence recruiter-style narrative.")
    recruiter_reasoning: List[str] = Field(..., description="3–5 bullet-point reasons justifying the evaluation.")
    resume_highlights: List[str] = Field(..., description="The strongest aspects of the resume.")
    red_flags: List[str] = Field(..., description="Weaknesses or missing elements. Empty list if none found.")


# ---------------------------------------------------------------------------
# Google ADK Agent definition
# ---------------------------------------------------------------------------

scoring_adk_agent = Agent(
    name="scoring_agent",
    model="gemini-2.5-flash",
    instruction="""
You are an experienced technical recruiter and hiring manager with 15+ years of
experience evaluating software engineering and technology candidates.

You have been given a structured JSON resume profile extracted by an automated
parser. Your job is to evaluate this candidate exactly as a real recruiter would
during the initial screening phase.

EVALUATION INSTRUCTIONS:

1. overall_recruiter_score (0-100)
   Holistically rate the candidate's profile quality considering all factors:
   technical depth, experience level, project quality, communication clarity,
   education, and professional completeness.

2. ats_score (0-100)
   Estimate how well this resume would survive an ATS filter.
   Consider: keyword richness, standard section names, completeness of contact info,
   measurable achievements, and absence of formatting red flags.

3. hiring_decision
   Choose EXACTLY ONE: "Hire", "Maybe Hire", or "Reject".
   Base this on whether you would move this candidate to the next round.

4. confidence_score (0-100)
   How confident are you in your evaluation given the data available?
   Low confidence when critical fields are missing; high when the profile is detailed.

5. category_scores — score each independently (0-100):
   - technical_skills : Depth, breadth, and relevance of hard skills.
   - projects         : Complexity, impact, and presentation of projects.
   - experience       : Relevance, seniority, tenure, and impact of work history.
   - communication    : Clarity, action verbs, measurable results, professionalism.
   - education        : Relevance and quality of academic credentials.

6. recruiter_summary
   Write a professional 3–5 sentence narrative a recruiter would put in a scorecard.
   Mention the candidate by name, their strongest area, and any notable concerns.

7. recruiter_reasoning
   Provide 3–5 concise bullet-point reasons that justify your overall evaluation.

8. resume_highlights
   List the 3–5 strongest positives noticed in the profile.

9. red_flags
   List weaknesses, gaps, or missing elements. Return an empty list if none exist.

STRICT REQUIREMENTS:
- Return ONLY valid JSON matching the schema.
- Do NOT include markdown fences, explanations, or extra text.
- All score values must be integers between 0 and 100.
- hiring_decision must be exactly one of: "Hire", "Maybe Hire", or "Reject".
- recruiter_reasoning must have between 3 and 5 items.
- resume_highlights must have at least 1 item.
    """,
    output_schema=RecruiterScoreSchema,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,  # Slightly higher than parser for nuanced reasoning
    ),
)


# ---------------------------------------------------------------------------
# Event-loop helper (shared pattern across all agents)
# ---------------------------------------------------------------------------

def _run_in_loop(coro):
    """Execute an async coroutine safely inside or outside a running event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        nest_asyncio.apply()
        return loop.run_until_complete(coro)

    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def score_candidate(resume_json: dict, api_key: Optional[str] = None) -> dict:
    """
    Evaluate a candidate's structured resume profile using the Scoring Agent.

    Args:
        resume_json: Structured candidate data returned by analyze_resume_text().
                     The agent does NOT re-parse the PDF.
        api_key:     Google Gemini API Key. Falls back to GEMINI_API_KEY env var.

    Returns:
        dict matching RecruiterScoreSchema.

    Raises:
        ValueError:   If no API key is available.
        RuntimeError: If the agent returns an empty response.
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "Google Gemini API Key is missing. "
            "Set GEMINI_API_KEY in your .env file or pass it explicitly."
        )
    os.environ["GEMINI_API_KEY"] = resolved_key

    resume_json_str = json.dumps(resume_json, indent=2, default=str)

    async def _run():
        session_service = InMemorySessionService()
        runner = Runner(
            app_name="AI_Recruiter",
            agent=scoring_adk_agent,
            session_service=session_service,
            auto_create_session=True,
        )

        result_text = ""
        async for event in runner.run_async(
            user_id="recruiter_pipeline",
            session_id="session_scoring",
            new_message=types.Content(
                parts=[types.Part(text=f"CANDIDATE PROFILE (Structured JSON):\n{resume_json_str}")]
            ),
        ):
            if event.content and event.content.parts:
                result_text = event.content.parts[0].text

        if not result_text:
            raise RuntimeError("Scoring Agent returned an empty response.")

        validated = RecruiterScoreSchema.model_validate_json(result_text)
        return validated.model_dump()

    return _run_in_loop(_run())
