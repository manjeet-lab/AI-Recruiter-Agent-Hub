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
import time
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
You are an experienced recruiter and hiring manager with 15+ years of experience
evaluating candidates across various industries.

You have been given a structured JSON resume profile extracted by an automated
parser. Your job is to evaluate this candidate exactly as a real recruiter would
during the initial screening phase.

IMPORTANT: DOMAIN-AWARE EVALUATION RULE
Locate the "Professional Domain" field in the structured profile. You must evaluate the candidate
strictly using the standards, expected skills, and typical profiles of that specific domain.
DO NOT use software-engineering/programming criteria unless the detected domain is "Software Engineering", "Cybersecurity", "Cloud Computing", or another directly software-related field.

Specific Domain Evaluation Guidelines:
- Software Engineering: Evaluate programming languages (Python, Java, etc.), framework knowledge (React, FastAPI), system design, software engineering practices, projects, and tech certifications.
- Biotechnology / Biomedical Engineering / Research / Healthcare: Evaluate laboratory skills (PCR, chromatography, assays, cell culture), clinical exposure, research experience, academic publications/patents, GMP/GLP compliance, and scientific tools. Do NOT look for coding/GitHub.
- Business Analytics / Data Science: Evaluate SQL, Excel, data visualization (Power BI, Tableau), statistical modeling, and data-driven problem-solving.
- Mechanical / Civil / Electrical Engineering: Evaluate CAD/CAM software (SolidWorks, AutoCAD), manufacturing/design processes, structural analysis tools (STAAD Pro, ANSYS), simulations, and physical project implementation.
- Finance / Accounting: Evaluate financial modeling, valuation, corporate accounting standards, certifications (CFA, CPA), Bloomberg terminal, and quantitative analysis skills.
- Marketing / HR / Sales / Business: Evaluate communication, campaign KPIs, SEO, brand strategy, stakeholder management, recruitment/sourcing strategies, and business operations.
- Other / Unknown / General Professional: If the domain is unknown, general, or unrecognized, evaluate using neutral professional criteria such as quality of career progression, leadership skills, academic prestige, verbal communication clarity, and business impact. Do NOT default to software engineering or programming expectations.

EVALUATION INSTRUCTIONS:

1. overall_recruiter_score (0-100)
   Holistically rate the candidate's profile quality considering all factors:
   technical/domain depth, experience level, project/research quality, communication clarity,
   education relevance, and professional completeness.

2. ats_score (0-100)
   Estimate how well this resume would survive an ATS filter for their specific industry.
   Consider: industry-specific keyword richness, standard section names, completeness of contact info,
   measurable achievements, and absence of formatting red flags.

3. hiring_decision
   Choose EXACTLY ONE: "Hire", "Maybe Hire", or "Reject".
   Base this on whether you would move this candidate to the next round for a role in their domain.

4. confidence_score (0-100)
   How confident are you in your evaluation given the data available?
   Low confidence when critical fields are missing; high when the profile is detailed.

5. category_scores — score each independently (0-100):
   - technical_skills : Depth, breadth, and relevance of professional/technical skills for their specific domain.
   - projects         : Quality, complexity, impact, and presentation of projects, designs, research, or models.
   - experience       : Relevance, seniority, tenure, and impact of work history in their specific profession.
   - communication    : Clarity, action verbs, measurable results, professionalism.
   - education        : Relevance and quality of academic credentials for their domain.

6. recruiter_summary
   Write a professional 3-5 sentence narrative a recruiter would put in a scorecard.
   Mention the candidate by name, their primary professional domain, their strongest area, and any notable concerns.

7. recruiter_reasoning
   Provide 3-5 concise bullet-point reasons that justify your overall evaluation, tailored to their domain.

8. resume_highlights
   List the 3-5 strongest positives noticed in the profile.

9. red_flags
   List weaknesses, gaps, or missing elements relevant to their profession. Return an empty list if none exist.

STRICT REQUIREMENTS:
- Return ONLY valid JSON matching this exact schema (no markdown fences, no extra text):
{
  "overall_recruiter_score": integer 0-100,
  "ats_score": integer 0-100,
  "hiring_decision": "Hire" | "Maybe Hire" | "Reject",
  "confidence_score": integer 0-100,
  "category_scores": {
    "technical_skills": integer 0-100,
    "projects": integer 0-100,
    "experience": integer 0-100,
    "communication": integer 0-100,
    "education": integer 0-100
  },
  "recruiter_summary": string,
  "recruiter_reasoning": [string, ...],
  "resume_highlights": [string, ...],
  "red_flags": [string, ...]
}
- All score values must be integers between 0 and 100.
- hiring_decision must be exactly one of: "Hire", "Maybe Hire", or "Reject".
- recruiter_reasoning must have between 3 and 5 items.
- resume_highlights must have at least 1 item.
    """,
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


def _run_with_retry(coro_factory, max_retries: int = 3, base_delay: float = 5.0):
    """
    Run a coroutine factory with exponential backoff retry logic.
    Retries on 503 UNAVAILABLE (model overload) errors from the Gemini API.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return _run_in_loop(coro_factory())
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            if "503" in err_str or "UNAVAILABLE" in err_str:
                if attempt < max_retries:
                    wait = base_delay * (2 ** attempt)
                    print(f"[Retry {attempt + 1}/{max_retries}] Model overloaded (503). "
                          f"Waiting {wait:.0f}s before retry...")
                    time.sleep(wait)
                    continue
            raise
    raise last_exc


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

        # Strip markdown fences if the model added them anyway
        clean = result_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        validated = RecruiterScoreSchema.model_validate_json(clean)
        return validated.model_dump()

    return _run_with_retry(_run)
