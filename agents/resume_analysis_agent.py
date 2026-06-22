"""
Resume Analysis Agent
=====================
Extracts structured candidate information from cleaned resume text.

Responsibilities (this agent ONLY):
  - Parse and categorize resume fields into a strongly-typed JSON schema.
  - Does NOT score, evaluate, or give feedback.

Flow:
  1. Receive plain-text resume content from the Orchestrator.
  2. Call Gemini via Google ADK with a strict output_schema.
  3. Validate and return a dict matching ResumeSchema.
"""

import os
import asyncio
import time
from typing import List, Optional

import nest_asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from pydantic import BaseModel, Field

# Load .env file so GEMINI_API_KEY is available when running outside Streamlit.
load_dotenv()


# ---------------------------------------------------------------------------
# Pydantic Schemas — define the exact JSON shape the agent must produce
# ---------------------------------------------------------------------------

class EducationItem(BaseModel):
    degree: Optional[str] = Field(None, description="Degree, e.g. B.S., M.S., Ph.D.")
    institution: Optional[str] = Field(None, description="University, College, or School name")
    graduation_year: Optional[str] = Field(None, description="Graduation year or date range")
    field_of_study: Optional[str] = Field(None, description="Major, concentration, or specialization")


class ExperienceItem(BaseModel):
    job_title: Optional[str] = Field(None, description="Title of the job/role")
    company: Optional[str] = Field(None, description="Name of the company or organization")
    start_date: Optional[str] = Field(None, description="Start date, e.g. MM/YYYY or Year")
    end_date: Optional[str] = Field(None, description="End date, e.g. MM/YYYY, Year, or 'Present'")
    description: List[str] = Field([], description="Bullet points summarizing responsibilities and achievements")


class ProjectItem(BaseModel):
    title: Optional[str] = Field(None, description="Name/title of the project")
    description: Optional[str] = Field(None, description="Brief summary of what the project does")
    technologies: List[str] = Field([], description="Tools, frameworks, and languages used")
    url: Optional[str] = Field(None, description="Link/URL to the project repository or demo")


class ResumeSchema(BaseModel):
    """Structured candidate profile extracted by the Resume Analysis Agent."""

    # Field aliases match the exact JSON keys used by downstream agents and the UI.
    full_name: Optional[str] = Field(None, alias="Full Name")
    email: Optional[str] = Field(None, alias="Email")
    phone_number: Optional[str] = Field(None, alias="Phone Number")
    professional_domain: Optional[str] = Field(None, alias="Professional Domain")
    skills: List[str] = Field([], alias="Skills")
    education: List[EducationItem] = Field([], alias="Education")
    work_experience: List[ExperienceItem] = Field([], alias="Work Experience")
    projects: List[ProjectItem] = Field([], alias="Projects")
    certifications: List[str] = Field([], alias="Certifications")
    technical_skills: List[str] = Field([], alias="Technical Skills")
    soft_skills: List[str] = Field([], alias="Soft Skills")
    languages: List[str] = Field([], alias="Languages")
    github_url: Optional[str] = Field(None, alias="GitHub URL")
    linkedin_url: Optional[str] = Field(None, alias="LinkedIn URL")

    class Config:
        populate_by_name = True


# ---------------------------------------------------------------------------
# Google ADK Agent definition
# ---------------------------------------------------------------------------

resume_analysis_adk_agent = Agent(
    name="resume_analysis_agent",
    model="gemini-2.5-flash",
    instruction="""
    You are an expert ATS (Applicant Tracking System) parser and recruiting assistant.
    Your task is to analyze the resume text provided by the user and extract all structural
    candidate details accurately. Missing fields must be set to null or empty lists as
    specified by the schema. Do not infer information that is not explicitly stated.

    Additionally, you must infer the candidate's primary professional domain from their education,
    work experience, projects, certifications, and skills. If multiple domains apply, select the single
    strongest/most dominant one.
    Examples of professional domains:
    - Software Engineering
    - Data Science
    - Business Analytics
    - Biotechnology
    - Biomedical Engineering
    - Mechanical Engineering
    - Civil Engineering
    - Electrical Engineering
    - Finance
    - Accounting
    - Marketing
    - Human Resources
    - Healthcare
    - Research
    - Education
    - Law
    - Sales
    - Product Management
    - Cybersecurity
    - Cloud Computing
    - UI/UX Design
    If the domain cannot be clearly identified, return a general/neutral professional domain descriptor (e.g. "General Professional" or similar) or null, but do not default to Software Engineering.

    Return ONLY a valid JSON object matching this schema (no markdown fences, no extra text):
    {
      "Full Name": string or null,
      "Email": string or null,
      "Phone Number": string or null,
      "Professional Domain": string or null,
      "Skills": [string, ...],
      "Education": [{"degree": ..., "institution": ..., "graduation_year": ..., "field_of_study": ...}, ...],
      "Work Experience": [{"job_title": ..., "company": ..., "start_date": ..., "end_date": ..., "description": [...]}, ...],
      "Projects": [{"title": ..., "description": ..., "technologies": [...], "url": ...}, ...],
      "Certifications": [string, ...],
      "Technical Skills": [string, ...],
      "Soft Skills": [string, ...],
      "Languages": [string, ...],
      "GitHub URL": string or null,
      "LinkedIn URL": string or null
    }
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,  # Low temperature for highly deterministic, consistent parsing
    ),
)


# ---------------------------------------------------------------------------
# Event-loop helper (shared pattern across all agents)
# ---------------------------------------------------------------------------

def _run_in_loop(coro):
    """
    Execute an async coroutine safely regardless of whether an event loop is
    already running (e.g. inside Streamlit's own loop).
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Streamlit runs its own event loop; nest_asyncio patches it so we can
        # call run_until_complete() from within an already-running loop.
        nest_asyncio.apply()
        return loop.run_until_complete(coro)

    return asyncio.run(coro)


def _run_with_retry(coro_factory, max_retries: int = 3, base_delay: float = 5.0):
    """
    Run a coroutine factory with exponential backoff retry logic.
    Retries on 503 UNAVAILABLE (model overload) errors from the Gemini API.

    Args:
        coro_factory: A zero-argument callable that returns a fresh coroutine each call.
        max_retries:  Number of retry attempts after the first failure (default: 3).
        base_delay:   Base wait time in seconds; doubles each retry (5s, 10s, 20s...).
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
            raise  # Non-retryable error — re-raise immediately
    raise last_exc


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def analyze_resume_text(resume_text: str, api_key: Optional[str] = None) -> dict:
    """
    Invoke the Resume Analysis ADK Agent to extract a structured candidate profile.

    Args:
        resume_text: Cleaned and normalised plain-text content from a resume PDF.
        api_key:     Google Gemini API Key.  Falls back to the GEMINI_API_KEY
                     environment variable when omitted.

    Returns:
        dict matching ResumeSchema (keys use the alias names, e.g. "Full Name").

    Raises:
        ValueError:  If no API key is available.
        RuntimeError: If the agent returns an empty response.
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "Google Gemini API Key is missing. "
            "Set GEMINI_API_KEY in your .env file or pass it explicitly."
        )
    os.environ["GEMINI_API_KEY"] = resolved_key

    async def _run():
        session_service = InMemorySessionService()
        runner = Runner(
            app_name="AI_Recruiter",
            agent=resume_analysis_adk_agent,
            session_service=session_service,
            auto_create_session=True,
        )

        result_text = ""
        async for event in runner.run_async(
            user_id="recruiter_pipeline",
            session_id="session_analysis",
            new_message=types.Content(parts=[types.Part(text=resume_text)]),
        ):
            if event.content and event.content.parts:
                result_text = event.content.parts[0].text

        if not result_text:
            raise RuntimeError("Resume Analysis Agent returned an empty response.")

        # Strip markdown fences if the model added them anyway
        clean = result_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        import json as _json
        raw = _json.loads(clean)
        parsed = ResumeSchema.model_validate(raw)
        return parsed.model_dump(by_alias=True)

    return _run_with_retry(_run)
