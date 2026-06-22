"""
Recommendation Agent
====================
Generates personalised career recommendations by synthesising the outputs of
three upstream agents:

  • Resume Analysis Agent  → resume_json   (structured candidate profile)
  • Scoring Agent          → scores_json   (scores, decision, red flags)
  • Feedback Agent         → feedback_json (strengths, weaknesses, suggestions)

Responsibilities (this agent ONLY):
  - Produce actionable skill, course, certification, project, and interview tips.
  - Build a realistic 30-day career roadmap.
  - Does NOT re-parse the PDF, recalculate scores, or regenerate feedback.

Flow:
  1. Receive all three dicts from the Orchestrator.
  2. Serialise them into a single rich context prompt.
  3. Call Gemini via Google ADK with a strict output_schema.
  4. Validate and return a dict matching RecommendationSchema.
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

class CourseRecommendation(BaseModel):
    """A single course recommendation with provider and rationale."""

    title: str = Field(..., description="Full name of the course or learning module.")
    provider: str = Field(..., description="Platform or organisation, e.g. Coursera, Udemy, Google.")
    reason: str = Field(
        ...,
        description=(
            "Concise 1–2 sentence explanation of why this course directly addresses "
            "a gap identified in the candidate's profile."
        ),
    )


class CareerRoadmap(BaseModel):
    """A structured 4-week action plan."""

    week_1: str = Field(..., description="Focus area and concrete tasks for Week 1 (Days 1–7).")
    week_2: str = Field(..., description="Focus area and concrete tasks for Week 2 (Days 8–14).")
    week_3: str = Field(..., description="Focus area and concrete tasks for Week 3 (Days 15–21).")
    week_4: str = Field(..., description="Focus area and concrete tasks for Week 4 (Days 22–30).")


class RecommendationSchema(BaseModel):
    """Complete career growth recommendations returned by the Recommendation Agent."""

    recommended_skills: List[str] = Field(
        ...,
        description=(
            "Skills the candidate should learn next based on profile gaps, "
            "red flags, and current market demand."
        ),
    )
    recommended_courses: List[CourseRecommendation] = Field(
        ...,
        description=(
            "3–5 specific, industry-recognised courses from reputable platforms. "
            "Each must directly address a gap in the candidate's profile."
        ),
    )
    recommended_certifications: List[str] = Field(
        ...,
        description=(
            "2–4 industry-recognised certifications that would significantly boost "
            "this candidate's profile and ATS score."
        ),
    )
    recommended_projects: List[str] = Field(
        ...,
        description=(
            "3–5 concrete, buildable project ideas that fill skill or portfolio gaps. "
            "Each must name the project, technologies used, and gap it addresses."
        ),
    )
    interview_preparation: List[str] = Field(
        ...,
        description=(
            "4–6 targeted interview preparation tips based on the candidate's "
            "tech stack, experience level, and recruiter evaluation."
        ),
    )
    career_roadmap: CareerRoadmap = Field(
        ...,
        description=(
            "A realistic 30-day action plan broken into 4 weekly sprints, "
            "each with a clear theme and 2–3 concrete deliverables."
        ),
    )


# ---------------------------------------------------------------------------
# Google ADK Agent definition
# ---------------------------------------------------------------------------

recommendation_adk_agent = Agent(
    name="recommendation_agent",
    model="gemini-2.5-flash",
    instruction="""
You are a senior career coach and recruiter with 15+ years of experience
helping professionals across various industries land their next role.

You have been given three structured JSON documents produced by an AI recruiting pipeline:
  1. A parsed resume profile (raw candidate data).
  2. A recruiter score evaluation (scores, decision, red flags, reasoning).
  3. Recruiter feedback (strengths, weaknesses, improvement suggestions).

Your task is to synthesise all of this information and generate a PERSONALISED,
ACTIONABLE career growth plan that will maximise this candidate's chances of
landing their target role.

IMPORTANT: DOMAIN-AWARE RECOMMENDATION RULE
Identify the "Professional Domain" field in the parsed resume profile. You MUST customize all recommendations, projects, skills, courses, certifications, interview preparation tips, and the 30-day roadmap to match this specific industry/profession.
Never recommend software-engineering-specific tools or frameworks (such as AWS, Docker, Kubernetes, React, FastAPI, Git, software system design, or coding exercises) unless the candidate's professional domain is "Software Engineering", "Data Science", "Cybersecurity", "Cloud Computing", or another directly computer science/software-related field.

Specific Domain Customization Guidelines:
- Software Engineering / Cloud / Cybersecurity:
  - Skills: Backend/Frontend engineering, API design, DevOps pipelines, Docker, Kubernetes, AWS/GCP, FastAPI, React.
  - Certifications: AWS Certified Solutions Architect, Certified Kubernetes Administrator (CKA), Security+.
  - Projects: Building full-stack web applications, designing distributed systems, CI/CD pipelines.
  - Interview Prep: Data structures & algorithms, LeetCode, system design, behavioral tips.
  - Roadmap: Coding sprints, repository setup, algorithm review.
- Data Science / Business Analytics:
  - Skills: SQL, Python/R for data analysis, machine learning algorithms, statistics, data visualization (Tableau, Power BI), A/B testing.
  - Certifications: Google Data Analytics Professional, Certified Analytics Professional (CAP), AWS Machine Learning.
  - Projects: Exploratory data analysis (EDA) on public datasets (e.g. Kaggle), building ML prediction models, Tableau dashboards.
  - Interview Prep: SQL query challenges, machine learning concepts, data modeling, case studies.
  - Roadmap: Learning data visualization tools, practicing SQL, cleaning datasets.
- Biotechnology / Biomedical Engineering / Research:
  - Skills: Lab techniques (PCR, qPCR, ELISA, cell culture), assay validation, molecular biology, genomics, bioinformatics, chromatography (HPLC/FPLC), GMP/GLP compliance, scientific writing.
  - Certifications: Regulatory Affairs Certification (RAC), ASCP Board of Certification, GCP (Good Clinical Practice) Certification.
  - Projects: Designing clinical trial protocols, documenting lab workflows, writing mock research/review papers, developing QA/QC compliance checklists for a GMP manufacturing line.
  - Interview Prep: Laboratory safety protocols, regulatory compliance questions (FDA guidelines), troubleshooting experiments, scientific presentation preparation.
  - Roadmap: Studying lab guidelines, drafting scientific reports, learning bioinformatics software.
- Mechanical / Civil / Electrical Engineering:
  - Skills: CAD/CAM tools (SolidWorks, AutoCAD, CATIA), Finite Element Analysis (FEA/ANSYS), structural analysis (STAAD Pro), GD&T, manufacturing/construction design, safety codes (ASME, IBC, NEC).
  - Certifications: FE (Fundamentals of Engineering) exam prep, LEED Green Associate, SolidWorks Certifications (CSWA/CSWP), Project Management Professional (PMP).
  - Projects: 3D CAD modeling of physical parts, FEA simulation reports, mock construction scheduling/management plans, structural design blueprints.
  - Interview Prep: Engineering principles, portfolio reviews, structural/thermal problem-solving, project management case studies.
  - Roadmap: Creating CAD portfolios, conducting simulation studies, studying local building/manufacturing codes.
- Finance / Accounting:
  - Skills: Financial modeling, discounted cash flow (DCF) analysis, corporate valuation, corporate accounting (GAAP/IFRS), portfolio management, risk analysis, Excel (VBA/macros), Bloomberg terminal.
  - Certifications: Chartered Financial Analyst (CFA) Level I/II/III, Certified Public Accountant (CPA), Chartered Accountant (CA), Financial Risk Manager (FRM).
  - Projects: Building DCF valuation models, portfolio performance reports, mock equity research write-ups, tax/audit compliance reviews.
  - Interview Prep: Finance case studies, valuation methodologies, accounting entry questions, market trends discussion.
  - Roadmap: Building Excel financial templates, reading financial statements, practicing valuation drills.
- Marketing / Sales / HR / Business Administration:
  - Skills: SEO, Google Analytics, content marketing, search engine marketing (SEM), brand strategy, CRM tools (Salesforce), sourcing/recruiting pipelines, payroll administration, employee relations.
  - Certifications: Google Analytics Individual Qualification (GAIQ), HubSpot Inbound Marketing, Professional in Human Resources (PHR), SHRM-CP.
  - Projects: Designing complete marketing campaigns, competitive market analyses, mock sourcing campaigns, HR policy manual templates.
  - Interview Prep: Marketing case studies, sales pitches, CRM scenarios, structured behavioral questions.
  - Roadmap: Analyzing industry metrics, drafting marketing/HR templates, building CRM dashboards.
- Healthcare / Nursing:
  - Skills: Clinical research, patient care protocols, medical terminology, HIPAA compliance, electronic medical records (EMR) systems, healthcare administration.
  - Certifications: Clinical Research Coordinator (CCRC), Certified Professional in Healthcare Quality (CPHQ), BLS/ACLS.
  - Projects: Designing healthcare quality improvement plans, hospital workflow optimization models, patient education material drafting.
  - Interview Prep: Clinical case scenario discussions, HIPAA guidelines, ethics-based situational questions.
  - Roadmap: Learning EMR systems, preparing clinical cheat-sheets, studying patient safety protocols.
- Unknown / General Professional:
  - If the domain is unknown, general, or unrecognized, focus on general, domain-agnostic professional development.
  - Recommend generic courses (e.g. project management, communication, leadership, business analysis) from Coursera or LinkedIn Learning.
  - Recommend general certifications (e.g. CAPM, PMP, Scrum Master).
  - Recommend general projects (e.g. process mapping, data analyses of spreadsheets, operational efficiency reports).
  - Do NOT suggest software-specific or programming skills.

RECOMMENDATION INSTRUCTIONS:

1. recommended_skills  (4-8 items)
   - Identify the most impactful skills this candidate should acquire next.
   - Prioritise skills that appear as red flags or gaps in scoring and feedback.
   - Include skills highly in-demand for the candidate's apparent target role.
   - Be specific: e.g. "HPLC/FPLC purification protocols" (Biotech) or "Financial modeling and DCF analysis" (Finance), not generic terms.

2. recommended_courses  (3-5 objects with title, provider, reason)
   - Recommend well-known courses from reputable platforms matching their domain (Coursera, Udemy, edX, LinkedIn Learning, or industry-specific sites).
   - Each must directly address a gap identified in the profile.
   - In the "reason" field, clearly link the course to the specific gap it closes.

3. recommended_certifications  (2-4 items)
   - Focus on certifications that would directly improve ATS score and credibility in their specific domain.

4. recommended_projects  (3-5 items)
   - Each description must: name the project, specify tools/technologies, explain the gap addressed.

5. interview_preparation  (4-6 items)
   - Be specific to the candidate's level and domain - not generic advice.

6. career_roadmap  (week_1, week_2, week_3, week_4)
   - Design a realistic, focused 30-day sprint.
   - Each week must have: a clear theme, 2-3 concrete measurable deliverables relevant to their domain.
   - Build progressively so each week builds on the previous one.

STRICT REQUIREMENTS:
- Return ONLY valid JSON matching this exact schema (no markdown fences, no extra text):
{
  "recommended_skills": [string, ...],
  "recommended_courses": [
    {"title": string, "provider": string, "reason": string},
    ...
  ],
  "recommended_certifications": [string, ...],
  "recommended_projects": [string, ...],
  "interview_preparation": [string, ...],
  "career_roadmap": {
    "week_1": string,
    "week_2": string,
    "week_3": string,
    "week_4": string
  }
}
- recommended_courses objects must have exactly: title, provider, reason.
- career_roadmap must have exactly: week_1, week_2, week_3, week_4.
- All text must be complete, professional sentences.
- Do NOT repeat advice from the recruiter feedback verbatim.
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.35,  # Slightly warmer for creative, varied recommendations
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

def generate_recommendations(
    resume_json: dict,
    scores_json: dict,
    feedback_json: dict,
    api_key: Optional[str] = None,
) -> dict:
    """
    Generate personalised career recommendations using outputs from all three
    prior agents via the Recommendation ADK Agent.

    Args:
        resume_json:   Structured candidate profile from Resume Analysis Agent.
        scores_json:   Scoring evaluation from Recruiter Scoring Agent.
        feedback_json: Recruiter feedback from Feedback Agent.
        api_key:       Google Gemini API Key. Falls back to GEMINI_API_KEY env var.

    Returns:
        dict matching RecommendationSchema.

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

    resume_str   = json.dumps(resume_json,   indent=2, default=str)
    scores_str   = json.dumps(scores_json,   indent=2, default=str)
    feedback_str = json.dumps(feedback_json, indent=2, default=str)

    async def _run():
        session_service = InMemorySessionService()
        runner = Runner(
            app_name="AI_Recruiter",
            agent=recommendation_adk_agent,
            session_service=session_service,
            auto_create_session=True,
        )

        prompt_input = (
            f"CANDIDATE RESUME PROFILE:\n{resume_str}\n\n"
            f"RECRUITER SCORE EVALUATION:\n{scores_str}\n\n"
            f"RECRUITER FEEDBACK:\n{feedback_str}"
        )
        result_text = ""
        async for event in runner.run_async(
            user_id="recruiter_pipeline",
            session_id="session_recommendation",
            new_message=types.Content(parts=[types.Part(text=prompt_input)]),
        ):
            if event.content and event.content.parts:
                result_text = event.content.parts[0].text

        if not result_text:
            raise RuntimeError("Recommendation Agent returned an empty response.")

        # Strip markdown fences if the model added them anyway
        clean = result_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        validated = RecommendationSchema.model_validate_json(clean)
        return validated.model_dump()

    return _run_with_retry(_run)
