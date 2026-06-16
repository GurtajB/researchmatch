from __future__ import annotations

from functools import lru_cache

from .data import Professor
from .matching import MatchResult
from .profile import StudentProfile


MODEL_NAME = "google/flan-t5-base"


@lru_cache(maxsize=1)
def _generator():
    from transformers import pipeline

    return pipeline("text2text-generation", model=MODEL_NAME, device=-1)


def _run_local_model(prompt: str, max_new_tokens: int = 120) -> str:
    pipe = _generator()
    result = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return result[0]["generated_text"].strip()


def heuristic_fit_line(profile: StudentProfile, result: MatchResult) -> str:
    terms = ", ".join(result.matched_terms[:5]) or "your stated interests"
    areas = ", ".join(result.professor.research_areas[:3])
    return f"This is a close fit to your profile because it connects {terms} with public research areas such as {areas}."


def generate_fit_line(profile: StudentProfile, result: MatchResult, use_local_llm: bool = False) -> str:
    if not use_local_llm:
        return heuristic_fit_line(profile, result)
    prompt = (
        "Write one factual sentence explaining why this professor's public research areas fit the student's "
        "interests. Do not mention availability or whether the professor accepts students.\n"
        f"Student: {profile.text_for_matching()}\n"
        f"Professor: {result.professor.name}, areas: {', '.join(result.professor.research_areas)}, "
        f"summary: {result.professor.summary}\n"
        "Sentence:"
    )
    try:
        return _run_local_model(prompt, max_new_tokens=80)
    except Exception:
        return heuristic_fit_line(profile, result)


def email_scaffold(profile: StudentProfile, professor: Professor, use_local_llm: bool = False) -> str:
    paper = professor.representative_papers[0]["title"] if professor.representative_papers else ""
    area = professor.research_areas[0] if professor.research_areas else "your research"
    skills = ", ".join(profile.skills[:5]) or "my current coursework and willingness to learn"
    courses = ", ".join(profile.courses[:4]) or "my recent coursework"
    paper_line = (
        f"I was especially interested in your paper/work on \"{paper}\"."
        if paper
        else f"I was especially interested in your public work on {area}."
    )

    fallback = f"""Subject: Research inquiry about {area}

Dear Professor {professor.name.split()[-1]},

My name is [Your Name], and I am a TAMS/UNT student interested in {profile.interests.strip()}.
{paper_line} I am reaching out because that work connects with my background in {skills} and courses such as {courses}.

I would appreciate the chance to ask whether there may be an appropriate way for me to learn more about your group's work, such as reading a recommended paper, attending a public lab meeting if available, or meeting briefly to discuss fit. I understand availability is unknown, and I will verify current projects before assuming there are openings.

Thank you for your time,
[Your Name]
"""
    if not use_local_llm:
        return fallback

    prompt = (
        "Draft an individualized, respectful research inquiry email scaffold. It must be edited by the student, "
        "must not claim the professor is accepting students, and must not sound reusable for many professors.\n"
        f"Professor: {professor.name}; areas: {', '.join(professor.research_areas)}; summary: {professor.summary}; "
        f"paper: {paper or 'none listed'}\n"
        f"Student: {profile.text_for_matching()}\n"
        "Email scaffold:"
    )
    try:
        generated = _run_local_model(prompt, max_new_tokens=220)
        if "accepting" in generated.lower() or "opening" in generated.lower():
            return fallback
        return generated
    except Exception:
        return fallback


if __name__ == "__main__":
    from pathlib import Path

    from .data import load_professors
    from .matching import TfidfEmbedder, match_professors
    from .profile import ResearchGoal

    professors = load_professors(Path(__file__).resolve().parents[1] / "professors.json")
    student = StudentProfile(
        interests="machine learning for health and medical data",
        skills=["Python", "machine learning"],
        courses=["biology", "statistics"],
        goal=ResearchGoal.CS_RESEARCH,
        time_availability="5 hours/week",
    )
    result = match_professors(student, professors, top_k=1, embedder=TfidfEmbedder())[0]
    print(generate_fit_line(student, result, use_local_llm=False))
    print(email_scaffold(student, result.professor, use_local_llm=False))
