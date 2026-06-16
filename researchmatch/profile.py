from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ResearchGoal(str, Enum):
    PUBLICATION = "publication"
    EXPERIENCE = "experience"
    MED_SCHOOL = "med-school"
    CS_RESEARCH = "CS-research"
    EXPLORE = "explore"


@dataclass
class StudentProfile:
    interests: str
    skills: list[str] = field(default_factory=list)
    courses: list[str] = field(default_factory=list)
    goal: ResearchGoal = ResearchGoal.EXPLORE
    time_availability: str = ""

    def validate(self) -> None:
        if len(self.interests.strip()) < 8:
            raise ValueError("Add a little more detail about your research interests.")
        if not isinstance(self.skills, list) or not isinstance(self.courses, list):
            raise ValueError("Skills and courses must be lists.")

    def text_for_matching(self) -> str:
        return " ".join(
            [
                self.interests,
                "skills: " + ", ".join(self.skills),
                "courses: " + ", ".join(self.courses),
                "goal: " + self.goal.value,
                "availability: " + self.time_availability,
            ]
        ).strip()

    def normalized_terms(self) -> set[str]:
        pieces = []
        pieces.extend(self.interests.replace("/", " ").replace(",", " ").split())
        pieces.extend(self.skills)
        pieces.extend(self.courses)
        return {normalize_term(piece) for piece in pieces if normalize_term(piece)}


def parse_csvish(value: str | list[str] | None) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [item.strip() for item in value if str(item).strip()]
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def normalize_term(value: str) -> str:
    return " ".join(
        "".join(ch.lower() if ch.isalnum() else " " for ch in value).split()
    )


if __name__ == "__main__":
    profile = StudentProfile(
        interests="I want to work on machine learning for medical images and health data.",
        skills=["Python", "machine learning", "statistics"],
        courses=["data structures", "biology"],
        goal=ResearchGoal.CS_RESEARCH,
        time_availability="6-8 hours/week",
    )
    profile.validate()
    print(profile)
    print(profile.text_for_matching())
