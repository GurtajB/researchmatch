from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id": str,
    "name": str,
    "dept": str,
    "research_areas": list,
    "summary": str,
    "representative_papers": list,
    "lab_url": str,
    "public_email": str,
    "objective_signals": list,
    "skills_relevant": list,
}


@dataclass
class Professor:
    id: str
    name: str
    dept: str
    research_areas: list[str]
    summary: str
    representative_papers: list[dict[str, Any]]
    lab_url: str = ""
    public_email: str = ""
    objective_signals: list[str] = field(default_factory=list)
    skills_relevant: list[str] = field(default_factory=list)
    title: str = ""
    recent_publications_status: str = ""
    researchmatch_priority: str = ""
    source_url: str = ""
    curation_notes: str = ""
    sample: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Professor":
        validate_professor(raw)
        known = {field_name for field_name in cls.__dataclass_fields__}
        clean = {key: value for key, value in raw.items() if key in known}
        return cls(**clean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "dept": self.dept,
            "title": self.title,
            "research_areas": self.research_areas,
            "summary": self.summary,
            "representative_papers": self.representative_papers,
            "lab_url": self.lab_url,
            "public_email": self.public_email,
            "objective_signals": self.objective_signals,
            "skills_relevant": self.skills_relevant,
            "recent_publications_status": self.recent_publications_status,
            "researchmatch_priority": self.researchmatch_priority,
            "source_url": self.source_url,
            "curation_notes": self.curation_notes,
            "sample": self.sample,
        }


def validate_professor(raw: dict[str, Any]) -> None:
    if not isinstance(raw, dict):
        raise ValueError("Each professor entry must be a JSON object.")
    for field_name, field_type in REQUIRED_FIELDS.items():
        if field_name not in raw:
            raise ValueError(f"Missing required field '{field_name}' in {raw.get('id', '<unknown>')}.")
        if not isinstance(raw[field_name], field_type):
            raise ValueError(
                f"Field '{field_name}' in {raw.get('id', '<unknown>')} must be {field_type.__name__}."
            )
    for list_field in ("research_areas", "objective_signals", "skills_relevant"):
        if not all(isinstance(item, str) for item in raw[list_field]):
            raise ValueError(f"Field '{list_field}' must contain only strings in {raw['id']}.")
    for paper in raw["representative_papers"]:
        if not isinstance(paper, dict) or "title" not in paper:
            raise ValueError(f"Each representative paper needs at least a title in {raw['id']}.")


def load_professors(path: str | Path = "professors.json") -> list[Professor]:
    data_path = Path(path)
    with data_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("professors.json must contain a list of professor objects.")
    professors = [Professor.from_dict(item) for item in raw]
    ids = [prof.id for prof in professors]
    if len(ids) != len(set(ids)):
        raise ValueError("Professor ids must be unique.")
    return professors


def save_professors(professors: list[Professor], path: str | Path = "professors.json") -> None:
    data_path = Path(path)
    with data_path.open("w", encoding="utf-8") as handle:
        json.dump([prof.to_dict() for prof in professors], handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def add_professor(path: str | Path, entry: dict[str, Any]) -> None:
    validate_professor(entry)
    professors = load_professors(path)
    if any(prof.id == entry["id"] for prof in professors):
        raise ValueError(f"Professor id already exists: {entry['id']}")
    professors.append(Professor.from_dict(entry))
    save_professors(professors, path)


def professor_embedding_text(professor: Professor) -> str:
    paper_titles = [str(paper.get("title", "")) for paper in professor.representative_papers]
    return " ".join(
        [
            professor.name,
            professor.dept,
            " ".join(professor.research_areas),
            professor.summary,
            " ".join(paper_titles),
        ]
    ).strip()


if __name__ == "__main__":
    data_file = Path(__file__).resolve().parents[1] / "professors.json"
    professors = load_professors(data_file)
    print(f"Loaded {len(professors)} professor records")
    print(professors[0].to_dict())
