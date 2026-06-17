from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import Professor, load_professors, professor_embedding_text
from .profile import ResearchGoal, StudentProfile, normalize_term


SEMANTIC_WEIGHT = 0.70
TAG_WEIGHT = 0.26
PRIORITY_BOOST = 0.04  # applied to professors flagged researchmatch_priority="high"


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )


class TfidfEmbedder:
    """Small fallback used for local smoke tests when sentence-transformers is absent."""

    name = "tfidf-fallback"

    def encode_pair(self, professor_texts: list[str], student_text: str) -> tuple[np.ndarray, np.ndarray]:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(professor_texts + [student_text])
        return matrix[:-1], matrix[-1:]

    def encode(self, texts: list[str]) -> np.ndarray:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        return vectorizer.fit_transform(texts).toarray()


@dataclass
class MatchResult:
    professor: Professor
    final_score: float
    semantic_score: float
    tag_score: float
    matched_terms: list[str]


def default_embedder(prefer_sentence_transformers: bool = True) -> Embedder:
    if prefer_sentence_transformers:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            pass
    return TfidfEmbedder()


def cache_key(professors: list[Professor], embedder_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(embedder_name.encode("utf-8"))
    for prof in professors:
        digest.update(professor_embedding_text(prof).encode("utf-8"))
    return digest.hexdigest()[:16]


def get_professor_embeddings(
    professors: list[Professor],
    embedder: Embedder,
    cache_dir: str | Path = ".cache",
) -> np.ndarray | None:
    if isinstance(embedder, TfidfEmbedder):
        return None
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    file_path = cache_path / f"professor_embeddings_{cache_key(professors, embedder.name)}.pkl"
    if file_path.exists():
        with file_path.open("rb") as handle:
            return pickle.load(handle)
    embeddings = embedder.encode([professor_embedding_text(prof) for prof in professors])
    with file_path.open("wb") as handle:
        pickle.dump(embeddings, handle)
    return embeddings


def professor_terms(professor: Professor) -> set[str]:
    terms = set()
    for value in professor.research_areas + professor.skills_relevant:
        norm = normalize_term(value)
        if norm:
            terms.add(norm)
        for token in norm.split():
            if len(token) > 2:
                terms.add(token)
    return terms


def tag_overlap(profile: StudentProfile, professor: Professor) -> tuple[float, list[str]]:
    student_terms = profile.normalized_terms()
    prof_terms = professor_terms(professor)
    direct = student_terms & prof_terms
    phrase_hits = {
        term for term in prof_terms if len(term) > 3 and term in normalize_term(profile.text_for_matching())
    }
    hits = sorted(direct | phrase_hits)
    denominator = max(4, min(12, len(prof_terms)))
    return min(1.0, len(hits) / denominator), hits[:12]


def match_professors(
    profile: StudentProfile,
    professors: list[Professor],
    top_k: int = 15,
    embedder: Embedder | None = None,
    cache_dir: str | Path = ".cache",
    prioritize_interest: bool = False,
) -> list[MatchResult]:
    profile.validate()
    if not professors:
        return []

    # Only show professors with publications in 2024 or later
    active = [p for p in professors if p.has_recent_pubs]
    if not active:
        active = professors  # fallback: show all if none pass the filter

    embedder = embedder or default_embedder()
    professor_texts = [professor_embedding_text(prof) for prof in active]
    student_text = profile.text_for_matching()

    if isinstance(embedder, TfidfEmbedder):
        prof_embeddings, student_embedding = embedder.encode_pair(professor_texts, student_text)
        semantic_scores = cosine_similarity(prof_embeddings, student_embedding).ravel()
    else:
        prof_embeddings = get_professor_embeddings(active, embedder, cache_dir=cache_dir)
        student_embedding = embedder.encode([student_text])
        semantic_scores = cosine_similarity(prof_embeddings, student_embedding).ravel()

    results: list[MatchResult] = []
    for professor, semantic in zip(active, semantic_scores):
        overlap_score, matched_terms = tag_overlap(profile, professor)
        priority_bonus = PRIORITY_BOOST if professor.researchmatch_priority == "high" else 0.0
        final_score = SEMANTIC_WEIGHT * float(semantic) + TAG_WEIGHT * overlap_score + priority_bonus
        results.append(
            MatchResult(
                professor=professor,
                final_score=max(0.0, min(1.0, final_score)),
                semantic_score=max(0.0, min(1.0, float(semantic))),
                tag_score=overlap_score,
                matched_terms=matched_terms,
            )
        )

    # Pull top K by keyword/semantic fit first
    top = sorted(results, key=lambda r: r.final_score, reverse=True)[:top_k]

    if prioritize_interest:
        # Sort by relevance score (best fit first)
        return sorted(top, key=lambda r: r.final_score, reverse=True)
    else:
        # Default: sort by h-index (highest first); h=None sorts last
        return sorted(top, key=lambda r: (r.professor.h_index is not None, r.professor.h_index or 0), reverse=True)


def _self_test() -> None:
    data_file = Path(__file__).resolve().parents[1] / "professors.json"
    professors = load_professors(data_file)
    profile = StudentProfile(
        interests="I want to use Python, machine learning, and computer vision for medical imaging or health data.",
        skills=["Python", "machine learning", "computer vision", "data analysis"],
        courses=["data structures", "statistics", "biology"],
        goal=ResearchGoal.CS_RESEARCH,
        time_availability="6-8 hours/week",
    )
    results = match_professors(profile, professors, top_k=5, embedder=TfidfEmbedder())
    print("Top matches for ML/computer vision/health profile:")
    for result in results:
        print(
            f"{result.professor.name} | {result.professor.dept} | "
            f"final={result.final_score:.3f} semantic={result.semantic_score:.3f} "
            f"tags={result.tag_score:.3f} terms={', '.join(result.matched_terms)}"
        )
    top_text = " ".join(results[0].professor.research_areas + results[0].professor.skills_relevant).lower()
    assert any(term in top_text for term in ("machine learning", "computer vision", "health", "medical", "bioinformatics"))


if __name__ == "__main__":
    _self_test()
