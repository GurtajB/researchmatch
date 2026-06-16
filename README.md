---
title: ResearchMatch
emoji: 🔬
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
short_description: Match UNT undergrads to professors by research fit
---

# ResearchMatch

ResearchMatch is a free, local Gradio app that helps TAMS / early-college and UNT undergrads find UNT professors whose public research areas match their stated interests, skills, coursework, and goals.

It ranks only fit-to-student. It does not rate professors as people and never claims a professor is accepting students.

## Run Locally

```bash
cd researchmatch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL printed in the terminal.

## Hugging Face Space Deploy

1. Create a free Hugging Face Space with the Gradio SDK.
2. Upload `app.py`, `requirements.txt`, `README.md`, the `researchmatch/` package folder, and `professors.json`.
3. The Space will install the pinned packages and launch `app.py`.

CPU is supported. Core matching uses local sentence-transformer embeddings. The FLAN-T5 wording features are optional and off by default because they can be slower on free CPU.

## Data Maintenance

`professors.json` is the source of truth. Each entry must include:

```json
{
  "id": "unique_id",
  "name": "Professor Name",
  "dept": "Department",
  "research_areas": ["machine learning", "bioinformatics"],
  "summary": "Plain-English public summary of the lab's work.",
  "representative_papers": [
    {"title": "Paper title", "year": 2025, "url": "https://..."}
  ],
  "lab_url": "https://...",
  "public_email": "public.email@unt.edu",
  "objective_signals": ["lab page lists projects - verify before outreach"],
  "skills_relevant": ["Python", "statistics"]
}
```

Extra audit fields such as `recent_publications_status`, `researchmatch_priority`, `source_url`, and `curation_notes` are allowed.

Do not add private information. Do not infer availability. Use only public, factual information and label signals as things the student must verify.

## Module Self-Tests

```bash
cd researchmatch
python -m researchmatch.data
python -m researchmatch.profile
python -m researchmatch.matching
python -m researchmatch.local_text
```

The matching self-test uses a TF-IDF fallback so it can run before `sentence-transformers` is installed. In deployment, the app uses `sentence-transformers/all-MiniLM-L6-v2` and caches professor embeddings on disk.

## What To Test With Real Users

- Students understand that rankings mean "closest fit to me," not professor quality.
- Students notice and understand the availability disclaimer.
- Top matches feel relevant for common profiles: bioinformatics, pre-med wet lab, AI/data science, cybersecurity, materials/chemistry, robotics/mechanical design, psychology/neuroscience.
- The score breakdown helps students see why a match appeared.
- Students edit the email scaffold into a specific message instead of copying it blindly.
- Curators can comfortably update `professors.json` without breaking validation.
