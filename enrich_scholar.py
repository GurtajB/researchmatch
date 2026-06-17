"""
Enrich professors.json with h-index and recent papers from Semantic Scholar.
Run: python enrich_scholar.py
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.request
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "professors.json"

# SSL context that works on macOS without certificates installed
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

BASE = "https://api.semanticscholar.org/graph/v1"
HEADERS = {"Accept": "application/json", "User-Agent": "ResearchMatch/1.0 (student project)"}


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=15, context=CTX) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 15 * (2 ** attempt)  # 15, 30, 60, 120, 240s
                print(f"  rate-limited, waiting {wait}s…", flush=True)
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            print(f"  network error: {e}", flush=True)
            time.sleep(8)
    return {}


def search_author(name: str, affil_hint: str = "University of North Texas") -> dict | None:
    q = urllib.request.quote(f"{name} {affil_hint}")
    url = f"{BASE}/author/search?query={q}&fields=name,hIndex,affiliations,paperCount&limit=5"
    data = get(url)
    candidates = data.get("data", [])
    # Pick the best match: name similarity + UNT affiliation
    name_lower = name.lower()
    for c in candidates:
        affils = " ".join(a.get("name", "").lower() for a in c.get("affiliations", []))
        if c.get("name", "").lower() == name_lower and ("north texas" in affils or "unt" in affils):
            return c
    # Fallback: just name match
    for c in candidates:
        if c.get("name", "").lower() == name_lower:
            return c
    return None


def get_recent_papers(author_id: str, n: int = 3) -> list[dict]:
    url = (
        f"{BASE}/author/{author_id}/papers"
        f"?fields=title,year,url,externalIds,citationCount&limit=50"
    )
    data = get(url)
    papers = data.get("data", [])
    # Sort by year desc, then citations
    papers.sort(key=lambda p: (p.get("year") or 0, p.get("citationCount") or 0), reverse=True)
    result = []
    for p in papers[:n]:
        title = p.get("title", "")
        year = p.get("year")
        url_out = p.get("url", "")
        # Prefer DOI link, then arXiv, then S2 URL
        ext = p.get("externalIds", {}) or {}
        if ext.get("DOI"):
            url_out = f"https://doi.org/{ext['DOI']}"
        elif ext.get("ArXiv"):
            url_out = f"https://arxiv.org/abs/{ext['ArXiv']}"
        result.append({"title": title, "year": str(year) if year else "", "url": url_out})
    return result


def main():
    with open(DATA_FILE) as f:
        profs = json.load(f)

    enriched = 0
    skipped = 0
    failed = 0

    for i, p in enumerate(profs):
        name = p.get("name", "")
        if not name:
            continue

        # Skip if already enriched
        if p.get("h_index") is not None:
            skipped += 1
            continue

        print(f"[{i+1}/{len(profs)}] {name}…", end=" ", flush=True)
        time.sleep(3.0)  # ~20 reqs/min, well under free tier limit

        author = search_author(name)
        if not author:
            print("not found")
            p["h_index"] = None
            failed += 1
            continue

        author_id = author["authorId"]
        h = author.get("hIndex")
        print(f"h={h}", end=" ", flush=True)

        time.sleep(3.0)
        papers = get_recent_papers(author_id, n=3)
        print(f"papers={len(papers)}")

        p["h_index"] = h
        if papers and not p.get("representative_papers"):
            p["representative_papers"] = papers
        elif papers:
            # Merge: keep existing titles, add URLs where missing
            existing_titles = {pp.get("title", "").lower() for pp in p["representative_papers"]}
            for paper in papers:
                if paper["title"].lower() not in existing_titles:
                    p["representative_papers"].append(paper)
            # Ensure URLs on existing papers
            for pp in p["representative_papers"]:
                if not pp.get("url"):
                    for paper in papers:
                        if paper["title"].lower() == pp.get("title", "").lower():
                            pp["url"] = paper["url"]

        enriched += 1

        # Save after every 10 to not lose progress
        if enriched % 10 == 0:
            with open(DATA_FILE, "w") as f:
                json.dump(profs, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"  ✓ saved checkpoint ({enriched} enriched so far)")

    with open(DATA_FILE, "w") as f:
        json.dump(profs, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nDone. enriched={enriched} skipped={skipped} not-found={failed}")


if __name__ == "__main__":
    main()
