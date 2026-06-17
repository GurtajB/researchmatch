"""
Enrich professors.json with h-index and recent papers from OpenAlex (open academic API).
Run: python enrich_scholar.py
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.request
import urllib.parse
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "professors.json"

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

BASE = "https://api.openalex.org"
# Include mailto for the polite pool (higher rate limit: 10 req/sec)
MAILTO = "gurtajboparai123@gmail.com"
HEADERS = {"Accept": "application/json", "User-Agent": f"ResearchMatch/1.0 (mailto:{MAILTO})"}


def get(url: str) -> dict:
    if "?" in url:
        url += f"&mailto={MAILTO}"
    else:
        url += f"?mailto={MAILTO}"
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=15, context=CTX) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10 * (2 ** attempt)
                print(f"  rate-limited, waiting {wait}s…", flush=True)
                time.sleep(wait)
            elif e.code == 404:
                return {}
            else:
                raise
        except Exception as e:
            print(f"  network error: {e}", flush=True)
            time.sleep(5)
    return {}


def search_author(name: str) -> dict | None:
    q = urllib.parse.quote(name)
    url = f"{BASE}/authors?search={q}&per_page=8"
    data = get(url)
    candidates = data.get("results", [])
    name_lower = name.lower()
    # Prefer exact name + UNT affiliation
    for c in candidates:
        affils = " ".join(
            x.get("institution", {}).get("display_name", "").lower()
            for x in c.get("affiliations", [])
        )
        if c.get("display_name", "").lower() == name_lower and "north texas" in affils:
            return c
    # Fallback: exact name match regardless of affiliation
    for c in candidates:
        if c.get("display_name", "").lower() == name_lower:
            return c
    return None


def get_top_papers(works_api_url: str, n: int = 3) -> list[dict]:
    url = f"{works_api_url}&sort=cited_by_count:desc&per_page={n}"
    data = get(url)
    papers = data.get("results", [])
    result = []
    for p in papers[:n]:
        title = p.get("title", "")
        year = p.get("publication_year")
        doi = p.get("doi", "")
        # doi already comes as full URL from OpenAlex
        url_out = doi if doi and doi.startswith("http") else (f"https://doi.org/{doi}" if doi else "")
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

        if p.get("h_index") is not None:
            skipped += 1
            continue

        print(f"[{i+1}/{len(profs)}] {name}…", end=" ", flush=True)
        time.sleep(0.15)  # polite pool allows 10 req/sec; 0.15s gives ~6/sec

        author = search_author(name)
        if not author:
            print("not found")
            p["h_index"] = None
            failed += 1
            continue

        stats = author.get("summary_stats", {}) or {}
        h = stats.get("h_index")
        works_count = author.get("works_count", 0)
        works_url = author.get("works_api_url", "")

        print(f"h={h} works={works_count}", end=" ", flush=True)

        papers = []
        if works_url:
            time.sleep(0.15)
            papers = get_top_papers(works_url, n=3)

        print(f"papers={len(papers)}")

        p["h_index"] = h

        # Add papers if none exist yet
        if papers and not any(pp.get("url") for pp in p.get("representative_papers", [])):
            existing_titles = {pp.get("title", "").lower() for pp in p.get("representative_papers", [])}
            for paper in papers:
                if paper["title"].lower() not in existing_titles:
                    p.setdefault("representative_papers", []).append(paper)

        enriched += 1

        if enriched % 20 == 0:
            with open(DATA_FILE, "w") as f:
                json.dump(profs, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"  ✓ checkpoint ({enriched} enriched)", flush=True)

    with open(DATA_FILE, "w") as f:
        json.dump(profs, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nDone. enriched={enriched} skipped={skipped} not-found={failed}")


if __name__ == "__main__":
    main()
