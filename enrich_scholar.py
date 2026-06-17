"""
Enrich professors.json with h-index, has_recent_pubs, and recent_papers via OpenAlex.
Run: python enrich_scholar.py
"""
from __future__ import annotations

import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "professors.json"

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

BASE = "https://api.openalex.org"
MAILTO = "gurtajboparai123@gmail.com"
HEADERS = {"Accept": "application/json", "User-Agent": f"ResearchMatch/1.0 (mailto:{MAILTO})"}
RECENT_YEARS = {2024, 2025}


def get(url: str) -> dict:
    sep = "&" if "?" in url else "?"
    full = f"{url}{sep}mailto={MAILTO}"
    req = urllib.request.Request(full, headers=HEADERS)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=15, context=CTX) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10 * (2 ** attempt)
                print(f"  rate-limited {wait}s…", flush=True)
                time.sleep(wait)
            elif e.code == 404:
                return {}
            else:
                raise
        except Exception as e:
            print(f"  net error: {e}", flush=True)
            time.sleep(5)
    return {}


def strip_middle_initial(name: str) -> str:
    """'Fred C. McMahan' → 'Fred McMahan'"""
    return re.sub(r"\b[A-Z]\.\s*", "", name).strip()


def search_author(name: str) -> dict | None:
    for query_name in [name, strip_middle_initial(name)]:
        if not query_name:
            continue
        q = urllib.parse.quote(query_name)
        data = get(f"{BASE}/authors?search={q}&per_page=8")
        candidates = data.get("results", [])
        name_lower = query_name.lower()
        # Prefer exact name + UNT affiliation
        for c in candidates:
            affils = " ".join(
                x.get("institution", {}).get("display_name", "").lower()
                for x in c.get("affiliations", [])
            )
            if c.get("display_name", "").lower() == name_lower and "north texas" in affils:
                return c
        # Exact name, any affiliation
        for c in candidates:
            if c.get("display_name", "").lower() == name_lower:
                return c
    return None


def recent_pub_count(author: dict) -> int:
    return sum(
        y.get("works_count", 0)
        for y in author.get("counts_by_year", [])
        if y.get("year") in RECENT_YEARS
    )


def get_recent_papers(works_api_url: str, n: int = 3) -> list[dict]:
    year_filter = ",".join(str(y) for y in sorted(RECENT_YEARS))
    url = (
        f"{works_api_url}"
        f"&filter=publication_year:{min(RECENT_YEARS)}-{max(RECENT_YEARS)}"
        f"&sort=publication_year:desc&per_page={n}"
    )
    data = get(url)
    result = []
    for p in data.get("results", [])[:n]:
        title = p.get("title") or ""
        year = p.get("publication_year")
        doi = p.get("doi") or ""
        ext = p.get("ids", {}) or {}
        arxiv = ext.get("arxiv", "")
        if doi.startswith("http"):
            url_out = doi
        elif doi:
            url_out = f"https://doi.org/{doi}"
        elif arxiv:
            arxiv_id = arxiv.replace("https://arxiv.org/abs/", "").replace("http://arxiv.org/abs/", "")
            url_out = f"https://arxiv.org/abs/{arxiv_id}"
        else:
            # Google Scholar fallback
            url_out = f"https://scholar.google.com/scholar?q={urllib.parse.quote(title)}"
        result.append({"title": title, "year": str(year) if year else "", "url": url_out})
    return result


def main():
    with open(DATA_FILE) as f:
        profs = json.load(f)

    enriched = skipped = failed = 0

    for i, p in enumerate(profs):
        name = p.get("name", "")
        if not name:
            continue

        # Skip if already fully enriched
        if "has_recent_pubs" in p and p.get("h_index") is not None:
            skipped += 1
            continue

        print(f"[{i+1}/{len(profs)}] {name}…", end=" ", flush=True)
        time.sleep(0.15)

        author = search_author(name)
        if not author:
            print("not found → fallback to status field")
            # Use existing recent_publications_status as proxy
            status = p.get("recent_publications_status", "")
            p["has_recent_pubs"] = status in ("verified_recent", "likely_active_needs_check")
            p.setdefault("recent_papers", [])
            if p.get("h_index") is None:
                p["h_index"] = None
            failed += 1
            continue

        stats = author.get("summary_stats", {}) or {}
        h = stats.get("h_index")
        rc = recent_pub_count(author)
        works_url = author.get("works_api_url", "")

        print(f"h={h} recent_pubs={rc}", end=" ", flush=True)

        recent_papers = []
        if works_url and rc > 0:
            time.sleep(0.15)
            recent_papers = get_recent_papers(works_url, n=3)

        print(f"papers={len(recent_papers)}")

        # Only update h_index if not already set or was null
        if p.get("h_index") is None:
            p["h_index"] = h

        p["has_recent_pubs"] = rc > 0
        p["recent_papers"] = recent_papers

        enriched += 1

        if enriched % 30 == 0:
            with open(DATA_FILE, "w") as f:
                json.dump(profs, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"  ✓ checkpoint ({enriched} enriched)", flush=True)

    with open(DATA_FILE, "w") as f:
        json.dump(profs, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nDone. enriched={enriched} fallback={failed} skipped={skipped}")
    # Summary
    with open(DATA_FILE) as f:
        final = json.load(f)
    active = [p for p in final if p.get("has_recent_pubs")]
    print(f"Professors with recent pubs (2024+): {len(active)}/{len(final)}")


if __name__ == "__main__":
    main()
