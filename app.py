from __future__ import annotations

import html
import os
from pathlib import Path

import gradio as gr

from researchmatch.data import load_professors
from researchmatch.local_text import email_scaffold, generate_fit_line
from researchmatch.matching import MatchResult, default_embedder, match_professors
from researchmatch.profile import ResearchGoal, StudentProfile, parse_csvish


APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "professors.json"

PROFESSORS = load_professors(DATA_FILE)
EMBEDDER = None

GOAL_CHOICES = [
    ("Just exploring my options", ResearchGoal.EXPLORE.value),
    ("Get hands-on research experience", ResearchGoal.EXPERIENCE.value),
    ("Publish research with a professor", ResearchGoal.PUBLICATION.value),
    ("Build a medical school application", ResearchGoal.MED_SCHOOL.value),
    ("Pursue CS / engineering research", ResearchGoal.CS_RESEARCH.value),
]

STATUS_MAP = {
    "verified_recent":          ("Recently active",           "st-green"),
    "likely_active_needs_check":("Likely active — verify",    "st-amber"),
    "needs_check":              ("Verify before outreach",    "st-amber"),
    "inactive_or_emeritus":     ("May be inactive",           "st-gray"),
    "":                         ("Status unknown",            "st-gray"),
}


def get_embedder():
    global EMBEDDER
    if EMBEDDER is None:
        EMBEDDER = default_embedder(prefer_sentence_transformers=True)
    return EMBEDDER


def esc(v) -> str:
    return html.escape(str(v or ""))


def fit_tier(score: float) -> str:
    if score >= 0.72: return "tier-high"
    if score >= 0.52: return "tier-mid"
    return "tier-low"


def area_pills(areas: list[str]) -> str:
    if not areas:
        return '<span class="tag tag-gray">Not listed</span>'
    return "".join(f'<span class="tag tag-blue">{esc(a)}</span>' for a in areas[:6])


def kw_pills(terms: list[str]) -> str:
    if not terms:
        return '<span class="tag tag-gray">No direct keyword overlap</span>'
    return "".join(f'<span class="tag tag-green">{esc(t)}</span>' for t in terms[:10])


def paper_link(paper: dict) -> str:
    url = paper.get("url", "")
    title = paper.get("title", "") or "Untitled"
    if not url:
        # Google Scholar search fallback
        url = "https://scholar.google.com/scholar?q=" + title.replace(" ", "+")
    return url


def papers_block(papers: list[dict], label: str = "Papers") -> str:
    if not papers:
        return ""
    rows = []
    for p in papers[:3]:
        title = esc(p.get("title", "Untitled"))
        year  = esc(p.get("year", ""))
        url   = paper_link(p)
        yr    = f' <span class="yr">({year})</span>' if year else ""
        rows.append(
            f'<div class="paper-row">'
            f'<a href="{esc(url)}" target="_blank" rel="noreferrer" class="paper-a">{title}</a>{yr}'
            f'</div>'
        )
    verify = '<p class="score-line" style="margin-top:6px">Links sourced from OpenAlex — verify on faculty page</p>'
    return f'<div class="section"><p class="sec-label">{label}</p>{"".join(rows)}{verify}</div>'


def h_index_badge(h: int | None) -> str:
    if h is None:
        return ""
    if h >= 30:
        cls = "hbadge-high"
    elif h >= 15:
        cls = "hbadge-mid"
    else:
        cls = "hbadge-low"
    return f'<span class="hbadge {cls}" title="Google Scholar h-index">h={h}</span>'


def result_card(r: MatchResult, profile: StudentProfile) -> str:
    p = r.professor
    status_text, status_cls = STATUS_MAP.get(p.recent_publications_status or "", ("Status unknown", "st-gray"))
    score_pct = round(r.final_score * 100)

    email_html = (
        f'<a href="mailto:{esc(p.public_email)}" class="cta-link">'
        f'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg>'
        f'{esc(p.public_email)}</a>'
    ) if p.public_email else '<span class="no-contact">Email not listed</span>'

    lab_html = (
        f'<a href="{esc(p.lab_url)}" target="_blank" rel="noreferrer" class="cta-link">'
        f'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>'
        f'Faculty page</a>'
    ) if p.lab_url else '<span class="no-contact">No link listed</span>'

    fit_line = generate_fit_line(profile, r, use_local_llm=False)

    signals_html = (
        "".join(f'<div class="sig-row">· {esc(s)}</div>' for s in p.objective_signals[:4])
        if p.objective_signals
        else '<div class="sig-row no-contact">Check their faculty page for current activity.</div>'
    )

    email_disclaimer = """
<div class="email-disclaimer">
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" style="flex-shrink:0;margin-top:1px"><circle cx="12" cy="12" r="10"/><path d="M12 8v4m0 4h.01"/></svg>
  <span><strong>Before sending:</strong> Read at least one of their recent papers first. Reference something specific about their actual work in your email — generic emails rarely get replies. Cold outreach works best when it shows genuine familiarity with what they study.</span>
</div>"""

    card_id = p.id.replace("-", "_")
    email_id = f"email_{card_id}"
    body_id = f"body_{card_id}"

    email_scaffold_html = f"""
<div class="email-drop">
  <button class="email-drop-btn" onclick="(function(b){{var d=document.getElementById('{email_id}');var open=d.style.display!=='none';d.style.display=open?'none':'block';b.innerHTML=open?'Generate outreach email ↓':'Hide email ↑'}})(this)">Generate outreach email ↓</button>
  <div id="{email_id}" style="display:none">
    {email_disclaimer}
    <div class="email-pre-wrap"><pre class="email-pre">{html.escape(email_scaffold(profile, p, use_local_llm=False))}</pre></div>
  </div>
</div>"""

    return f"""
<div class="card">
  <div class="card-head">
    <div class="card-head-left">
      <div class="prof-name-row">
        <h3 class="prof-name">{esc(p.name or "Unknown")}</h3>
        {h_index_badge(p.h_index)}
      </div>
      <p class="prof-sub">
        {esc(p.title or "Faculty")}
        <span class="dot">·</span>
        {esc(p.dept or "Dept unknown")}
        <span class="dot">·</span>
        <span class="{status_cls}">{status_text}</span>
      </p>
    </div>
    <div class="fit-badge {fit_tier(r.final_score)}">
      <span class="fit-num">{score_pct}%</span>
      <span class="fit-word">match</span>
    </div>
  </div>

  <div class="card-summary-row" onclick="(function(r,b){{var d=document.getElementById('{body_id}');var open=d.style.display!=='none';d.style.display=open?'none':'block';r.classList.toggle('card-open',!open);b.innerHTML=open?'▸':'▾'}})(this,this.querySelector('.card-summary-chevron'))">
    <span class="card-summary-text">{esc(p.summary or "No summary available.")}</span>
    <span class="card-summary-chevron">▸</span>
  </div>

  <div id="{body_id}" class="card-body" style="display:none">
    <div class="section">
      <p class="sec-label">Research areas</p>
      <div class="tag-row">{area_pills(p.research_areas)}</div>
    </div>

    <div class="section reason-box">
      <p class="sec-label">Why this fits your brief</p>
      <p class="reason-text">{esc(fit_line)}</p>
    </div>

    <div class="section">
      <p class="sec-label">Matched keywords</p>
      <div class="tag-row">{kw_pills(r.matched_terms)}</div>
      <p class="score-line">Semantic {round(r.semantic_score*100)}% · Keyword {round(r.tag_score*100)}%</p>
    </div>

    <div class="section contact-section">
      <p class="sec-label">Contact</p>
      <div class="contact-row">{email_html}{lab_html}</div>
    </div>

    <div class="section">
      <p class="sec-label">Public signals to verify</p>
      <div class="signals">{signals_html}</div>
    </div>

    {papers_block(p.recent_papers, "Recent publications (2024+)") if p.recent_papers else papers_block(p.representative_papers, "Publications")}
    {email_scaffold_html}
  </div>
</div>
"""


TOP_K = 15


def find_matches(interests: str, skills: str, goal: str, prioritize_interest: bool = False) -> str:
    top_k = TOP_K
    interests = interests.strip()
    if len(interests) < 8:
        return '<div class="empty-state"><strong>Add a bit more detail</strong> — even one sentence about what excites you helps the matching a lot.</div>'
    try:
        profile = StudentProfile(
            interests=interests,
            skills=parse_csvish(skills),
            courses=[],
            goal=ResearchGoal(goal),
            time_availability="",
        )
        results = match_professors(
            profile, PROFESSORS, top_k=int(top_k),
            embedder=get_embedder(), cache_dir=APP_DIR / ".cache",
            prioritize_interest=bool(prioritize_interest),
        )
    except Exception as exc:
        return f'<div class="empty-state"><strong>Error:</strong> {esc(str(exc))}</div>'

    if not results:
        return '<div class="empty-state">No matches found — try being more specific about your interests.</div>'

    sort_label = "sorted by interest fit" if prioritize_interest else "sorted by h-index"
    n = len(results)
    cards = "\n".join(result_card(r, profile) for r in results)
    return f"""
<div class="results-meta">
  {n} professor{"s" if n != 1 else ""} with 2024+ publications matching your brief
  <span class="sort-badge">↕ {sort_label}</span>
</div>
{cards}
<div class="results-footer">Unofficial student-built tool · Verify all information · Faculty availability is always unknown · Outreach should be individual and specific</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# CSS — full professional design system
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Tokens ── */
:root {
  --green:        #00853E;
  --green-d:      #006B32;
  --green-dd:     #005028;
  --green-bg:     #F0FBF4;
  --green-border: #C3E6D3;
  --blue-bg:      #EFF6FF;
  --blue-border:  #BFDBFE;
  --blue-ink:     #1D4ED8;
  --amber:        #92400E;
  --amber-bg:     #FFFBEB;
  --amber-border: #FDE68A;
  --gray-bg:      #F8FAFC;
  --gray-border:  #E2E8F0;
  --gray-ink:     #64748B;
  --ink:          #0F172A;
  --ink-2:        #1E293B;
  --ink-3:        #334155;
  --ink-4:        #475569;
  --muted:        #64748B;
  --subtle:       #94A3B8;
  --border:       #E2E8F0;
  --border-strong:#CBD5E1;
  --white:        #FFFFFF;
  --page:         #F8FAFC;
  --card-shadow:  0 1px 3px rgba(15,23,42,.06), 0 1px 2px rgba(15,23,42,.04);
  --card-shadow-h:0 4px 20px rgba(15,23,42,.10), 0 1px 4px rgba(15,23,42,.06);
}

/* ── Font everywhere ── */
*, *::before, *::after { box-sizing: border-box; }

body,
.gradio-container,
.gradio-container * {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
               system-ui, sans-serif !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* ── Strip all Gradio chrome ── */
body, .gradio-container {
  background: var(--page) !important;
  color: var(--ink) !important;
  margin: 0 !important;
  padding: 0 !important;
}

.gradio-container,
.gradio-container .contain,
.gradio-container .main,
.gradio-container main {
  max-width: none !important;
  width: 100% !important;
  padding: 0 !important;
  gap: 0 !important;
}

footer { display: none !important; }

/* ── Shell ── */
#shell {
  width: 100%;
  min-height: 100vh;
  padding-bottom: 80px;
}

/* ─────────────────────────────────────────────────
   NAV
───────────────────────────────────────────────── */
.rm-nav {
  position: sticky;
  top: 0;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 60px;
  padding: 0 clamp(20px, 4vw, 56px);
  background: rgba(255,255,255,0.92);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
}

.rm-nav-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
}

.rm-nav-logo {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  background: var(--green);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.rm-nav-logo svg { display: block; }

.rm-nav-wordmark {
  font-size: 1.15rem;
  font-weight: 800;
  color: var(--ink);
  letter-spacing: -0.03em;
  line-height: 1;
}

.rm-nav-badge {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--white);
  background: var(--green);
  border-radius: 999px;
  padding: 2px 8px;
  letter-spacing: 0.01em;
}

.rm-nav-right {
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--muted);
}

/* ─────────────────────────────────────────────────
   HERO
───────────────────────────────────────────────── */
.rm-hero {
  padding: clamp(32px, 5vw, 56px) clamp(20px, 4vw, 56px) clamp(20px, 3vw, 32px);
  max-width: 900px;
}

.rm-hero-label {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-size: 0.76rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--green);
  margin-bottom: 18px;
}

.rm-hero-label-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--green);
}

.rm-wordmark {
  font-size: clamp(3.6rem, 9vw, 7rem);
  font-weight: 900;
  letter-spacing: -0.04em;
  line-height: 0.95;
  color: var(--ink);
  margin: 0 0 20px;
}

.rm-wordmark-accent {
  color: var(--green);
}

.rm-hero-sub {
  font-size: clamp(1rem, 2.2vw, 1.2rem);
  font-weight: 400;
  color: var(--ink-4);
  line-height: 1.65;
  max-width: 600px;
  margin: 0 0 28px;
}

.rm-hero-stats {
  display: flex;
  align-items: center;
  gap: 24px;
  flex-wrap: wrap;
}

.rm-stat {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 0.86rem;
  font-weight: 500;
  color: var(--ink-4);
}

.rm-stat-num {
  font-size: 1.05rem;
  font-weight: 800;
  color: var(--ink);
}

.rm-stat-sep {
  width: 1px;
  height: 20px;
  background: var(--border-strong);
}

/* ─────────────────────────────────────────────────
   MAIN LAYOUT
───────────────────────────────────────────────── */
.rm-main {
  display: grid;
  grid-template-columns: minmax(0, 1.8fr) minmax(280px, 1fr);
  gap: 0;
  padding: 0 clamp(20px, 4vw, 56px);
  align-items: start;
}

/* ─────────────────────────────────────────────────
   LEFT COL (form + results)
───────────────────────────────────────────────── */
.rm-left {
  padding-right: clamp(24px, 3vw, 48px);
  border-right: 1px solid var(--border);
  padding-bottom: 60px;
}

/* Form card */
.rm-form-card {
  background: var(--white);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--card-shadow);
  padding: clamp(20px, 3vw, 32px);
  margin-bottom: 32px;
}

.rm-form-step {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 20px;
}

.rm-step-num {
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: var(--green);
  color: var(--white);
  font-size: 0.72rem;
  font-weight: 800;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.rm-form-step h2 {
  font-size: 1rem;
  font-weight: 700;
  color: var(--ink);
  margin: 0;
  letter-spacing: -0.02em;
}

.rm-form-step p {
  font-size: 0.82rem;
  color: var(--muted);
  margin: 1px 0 0;
}

/* Results section title */
.rm-results-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 20px;
  padding-top: 8px;
}

/* ── Override Gradio dark-mode injection — this is the real culprit ── */
/* Gradio detects OS dark mode and re-applies --block-background-fill: var(--neutral-800).
   We must override with !important in every possible selector Gradio might use. */
/* ── Override Gradio dark-mode injection everywhere it might apply ── */
:root,
.dark,
.light,
.gradio-container,
[data-theme="dark"],
[data-theme="light"] {
  --block-background-fill: transparent !important;
  --block-label-background-fill: transparent !important;
  --block-border-width: 0px !important;
  --block-label-border-width: 0px !important;
  --block-border-color: transparent !important;
  --block-label-border-color: transparent !important;
  --block-label-text-color: #0F172A !important;
  --block-label-padding: 0px 0px 8px 0px !important;
  --block-label-margin: 0px !important;
  --block-padding: 0px !important;
  --block-shadow: none !important;
  --block-label-shadow: none !important;
  --input-background-fill: #FFFFFF !important;
  --input-border-color: #E2E8F0 !important;
  --body-background-fill: #F8FAFC !important;
  --body-text-color: #0F172A !important;
  --background-fill-primary: #FFFFFF !important;
  --background-fill-secondary: transparent !important;
  --border-color-primary: #E2E8F0 !important;
  --button-primary-background-fill: #00853E !important;
  --button-primary-background-fill-hover: #006B32 !important;
  --button-primary-text-color: #FFFFFF !important;
  --button-primary-border-color: transparent !important;
  --neutral-200: #E2E8F0 !important;
  --neutral-800: transparent !important;
}

@media (prefers-color-scheme: dark) {
  :root {
    --block-background-fill: transparent !important;
    --block-label-background-fill: transparent !important;
    --background-fill-secondary: transparent !important;
    --body-background-fill: #F8FAFC !important;
    --body-text-color: #0F172A !important;
    --background-fill-primary: #FFFFFF !important;
    --neutral-800: transparent !important;
  }
}

/* ── Wipe dark Gradio structural wrappers — but NOT our custom HTML ── */
#rm-left .block,
#rm-left .block > *,
#rm-left .form-group,
#rm-left .label-wrap,
#rm-left .block > .wrap,
#rm-left .block > .wrap > .wrap,
#rm-sidebar .block,
#rm-sidebar .block > *,
#rm-sidebar .label-wrap {
  background-color: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

/* ── Labels: big, bold, BLACK — hardcoded, no CSS vars that dark mode can touch ── */
#rm-left label,
#rm-left label *,
#rm-left span[class*="svelte"],
.gradio-container #rm-left label,
.gradio-container #rm-left label span {
  color: #0F172A !important;
  font-size: 1rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.02em !important;
  background: transparent !important;
  background-color: transparent !important;
  padding: 0 0 8px 4px !important;
  margin: 0 !important;
  border: 0 !important;
  box-shadow: none !important;
  visibility: visible !important;
  opacity: 1 !important;
}

/* ── Inputs: white bg, dark text — hardcoded ── */
#rm-left textarea,
#rm-left input[type="text"],
#rm-left input[type="number"],
#rm-left select {
  background-color: #FFFFFF !important;
  border: 1.5px solid #E2E8F0 !important;
  border-radius: 10px !important;
  color: #0F172A !important;
  font-size: 0.95rem !important;
  font-weight: 400 !important;
  line-height: 1.55 !important;
  padding: 12px 14px !important;
  box-shadow: none !important;
  outline: none !important;
  width: 100% !important;
  display: block !important;
}

#rm-left textarea { min-height: 130px !important; resize: vertical !important; }

#rm-left textarea:focus,
#rm-left input:focus {
  border-color: #00853E !important;
  box-shadow: 0 0 0 3px rgba(0,133,62,0.12) !important;
}

#rm-left textarea::placeholder,
#rm-left input::placeholder {
  color: #94A3B8 !important;
}

/* ── Dropdown ── */
#rm-left .wrap {
  border: 1.5px solid #E2E8F0 !important;
  border-radius: 10px !important;
  background-color: #FFFFFF !important;
  box-shadow: none !important;
}
#rm-left .wrap:focus-within { border-color: #00853E !important; box-shadow: 0 0 0 3px rgba(0,133,62,0.12) !important; }
#rm-left .wrap > .wrap { border: 0 !important; border-radius: 0 !important; background: transparent !important; }

/* ── Block containers: transparent, no remnant Gradio backgrounds ── */
#rm-left .block,
#rm-left .form-group {
  background: transparent !important;
  background-color: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin-bottom: 20px !important;
}

/* ── Re-apply white to our own card ── */
.rm-form-card { background-color: #FFFFFF !important; }
.rm-side-card  { background-color: #FFFFFF !important; }

/* ── RUN BUTTON: full green, fully visible — hardcoded, no CSS vars ── */
#rm-run,
#rm-run > div,
#rm-run > div > button,
#rm-run button,
.gradio-container #rm-run button {
  background: #00853E !important;
  background-color: #00853E !important;
  color: #FFFFFF !important;
  border: 0 !important;
  border-radius: 12px !important;
  width: 100% !important;
  height: 52px !important;
  font-size: 1.05rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.01em !important;
  cursor: pointer !important;
  box-shadow: 0 2px 12px rgba(0,133,62,0.35) !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  visibility: visible !important;
  opacity: 1 !important;
  text-shadow: none !important;
}
#rm-run button:hover,
.gradio-container #rm-run button:hover {
  background: #006B32 !important;
  background-color: #006B32 !important;
  box-shadow: 0 4px 16px rgba(0,133,62,0.45) !important;
  transform: translateY(-1px) !important;
}
#rm-run button:active { transform: translateY(0) !important; }

/* Output wrapper */
#rm-output,
#rm-output > .wrap,
#rm-output > div {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
}

/* ─────────────────────────────────────────────────
   RIGHT SIDEBAR
───────────────────────────────────────────────── */
.rm-sidebar {
  padding-left: clamp(24px, 3vw, 40px);
  padding-top: 0;
  position: sticky;
  top: 76px;
  padding-bottom: 40px;
}

.rm-side-card {
  background: var(--white);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px 22px;
  margin-bottom: 14px;
  box-shadow: var(--card-shadow);
}

.rm-side-title {
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--muted);
  margin-bottom: 14px;
}

.tip-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 0;
  margin: 0;
}

.tip-list li {
  font-size: 0.875rem;
  font-weight: 400;
  color: var(--ink-3);
  line-height: 1.5;
  padding-left: 16px;
  position: relative;
}

.tip-list li::before {
  content: "→";
  position: absolute;
  left: 0;
  color: var(--green);
  font-weight: 700;
  font-size: 0.8rem;
}

.disclaimer {
  font-size: 0.82rem;
  color: var(--ink-3);
  line-height: 1.55;
}

.disclaimer strong { color: var(--ink); }

/* Force all text inside our custom HTML to be readable */
.rm-side-card,
.rm-side-card * {
  color: var(--ink-3);
}

.rm-side-card .rm-side-title { color: var(--muted); }
.rm-side-card .tip-list li::before { color: var(--green); }
.rm-side-card em, .rm-side-card i,
.tip-list em, .tip-list i,
.rm-hero-sub em, .rm-hero-sub i {
  font-style: italic;
  color: var(--ink-2);
}

/* Force visible text on ALL our HTML output */
.card, .card *,
.empty-state, .empty-state *,
.results-meta,
.results-footer {
  color: inherit;
}

.card .prof-name { color: var(--ink); }
.card .prof-sub  { color: var(--muted); }
.card .summary-text { color: var(--ink-3); }
.card .sec-label { color: var(--subtle); }
.card .reason-text { color: var(--ink-3); }
.card .score-line { color: var(--subtle); }
.card .sig-row { color: var(--ink-3); }
.card .paper-row { color: var(--ink-3); }

/* ─────────────────────────────────────────────────
   RESULT CARDS
───────────────────────────────────────────────── */
.results-meta {
  font-size: 0.84rem;
  font-weight: 600;
  color: var(--muted);
  margin-bottom: 14px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.sort-badge {
  display: inline-block;
  font-size: 0.75rem;
  font-weight: 700;
  padding: 2px 10px;
  border-radius: 999px;
  background: var(--blue-bg);
  color: var(--blue-ink);
  border: 1px solid var(--blue-border);
}

/* Priority toggle checkbox */
#rm-priority-toggle,
#rm-priority-toggle label,
#rm-priority-toggle input,
#rm-priority-toggle span {
  color: var(--ink-3) !important;
  font-size: 0.875rem !important;
  font-weight: 500 !important;
}
#rm-priority-toggle { margin-bottom: 14px !important; }

.card {
  background: var(--white);
  border: 1px solid var(--border);
  border-radius: 14px;
  margin-bottom: 14px;
  box-shadow: var(--card-shadow);
  overflow: hidden;
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.card:hover {
  box-shadow: var(--card-shadow-h);
  transform: translateY(-1px);
}

/* Card header */
.card-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 20px 22px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--gray-bg);
}

.prof-name {
  font-size: 1.15rem;
  font-weight: 800;
  color: var(--ink);
  letter-spacing: -0.025em;
  line-height: 1.2;
  margin: 0 0 5px;
}

.prof-sub {
  font-size: 0.82rem;
  color: var(--muted);
  margin: 0;
  line-height: 1.4;
}

.dot {
  margin: 0 4px;
  color: var(--border-strong);
}

.st-green { color: var(--green-d);  font-weight: 600; }
.st-amber { color: var(--amber);    font-weight: 600; }
.st-gray  { color: var(--gray-ink); font-weight: 500; }

/* Fit badge */
.fit-badge {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 72px;
  padding: 10px 14px;
  border-radius: 10px;
  flex-shrink: 0;
  line-height: 1;
}

.fit-num {
  font-size: 1.35rem;
  font-weight: 900;
  letter-spacing: -0.03em;
  display: block;
}

.fit-word {
  font-size: 0.66rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  display: block;
  margin-top: 2px;
}

.tier-high {
  background: var(--green-bg);
  border: 1px solid var(--green-border);
  color: var(--green-d);
}

.tier-mid {
  background: var(--amber-bg);
  border: 1px solid var(--amber-border);
  color: var(--amber);
}

.tier-low {
  background: var(--gray-bg);
  border: 1px solid var(--gray-border);
  color: var(--gray-ink);
}

/* Card body */
.card-body {
  padding: 18px 22px 20px;
}

.summary-text {
  font-size: 0.935rem;
  color: var(--ink-3);
  line-height: 1.65;
  margin: 0 0 16px;
}

/* Sections inside card */
.section {
  margin-bottom: 14px;
}

.section:last-child { margin-bottom: 0; }

.sec-label {
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--subtle);
  margin: 0 0 7px;
}

/* Reason box */
.reason-box {
  background: var(--green-bg);
  border: 1px solid var(--green-border);
  border-radius: 10px;
  padding: 12px 14px;
}

.reason-box .sec-label { color: var(--green-d); }

.reason-text {
  font-size: 0.9rem;
  color: var(--ink-3);
  line-height: 1.6;
  margin: 0;
}

/* Tag pills */
.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-bottom: 6px;
}

.tag {
  display: inline-block;
  font-size: 0.78rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 999px;
  line-height: 1.5;
}

.tag-green {
  background: var(--green-bg);
  color: var(--green-d);
  border: 1px solid var(--green-border);
}

.tag-blue {
  background: var(--blue-bg);
  color: var(--blue-ink);
  border: 1px solid var(--blue-border);
}

.tag-gray {
  background: var(--gray-bg);
  color: var(--gray-ink);
  border: 1px solid var(--gray-border);
}

.score-line {
  font-size: 0.76rem;
  color: var(--subtle);
  margin: 4px 0 0;
}

/* Contact */
.contact-section {
  border-top: 1px solid var(--border);
  padding-top: 14px;
  margin-top: 4px;
}

.contact-row {
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
}

.cta-link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.87rem;
  font-weight: 600;
  color: var(--green-d) !important;
  text-decoration: none;
  transition: color 0.1s;
}

.cta-link:hover {
  color: var(--green-dd) !important;
  text-decoration: underline;
}

.no-contact {
  font-size: 0.87rem;
  color: var(--subtle);
}

/* Signals */
.signals {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.sig-row {
  font-size: 0.87rem;
  color: var(--ink-3);
  line-height: 1.5;
}

/* Papers */
.paper-row {
  font-size: 0.87rem;
  color: var(--ink-3);
  line-height: 1.5;
  margin-bottom: 5px;
}

.paper-a {
  color: var(--green-d) !important;
  text-decoration: none;
  font-weight: 500;
}
.paper-a:hover { text-decoration: underline; }

.yr { color: var(--subtle); }

/* Email scaffold */
.email-drop {
  border-top: 1px solid var(--border);
  padding-top: 14px;
  margin-top: 14px;
}

.email-drop-btn {
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--green-d);
  cursor: pointer;
  background: none;
  border: none;
  padding: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  user-select: none;
  font-family: inherit;
}

.email-drop-btn:hover { color: var(--green-dd); text-decoration: underline; }

.email-pre-wrap { margin-top: 12px; }

.email-pre {
  font-family: "SF Mono", "Fira Code", "Consolas", monospace !important;
  font-size: 0.8rem !important;
  line-height: 1.6 !important;
  white-space: pre-wrap;
  background: var(--gray-bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  color: var(--ink-2);
  margin: 0;
}

/* ─────────────────────────────────────────────────
   SCROLL HINT
───────────────────────────────────────────────── */
.scroll-hint {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 16px 0 8px;
  padding: 10px 14px;
  background: var(--green-bg);
  border: 1px solid var(--green-border);
  border-radius: 10px;
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--green-d);
}

.scroll-hint-arrow {
  font-size: 1.1rem;
  animation: bounce-down 1.4s ease-in-out infinite;
}

@keyframes bounce-down {
  0%, 100% { transform: translateY(0); }
  50%       { transform: translateY(5px); }
}

/* ─────────────────────────────────────────────────
   COLLAPSIBLE CARD TOGGLE ROW
───────────────────────────────────────────────── */
.card-summary-row {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 13px 22px;
  cursor: pointer;
  user-select: none;
  background: var(--white);
  border-top: 1px solid var(--border);
  transition: background 0.15s;
}

.card-summary-row:hover { background: var(--gray-bg); }

.card-summary-text {
  flex: 1;
  font-size: 0.92rem;
  color: var(--ink-3);
  line-height: 1.55;
}

.card-summary-chevron {
  font-size: 0.85rem;
  color: var(--green-d);
  font-weight: 700;
  margin-top: 2px;
  flex-shrink: 0;
  transition: transform 0.15s;
}

.card-open .card-summary-chevron { transform: rotate(90deg); }

/* ─────────────────────────────────────────────────
   H-INDEX BADGE
───────────────────────────────────────────────── */
.prof-name-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 5px;
}

.prof-name-row .prof-name {
  margin-bottom: 0;
}

.hbadge {
  font-size: 0.72rem;
  font-weight: 800;
  padding: 2px 9px;
  border-radius: 999px;
  letter-spacing: 0.02em;
  white-space: nowrap;
}

.hbadge-high {
  background: #ECFDF5;
  color: #065F46;
  border: 1px solid #6EE7B7;
}

.hbadge-mid {
  background: var(--blue-bg);
  color: var(--blue-ink);
  border: 1px solid var(--blue-border);
}

.hbadge-low {
  background: var(--gray-bg);
  color: var(--gray-ink);
  border: 1px solid var(--gray-border);
}

/* ─────────────────────────────────────────────────
   EMAIL DISCLAIMER
───────────────────────────────────────────────── */
.email-disclaimer {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin: 12px 0 10px;
  padding: 10px 13px;
  background: var(--amber-bg);
  border: 1px solid var(--amber-border);
  border-radius: 9px;
  font-size: 0.83rem;
  color: var(--amber);
  line-height: 1.5;
}

.email-disclaimer strong { color: #78350F; }

/* Empty / error states */
.empty-state {
  background: var(--white);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 32px 24px;
  color: var(--muted);
  font-size: 0.94rem;
  line-height: 1.6;
  text-align: center;
}

.empty-state strong { color: var(--ink); }

/* Results footer */
.results-footer {
  margin-top: 20px;
  padding: 14px 16px;
  background: var(--amber-bg);
  border: 1px solid var(--amber-border);
  border-radius: 10px;
  font-size: 0.81rem;
  color: var(--amber);
  line-height: 1.5;
}

/* ─────────────────────────────────────────────────
   RESPONSIVE
───────────────────────────────────────────────── */
@media (max-width: 860px) {
  .rm-main { grid-template-columns: 1fr; padding: 0 16px; }

  .rm-left {
    border-right: 0;
    padding-right: 0;
    border-bottom: 1px solid var(--border);
    padding-bottom: 32px;
    margin-bottom: 32px;
  }

  .rm-sidebar {
    padding-left: 0;
    position: static;
  }

  .rm-wordmark { font-size: clamp(2.6rem, 12vw, 4rem); }

  .card-head { flex-direction: column; gap: 10px; }
  .fit-badge { flex-direction: row; gap: 6px; width: auto; align-self: flex-start; }
  .fit-word { font-size: 0.75rem; margin-top: 0; }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
n_profs = len(PROFESSORS)
n_depts = len({p.dept for p in PROFESSORS})

with gr.Blocks(
    title="ResearchMatch — UNT",
    css=CSS,
    theme=gr.themes.Base(
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        # Kill all dark block backgrounds at the theme level
        block_background_fill="transparent",
        block_label_background_fill="transparent",
        block_border_width="0px",
        block_label_border_width="0px",
        block_border_color="transparent",
        block_label_text_color="#1E293B",
        block_label_padding="0px 0px 6px 0px",
        block_label_margin="0px",
        block_padding="0px",
        input_background_fill="#FFFFFF",
        input_border_color="#E2E8F0",
        input_border_width="1.5px",
        input_shadow="none",
        input_shadow_focus="0 0 0 3px rgba(0,133,62,0.12)",
        slider_color="#00853E",
        color_accent="#00853E",
        color_accent_soft="#F0FBF4",
        body_background_fill="#F8FAFC",
        body_text_color="#0F172A",
    ),
) as demo:

    with gr.Column(elem_id="shell"):

        # ── Nav ──
        gr.HTML(f"""
        <nav class="rm-nav">
          <div class="rm-nav-brand">
            <div class="rm-nav-logo">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
              </svg>
            </div>
            <span class="rm-nav-wordmark">ResearchMatch</span>
            <span class="rm-nav-badge">UNT</span>
          </div>
          <div class="rm-nav-right">University of North Texas · Unofficial student tool</div>
        </nav>
        """)

        # ── Hero ──
        gr.HTML(f"""
        <div class="rm-hero">
          <div class="rm-hero-label">
            <div class="rm-hero-label-dot"></div>
            Find your research fit
          </div>
          <h1 class="rm-wordmark">Research<span class="rm-wordmark-accent">Match</span></h1>
          <p class="rm-hero-sub">
            Tell us what excites you. We'll rank UNT professors by how closely
            their published work matches your interests — so you can reach out
            to the right people with something real to say.
          </p>
          <div class="rm-hero-stats">
            <div class="rm-stat">
              <span class="rm-stat-num">{n_profs}</span>
              professors indexed
            </div>
            <div class="rm-stat-sep"></div>
            <div class="rm-stat">
              <span class="rm-stat-num">{n_depts}</span>
              departments
            </div>
            <div class="rm-stat-sep"></div>
            <div class="rm-stat">
              Semantic + keyword matching
            </div>
          </div>
        </div>
        """)

        # ── Main two-column ──
        with gr.Row(elem_classes=["rm-main"]):

            # Left: form + results
            with gr.Column(elem_id="rm-left"):

                # Form card
                with gr.Column(elem_classes=["rm-form-card"]):
                    gr.HTML("""
                    <div class="rm-form-step">
                      <div class="rm-step-num">1</div>
                      <div>
                        <h2>Describe your research interests</h2>
                        <p>Specific topics, questions, or methods work better than broad fields.</p>
                      </div>
                    </div>
                    """)

                    interests = gr.Textbox(
                        label="What research sounds exciting to you?",
                        lines=5,
                        placeholder=(
                            'Be specific — e.g. "I want to use machine learning to analyze '
                            'brain signals, study antibiotic resistance in bacteria, '
                            'or build augmented reality tools for accessibility."'
                        ),
                    )
                    skills = gr.Textbox(
                        label="Skills or tools (optional)",
                        placeholder="Python, R, biology lab, statistics, data analysis…",
                    )

                    goal = gr.Dropdown(
                        label="Current goal",
                        choices=GOAL_CHOICES,
                        value=ResearchGoal.EXPLORE.value,
                    )

                    prioritize_interest = gr.Checkbox(
                        label="Prioritize interest fit over h-index",
                        value=False,
                        elem_id="rm-priority-toggle",
                        info="Default: sorted by h-index (most prolific matching profs first). Toggle to sort by how closely their work matches your interests instead.",
                    )

                    run_btn = gr.Button("Find matches →", elem_id="rm-run", variant="primary")

                gr.HTML("""
                <div class="scroll-hint" id="rm-scroll-hint">
                  <div class="scroll-hint-arrow">↓</div>
                  <span>Your matches appear below — scroll down to see them</span>
                </div>
                """)

                # Results
                output = gr.HTML(
                    value='<div class="empty-state">Your matched professors will appear here — fill in your interests above and click <strong>Find matches</strong>.</div>',
                    elem_id="rm-output",
                )

            # Right: sidebar
            with gr.Column(elem_id="rm-sidebar", scale=1):
                gr.HTML("""
                <div style="padding-top:0;">

                  <div class="rm-side-card">
                    <div class="rm-side-title">How it works</div>
                    <ul class="tip-list">
                      <li>Describe your interests in plain language — topics, methods, or questions you're curious about.</li>
                      <li>We compare your brief to each professor's public research using AI-powered semantic matching plus keyword overlap.</li>
                      <li>Results are ranked by fit to <em>your brief only</em>, not by who is the "best" professor.</li>
                      <li>Use the match to shortlist, then verify on their faculty page before reaching out.</li>
                    </ul>
                  </div>

                  <div class="rm-side-card">
                    <div class="rm-side-title">Tips for better matches</div>
                    <ul class="tip-list">
                      <li>Name specific topics — "antibiotic resistance in E. coli" beats just "biology."</li>
                      <li>Mention methods, even ones you haven't used yet.</li>
                      <li>Add skills — even "Excel" or "literature review" helps keyword matching.</li>
                      <li>Try different phrasings if the first results feel off.</li>
                    </ul>
                  </div>

                  <div class="rm-side-card">
                    <div class="rm-side-title">Before you reach out</div>
                    <ul class="tip-list">
                      <li>Read at least one of their papers first — cold emails without this rarely work.</li>
                      <li>Check their lab page for current openings or student policies.</li>
                      <li>Email one professor at a time, not a mass blast.</li>
                      <li>Mention something specific about their actual work, not just your goals.</li>
                    </ul>
                  </div>

                  <div class="rm-side-card">
                    <p class="disclaimer">
                      <strong>Unofficial student-built tool.</strong> Data curated from public UNT faculty sources.
                      Professor availability is always unknown.
                      Verify all information before outreach.
                    </p>
                  </div>

                </div>
                """)

        run_btn.click(
            fn=find_matches,
            inputs=[interests, skills, goal, prioritize_interest],
            outputs=output,
        )


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port)
