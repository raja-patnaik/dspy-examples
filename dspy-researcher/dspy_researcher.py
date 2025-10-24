from __future__ import annotations

import os
import re
import json
import asyncio
import operator
from typing import Any, Dict, List, Optional, Tuple, Annotated
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from dateutil import parser as dtparse

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

import dspy
from dspy.teleprompt import GEPA

from exa_py import Exa

# ----------------------------
# Configuration
# ----------------------------

MAX_ROUNDS = int(os.environ.get("RR_MAX_ROUNDS", "1"))              # writer<->research loop rounds
SEARCH_RESULTS_PER_QUERY = int(os.environ.get("RR_SEARCH_K", "6"))  # per query
MAX_CONTENT_CHARS_PER_SOURCE = int(os.environ.get("RR_MAX_CHARS", "12000"))

WRITER_MODEL = os.environ.get("GEMINI_WRITER_MODEL", "gemini/gemini-flash-latest")
RESEARCH_MODEL = os.environ.get("GEMINI_RESEARCH_MODEL", "gemini/gemini-flash-latest")
REFLECTION_MODEL = os.environ.get("GEMINI_REFLECTION_MODEL", WRITER_MODEL)

FALLBACK_WRITER = "gemini/gemini/gemini-flash-latest"
FALLBACK_RESEARCH = "gemini/gemini/gemini-flash-latest"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EXA_API_KEY = os.environ.get("EXA_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")
if not EXA_API_KEY:
    raise RuntimeError("EXA_API_KEY is not set. Get one from dashboard.exa.ai")

# Initialize Exa once (thread-safe to call via to_thread)
EXA = Exa(EXA_API_KEY)

# ----------------------------
# Utilities & data models
# ----------------------------

def short_host(u: str) -> str:
    try:
        return urlparse(u).netloc.replace("www.", "")
    except Exception:
        return u

def clamp(n: float, lo=0.0, hi=1.0) -> float:
    return max(lo, min(hi, n))

def safe_json_loads(s: str, fallback=None):
    try:
        return json.loads(s)
    except Exception:
        return fallback

class SectionSpec(BaseModel):
    name: str
    instructions: str

class SourceDoc(BaseModel):
    url: str
    title: Optional[str] = None
    site: Optional[str] = None
    published: Optional[str] = None
    content: Optional[str] = None

class ResearchSummary(BaseModel):
    section: str
    query: str
    bullets: List[str] = Field(default_factory=list)
    sources: List[SourceDoc] = Field(default_factory=list)

class ReviewReport(BaseModel):
    pass_checks: bool
    summary: str
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

class EvalResult(BaseModel):
    score: float
    breakdown: Dict[str, float]
    notes: str

# ----------------------------
# DSPy model setup
# ----------------------------

def _make_lm(model_name: str, api_key: str, temperature: float = 0.3, model_type: str = "chat", max_tokens: int = 65536):
    """Create a DSPy LM via LiteLLM provider strings (e.g., 'gemini/gemini-2.5-pro-preview-03-25')."""
    try:
        return dspy.LM(model_name, api_key=api_key, temperature=temperature, model_type=model_type, max_tokens=max_tokens)
    except Exception:
        if "pro" in model_name:
            return dspy.LM(FALLBACK_WRITER, api_key=api_key, temperature=temperature, model_type=model_type, max_tokens=max_tokens)
        return dspy.LM(FALLBACK_RESEARCH, api_key=api_key, temperature=temperature, model_type=model_type, max_tokens=max_tokens)

WRITER_LM   = _make_lm(WRITER_MODEL, GEMINI_API_KEY, temperature=0.2)
RESEARCH_LM = _make_lm(RESEARCH_MODEL, GEMINI_API_KEY, temperature=0.4)
REFLECT_LM  = _make_lm(REFLECTION_MODEL, GEMINI_API_KEY, temperature=0.8)

dspy.configure(lm=RESEARCH_LM, cache=True)

# ----------------------------
# DSPy Signatures (instructions)
# ----------------------------

class QueryGenSig(dspy.Signature):
    """Produce 4–8 diverse Exa search queries for a section (use quoted phrases, site:, intitle:, date ranges). Return a JSON list of strings."""
    section_title = dspy.InputField()
    section_instructions = dspy.InputField()
    queries_json = dspy.OutputField()

class SummarizeSig(dspy.Signature):
    """Summarize source texts into evidence bullets for the section.
    OUTPUT JSON: {"bullets": ["...", "..."]}. Cite as [S#] (matching the per-query ordering). Keep bullets concise & factual.
    """
    prompt = dspy.InputField()
    sources_digest = dspy.InputField()
    output_json = dspy.OutputField()

class WriteSectionSig(dspy.Signature):
    """Write a polished Markdown section '# {section_title}' using [n] numeric citations only. Avoid bare URLs. Return ONLY the section Markdown."""
    section_title = dspy.InputField()
    section_instructions = dspy.InputField()
    evidence_digest = dspy.InputField()
    output_markdown = dspy.OutputField()

class GapAnalysisSig(dspy.Signature):
    """Given current bullets, decide if more research is needed. OUTPUT JSON: {"need_more": bool, "followup_queries": ["..."]}"""
    section_title = dspy.InputField()
    bullets_digest = dspy.InputField()
    output_json = dspy.OutputField()

class CiteFixSig(dspy.Signature):
    """Fix citations: ensure only [n] numeric citations (no [S#] or raw URLs). Return ONLY the corrected Markdown body."""
    markdown_body = dspy.InputField()
    id_map_notes = dspy.InputField()
    fixed_markdown = dspy.OutputField()

class ReviewSig(dspy.Signature):
    """Review the full report for coverage, correctness, clarity, neutrality, structure, citation hygiene. OUTPUT JSON: {pass_checks, issues, suggestions, summary}"""
    report_md = dspy.InputField()
    output_json = dspy.OutputField()

class ReviseSig(dspy.Signature):
    """Apply review suggestions to the report without adding new unsupported facts. Return the improved Markdown body (no References)."""
    report_md = dspy.InputField()
    suggestions = dspy.InputField()
    improved_md = dspy.OutputField()

QUERY_GEN      = dspy.ChainOfThought(QueryGenSig)
SUMMARIZER     = dspy.Predict(SummarizeSig)
WRITE_SECTION  = dspy.ChainOfThought(WriteSectionSig)
GAP_ANALYZER   = dspy.Predict(GapAnalysisSig)
CITE_FIXER     = dspy.Predict(CiteFixSig)
REVIEWER       = dspy.Predict(ReviewSig)
REVISER        = dspy.ChainOfThought(ReviseSig)

# ----------------------------
# GEPA: fast optimization
# ----------------------------

def heuristic_report_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> float:
    """LLM-free shaping signal for GEPA."""
    text = ""
    if hasattr(pred, "output_markdown"): text = pred.output_markdown or ""
    elif hasattr(pred, "fixed_markdown"): text = pred.fixed_markdown or ""
    elif hasattr(pred, "queries_json"): text = pred.queries_json or ""
    elif hasattr(pred, "output_json"): text = pred.output_json or ""
    score, notes = 0.0, []
    if hasattr(pred, "queries_json"):
        data = safe_json_loads(text, [])
        uniq = len(set([q.strip().lower() for q in data if isinstance(q, str)]))
        has_ops = any(("site:" in (q or "").lower() or "intitle:" in (q or "").lower() or '"' in (q or "")) for q in data if isinstance(q, str))
        score = 0.3*clamp(uniq/8) + 0.2*(1 if 4 <= uniq <= 10 else 0) + 0.5*(1 if has_ops else 0)
        if uniq < 4: notes.append("Add 6–8 diverse queries.")
        if not has_ops: notes.append("Use operators like site:, intitle:, \"quoted\".")
    elif hasattr(pred, "output_markdown"):
        has_h1 = bool(re.search(r"^#\s+", text, flags=re.M))
        cites = len(re.findall(r"\[\d+\]", text))
        urls_inline = bool(re.search(r"https?://", text))
        too_short = len(text) < 700
        score = (0.25*(1 if has_h1 else 0) + 0.35*clamp(cites/6) + 0.15*(0 if urls_inline else 1) + 0.25*(0 if too_short else 1))
        if not has_h1: notes.append("Start with H1.")
        if cites < 3: notes.append("Add more [n] citations.")
        if urls_inline: notes.append("No bare URLs in body.")
        if too_short: notes.append("Increase depth (>=700 chars).")
    elif hasattr(pred, "fixed_markdown"):
        leftovers = bool(re.search(r"\[S\d+\]", text))
        bracket_nums = bool(re.search(r"\[\d+\]", text))
        score = 0.6*(1 if bracket_nums else 0) + 0.4*(0 if leftovers else 1)
        if leftovers: notes.append("Replace [S#] with [n].")
    elif hasattr(pred, "output_json"):
        ok = safe_json_loads(text) is not None
        score = 1.0 if ok else 0.2
        if not ok: notes.append("Return valid JSON.")
    else:
        score = 0.5; notes.append("Improve structure.")
    return float(clamp(score, 0, 1))

def optimize_with_gepa() -> None:
    """Optimize each DSPy module with its specific training set."""
    # Map each module to its training set function
    module_trainsets = {
        QUERY_GEN: trainset_query_gen(),
        SUMMARIZER: trainset_summarizer(),
        WRITE_SECTION: trainset_write_section(),
        GAP_ANALYZER: trainset_gap_analyzer(),
        CITE_FIXER: trainset_cite_fixer(),
        REVIEWER: trainset_reviewer(),
        REVISER: trainset_reviser(),
    }

    tele = GEPA(metric=heuristic_report_metric, auto="light", reflection_lm=REFLECT_LM, track_stats=False)

    for module, trainset in module_trainsets.items():
        module_name = type(module).__name__
        print(f"[GEPA] Optimizing {module_name} with {len(trainset)} examples...")
        try:
            tele.compile(student=module, trainset=trainset)
            print(f"[GEPA] ✓ {module_name} optimization complete")
        except Exception as e:
            print(f"[GEPA] ✗ Skipped {module_name}: {e}")

# Module-specific training sets for GEPA

def trainset_query_gen() -> List[dspy.Example]:
    """Training examples for QUERY_GEN: diverse search queries with operators."""
    return [
        dspy.Example(
            section_title="Market Size",
            section_instructions="TAM, SAM, SOM 2019–2025 with figures and growth rates",
            queries_json='["market size TAM 2024", "site:.gov industry market size", "intitle:forecast 2024..2025", "\\"total addressable market\\" 2024", "SAM SOM market sizing report", "site:statista.com market size trends"]'
        ).with_inputs('section_title', 'section_instructions'),
        dspy.Example(
            section_title="Technology Stack",
            section_instructions="Current tools, frameworks, and infrastructure with version numbers",
            queries_json='["site:github.com popular frameworks 2024", "intitle:\\"tech stack\\" comparison", "\\"technology adoption\\" trends 2024", "infrastructure tools benchmarks", "site:stackoverflow.com framework usage statistics", "developer survey 2024 tools"]'
        ).with_inputs('section_title', 'section_instructions'),
        dspy.Example(
            section_title="Regulatory Landscape",
            section_instructions="Key regulations, compliance requirements, and policy changes 2023-2025",
            queries_json='["site:.gov regulations 2024", "compliance requirements industry", "intitle:\\"policy changes\\" 2024..2025", "\\"regulatory framework\\" updates", "site:.org legal compliance guidelines", "data protection regulations 2024"]'
        ).with_inputs('section_title', 'section_instructions'),
    ]

def trainset_summarizer() -> List[dspy.Example]:
    """Training examples for SUMMARIZER: source texts to cited bullets."""
    return [
        dspy.Example(
            prompt="Summarize for section 'Market Growth'. Cite using [S#] matching the source indices above.",
            sources_digest="S1 | Market Report 2024 — example.com\nThe global market grew 15% YoY in Q3 2024, reaching $500B valuation.\n\nS2 | Industry Analysis — research.org\nAdoption rates increased from 23% in 2023 to 34% in 2024, driven by enterprise demand.",
            output_json='{"bullets": ["Global market grew 15% YoY in Q3 2024, reaching $500B valuation [S1]", "Adoption rates increased from 23% (2023) to 34% (2024), driven by enterprise demand [S2]"]}'
        ).with_inputs('prompt', 'sources_digest'),
        dspy.Example(
            prompt="Summarize for section 'Key Risks'. Cite using [S#] matching the source indices above.",
            sources_digest="S1 | Risk Assessment Report — analyst.com\nSupply chain disruptions pose the highest risk, with 67% of companies reporting delays.\n\nS2 | Security Analysis — security.org\nCybersecurity incidents increased 45% in H1 2024 compared to 2023.",
            output_json='{"bullets": ["Supply chain disruptions pose highest risk, with 67% of companies reporting delays [S1]", "Cybersecurity incidents increased 45% in H1 2024 vs 2023 [S2]"]}'
        ).with_inputs('prompt', 'sources_digest'),
    ]

def trainset_write_section() -> List[dspy.Example]:
    """Training examples for WRITE_SECTION: evidence to markdown with citations."""
    return [
        dspy.Example(
            section_title="Market Trends",
            section_instructions="2024-2025 trends with specific data points and citations",
            evidence_digest="Evidence for 'Market Trends':\n- Cloud adoption growing 25% annually [1]\n- AI integration became standard in 78% of new products [2]\n- Mobile-first approach now 89% of development strategies [3]",
            output_markdown='# Market Trends\n\nThe market has experienced significant transformation in 2024-2025. Cloud adoption continues to accelerate at 25% annual growth [1], fundamentally changing infrastructure strategies across enterprises. Artificial intelligence has moved from experimental to essential, with 78% of new products now integrating AI capabilities as a standard feature [2].\n\nDevelopment approaches have also shifted dramatically toward mobile-first strategies, with 89% of teams now prioritizing mobile experiences in their product roadmaps [3]. This represents a maturation of mobile technology from an afterthought to the primary delivery platform for most digital services.'
        ).with_inputs('section_title', 'section_instructions', 'evidence_digest'),
        dspy.Example(
            section_title="Competitive Landscape",
            section_instructions="Top 5 players with market share and differentiators",
            evidence_digest="Evidence for 'Competitive Landscape':\n- Company A leads with 32% market share, strong in enterprise [1]\n- Company B holds 24%, focused on SMB segment [2]\n- Company C at 18%, competing on price and features [3]\n- Companies D and E combined hold 15%, niche players [4]",
            output_markdown='# Competitive Landscape\n\nThe market is led by Company A with a commanding 32% market share, leveraging strong enterprise relationships and established trust [1]. Company B follows with 24% market share, having carved out a successful position in the SMB segment through targeted product features and pricing [2].\n\nCompany C holds 18% of the market, competing aggressively on both price and feature completeness [3]. The remaining market is fragmented among smaller players, with Companies D and E combining for 15% share, each serving specific niche segments [4].'
        ).with_inputs('section_title', 'section_instructions', 'evidence_digest'),
    ]

def trainset_gap_analyzer() -> List[dspy.Example]:
    """Training examples for GAP_ANALYZER: decide if more research needed."""
    return [
        dspy.Example(
            section_title="Market Size",
            bullets_digest="- Market size $500B in 2024\n- Growing 15% YoY\n- North America 45% share\n- Asia Pacific 30% share\n- Europe 20% share\n- Forecast to reach $850B by 2027",
            output_json='{"need_more": false, "followup_queries": []}'
        ).with_inputs('section_title', 'bullets_digest'),
        dspy.Example(
            section_title="Technology Stack",
            bullets_digest="- React most popular framework\n- Python growing in backend",
            output_json='{"need_more": true, "followup_queries": ["infrastructure technologies 2024", "database trends and adoption", "cloud platform market share", "DevOps tool usage statistics"]}'
        ).with_inputs('section_title', 'bullets_digest'),
        dspy.Example(
            section_title="Risks",
            bullets_digest="No bullets yet.",
            output_json='{"need_more": true, "followup_queries": ["industry risks 2024", "compliance challenges", "supply chain vulnerabilities", "cybersecurity threats report", "market disruption factors"]}'
        ).with_inputs('section_title', 'bullets_digest'),
    ]

def trainset_cite_fixer() -> List[dspy.Example]:
    """Training examples for CITE_FIXER: convert [S#] to [n]."""
    return [
        dspy.Example(
            markdown_body="# Findings\n\nResearch shows significant growth [S1] with adoption increasing [S2]. The trend continues [S1] into 2025.",
            id_map_notes="1 -> https://example.com/report\n2 -> https://research.org/study",
            fixed_markdown="# Findings\n\nResearch shows significant growth [1] with adoption increasing [2]. The trend continues [1] into 2025."
        ).with_inputs('markdown_body', 'id_map_notes'),
        dspy.Example(
            markdown_body="# Market Analysis\n\nThe market leader [S3] holds 40% share, while challenger [S1] has 25%. Recent data [S2] confirms this distribution [S3].",
            id_map_notes="1 -> https://news.com/article\n2 -> https://data.org/stats\n3 -> https://market.com/report",
            fixed_markdown="# Market Analysis\n\nThe market leader [3] holds 40% share, while challenger [1] has 25%. Recent data [2] confirms this distribution [3]."
        ).with_inputs('markdown_body', 'id_map_notes'),
    ]

def trainset_reviewer() -> List[dspy.Example]:
    """Training examples for REVIEWER: review reports for quality."""
    return [
        dspy.Example(
            report_md="# Executive Summary\n\nThis report analyzes market trends based on comprehensive research [1][2][3].\n\n# Market Size\n\nThe global market reached $500B in 2024, growing 15% YoY [1]. Regional distribution shows North America at 45%, Asia Pacific 30%, and Europe 20% [2].\n\n## References\n[1] Market Report 2024 — example.com. https://example.com/report\n[2] Regional Analysis — research.org. https://research.org/study\n[3] Industry Forecast — analyst.com. https://analyst.com/forecast",
            output_json='{"pass_checks": true, "issues": [], "suggestions": ["Consider adding specific date ranges for projections", "Could expand on Asia Pacific growth drivers"], "summary": "Well-structured report with good citation coverage and clear regional breakdown."}'
        ).with_inputs('report_md'),
        dspy.Example(
            report_md="# Market Overview\n\nThe market is growing fast. Many companies are entering the space. https://example.com shows good data.\n\n## References",
            output_json='{"pass_checks": false, "issues": ["No specific data or citations in body", "Bare URL in prose instead of numeric citation", "Empty references section", "Vague language without concrete facts"], "suggestions": ["Add specific growth percentages with citations", "Replace bare URL with [n] citation and add to references", "Include concrete company names and market shares", "Provide time-specific data points"], "summary": "Report lacks specificity, proper citations, and concrete data. Needs major revision."}'
        ).with_inputs('report_md'),
    ]

def trainset_reviser() -> List[dspy.Example]:
    """Training examples for REVISER: improve reports based on feedback."""
    return [
        dspy.Example(
            report_md="# Market Size\n\nThe market is big and growing. Many companies participate.\n\n## References\n[1] Market Report — example.com. https://example.com",
            suggestions="- Add specific market size figures with citations\n- Include growth percentages and timeframes\n- Specify key player names",
            improved_md="# Market Size\n\nThe global market reached $500 billion in 2024, representing 15% year-over-year growth [1]. The market is characterized by strong participation from both established enterprises and emerging startups, with the top 5 players accounting for 68% of total market share [1]."
        ).with_inputs('report_md', 'suggestions'),
        dspy.Example(
            report_md="# Technology Trends\n\nAI is popular. Cloud computing is used a lot. https://tech.com has more info.\n\n## References\n[1] Tech Report — tech.com. https://tech.com",
            suggestions="- Replace bare URL with proper citation\n- Add specific adoption percentages\n- Include concrete technology names",
            improved_md="# Technology Trends\n\nArtificial intelligence adoption has reached 78% among enterprise software products in 2024 [1]. Cloud infrastructure utilization continues to grow, with 89% of organizations now operating hybrid or multi-cloud environments [1]. Key technologies driving this transformation include transformer-based language models, containerization platforms like Kubernetes, and serverless computing architectures [1]."
        ).with_inputs('report_md', 'suggestions'),
    ]

# ----------------------------
# Exa search + contents
# ----------------------------

async def exa_search_and_contents(query: str, k: int) -> List[SourceDoc]:
    """
    Use Exa's search_and_contents to get top-k results with full text.
    Python SDK fields: result.url, result.title, result.published_date, result.text. (See SDK spec.)
    """
    def _call():
        # You can add: category="news"/"research paper", start_published_date=..., type="auto"
        return EXA.search_and_contents(query, text=True, num_results=k)
    try:
        resp = await asyncio.to_thread(_call)
    except Exception as e:
        print(f"[Exa] search failed for '{query}': {e}")
        return []
    docs: List[SourceDoc] = []
    for r in getattr(resp, "results", []) or []:
        text = (getattr(r, "text", None) or "")[:MAX_CONTENT_CHARS_PER_SOURCE]
        if not text or len(text) < 200:
            continue
        url = getattr(r, "url", None) or getattr(r, "id", None)
        title = getattr(r, "title", None)
        pub = getattr(r, "published_date", None)
        # Normalize published to ISO date if possible
        if pub:
            try: pub = dtparse.parse(pub).date().isoformat()
            except Exception: pass
        docs.append(SourceDoc(url=url, title=title, site=short_host(url), published=pub, content=text))
    return docs

# ----------------------------
# Citation registry
# ----------------------------

class CitationRegistry:
    def __init__(self): self.url_to_id: Dict[str, int] = {}; self.ordered: List[str] = []
    def assign(self, url: str) -> int:
        if url not in self.url_to_id:
            self.url_to_id[url] = len(self.ordered) + 1
            self.ordered.append(url)
        return self.url_to_id[url]
    def references_markdown(self, url_to_doc: Dict[str, SourceDoc]) -> str:
        lines = ["## References"]
        for u in self.ordered:
            idx = self.url_to_id[u]; doc = url_to_doc.get(u) or SourceDoc(url=u)
            label = doc.title or u; site = f" — {doc.site}" if doc.site else ""
            dt = f" (published {doc.published})" if doc.published else ""
            lines.append(f"[{idx}] {label}{site}{dt}. {u}")
        return "\n".join(lines)

# ----------------------------
# Graph state
# ----------------------------

class GraphState(TypedDict):
    topic: str
    sections: List[SectionSpec]
    round: int
    queries: Annotated[List[Dict[str, str]], operator.add]            # [{"section","query"}]
    research: Annotated[List[ResearchSummary], operator.add]          # append
    drafts: Annotated[Dict[str, str], operator.or_]                   # {section: markdown}
    cite_maps: Annotated[Dict[str, Dict[int, str]], operator.or_]     # {section: {local_num: url}}
    used_urls: Annotated[List[str], operator.add]                     # optional
    report_md: Optional[str]
    references_md: Optional[str]
    eval_result: Optional[EvalResult]

# ----------------------------
# Nodes (agents)
# ----------------------------

def plan_queries(state: GraphState) -> GraphState:
    print("\n" + "="*80)
    print(f"[QUERY] Planning search queries for {len(state['sections'])} sections...")
    print("="*80)

    new_queries: List[Dict[str, str]] = []
    with dspy.context(lm=WRITER_LM):
        for sec in state["sections"]:
            q = QUERY_GEN(section_title=sec.name, section_instructions=sec.instructions)
            data = safe_json_loads(q.queries_json, [])
            uniq, seen = [], set()
            for s in data:
                if isinstance(s, str):
                    s2 = s.strip()
                    if s2 and s2.lower() not in seen:
                        uniq.append(s2); seen.add(s2.lower())
                    if len(uniq) >= 8: break
            if not uniq:
                uniq = [f'{sec.name} overview', f'{sec.name} trends 2024..2025', f'"{sec.name}" case studies', f'intitle:{sec.name} report PDF', f'site:.gov {sec.name}', f'site:.org {sec.name}']
            print(f"[QUERY] {sec.name}: {len(uniq)} queries generated")
            for u in uniq[:8]:
                new_queries.append({"section": sec.name, "query": u})

    print(f"[QUERY] Total queries planned: {len(new_queries)}\n")
    return {"queries": new_queries}

def route_queries(state: GraphState):
    return [Send("search_node", {"section": item["section"], "query": item["query"]})
            for item in state.get("queries", [])]

async def search_node(state: GraphState) -> GraphState:
    """Exa search + contents + summarize (Flash)."""
    section, query = state["section"], state["query"]
    print(f"[SEARCH] '{section}': {query[:60]}{'...' if len(query) > 60 else ''}")

    docs = await exa_search_and_contents(query, k=SEARCH_RESULTS_PER_QUERY)
    print(f"[SEARCH] → Found {len(docs)} documents")

    # Build digest for LLM summarization: S1,S2,... per query
    pieces = []
    for i, d in enumerate(docs, start=1):
        excerpt = (d.content or "")[:2000]
        pieces.append(f"S{i} | {d.title or d.url} — {d.site or ''}\n{excerpt}")
    sources_digest = "\n\n".join(pieces) if pieces else "NO_SOURCES"
    prompt = f"Summarize for section '{section}'. Cite using [S#] matching the source indices above."

    with dspy.context(lm=RESEARCH_LM):
        out = SUMMARIZER(prompt=prompt, sources_digest=sources_digest)
    js = safe_json_loads(out.output_json, {}) or {}
    bullets = js.get("bullets", [])

    print(f"[SEARCH] → Extracted {len(bullets)} evidence bullets\n")
    return {"research": [ResearchSummary(section=section, query=query, bullets=bullets, sources=docs)]}

def merge_and_gap_analyze(state: GraphState) -> GraphState:
    print("\n" + "="*80)
    print(f"[GAP] Analyzing research coverage (Round {state['round'] + 1}/{MAX_ROUNDS})...")
    print("="*80)

    sec_to_bullets: Dict[str, List[str]] = {}
    for rs in state.get("research", []):
        sec_to_bullets.setdefault(rs.section, []).extend(rs.bullets or [])

    followups: List[Dict[str, str]] = []
    with dspy.context(lm=WRITER_LM):
        for sec in state["sections"]:
            bullets = sec_to_bullets.get(sec.name, [])
            digest = "\n".join(f"- {b}" for b in bullets[:50]) if bullets else "No bullets yet."
            print(f"[GAP] '{sec.name}': {len(bullets)} bullets collected")

            ga = GAP_ANALYZER(section_title=sec.name, bullets_digest=digest)
            j = safe_json_loads(ga.output_json, {}) or {}
            if j.get("need_more") and isinstance(j.get("followup_queries"), list):
                new_queries = [q for q in j["followup_queries"][:5] if isinstance(q, str) and q.strip()]
                if new_queries:
                    print(f"[GAP] → Needs {len(new_queries)} more queries")
                    for q in new_queries:
                        followups.append({"section": sec.name, "query": q.strip()})

    if followups and state["round"] + 1 < MAX_ROUNDS:
        print(f"\n[GAP] DECISION: Continue research with {len(followups)} followup queries\n")
        return {"queries": followups, "round": state["round"] + 1}

    print("\n[GAP] DECISION: Research complete, proceeding to writing\n")
    return {"queries": []}  # clear queries and proceed to writing

def route_or_write(state: GraphState):
    if any(state.get("queries", [])) and state["round"] > 0:
        return route_queries(state)   # more research
    # else: write each section in parallel
    return [Send("write_section_node", {
        "section": s.name,
        "sections": state["sections"],
        "research": state.get("research", [])
    }) for s in state["sections"]]

def _build_evidence_digest(section: str, research: List[ResearchSummary]) -> Tuple[str, Dict[int, str]]:
    """
    Merge bullets across queries and create a local S#->url mapping for the writer pass.
    We number S1..Sk per research summary (query) and rewrite bullets remains [S#] (writer learns mapping).
    """
    lines = [f"Evidence for '{section}':"]
    s_to_url_global: Dict[int, str] = {}
    next_num = 1
    for rs in research:
        if rs.section != section: continue
        # map local S# for this query block to absolute local numbers for the section
        local_map = {}
        for d in rs.sources:
            local_map[f"S{len(local_map)+1}"] = d.url
        for b in rs.bullets:
            bb = b
            for s_id, url in local_map.items():
                # assign a stable number for this section for each url
                if url not in s_to_url_global.values():
                    s_to_url_global[next_num] = url; assigned = next_num; next_num += 1
                else:
                    # find existing number for this url
                    assigned = [k for k,v in s_to_url_global.items() if v == url][0]
                bb = re.sub(rf"\[{s_id}\]", f"[{assigned}]", bb)
            lines.append(f"- {bb}")
    return "\n".join(lines), s_to_url_global  # evidence digest, local map num->url

def write_section_node(state: GraphState) -> GraphState:
    section = state["section"]
    sec_spec = next((s for s in state["sections"] if s.name == section), None)
    if not sec_spec: return {}

    print(f"[WRITE] Drafting section: '{section}'")

    edigest, local_num_to_url = _build_evidence_digest(section, state.get("research", []))

    with dspy.context(lm=WRITER_LM):
        w = WRITE_SECTION(section_title=sec_spec.name, section_instructions=sec_spec.instructions, evidence_digest=edigest)
        md = w.output_markdown or f"# {section}\n\n*(No content was generated.)*\n"

    with dspy.context(lm=WRITER_LM):
        fixed = CITE_FIXER(markdown_body=md, id_map_notes="\n".join(f"{k} -> {v}" for k, v in local_num_to_url.items()))
        md2 = fixed.fixed_markdown or md

    used_ids = sorted(set(int(x) for x in re.findall(r"\[(\d+)\]", md2)))
    urls = [local_num_to_url.get(i) for i in used_ids if i in local_num_to_url]

    char_count = len(md2)
    cite_count = len(used_ids)
    print(f"[WRITE] → {char_count} chars, {cite_count} citations\n")

    # store section draft, local citation map (for final global renumber), and used urls
    return {
        "drafts": {section: md2},
        "cite_maps": {section: local_num_to_url},
        "used_urls": [u for u in urls if u]
    }

def assemble_and_review(state: GraphState) -> GraphState:
    print("\n" + "="*80)
    print("[REVIEW] Assembling and reviewing final report...")
    print("="*80)

    order = [s.name for s in state["sections"]]
    global_reg = CitationRegistry()
    url_to_doc: Dict[str, SourceDoc] = {}

    # Build a metadata table for all discovered sources
    for rs in state.get("research", []):
        for d in rs.sources:
            url_to_doc[d.url] = d

    print(f"[REVIEW] Assembling {len(order)} sections...")

    # Renumber citations globally in order of their first appearance across sections
    def renumber_section(md: str, map_local: Dict[int, str]) -> str:
        def _repl(m):
            old_num = int(m.group(1))
            url = map_local.get(old_num)
            if not url: return m.group(0)
            new_num = global_reg.assign(url)
            return f"[{new_num}]"
        # only replace inside the body (no references exist yet)
        return re.sub(r"\[(\d+)\]", _repl, md)

    parts = []
    for sec in order:
        body = state["drafts"].get(sec, "")
        local_map = state.get("cite_maps", {}).get(sec, {})
        parts.append(renumber_section(body, local_map))
    body_renumbered = "\n\n".join(parts).strip()

    refs = global_reg.references_markdown(url_to_doc)
    full_md = f"{body_renumbered}\n\n{refs}"

    total_citations = len(global_reg.ordered)
    total_chars = len(full_md)
    print(f"[REVIEW] → {total_chars} chars, {total_citations} unique sources")

    # Review & optional revise
    print("[REVIEW] Running quality review...")
    with dspy.context(lm=WRITER_LM):
        rv = REVIEWER(report_md=full_md)
    rj = safe_json_loads(rv.output_json, {}) or {}

    pass_checks = rj.get("pass_checks", False)
    issues = rj.get("issues", [])
    suggestions = rj.get("suggestions", [])

    print(f"[REVIEW] → Pass: {pass_checks}, Issues: {len(issues)}, Suggestions: {len(suggestions)}")

    if not pass_checks and suggestions:
        print(f"[REVIEW] Applying {len(suggestions)} revision suggestions...")
        with dspy.context(lm=WRITER_LM):
            rev = REVISER(report_md=full_md, suggestions="\n".join(f"- {s}" for s in suggestions))
        full_md = (rev.improved_md or body_renumbered).strip() + "\n\n" + refs
        print("[REVIEW] → Revision complete")

    print()
    return {"report_md": full_md, "references_md": refs}

# ----------------------------
# Evaluation
# ----------------------------

DEFAULT_EVAL_QUESTIONS = [
    "Does each section follow the instructions and include concrete facts?",
    "Are all nontrivial claims cited with [n] and do references look reputable?",
    "Is the structure clear with helpful headings/subheadings?",
    "Are there explicit dates for time-sensitive facts?",
    "Are there at least 2–3 sources per major section?",
    "Are URLs omitted from the prose (only numeric citations)?",
    "Is there any hallucination smell?",
]

def eval_report_simple(md: str) -> EvalResult:
    checks = {}
    checks["has_h1"] = 1.0 if re.search(r"^#\s+", md, flags=re.M) else 0.0
    cites = len(re.findall(r"\[\d+\]", md))
    checks["enough_cites"] = clamp(cites/10)
    checks["no_raw_urls"] = 1.0 if not re.search(r"https?://", md.split("## References")[0]) else 0.0
    checks["has_refs"] = 1.0 if "## References" in md else 0.0
    checks["length_ok"] = 1.0 if len(md) >= 2000 else 0.4 if len(md) >= 1200 else 0.1
    score = sum(checks.values())/len(checks)
    return EvalResult(score=score, breakdown=checks, notes=f"{cites} inline citations; {len(md)} chars.")

# ----------------------------
# Build graph
# ----------------------------

def build_graph() -> Any:
    graph = StateGraph(GraphState)
    graph.add_node("plan_queries", plan_queries)
    graph.add_node("search_node", search_node)  # async
    graph.add_node("merge_and_gap_analyze", merge_and_gap_analyze)
    graph.add_node("write_section_node", write_section_node)
    graph.add_node("assemble_and_review", assemble_and_review)

    graph.add_edge(START, "plan_queries")
    graph.add_conditional_edges("plan_queries", route_queries, ["search_node"])
    graph.add_edge("search_node", "merge_and_gap_analyze")
    graph.add_conditional_edges("merge_and_gap_analyze", route_or_write, ["search_node", "write_section_node"])
    graph.add_edge("write_section_node", "assemble_and_review")
    graph.add_edge("assemble_and_review", END)
    return graph.compile()

# ----------------------------
# Run end-to-end
# ----------------------------

async def run_pipeline(topic: str, sections: List[SectionSpec], optimization: bool = False) -> Dict[str, Any]:
    print("\n" + "="*80)
    print(f"RESEARCH PIPELINE: {topic}")
    print("="*80)
    print(f"Sections: {', '.join([s.name for s in sections])}")
    print("="*80 + "\n")

    # GEPA optimization with module-specific training sets
    if optimization:
        print("[GEPA] Starting prompt optimization...")
        optimize_with_gepa()
        print("[GEPA] Optimization complete!\n")

    print("[PIPELINE] Building research graph...")
    app = build_graph()
    initial_state: GraphState = {
        "topic": topic,
        "sections": sections,
        "round": 0,
        "queries": [],
        "research": [],
        "drafts": {},
        "cite_maps": {},
        "used_urls": [],
        "report_md": None,
        "references_md": None,
        "eval_result": None,
    }

    print("[PIPELINE] Executing multi-agent research workflow...\n")
    final_state: GraphState = await app.ainvoke(initial_state)

    print("\n" + "="*80)
    print("[PIPELINE] Evaluating report quality...")
    print("="*80)

    # Evaluate
    md = final_state.get("report_md") or ""
    final_state["eval_result"] = eval_report_simple(md)

    # Save
    print("[PIPELINE] Saving report to ./report.md")
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("[PIPELINE] ✓ Pipeline complete!\n")
    return final_state

# ----------------------------
# Example usage
# ----------------------------

SECTIONS = [
    SectionSpec(
        name="Executive Summary",
        instructions="In 180–250 words, summarize the most decision-relevant takeaways. No citations here unless needed for key numbers."
    ),
    SectionSpec(
        name="Market Landscape",
        instructions="Define the space; 2023–2025 trends; include 4+ specific figures with sources."
    ),
    SectionSpec(
        name="Key Players & Differentiation",
        instructions="Compare 5–7 players; list 1–2 distinctive capabilities each; add 2–3 objective benchmarks with citations."
    ),
    SectionSpec(
        name="Risks & Open Questions",
        instructions="Top risks, unknowns, and watch items; cite evidence; use bullets."
    ),
    SectionSpec(
        name="Outlook (12–24 months)",
        instructions="3–5 grounded predictions with supporting evidence and explicit dates; include leading indicators to track."
    ),
]

if __name__ == "__main__":
    topic = "State of Edge AI Acceleration (2024–2025)"
    try:
        final = asyncio.run(run_pipeline(
            topic=topic, 
            sections=SECTIONS, 
            # optimization=True         # Uncomment to enable GEPA optimization
        ))
    except Exception as e:
        print("Pipeline failed:", e)
        raise

    print("\n" + "="*88)
    print("FINAL MARKDOWN (also saved to ./report.md):")
    print("="*88 + "\n")
    print(final.get("report_md", ""))

    ev: EvalResult = final.get("eval_result") or EvalResult(score=0.0, breakdown={}, notes="")
    print("\n" + "-"*88)
    print("EVALUATION (quick heuristic):")
    print("-"*88)
    print(f"Score: {ev.score:.2f}")
    print("Breakdown:", json.dumps(ev.breakdown, indent=2))
    print("Notes:", ev.notes)
