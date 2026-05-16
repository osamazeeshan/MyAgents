"""Tool functions that research agents can call."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import function_tool

from .config import load_settings

_SAFE_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")
_HTTP_TIMEOUT_SECONDS = 20
_USER_AGENT = "research-agents/0.1 (+https://github.com/openai/openai-agents-python)"
_MAX_VERIFIED_PAPER_RESULTS = 20


PaperRecord = dict[str, Any]


def _slugify(value: str) -> str:
    slug = _SAFE_SLUG_PATTERN.sub("-", value.strip().lower()).strip("-._")
    return slug or "research-note"


def _fetch_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _paper_key(paper: PaperRecord) -> str:
    doi = _clean_text(paper.get("doi")).lower()
    if doi:
        return f"doi:{doi}"
    return f"title:{_normalize_title(_clean_text(paper.get('title')))}"


def _format_authors(authors: list[str], limit: int = 8) -> str:
    authors = [author for author in authors if author]
    if not authors:
        return "Unknown authors"
    if len(authors) <= limit:
        return ", ".join(authors)
    return f"{', '.join(authors[:limit])}, et al."


def _score_paper(paper: PaperRecord, topic: str) -> tuple[int, int, int, str]:
    topic_terms = {term for term in re.findall(r"[a-z0-9]+", topic.lower()) if len(term) > 2}
    title_terms = set(re.findall(r"[a-z0-9]+", _clean_text(paper.get("title")).lower()))
    abstract_terms = set(re.findall(r"[a-z0-9]+", _clean_text(paper.get("abstract")).lower()))
    overlap = len(topic_terms & (title_terms | abstract_terms))
    citations = int(paper.get("citation_count") or 0)
    year = int(paper.get("year") or 0)
    return (overlap, citations, year, _clean_text(paper.get("title")))


def _semantic_scholar_search(topic: str, start_year: int, end_year: int, limit: int) -> list[PaperRecord]:
    fields = ",".join(
        [
            "title",
            "authors",
            "year",
            "venue",
            "url",
            "externalIds",
            "abstract",
            "citationCount",
            "publicationTypes",
            "openAccessPdf",
        ]
    )
    params = urllib.parse.urlencode(
        {
            "query": topic,
            "year": f"{start_year}-{end_year}",
            "limit": min(max(limit, 1), 100),
            "fields": fields,
        }
    )
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    data = _fetch_json(url)
    records: list[PaperRecord] = []
    for item in data.get("data", []):
        title = _clean_text(item.get("title"))
        year = item.get("year")
        if not title or not year or not (start_year <= int(year) <= end_year):
            continue
        external_ids = item.get("externalIds") or {}
        doi = _clean_text(external_ids.get("DOI"))
        arxiv_id = _clean_text(external_ids.get("ArXiv"))
        pdf = item.get("openAccessPdf") or {}
        records.append(
            {
                "title": title,
                "authors": [_clean_text(author.get("name")) for author in item.get("authors", [])],
                "year": int(year),
                "venue": _clean_text(item.get("venue")) or "Semantic Scholar record",
                "url": _clean_text(item.get("url")) or (f"https://doi.org/{doi}" if doi else ""),
                "doi": doi,
                "arxiv_id": arxiv_id,
                "abstract": _clean_text(item.get("abstract")),
                "citation_count": int(item.get("citationCount") or 0),
                "source": "Semantic Scholar",
                "pdf_url": _clean_text(pdf.get("url")),
            }
        )
    return records


def _openalex_search(topic: str, start_year: int, end_year: int, limit: int) -> list[PaperRecord]:
    params = urllib.parse.urlencode(
        {
            "search": topic,
            "filter": ",".join(
                [
                    f"from_publication_date:{start_year}-01-01",
                    f"to_publication_date:{end_year}-12-31",
                    "type:article|preprint",
                ]
            ),
            "per-page": min(max(limit, 1), 200),
            "sort": "relevance_score:desc",
        }
    )
    url = f"https://api.openalex.org/works?{params}"
    data = _fetch_json(url)
    records: list[PaperRecord] = []
    for item in data.get("results", []):
        title = _clean_text(item.get("title") or item.get("display_name"))
        year = item.get("publication_year")
        if not title or not year or not (start_year <= int(year) <= end_year):
            continue
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}
        doi_url = _clean_text(item.get("doi"))
        doi = doi_url.removeprefix("https://doi.org/") if doi_url else ""
        authorships = item.get("authorships") or []
        authors = [
            _clean_text((authorship.get("author") or {}).get("display_name"))
            for authorship in authorships
        ]
        records.append(
            {
                "title": title,
                "authors": authors,
                "year": int(year),
                "venue": _clean_text(source.get("display_name")) or "OpenAlex record",
                "url": _clean_text(item.get("id")) or doi_url,
                "doi": doi,
                "arxiv_id": "",
                "abstract": "",
                "citation_count": int(item.get("cited_by_count") or 0),
                "source": "OpenAlex",
                "pdf_url": _clean_text((primary_location.get("pdf_url") or "")),
            }
        )
    return records


def _crossref_search(topic: str, start_year: int, end_year: int, limit: int) -> list[PaperRecord]:
    params = urllib.parse.urlencode(
        {
            "query.title": topic,
            "filter": f"from-pub-date:{start_year}-01-01,until-pub-date:{end_year}-12-31",
            "rows": min(max(limit, 1), 100),
            "sort": "relevance",
            "order": "desc",
        }
    )
    url = f"https://api.crossref.org/works?{params}"
    data = _fetch_json(url)
    records: list[PaperRecord] = []
    for item in (data.get("message") or {}).get("items", []):
        titles = item.get("title") or []
        title = _clean_text(titles[0] if titles else "")
        issued = item.get("issued") or item.get("published-print") or item.get("published-online") or {}
        date_parts = issued.get("date-parts") or []
        year = date_parts[0][0] if date_parts and date_parts[0] else None
        if not title or not year or not (start_year <= int(year) <= end_year):
            continue
        authors = []
        for author in item.get("author") or []:
            given = _clean_text(author.get("given"))
            family = _clean_text(author.get("family"))
            authors.append(_clean_text(f"{given} {family}"))
        doi = _clean_text(item.get("DOI"))
        container_titles = item.get("container-title") or []
        records.append(
            {
                "title": title,
                "authors": authors,
                "year": int(year),
                "venue": _clean_text(container_titles[0] if container_titles else "Crossref record"),
                "url": _clean_text(item.get("URL")) or (f"https://doi.org/{doi}" if doi else ""),
                "doi": doi,
                "arxiv_id": "",
                "abstract": _clean_text(item.get("abstract")),
                "citation_count": int(item.get("is-referenced-by-count") or 0),
                "source": "Crossref",
                "pdf_url": "",
            }
        )
    return records


def _arxiv_search(topic: str, start_year: int, end_year: int, limit: int) -> list[PaperRecord]:
    query = urllib.parse.quote(f'all:"{topic}"')
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query={query}&start=0&max_results={min(max(limit, 1), 50)}"
        "&sortBy=relevance&sortOrder=descending"
    )
    text = _fetch_text(url)
    root = ET.fromstring(text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    records: list[PaperRecord] = []
    for entry in root.findall("atom:entry", ns):
        title = _clean_text(entry.findtext("atom:title", default="", namespaces=ns))
        published = _clean_text(entry.findtext("atom:published", default="", namespaces=ns))
        year = int(published[:4]) if published[:4].isdigit() else 0
        if not title or not (start_year <= year <= end_year):
            continue
        authors = [
            _clean_text(author.findtext("atom:name", default="", namespaces=ns))
            for author in entry.findall("atom:author", ns)
        ]
        entry_id = _clean_text(entry.findtext("atom:id", default="", namespaces=ns))
        arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else ""
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = _clean_text(link.attrib.get("href"))
        records.append(
            {
                "title": title,
                "authors": authors,
                "year": year,
                "venue": "arXiv preprint",
                "url": entry_id,
                "doi": "",
                "arxiv_id": arxiv_id,
                "abstract": _clean_text(entry.findtext("atom:summary", default="", namespaces=ns)),
                "citation_count": 0,
                "source": "arXiv",
                "pdf_url": pdf_url,
            }
        )
    return records


def _deduplicate_papers(papers: list[PaperRecord]) -> list[PaperRecord]:
    merged: dict[str, PaperRecord] = {}
    for paper in papers:
        key = _paper_key(paper)
        if key == "title:":
            continue
        existing = merged.get(key)
        if existing is None:
            merged[key] = paper
            continue
        existing_sources = set(_clean_text(existing.get("source")).split(" + "))
        existing_sources.add(_clean_text(paper.get("source")))
        existing["source"] = " + ".join(sorted(source for source in existing_sources if source))
        for field in ["doi", "arxiv_id", "abstract", "url", "pdf_url", "venue"]:
            if not existing.get(field) and paper.get(field):
                existing[field] = paper[field]
        existing["citation_count"] = max(
            int(existing.get("citation_count") or 0), int(paper.get("citation_count") or 0)
        )
        if len(paper.get("authors") or []) > len(existing.get("authors") or []):
            existing["authors"] = paper["authors"]
    return list(merged.values())


def search_verified_recent_papers_markdown(
    topic: str,
    start_year: int,
    end_year: int,
    max_results: int = _MAX_VERIFIED_PAPER_RESULTS,
) -> str:
    """Search multiple scholarly indexes and return only traceable paper records."""

    topic = topic.strip()
    if not topic:
        raise ValueError("topic must not be empty")
    if topic.isdigit():
        raise ValueError(
            "topic must be resolved before searching; got only a numeric menu selection"
        )
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    max_results = min(max(max_results, 1), _MAX_VERIFIED_PAPER_RESULTS)

    errors: list[str] = []
    papers: list[PaperRecord] = []
    sources = [
        ("Semantic Scholar", _semantic_scholar_search),
        ("OpenAlex", _openalex_search),
        ("Crossref", _crossref_search),
        ("arXiv", _arxiv_search),
    ]
    per_source_limit = max(max_results, 10)
    for name, search in sources:
        try:
            papers.extend(search(topic, start_year, end_year, per_source_limit))
            if name == "arXiv":
                # arXiv politely asks clients not to make rapid repeated requests.
                time.sleep(0.25)
        except (urllib.error.URLError, TimeoutError, ValueError, ET.ParseError, json.JSONDecodeError) as exc:
            errors.append(f"{name}: {exc}")

    papers = _deduplicate_papers(papers)
    papers.sort(key=lambda paper: _score_paper(paper, topic), reverse=True)
    papers = papers[: max(max_results, 1)]

    lines = [
        f"# Verified paper search for: {topic}",
        "",
        f"Year window: {start_year}-{end_year}",
        f"Result cap: {max_results} most relevant, recent, and/or highly cited papers.",
        "Search sources: Semantic Scholar Graph API, OpenAlex Works API, Crossref Works API, arXiv API.",
        "Verification rule: every listed paper must include a source URL, DOI, or arXiv ID returned by an external index.",
        "",
    ]
    if errors:
        lines.extend(["## Source warnings", ""])
        lines.extend(f"- {error}" for error in errors)
        lines.append("")

    if not papers:
        lines.extend(
            [
                "## Verified papers",
                "",
                "No verified papers were returned by the configured scholarly indexes.",
                "Broaden the topic wording or run the workflow with hosted web search enabled.",
            ]
        )
        return "\n".join(lines)

    lines.extend(["## Verified papers", ""])
    for index, paper in enumerate(papers, start=1):
        doi = _clean_text(paper.get("doi"))
        arxiv_id = _clean_text(paper.get("arxiv_id"))
        url = _clean_text(paper.get("url")) or (f"https://doi.org/{doi}" if doi else "")
        identifiers = []
        if doi:
            identifiers.append(f"DOI: {doi}")
        if arxiv_id:
            identifiers.append(f"arXiv: {arxiv_id}")
        if paper.get("pdf_url"):
            identifiers.append(f"PDF: {paper['pdf_url']}")
        abstract = _clean_text(paper.get("abstract"))
        if len(abstract) > 700:
            abstract = f"{abstract[:697].rstrip()}..."
        lines.extend(
            [
                f"{index}. **{_clean_text(paper.get('title'))}** ({paper.get('year')})",
                f"   - Authors: {_format_authors(paper.get('authors') or [])}",
                f"   - Venue/source: {_clean_text(paper.get('venue'))}",
                f"   - Record URL: {url or 'Unavailable'}",
                f"   - Identifiers: {', '.join(identifiers) if identifiers else 'External index record only'}",
                f"   - Citation count in index: {int(paper.get('citation_count') or 0)}",
                f"   - Found via: {_clean_text(paper.get('source'))}",
            ]
        )
        if abstract:
            lines.append(f"   - Abstract snippet: {abstract}")
        lines.append("")

    return "\n".join(lines).rstrip()


@function_tool
def save_research_note(title: str, body: str) -> str:
    """Save a markdown research note locally and return its file path."""

    settings = load_settings()
    settings.notes_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = settings.notes_dir / f"{timestamp}-{_slugify(title)}.md"
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")
    return str(path)


@function_tool
def build_literature_search_query(topic: str, method: str = "broad") -> str:
    """Create a reusable literature-search query for a research topic."""

    topic = topic.strip()
    if method == "systematic":
        return f'("{topic}" OR related terminology) AND (review OR meta-analysis OR benchmark OR dataset)'
    if method == "recent":
        return f'("{topic}") AND (2024 OR 2025 OR 2026) AND (paper OR preprint OR proceedings)'
    return f'("{topic}") AND (survey OR benchmark OR framework OR evaluation OR evidence)'


@function_tool
def search_verified_recent_papers(
    topic: str,
    start_year: int,
    end_year: int,
    max_results: int = _MAX_VERIFIED_PAPER_RESULTS,
) -> str:
    """Search scholarly indexes and return only papers with external verification links."""

    return search_verified_recent_papers_markdown(
        topic=topic,
        start_year=start_year,
        end_year=end_year,
        max_results=max_results,
    )
