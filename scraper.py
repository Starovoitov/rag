from __future__ import annotations

import re

import requests
import trafilatura

from parser.models import ParsedDocument, SourceSpec


def scrape_source(source: SourceSpec, timeout: int = 30) -> ParsedDocument:
    """Fetch a URL and extract main readable text with fallback HTML cleaning."""
    response = requests.get(source.url, timeout=timeout)
    response.raise_for_status()

    downloaded = trafilatura.extract(
        response.text,
        include_comments=False,
        include_links=False,
        output_format="txt",
    )
    text = downloaded or fallback_clean_html(response.text)
    title = extract_title(response.text, source.url)

    return ParsedDocument(source=source, title=title, text=text)


def extract_title(html: str, default: str) -> str:
    """Extract HTML title text, falling back to provided default string."""
    match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return default
    return " ".join(match.group(1).split()).strip() or default


def fallback_clean_html(html: str) -> str:
    """Convert raw HTML into plain text when structured extraction fails."""
    no_script = re.sub(
        r"<(script|style).*?>.*?</\1>", " ", html, flags=re.IGNORECASE | re.DOTALL
    )
    no_tags = re.sub(r"<[^>]+>", " ", no_script)
    no_entities = re.sub(r"&[a-zA-Z0-9#]+;", " ", no_tags)
    return " ".join(no_entities.split()).strip()

