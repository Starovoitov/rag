from __future__ import annotations

import re
import time

import requests
try:
    import trafilatura
except ModuleNotFoundError:  # pragma: no cover
    trafilatura = None

from parser.models import ParsedDocument, SourceSpec

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def scrape_source(source: SourceSpec, timeout: int = 20, max_retries: int = 3) -> ParsedDocument:
    """Fetch a URL and extract main readable text with fallback HTML cleaning."""
    last_error: Exception | None = None
    response: requests.Response | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(source.url, headers=HEADERS, timeout=timeout)
            if response.status_code == 200:
                break
            last_error = requests.HTTPError(f"status={response.status_code} for url={source.url}")
        except requests.RequestException as exc:
            last_error = exc
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    if response is None or response.status_code != 200:
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to fetch source: {source.url}")
    response.raise_for_status()

    downloaded = None
    if trafilatura is not None:
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

