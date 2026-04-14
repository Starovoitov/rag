from __future__ import annotations

import re


NOISE_PATTERNS = [
    r"cookie(s)?\s+policy",
    r"accept\s+all\s+cookies",
    r"sign\s+in",
    r"skip\s+to\s+content",
    r"table\s+of\s+contents",
    r"all\s+rights\s+reserved",
]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    cleaned_lines = []
    for line in lines:
        compact = re.sub(r"\s+", " ", line).strip()
        if not compact:
            continue
        if is_noise(compact):
            continue
        cleaned_lines.append(compact)
    return "\n".join(cleaned_lines)


def is_noise(line: str) -> bool:
    """Check whether a line matches configured boilerplate noise patterns."""
    lowered = line.lower()
    return any(re.search(pattern, lowered) for pattern in NOISE_PATTERNS)

