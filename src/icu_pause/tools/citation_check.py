"""Deterministic citation verification for ICU-PAUSE output.

Three checks:
1. **Provenance** — every parenthetical source tag in the final output
   exists in the cite_registry (catches fabricated timestamps).
2. **Preservation** — source tags emitted by domain agents survive into
   the Intensivist's final output (catches rewrite-stripping).
3. **Deduplication** — collapses consecutive identical cite tags so
   "BP 118/72 (vital 1/17 08:00), HR 84 (vital 1/17 08:00)" becomes
   "BP 118/72, HR 84 (vital 1/17 08:00)".
"""

from __future__ import annotations

import re
from typing import Any

from icu_pause.data.context import CITE_PATTERN
from icu_pause.schemas.icu_pause import AgentSnippet


def check_citation_provenance(
    merged_sections: dict[str, str],
    cite_registry: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Verify every source tag in output exists in structured data.

    Returns a list of warning strings for unverifiable citations.
    """
    issues: list[str] = []
    for section_key, text in merged_sections.items():
        for match in CITE_PATTERN.finditer(text):
            if match.group() not in cite_registry:
                issues.append(
                    f"CITATION: Unverifiable source tag {match.group()} "
                    f"in section {section_key}"
                )
    return issues


def check_citation_preservation(
    agent_snippets: list[AgentSnippet],
    merged_sections: dict[str, str],
) -> list[str]:
    """Log agent citations that were stripped by the Intensivist.

    Non-blocking / informational — measures the preservation rate so you
    can tell whether the Intensivist prompt is working.

    Extracts cite tags from final output as a set (not substring search)
    to avoid false-flags from whitespace or punctuation differences.
    """
    merged_text = " ".join(merged_sections.values())
    final_cites: set[str] = set(CITE_PATTERN.findall(merged_text))

    dropped: list[str] = []
    for snippet in agent_snippets:
        for section in snippet.sections:
            for match in CITE_PATTERN.finditer(section.content):
                if match.group() not in final_cites:
                    dropped.append(
                        f"CITATION_DROPPED: {snippet.agent_name} emitted "
                        f"{match.group()}, absent from final output "
                        f"(section {section.section})"
                    )
    return dropped


# ---------------------------------------------------------------------------
# Citation expansion (split concatenated multi-source parens)
# ---------------------------------------------------------------------------


# Match a paren containing 2+ source tags joined by "; ", e.g.:
#   (assess 1-11 12:00; vital 1-12 07:00; resp 1-12 06:00)
# The renderer's CITE_PATTERN expects one source per paren, so we expand these.
_CONCAT_CITE_PATTERN = re.compile(
    r"\("
    r"(?:lab|vital|med|resp|assess|code|proc) \d{1,2}-\d{2} \d{2}:\d{2}"
    r"(?:; (?:lab|vital|med|resp|assess|code|proc) \d{1,2}-\d{2} \d{2}:\d{2})+"
    r"\)"
)


def expand_concatenated_citations(text: str) -> str:
    """Split semicolon-concatenated citation parens into separate parens.

    Before: "FiO2 0.3 (assess 1-11 12:00; vital 1-12 07:00; resp 1-12 06:00)"
    After:  "FiO2 0.3 (assess 1-11 12:00) (vital 1-12 07:00) (resp 1-12 06:00)"

    The intensivist sometimes consolidates multiple source tags into one paren,
    which the canonical CITE_PATTERN does not match. Expansion is run before
    deduplicate_citations so downstream renderers and judges see one tag per
    paren.
    """
    def _split(match: re.Match[str]) -> str:
        inner = match.group(0)[1:-1]  # strip outer parens
        parts = inner.split("; ")
        return " ".join(f"({p})" for p in parts)

    return _CONCAT_CITE_PATTERN.sub(_split, text)


# ---------------------------------------------------------------------------
# Citation deduplication
# ---------------------------------------------------------------------------


def deduplicate_citations(text: str) -> str:
    """Collapse consecutive identical cite tags into grouped form.

    Before: "BP 118/72 (vital 1/17 08:00), HR 84 (vital 1/17 08:00)"
    After:  "BP 118/72, HR 84 (vital 1/17 08:00)"

    Strategy: find all cite-tag positions, identify runs where the same tag
    appears with only ", value" separating them, then remove all but the last
    tag in each run.
    """
    # Find all cite tags and their spans
    matches = list(CITE_PATTERN.finditer(text))
    if len(matches) < 2:
        return text

    # Work backwards so span offsets remain valid as we delete
    removals: list[tuple[int, int]] = []  # (start, end) spans to delete
    i = len(matches) - 1
    while i > 0:
        curr = matches[i]
        prev = matches[i - 1]
        # Same tag?
        if curr.group() == prev.group():
            # Check that the text between them is just ", value-text"
            between = text[prev.end():curr.start()]
            # Should be: optional separator (comma/semicolon + space) + value text
            # i.e., no sentence breaks or other cite tags
            stripped = between.strip()
            if stripped and stripped[0] in (",", ";"):
                # Remove the earlier tag (keep the later one)
                removals.append((prev.start(), prev.end()))
                # Also remove any trailing whitespace before the comma
                # that preceded the removed tag's text
        i -= 1

    if not removals:
        return text

    # Apply removals back-to-front
    result = list(text)
    for start, end in sorted(removals, reverse=True):
        # Remove the cite tag and any leading space before it
        rm_start = start
        while rm_start > 0 and result[rm_start - 1] == " ":
            rm_start -= 1
        del result[rm_start:end]

    return "".join(result)
