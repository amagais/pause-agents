"""Section I readability lint helpers.

Pure functions for the readability rules added to
``config/prompts/intensivist.yaml`` (GRAMMATICAL-SEAM RULE +
"CURRENTLY" CLAUSE ACTIONABLE FRAMING). Used by:

* ``tests/test_section_i_readability.py`` — fixture-based unit tests.
* ``scripts/audit_section_i_readability.py`` — observational pass
  over ``.brief.json`` files (HPC).
"""

from __future__ import annotations

import re

# Pattern A — ``with X after Y were/was {stopped,held,started}``.
# ``[^.;\n]`` bounds the X / Y fillers so the regex cannot cross
# sentence or semicolon-clause boundaries (avoids the false-positive
# where ``with`` and ``were stopped`` sit in different sentences).
_SEAM_PATTERN_A = re.compile(
    r"\bwith\s+[^.;\n]{1,150}?\s+after\s+[^.;\n]{1,150}?\s+(?:were|was)\s+"
    r"(?:stopped|held|started)\b",
    re.IGNORECASE,
)

_PLACEHOLDER_PHRASES: tuple[str, ...] = (
    "reconciliation needs",
    "ongoing plan",
    "borderline low",
    "borderline high",
)

# Bare ``TBD`` / ``to be determined`` is forbidden UNLESS followed by
# ``pending <specifier>`` within the same clause (60-char no-clause-break
# window).
_TBD_BARE_PATTERN = re.compile(
    r"\b(?:TBD|to be determined)\b(?![^.;\n]{0,60}\bpending\b)",
    re.IGNORECASE,
)

# Clinical variables that, when named in the "Currently" clause,
# require an adjacent numeric value.
_NUMERIC_REQUIRING_VARIABLES: tuple[str, ...] = (
    "SpO2",
    "SpO₂",
    "sodium",
    "potassium",
    "MAP",
    "lactate",
)

_NUMERIC_PATTERN = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")

_CURRENTLY_RE = re.compile(
    r"\bCurrently\b[^\n]*(?:\n(?!\n)[^\n]*)*",
    re.IGNORECASE,
)


def find_with_x_after_y_seams(text: str) -> list[str]:
    """Return matches of the ``with X after Y were/was Z`` seam
    (Pattern A from the grammatical-seam rule). Empty list = clean.
    """
    return [m.group(0) for m in _SEAM_PATTERN_A.finditer(text)]


def find_placeholder_phrases(text: str) -> list[str]:
    """Return forbidden placeholder phrases found in ``text``.

    Covers: ``reconciliation needs``, ``ongoing plan``, ``borderline
    low``, ``borderline high``, and bare ``TBD`` / ``to be determined``
    (the bare form is the one NOT followed by ``pending <specifier>``
    within the same clause).
    """
    hits: list[str] = []
    lower = text.lower()
    for phrase in _PLACEHOLDER_PHRASES:
        if phrase in lower:
            hits.append(phrase)
    hits.extend(m.group(0) for m in _TBD_BARE_PATTERN.finditer(text))
    return hits


def find_currently_clause_missing_numerics(currently_clause: str) -> list[str]:
    """If the clause names a numeric clinical variable but contains
    no numeric value, return the variables found unaccompanied.
    Empty list = clean (either no numeric variable named, or a number
    is present alongside).

    Parenthesized citations (``(exam-vitals 9-22 09:00)``,
    ``(lab-bmp 9-22)``) are stripped before the numeric check —
    dates/times inside a citation gesture at provenance, not at the
    clinical value the rule is asking for. Variable-name detection
    runs on the full clause (variable names are not inside parens in
    normal usage).
    """
    variables_present = [
        v
        for v in _NUMERIC_REQUIRING_VARIABLES
        if re.search(rf"\b{re.escape(v)}\b", currently_clause, re.IGNORECASE)
    ]
    if not variables_present:
        return []
    stripped = re.sub(r"\([^)]*\)", "", currently_clause)
    if _NUMERIC_PATTERN.search(stripped):
        return []
    return variables_present


def extract_currently_clause(section_i_body: str) -> str | None:
    """Extract the ``Currently …`` clause from a Section I body.

    Runs from ``Currently`` to end-of-paragraph (double newline) or
    end-of-string. Returns None when no such clause is found.
    """
    m = _CURRENTLY_RE.search(section_i_body)
    return m.group(0) if m else None
