"""Warning filter and rendering helper (vendored copy).

VENDORED from amagais/icu_pause_agents
  upstream paths: src/icu_pause/schemas/icu_pause.py (Warning model + classifier)
                  src/icu_pause/rendering/formatter.py (render_warnings_for_clinician)

The main repo is the single source of truth. This deployed reviewer app has
no `icu_pause` package dependency, so the routing rules are inlined here.
Keep `CLINICIAN_FACING_CATEGORIES`, `SEVERITY_ORDER`, and the legacy
classifier regexes in sync with upstream — see also
scripts/backfill_warnings.py which uses the same heuristic to migrate
existing on-disk output.json files.
"""

from __future__ import annotations

import re
from typing import Any

CLINICIAN_FACING_CATEGORIES = frozenset({
    "safety_flag",
    "cross_domain_conflict",
    "data_gap",
    "deterministic_override",
})

SEVERITY_ORDER = {
    "safety_critical": 0,
    "clinical": 1,
    "logistical": 2,
    "info": 3,
}

SEVERITY_LABEL = {
    "safety_critical": "SAFETY-CRITICAL",
    "clinical": "CLINICAL",
    "logistical": "LOGISTICAL",
    "info": "INFO",
}

CATEGORY_LABEL = {
    "safety_flag": "Safety Flag",
    "cross_domain_conflict": "Cross-Domain Conflict",
    "data_gap": "Data Gap",
    "deterministic_override": "Deterministic Override",
    "editorial_revision": "Editorial Revision",
    "self_critique": "Self-Critique",
    "qa_process": "QA Process",
}


_EDITORIAL_VERBS = re.compile(
    r"\b(revis(?:ed?|ing)|removed?|moved?|standardiz(?:ed?|ing)|generaliz(?:ed?|ing)|"
    r"soften(?:ed?|ing)|reformat(?:ted?|ting)?|reorganiz(?:ed?|ing)|reorder(?:ed?|ing)|"
    r"retain(?:ed?|ing)|relocat(?:ed?|ing))\b",
    re.IGNORECASE,
)
_NO_ISSUE = re.compile(
    r"\b(no (issues?|errors?|concerns?|discrepan|conflict|contradict|hallucin)|"
    r"not (found|detected)|all required|none found)\b",
    re.IGNORECASE,
)
_DATA_GAP_HINTS = re.compile(
    r"(\bverify\b.{0,60}\b(bedside|at transfer|before transfer|in source|prior to transfer)\b|"
    r"\bnot (confirmed|documented)\b|\bundocument|\bmissing (from|in) (source|data)\b|"
    r"\bdata gap\b)",
    re.IGNORECASE,
)
_CONFLICT_HINTS = re.compile(
    r"\b(mismatch|inconsisten|contradict|disagree|vs\.?\s)",
    re.IGNORECASE,
)
_SAFETY_HINTS = re.compile(
    r"\b(additive (risk|respiratory)|interaction|respiratory depression|"
    r"high.risk|isolation precaution|airway risk)\b",
    re.IGNORECASE,
)


def _classify_legacy(message: str) -> dict:
    """Heuristic classifier for legacy free-text warning strings.

    Mirrors `classify_legacy_warning` in the upstream schema. Returns a dict
    representation of the Warning so downstream display code can stay
    untyped-dict friendly.
    """
    text = message.strip()
    lower = text.lower()
    if text.startswith("CITATION_DROPPED:"):
        return _w("qa_process", "info", text)
    if text.startswith("CITATION:"):
        return _w("safety_flag", "clinical", text)
    if "execution failed" in lower or "agent failed" in lower:
        return _w("qa_process", "info", text)
    if _NO_ISSUE.search(text):
        return _w("self_critique", "info", text)
    if _EDITORIAL_VERBS.search(text):
        return _w("editorial_revision", "info", text)
    if _DATA_GAP_HINTS.search(text):
        return _w("data_gap", "clinical", text)
    if _CONFLICT_HINTS.search(text):
        return _w("cross_domain_conflict", "clinical", text)
    if _SAFETY_HINTS.search(text):
        return _w("safety_flag", "clinical", text)
    return _w("safety_flag", "clinical", text)


def _w(category: str, severity: str, message: str) -> dict:
    return {
        "category": category,
        "severity": severity,
        "message": message,
        "source_agent": "legacy",
    }


def _coerce(item: Any) -> dict:
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return _classify_legacy(item)
    return _classify_legacy(str(item))


def render_warnings_for_clinician(warnings: list[Any] | None) -> list[dict]:
    """Filter warnings to clinician-facing categories, sorted by severity.

    Accepts dicts (modern), strings (legacy on-disk JSON), or any mix.
    Returns a list of dict-shaped warnings with category/severity/message
    fields suitable for display.
    """
    if not warnings:
        return []
    coerced = [_coerce(w) for w in warnings]
    visible = [w for w in coerced if w.get("category") in CLINICIAN_FACING_CATEGORIES]
    visible.sort(key=lambda w: SEVERITY_ORDER.get(w.get("severity", "info"), 99))
    return visible


def has_audit_only_warnings(warnings: list[Any] | None) -> bool:
    """True if there are warnings filtered out for being audit-only.

    Used by the dev-mode panel to decide whether to surface an "Audit-only
    warnings" expander.
    """
    if not warnings:
        return False
    for item in warnings:
        c = _coerce(item)
        if c.get("category") not in CLINICIAN_FACING_CATEGORIES:
            return True
    return False


def audit_only_warnings(warnings: list[Any] | None) -> list[dict]:
    """The complement of render_warnings_for_clinician — audit-only entries."""
    if not warnings:
        return []
    coerced = [_coerce(w) for w in warnings]
    return [w for w in coerced if w.get("category") not in CLINICIAN_FACING_CATEGORIES]
