"""Renders ICU-PAUSE output as clinical text matching the ICU-PAUSE template."""

from __future__ import annotations

from typing import Any

from icu_pause.schemas.icu_pause import (
    CLINICIAN_FACING_CATEGORIES,
    SEVERITY_ORDER,
    Warning,
    WarningCategory,
    WarningSeverity,
    classify_legacy_warning,
)


def render_warnings_for_clinician(
    warnings: list[Warning | dict | str] | None,
) -> list[Warning]:
    """Filter warnings to clinician-facing categories, sorted by severity.

    Accepts on-disk dicts (post model_dump), live Warning objects, or legacy
    plain strings (from output.json files written before the structured-warning
    refactor — coerced via classify_legacy_warning so iteration-1 cases are
    comparable to later runs).
    """
    if not warnings:
        return []
    coerced: list[Warning] = []
    for item in warnings:
        if isinstance(item, Warning):
            coerced.append(item)
        elif isinstance(item, dict):
            coerced.append(Warning.model_validate(item))
        elif isinstance(item, str):
            coerced.append(classify_legacy_warning(item))
    visible = [w for w in coerced if w.category in CLINICIAN_FACING_CATEGORIES]
    visible.sort(key=lambda w: SEVERITY_ORDER[w.severity])
    return visible


def _format_clinician_warning_line(w: Warning) -> str:
    """One-line plain-text rendering used by formatter.py and doc_export.py text mode."""
    return f"[{w.severity.value.upper()} \u00b7 {w.category.value}] {w.message}"

# ICU-PAUSE sections in mnemonic order with display labels
SECTION_ORDER = [
    ("I", "I", "ICU Admission Reason & Brief ICU Course"),
    ("C", "C", "Code Status / DPOA Info / Goals of Care / ACP Note"),
    ("U_unprescribing", "U", "Unprescribing & Pertinent High-Risk Medications"),
    ("P", "P", "Pending Tests at the Time of Transfer"),
    ("A", "A", "Active Consultants, including Rehab"),
    ("U_uncertainty", "U", "Uncertainty Measure / Diagnostic Pause"),
    ("S", "S", "Summary of Major Problems and To-Do's"),
    ("E", "E", "Exam at the Time of Transfer, incl. Lines/Drains/Airways & Data Review"),
]


_BUCKET_LABELS = {
    "pre_transfer": "\U0001f6cf\ufe0f BEFORE TRANSFER (ICU team):",      # 🛏️
    "ward_ongoing": "\U0001f3e5 ON THE WARD (receiving team):",           # 🏥
    "discharge": "\U0001f3e0 AT DISCHARGE (case manager/team):",          # 🏠
}

_BUCKET_ORDER = ["pre_transfer", "ward_ongoing", "discharge"]


def render_todo_checklist(todo_items: list[dict[str, str] | str]) -> list[str]:
    """Render todo_checklist items grouped by temporal bucket.

    Accepts both new format (list[dict]) and legacy format (list[str]).
    Returns lines ready to join with newline.
    """
    # Normalize: support legacy list[str] format
    normalized: list[dict[str, str]] = []
    for item in todo_items:
        if isinstance(item, str):
            normalized.append({"bucket": "ward_ongoing", "text": item})
        else:
            normalized.append(item)

    # Group by bucket
    buckets: dict[str, list[str]] = {b: [] for b in _BUCKET_ORDER}
    for item in normalized:
        bucket = item.get("bucket", "ward_ongoing")
        if bucket not in buckets:
            bucket = "ward_ongoing"
        buckets[bucket].append(item["text"])

    lines: list[str] = []
    for bucket_key in _BUCKET_ORDER:
        items = buckets[bucket_key]
        if not items:
            continue
        lines.append(f"  {_BUCKET_LABELS[bucket_key]}")
        for i, text in enumerate(items, 1):
            lines.append(f"    [ ] {i}. {text}")
    return lines


def render_icu_pause_text(output: dict[str, Any]) -> str:
    """Render the ICU-PAUSE output dict as a formatted clinical text document.

    Follows the ICU-PAUSE template structure from the ATS ICU-PAUSE Framework.
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 70)
    lines.append("ICU to Ward Transfer Summary (ICU-PAUSE Framework)")
    lines.append("=" * 70)
    lines.append(f"Hospitalization ID: {output.get('hospitalization_id', 'N/A')}")
    lines.append(f"Generated: {output.get('generated_at', 'N/A')}")
    lines.append("")

    # Sections
    sections = output.get("sections", {})
    for key, letter, label in SECTION_ORDER:
        content = sections.get(key, "Not enough information from structured data.")
        lines.append(f"{letter}  {label}")
        lines.append("-" * 50)
        lines.append(content)
        lines.append("")

    # To-Do Checklist
    todo_items = output.get("todo_checklist", [])
    if todo_items:
        lines.append("=" * 70)
        lines.append("TO-DO LIST")
        lines.append("-" * 50)
        lines.extend(render_todo_checklist(todo_items))
        lines.append("")

    # QA Issues
    qa_issues = output.get("qa_issues", [])
    if qa_issues:
        lines.append("=" * 70)
        lines.append("QA ISSUES (Review Required)")
        lines.append("-" * 50)
        for issue in qa_issues:
            lines.append(f"  ! {issue}")
        lines.append("")

    # Warnings — clinician-facing categories only, sorted by severity.
    visible_warnings = render_warnings_for_clinician(output.get("warnings", []))
    if visible_warnings:
        lines.append("WARNINGS:")
        for w in visible_warnings:
            lines.append(f"  * {_format_clinician_warning_line(w)}")
        lines.append("")

    # Metadata
    meta = output.get("metadata", {})
    if meta:
        filled = meta.get("sections_filled", "?")
        total = meta.get("sections_total", "?")
        agents = meta.get("agent_count", "?")
        lines.append(f"[{filled}/{total} sections filled | {agents} agents ran]")

    return "\n".join(lines)
