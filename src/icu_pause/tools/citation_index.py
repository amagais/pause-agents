"""Build a trimmed, renderer-ready index of citation tags → source data.

The raw ``cite_registry`` maps each tag to the full source row (dozens of
columns).  Persisting all of that into ``ICUPauseOutput.metadata`` would bloat
the output JSON — and most fields aren't needed to render a tooltip.

This module filters the registry to only tags that actually appear in the
final rendered text, then trims each row to the minimal shape a renderer
needs: ``source_type``, ``time``, ``label``, ``value``, ``unit``, ``tier``.

Ordering guarantee (per design discussion):
    render text → extract referenced tags → filter registry to those tags
    → tags in text but absent from filtered index are flagged ``unverified``

This ordering means a tag that's correctly in ``cite_registry`` can't be
mis-labelled ``unverified`` just because another provenance path stripped it.
"""

from __future__ import annotations

from typing import Any, Callable

from icu_pause.data.context import CITE_PATTERN
from icu_pause.schemas.icu_pause import CitationEntry, CitationRow


# ---------------------------------------------------------------------------
# Per-source-type row-trimming dispatch
# ---------------------------------------------------------------------------
#
# Each trim function takes a source row (dict) and returns (label, value, unit).
# Time and source_type are added by the caller.  Missing fields → None.


def _first(row: dict[str, Any], *keys: str) -> Any:
    """Return the first non-None value for any of *keys*, else None."""
    for k in keys:
        v = row.get(k)
        if v is not None and v != "":
            return v
    return None


def _fmt_value(v: Any) -> str | None:
    """Stringify a scalar value for tooltip display."""
    if v is None:
        return None
    if isinstance(v, float):
        # Trim trailing zeros: 21.0 → "21", 7.35 → "7.35"
        if v.is_integer():
            return str(int(v))
        return f"{v:g}"
    return str(v)


def _trim_lab(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    label = _first(row, "lab_name", "lab_category")
    value = _fmt_value(_first(row, "lab_value_numeric", "lab_value"))
    unit = _first(row, "reference_unit")
    return label, value, unit


def _trim_vital(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    label = _first(row, "vital_category", "vital_name")
    # Bucketed-trend rows have mean; raw rows have vital_value.
    value = _fmt_value(_first(row, "mean", "vital_value"))
    # Units aren't stored per-row in CLIF vitals; infer from category for
    # the common ones so the tooltip isn't dimensionless.
    unit = _VITAL_UNITS.get(str(label).lower()) if label else None
    return label, value, unit


def _trim_med(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    label = _first(row, "med_name", "med_category")
    value = _fmt_value(_first(row, "med_dose", "dose"))
    unit = _first(row, "med_dose_unit", "dose_unit")
    return label, value, unit


def _trim_resp(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    # Respiratory rows describe a device configuration; fold the most
    # decision-critical fields (device + FiO2) into label/value.
    device = _first(row, "device_category", "device_name")
    fio2 = _fmt_value(_first(row, "fio2_set", "fio2"))
    if device and fio2:
        return str(device), f"FiO2 {fio2}", None
    if device:
        return str(device), None, None
    if fio2:
        return "FiO2", fio2, None
    return None, None, None


def _trim_assess(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    label = _first(row, "assessment_category", "assessment_name")
    value = _fmt_value(_first(row, "assessment_value", "numerical_value"))
    return label, value, None


def _trim_code(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    label = "Code status"
    value = _first(row, "code_status_category", "code_status_name", "code_status")
    return label, _fmt_value(value), None


def _trim_proc(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    # procedure_code is stripped by context._drop_admin_columns("proc") — do
    # not add it back to the fallback without also removing it from the drop
    # set in context.py.  The admin drop + trim read-set must stay disjoint.
    label = _first(row, "procedure_name", "procedure_category")
    return label, None, None


# Phase 3 transfer-exam sub-block trimmers.
#
# Unlike the raw-CLIF source types above, the exam-* registries are
# populated with rows the deterministic builder has already normalized:
# each row carries ``label``, ``value``, ``unit`` fields directly. The
# trimmer is therefore near-identity. Per-row ``time`` is read separately
# via build_citation_index so the tooltip can show divergent timestamps
# when sub-block components arrive at different moments.


def _trim_exam_passthrough(
    row: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    return row.get("label"), row.get("value"), row.get("unit")


# Human-readable display names for each note_type. Keep in sync with
# AGENT_NOTE_ROUTING keys in config.py.
_NOTE_TYPE_LABELS: dict[str, str] = {
    "progress_note": "Progress note",
    "hp_note": "H&P",
    "consults_note": "Consult note",
    "plan_of_care_note": "Plan of care",
    "nursing_note": "Nursing note",
    "case_management_note": "Case management note",
    "social_work_note": "Social work note",
    "therapy_note": "Therapy note",
}


def _trim_note(
    row: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    """Trim a note row to (label, value, unit) for the tooltip.

    label = human-readable note_type, sourced from the row's display
            string when present (falls back to _NOTE_TYPE_LABELS, then
            to the raw note_type token).
    value = service / specialty attribution; falls back to author-type
            then note_id when no service info is on the row.
    unit  = None (notes are not dimensioned).

    Attribution-field priority is based on the actual routed-note row
    schema verified against the 2026-05-29 Path 1 smoke (see
    project_icu_pause_path1_smoke_observations.md). The previous
    placeholder field list (``author`` / ``author_name`` / ``provider``
    / ``service``) didn't match the row schema and silently fell back
    to note_id everywhere — opaque numeric IDs in the reviewer
    tooltip, which worsens PDSQI-9 Currency/Accuracy grading and IRR
    on cited claims.

    Priority rationale:
      1. note_author_service — "Critical Care Medicine", "Nephrology",
         "Cardiology": the team identity the receiving clinician most
         quickly contextualizes.
      2. note_author_specialty — "Pulmonology", "Hematology": specialty
         attribution when service is missing.
      3. note_author_type — "Physician", "Nurse Practitioner": coarse
         author-class fallback.
      4. note_id (first 12 chars) — last-resort identifier so the
         tooltip is never empty.

    The cite tag itself carries the date/time; the reviewer-app's
    note_renderer / source_renderer handles click-through to the body.
    """
    nt = row.get("note_type")
    label = _NOTE_TYPE_LABELS.get(nt, str(nt) if nt else "Note")
    value = _first(
        row,
        "note_author_service",
        "note_author_specialty",
        "note_author_type",
    )
    if value is None:
        note_id = row.get("note_id")
        if note_id is not None:
            value = str(note_id)[:12]
    return label, _fmt_value(value), None


_TRIMMERS: dict[str, Callable[[dict[str, Any]], tuple[str | None, str | None, str | None]]] = {
    "lab": _trim_lab,
    "vital": _trim_vital,
    "med": _trim_med,
    "resp": _trim_resp,
    "assess": _trim_assess,
    "code": _trim_code,
    "proc": _trim_proc,
    "exam-vitals": _trim_exam_passthrough,
    "exam-neuro": _trim_exam_passthrough,
    "exam-resp": _trim_exam_passthrough,
    "progress_note": _trim_note,
    "hp_note": _trim_note,
    "consults_note": _trim_note,
    "plan_of_care_note": _trim_note,
    "nursing_note": _trim_note,
    "case_management_note": _trim_note,
    "social_work_note": _trim_note,
    "therapy_note": _trim_note,
}


_VITAL_UNITS = {
    "heart_rate": "bpm",
    "hr": "bpm",
    "respiratory_rate": "/min",
    "rr": "/min",
    "sbp": "mmHg",
    "dbp": "mmHg",
    "map": "mmHg",
    "spo2": "%",
    "temp_c": "°C",
    "temp_f": "°F",
    "weight_kg": "kg",
}


# ---------------------------------------------------------------------------
# Time-field lookup per source_type (matches _add_cite_fields in context.py)
# ---------------------------------------------------------------------------

_TIME_FIELD = {
    "lab": "lab_result_dttm",
    "vital": ("bucket_end", "recorded_dttm"),
    "med": "admin_dttm",
    "resp": "recorded_dttm",
    "assess": "recorded_dttm",
    "code": ("start_dttm", "code_status_dttm"),
    "proc": "procedure_dttm",
    # Exam blocks normalize their rows with a ``time`` field directly.
    "exam-vitals": "time",
    "exam-neuro": "time",
    "exam-resp": "time",
    # Notes: prefer revision_dttm (matches sort precedence in
    # serialize_to_json) with creation_dttm fallback for rows missing
    # a revision timestamp.
    "progress_note": ("revision_dttm", "creation_dttm"),
    "hp_note": ("revision_dttm", "creation_dttm"),
    "consults_note": ("revision_dttm", "creation_dttm"),
    "plan_of_care_note": ("revision_dttm", "creation_dttm"),
    "nursing_note": ("revision_dttm", "creation_dttm"),
    "case_management_note": ("revision_dttm", "creation_dttm"),
    "social_work_note": ("revision_dttm", "creation_dttm"),
    "therapy_note": ("revision_dttm", "creation_dttm"),
}


def _extract_time(row: dict[str, Any], source_type: str) -> str:
    fields = _TIME_FIELD.get(source_type, "recorded_dttm")
    if isinstance(fields, str):
        fields = (fields,)
    for f in fields:
        val = row.get(f)
        if val:
            return str(val)
    return ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _parse_source_type(tag: str) -> str:
    """Extract the source_type token from a tag like ``"(vital 1-12 07:00)"``."""
    # Tag format guaranteed by CITE_PATTERN: "(<type> <date> <time>)"
    # Strip leading "(" and take the first whitespace-delimited token.
    return tag[1:].split(" ", 1)[0]


def build_citation_index(
    rendered_sections: dict[str, str],
    cite_registry: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Build ``metadata.citation_index`` from rendered sections + registry.

    Args:
        rendered_sections: The final ``merged_sections`` that will ship to the
            reviewer.  Only tags found in this text are included in the index.
        cite_registry: The full tag → source rows registry from data retrieval.

    Returns:
        Dict of tag → ``CitationEntry.model_dump()`` (serialized for JSON).
        Tags that appear in the text but aren't in the registry are included
        with ``tier='unverified'`` and null content fields so renderers can
        still decorate them with a warning affordance.
    """
    # 1. Scan rendered text for every citation tag actually used.
    referenced: set[str] = set()
    for text in rendered_sections.values():
        if not text:
            continue
        referenced.update(CITE_PATTERN.findall(text))

    index: dict[str, dict[str, Any]] = {}

    for tag in referenced:
        rows = cite_registry.get(tag)
        source_type = _parse_source_type(tag)
        if not rows:
            # Tag appears in text but has no backing row → unverified.
            # We still emit the source_type so renderers can show what
            # the agent claimed to be citing.
            index[tag] = CitationEntry(
                source_type=source_type,  # type: ignore[arg-type]
                time="",
                label=None,
                value=None,
                unit=None,
                tier="unverified",
            ).model_dump()
            continue

        # Multiple rows can share a tag (bucketed vitals at the same
        # bucket_end, several labs drawn together, etc.). Trim every row
        # and keep the full list under ``rows``; the singular label/value/
        # unit fields mirror rows[0] for backwards compat with on-disk
        # output.json from before this change. Previously we kept only
        # rows[0] — that produced a single-vital tooltip ("dbp 60.5 mmHg")
        # under a sentence that cited HR/BP/MAP/SpO2 together, masking the
        # other siblings entirely.
        #
        # Per-row ``time`` is populated for all source types — the tooltip
        # uses it to render per-row timestamps when they differ from the
        # tag's anchor (Phase 3: exam-* blocks aggregate rows across a 4 h
        # window so component timestamps can disagree; other source types
        # the value is identical to the anchor by construction but it
        # costs ~10 bytes per row to set it for them too).
        trimmer = _TRIMMERS.get(source_type)
        trimmed_rows: list[CitationRow] = []
        # Dedup by (label, value, unit, time) tuple. cite_registry is
        # built across 1 corpus-level + N per-agent serialize_to_json
        # calls (workflow.py:234 + workflow.py:293-297), each appending
        # to the same registry — so a tag can hold many copies of the
        # same logical source row (Path 1 smoke 2026-05-29 saw rows=9
        # for a single H&P, rows=72 for a single lab tag). Without
        # dedup, format_tooltip renders identical rows N times,
        # producing tooltips like "Progress Notes Critical Care
        # Medicine; Progress Notes Critical Care Medicine; ... (×9)".
        # Distinct-content siblings (HR / SBP / DBP at the same bucket,
        # the original sibling-rows use case) survive dedup because
        # their (label, value, unit) tuples differ.
        #
        # Dedup the INDEX VIEW only; cite_registry is left intact so
        # downstream consumers that count registry rows still see them.
        seen_keys: set[tuple] = set()
        if trimmer is not None:
            for r in rows:
                label, value, unit = trimmer(r)
                # Skip rows that trim to nothing — they add no tooltip
                # value and would clutter the sibling list.
                if label is None and value is None and unit is None:
                    continue
                row_time = _extract_time(r, source_type) or None
                key = (label, value, unit, row_time)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                trimmed_rows.append(
                    CitationRow(
                        label=label,
                        value=value,
                        unit=unit,
                        time=row_time,
                    )
                )

        if trimmed_rows:
            primary = trimmed_rows[0]
        else:
            primary = CitationRow()

        index[tag] = CitationEntry(
            source_type=source_type,  # type: ignore[arg-type]
            time=_extract_time(rows[0], source_type),
            label=primary.label,
            value=primary.value,
            unit=primary.unit,
            rows=trimmed_rows,
            tier="decision_critical",
        ).model_dump()

    return index
