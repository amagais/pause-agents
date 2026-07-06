"""Pure-python builder for the display-friendly source bundle.

Faithful copy of ``build_source_bundle_from_metadata`` from
``review_app/scripts/prepare_cases.py``, extracted here WITHOUT the azure-coupled
imports so it runs on HPC. The LLM-as-judge must validate against the *same*
reshaped source the human reviewers saw (so judge↔human concordance is measured
on the same information), and the reviewers' ``source_bundle.json`` is built by
exactly this reshaping. Redaction is intentionally NOT applied here — it was a
reviewer-only protection; the judge uses the unredacted data and the clinical
content is identical.

DUP NOTE: ``review_app/scripts/prepare_cases.py`` keeps its own copy. Editing
that file would trigger the review-app GitHub Actions deploy (path-filtered to
``review_app/**``) during the active clinician rollout, so the two are kept in
sync by hand for now. Once the rollout ends, dedupe by importing this module
from ``prepare_cases``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_source_bundle_from_metadata(
    source_data: dict,
    absence_reasons: dict[str, dict[str, int]] | None = None,
    lookback_hours: int = 48,
) -> dict:
    """Build a display-friendly source bundle from
    ``pipeline_output.metadata.source_data``. No CLIF parquet needed.

    Mirrors ``review_app/scripts/prepare_cases.py:build_source_bundle_from_metadata``.
    """
    bundle: dict[str, Any] = {}

    # Demographics
    bundle["demographics"] = source_data.get("demographics", {})

    # Notes window (banner metadata for the reviewer panel). reference_dttm
    # comes from demographics; lookback_hours defaults to the pipeline default.
    bundle["notes_window"] = {
        "reference_dttm": (source_data.get("demographics") or {}).get("reference_dttm"),
        "lookback_hours": lookback_hours,
    }
    bundle["notes_absence_reasons"] = absence_reasons or {}

    # Vitals — {bucketed_trends: [...], recent_raw: [...]} in metadata.
    vitals_raw = source_data.get("vitals", {})
    if isinstance(vitals_raw, dict):
        bundle["vitals_summary"] = {
            "bucketed_trends": vitals_raw.get("bucketed_trends", []) or [],
            "recent_raw": vitals_raw.get("recent_raw", []) or [],
        }
    elif isinstance(vitals_raw, list):
        logger.warning("vitals_summary: legacy list shape; regenerate brief to "
                       "surface recent_raw alongside bucketed_trends")
        bundle["vitals_summary"] = vitals_raw
    else:
        bundle["vitals_summary"] = {"bucketed_trends": [], "recent_raw": []}

    # Transfer-exam block (deterministic Section E snapshot)
    bundle["transfer_exam_block"] = source_data.get("transfer_exam_block", "") or ""

    # Labs
    bundle["labs_recent"] = source_data.get("labs", []) or []

    # Meds — {continuous: [...], intermittent: [...]}
    meds = source_data.get("meds", {})
    if isinstance(meds, dict):
        bundle["meds_continuous"] = meds.get("continuous", []) or []
        bundle["meds_intermittent"] = meds.get("intermittent", []) or []
    else:
        bundle["meds_continuous"] = []
        bundle["meds_intermittent"] = []

    # Respiratory
    bundle["respiratory_support"] = source_data.get("respiratory", []) or []

    # Assessments
    bundle["assessments"] = source_data.get("assessments", []) or []

    # Code status
    bundle["code_status"] = source_data.get("code_status", []) or []

    # Diagnoses (ICD codes — no human-readable names in metadata)
    bundle["diagnoses"] = source_data.get("diagnoses", []) or []

    # Microbiology
    bundle["microbiology"] = source_data.get("microbiology", []) or []

    # Procedures
    bundle["procedures"] = source_data.get("procedures", []) or []

    # Clinical notes — normalise to {note_type: [note_dict, …]}
    notes_raw = source_data.get("notes", [])
    if isinstance(notes_raw, dict):
        bundle["clinical_notes"] = notes_raw
    elif isinstance(notes_raw, list):
        grouped: dict[str, list[dict]] = {}
        for note in notes_raw:
            ntype = note.get("note_type", "other")
            grouped.setdefault(ntype, []).append(note)
        bundle["clinical_notes"] = grouped
    else:
        bundle["clinical_notes"] = {}

    return bundle


def attach_cite_fields_to_bundle(
    bundle: dict,
    timezone_name: str = "America/Chicago",
) -> dict:
    """Inject the canonical ``cite`` tag into each *structured* source row, in display tz.

    Mirrors ``context.serialize_to_json``'s per-section ``_add_cite_fields``
    calls so the LLM judge's structured source rows carry the *exact*
    ``(source_type M-DD HH:MM)`` tags the brief cites — the same tags the
    review_app source table showed the human reviewers. Without this the judge
    sees raw UTC timestamps while the brief cites America/Chicago local time, so
    citations never textually match (the ``cited`` floor / "not in source"
    artifact). Verification on the judge side becomes a direct ``cite``-string
    match, with no timezone/format reasoning required.

    Clinical notes are deliberately excluded — they are verified by type+content,
    matching how the review_app presented them (see the clinical_notes comment
    below and pdsqi9 `cited` "HOW TO VERIFY").

    Mutates ``bundle`` rows in place (adds a ``cite`` key) and returns it. Rows
    without a parseable timestamp in the mapped column get no cite (silently
    skipped by ``_add_cite_fields``). Source-type/time-col mapping is copied
    from ``context.serialize_to_json`` — keep in sync if that changes.
    """
    from zoneinfo import ZoneInfo

    from icu_pause.data.context import _add_cite_fields

    tz = ZoneInfo(timezone_name)
    registry: dict = {}

    def _cite(rows, source_type, time_col) -> int:
        if not isinstance(rows, list) or not rows:
            return 0
        try:
            _add_cite_fields(rows, source_type, time_col, tz, registry)
        except AssertionError:
            logger.warning("attach_cite: unknown source_type %r — skipped", source_type)
            return 0
        return sum(1 for r in rows if isinstance(r, dict) and r.get("cite"))

    n = 0
    n += _cite(bundle.get("labs_recent"), "lab", "lab_result_dttm")
    vs = bundle.get("vitals_summary")
    if isinstance(vs, dict):
        n += _cite(vs.get("recent_raw"), "vital", "recorded_dttm")
        n += _cite(vs.get("bucketed_trends"), "vital", "bucket_end")
    n += _cite(bundle.get("meds_continuous"), "med", "admin_dttm")
    n += _cite(bundle.get("meds_intermittent"), "med", "admin_dttm")
    n += _cite(bundle.get("respiratory_support"), "resp", "recorded_dttm")
    n += _cite(bundle.get("assessments"), "assess", "recorded_dttm")
    n += _cite(bundle.get("code_status"), "code", "code_status_dttm")
    n += _cite(bundle.get("procedures"), "proc", "procedure_dttm")

    # Clinical notes are intentionally NOT given cite tags. The review_app
    # suppresses per-row note timestamps (review_app/display/citations.py
    # `_is_note_tag`) because a note tag's display-tz anchor (revision_dttm /
    # creation_dttm) diverges from the row's stored time; reviewers therefore
    # verified note citations by TYPE + CONTENT, never by timestamp. The judge
    # mirrors that (see pdsqi9 `cited` "HOW TO VERIFY"). Attaching a (possibly
    # divergent, possibly capped-out) note cite here would reintroduce exactly
    # the timestamp mismatch the review_app hides. The clinical_notes content is
    # already in the bundle for the judge's content check.

    if n == 0:
        logger.warning(
            "attach_cite_fields_to_bundle: 0 structured cite tags attached — "
            "verify source rows carry the expected *_dttm columns "
            "(labs.lab_result_dttm, resp/vital/assess.recorded_dttm, "
            "meds.admin_dttm, code_status.code_status_dttm, ...)")
    return bundle
