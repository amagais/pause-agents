"""Render source_bundle data as collapsible Streamlit panels."""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from display._datetimes import (
    DISPLAY_TZ as _SHARED_DISPLAY_TZ,
    format_compact as _format_compact_display,
    parse_iso_to_display as _parse_iso_to_display,
)


# Common EHR section headers (ALL-CAPS or Title-Case words ending in a colon).
# When a note arrives as a single wall of text, we inject paragraph breaks
# before these markers so reviewers can actually read it.
_SECTION_HEADER_RE = re.compile(
    r"(?<!\n)(?<!^)("
    r"[A-Z][A-Z /&\-]{2,}:"                              # ALLCAPS HEADERS:
    r"|(?:Chief Complaint|History of Present Illness|HPI|Past Medical History|PMH|"
    r"Past Surgical History|PSH|Medications|Allergies|Social History|Family History|"
    r"Review of Systems|ROS|Physical Exam|PE|Vital Signs|Vitals|Assessment|Plan|"
    r"Impression|Labs?|Imaging|Procedures?|Hospital Course|Discharge Summary|"
    r"Subjective|Objective|A/P|A&P|Assessment and Plan|Problem List):"
    r")"
)

# Sentence boundary: period/question/exclamation followed by space + capital letter.
# Used only as a last-resort break when a note has no newlines and no headers.
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _format_note_text(text: str) -> str:
    """Return HTML-escaped note text with readable line breaks.

    If the note already has newlines, preserve them (pre-wrap CSS will render
    them). If the note is one long blob, inject breaks before section headers;
    fall back to sentence-level breaks if no headers are found.
    """
    if not text:
        return ""

    escaped = html.escape(text)

    newline_count = escaped.count("\n")
    # One newline per ~200 chars is enough structure — leave it alone.
    if newline_count >= max(3, len(escaped) // 400):
        return escaped

    # Inject double newline before recognizable section headers.
    with_headers = _SECTION_HEADER_RE.sub(r"\n\n\1", escaped)

    if with_headers.count("\n") >= 3:
        return with_headers.strip()

    # Last resort: break on sentence boundaries so at least it wraps into paragraphs.
    return _SENTENCE_BREAK_RE.sub("\n", with_headers).strip()


# Per-row columns that are constant across an entire hospitalization's data
# and are filtered out of agent input — hide them in the reviewer display so
# what the reviewer sees mirrors what the agents see.
#
# Note: post the admin-column drop in context._drop_admin_columns, most of
# these (hospitalization_id, provider ids, etc.) are already stripped at
# serialization time.  Keeping the frozenset as a defence-in-depth guard
# for cases that bypass serialize_to_json (older cached outputs, tests).
_HIDDEN_COLUMNS: frozenset[str] = frozenset({
    "hospitalization_id",
    "patient_id",
})


# Column-name heuristics: anything matching this pattern is treated as a
# timestamp and reformatted to ``M-DD HH:MM`` at display time — the same
# compact form the agent's cite tags use.  The underlying row still holds
# the ISO string (agents compute intervals from it); we only change the
# display.
_DTTM_COL_RE = re.compile(r"_dttm$|^dttm$|_end$", re.IGNORECASE)


# Display TZ for all source-table timestamps. Imported from
# ``display/_datetimes.py`` so the source table and citation tooltips
# can't drift in TZ handling — see that module's docstring for why this
# matters.
_DISPLAY_TZ = _SHARED_DISPLAY_TZ


def _compact_dttm(val: Any) -> Any:
    """Reformat an ISO timestamp string to ``M-DD HH:MM`` in America/Chicago.

    Matches the agent cite-tag format from ``data/context.py`` so reviewers
    see the same hour in the source table and the cited evidence.
    Naive timestamps are assumed to be UTC (mirrors cite-tag handling).
    Returns the input unchanged if parsing fails.

    Thin wrapper over the shared ``_datetimes`` helpers so the source
    table and citation tooltips parse + format ISO strings via the same
    code path.
    """
    if val is None or val == "":
        return val
    dt = _parse_iso_to_display(val)
    if dt is None:
        return val
    return _format_compact_display(dt)


def _strip_hidden(rows: list[dict]) -> list[dict]:
    """Return row dicts with per-case-constant identifier columns removed.

    Also compacts any column matching ``_DTTM_COL_RE`` to the ``M-DD HH:MM``
    form — matches the cite-tag time format agents actually reason in.
    """
    if not rows:
        return rows
    out: list[dict] = []
    for row in rows:
        trimmed: dict[str, Any] = {}
        for k, v in row.items():
            if k in _HIDDEN_COLUMNS:
                continue
            if _DTTM_COL_RE.search(k):
                trimmed[k] = _compact_dttm(v)
            else:
                trimmed[k] = v
        out.append(trimmed)
    return out


def _format_note_caption(note: dict[str, Any]) -> str:
    """Build the per-note header caption: single timestamp + service attribution.

    Always shows one timestamp: revision_dttm when present, otherwise
    creation_dttm. The cite tag the model emits inline anchors on the
    same field (see _CITE_SOURCE_TYPES note rows in data/context.py),
    so this caption and the tooltip cross-reference one number.

    creation_dttm is deliberately not surfaced even when it differs
    from revision_dttm. Reviewers were reading the panel's "created
    M-DD HH:MM" alongside the tooltip's anchor time as a creation-vs-
    revised contradiction and flagging notes as inaccurate, when in
    fact only revision_dttm reaches the tooltip and both surfaces are
    consistent. The Gate 2 48h relevance window is still documented
    in the methods-key expander below.

    Service attribution matches the cite-tooltip priority order
    (note_author_service → note_author_specialty → note_author_type)
    so the panel header and the cite tooltip read consistently.
    """
    revision_raw = note.get("revision_dttm") or ""
    creation_raw = note.get("creation_dttm") or ""
    anchor = _compact_dttm(revision_raw) or _compact_dttm(creation_raw) or ""
    date_part = f"Date: {anchor}"

    service = (
        note.get("note_author_service")
        or note.get("note_author_specialty")
        or note.get("note_author_type")
    )
    if service:
        return f"{date_part} · {service}"
    return date_part


def _render_note_body(text: str) -> None:
    """Render a single note body inside a scroll-friendly, wrapped block."""
    formatted = _format_note_text(text)
    st.markdown(
        "<div style='white-space: pre-wrap; font-family: -apple-system, BlinkMacSystemFont, "
        "\"Segoe UI\", sans-serif; font-size: 0.92rem; line-height: 1.5; "
        "color: #1c1c1e; background: #f8f9fa; padding: 12px 14px; border-radius: 4px; "
        "border: 1px solid #e9ecef; max-height: 520px; overflow-y: auto;'>"
        f"{formatted}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_source(source: dict[str, Any]) -> None:
    """
    Render all populated sections of a source_bundle as collapsible expanders.
    Collapsed by default to keep the UI clean.
    """
    demo = source.get("demographics", {})
    _render_demographics(demo)
    _render_transfer_exam_block(source.get("transfer_exam_block", ""))
    _render_vitals(source.get("vitals_summary", []))
    _render_labs(source.get("labs_recent", []))
    _render_meds_continuous(source.get("meds_continuous", []))
    _render_meds_intermittent(source.get("meds_intermittent", []))
    _render_respiratory(source.get("respiratory_support", []))
    _render_assessments(source.get("assessments", []))
    _render_code_status(source.get("code_status", []))
    _render_diagnoses(source.get("diagnoses", []))
    _render_microbiology(source.get("microbiology", []))
    _render_procedures(source.get("procedures", []))
    _render_notes(
        source.get("clinical_notes", {}),
        notes_window=source.get("notes_window"),
        absence_reasons=source.get("notes_absence_reasons"),
    )


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def _render_demographics(demo: dict) -> None:
    if not demo:
        return
    with st.expander("Demographics & Admission", expanded=False):
        cols = st.columns(3)
        cols[0].metric(
            "Age",
            demo.get("age_at_icu_admission") or demo.get("age_at_admission", "—"),
        )
        cols[1].metric("Sex", demo.get("sex_category", "—"))
        cols[2].metric("Admit Type", demo.get("admission_type_category", "—"))
        st.caption(f"ICU admission: {demo.get('icu_admission_dttm', '—')}")
        # Show reference_dttm (transfer note creation time = the moment the
        # pipeline froze the patient snapshot) instead of icu_discharge_dttm,
        # which is post-transfer and leaks future data. Falls back to '—' for
        # legacy cases generated before reference_dttm was added to the
        # demographics dict — rerun the pipeline to populate it.
        st.caption(f"Data as of (transfer note time): {demo.get('reference_dttm', '—')}")
        los = demo.get("icu_los_hours")
        if los is not None:
            st.caption(f"ICU LOS: {los:.1f} hours ({los/24:.1f} days)")


def _render_transfer_exam_block(block: str) -> None:
    """Surface the deterministic Section E exam block as the reviewer sees it.

    The intensivist is instructed to copy this string verbatim into Section E.
    Exposing it here lets the reviewer audit whether the rendered E section
    obeyed that instruction. Empty string until the Phase-3 builder ships.
    """
    if not block:
        return
    with st.expander("Transfer exam block (intensivist input)", expanded=False):
        st.code(block, language=None)


def _render_vitals(vitals) -> None:
    """Render vitals data — accepts both the new dict shape and legacy list.

    New shape (post-Phase-1): ``{"bucketed_trends": [...], "recent_raw": [...]}``
    surfaces both views in separate expanders so the reviewer can audit
    snapshot-style sentences against raw measurements the bucket mean
    smooths over.

    Legacy list shape: pre-Phase-1 on-disk briefs and the demo page; rendered
    as a single bucketed-trends expander to preserve existing behavior.
    """
    if not vitals:
        return
    if isinstance(vitals, dict):
        bucketed = vitals.get("bucketed_trends") or []
        recent_raw = vitals.get("recent_raw") or []
        if bucketed:
            with st.expander("Vitals (8-hour bucket means)", expanded=False):
                st.caption(
                    "Smoothed means across 8 h windows. Audit narrative "
                    "**trend** claims here (e.g. 'BP has been climbing')."
                )
                df = pd.DataFrame(_strip_hidden(bucketed))
                st.dataframe(df, use_container_width=True, hide_index=True)
        if recent_raw:
            with st.expander(
                "Vitals (point-in-time measurements)", expanded=False
            ):
                st.caption(
                    "Individual measurements as the agent saw them. Audit "
                    "**snapshot** statements here — Section E transfer-exam "
                    "block, 'at transfer' lines, etc."
                )
                df = pd.DataFrame(_strip_hidden(recent_raw))
                st.dataframe(df, use_container_width=True, hide_index=True)
        return
    # Legacy list shape — render as single bucketed expander.
    with st.expander("Vitals (8-hour bucket means)", expanded=False):
        df = pd.DataFrame(_strip_hidden(vitals))
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_labs(labs: list[dict]) -> None:
    if not labs:
        return
    with st.expander("Labs (recent)", expanded=False):
        # Derive a status column so pending rows are visible without
        # scrolling past resulted ones; null the redundant "pending"
        # marker from lab_value so the column only carries real
        # qualitative results (e.g., "positive") when present.
        rows: list[dict] = []
        for r in _strip_hidden(labs):
            r = dict(r)
            lv = r.get("lab_value")
            is_pending = isinstance(lv, str) and lv.strip().lower() == "pending"
            r["status"] = "pending" if is_pending else "resulted"
            if is_pending:
                r["lab_value"] = None
            rows.append(r)
        # Sort pending first so reviewers see them at the top of the table.
        rows.sort(key=lambda r: 0 if r.get("status") == "pending" else 1)
        df = pd.DataFrame(rows)
        if "status" in df.columns:
            cols = ["status"] + [c for c in df.columns if c != "status"]
            df = df[cols]
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_meds_continuous(meds: list[dict]) -> None:
    if not meds:
        return
    with st.expander("Continuous Infusions", expanded=False):
        df = pd.DataFrame(_strip_hidden(meds))
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_meds_intermittent(meds: list[dict]) -> None:
    if not meds:
        return
    with st.expander("Intermittent Medications", expanded=False):
        df = pd.DataFrame(_strip_hidden(meds))
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_respiratory(resp: list[dict]) -> None:
    if not resp:
        return
    with st.expander("Respiratory Support", expanded=False):
        df = pd.DataFrame(_strip_hidden(resp))
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_assessments(items: list[dict]) -> None:
    if not items:
        return
    with st.expander("Patient Assessments (RASS, GCS, Pain, Delirium)", expanded=False):
        df = pd.DataFrame(_strip_hidden(items))
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_code_status(items: list[dict]) -> None:
    if not items:
        return
    with st.expander("Code Status", expanded=False):
        for cs in items:
            when = _compact_dttm(cs.get("recorded_dttm") or cs.get("start_dttm") or "—")
            status = cs.get("code_status_category") or cs.get("code_status") or "—"
            st.markdown(f"- **{status}** ({when})")


def _render_diagnoses(items: list[dict]) -> None:
    if not items:
        return
    with st.expander("Hospital Diagnoses (ICD)", expanded=False):
        for d in items:
            st.markdown(f"- {d.get('diagnosis_name', d.get('icd_code', '—'))}")


_MICRO_ORGANISM_FIELDS: tuple[str, ...] = (
    "organism", "organism_name", "organism_category",
)


def _render_microbiology(items: list[dict]) -> None:
    if not items:
        return
    with st.expander("Microbiology / Cultures", expanded=False):
        # Derive a status column so pending cultures are visible without
        # scrolling past resulted ones; mirrors _render_labs. Pending
        # marker is the literal "pending" written into organism fields
        # by retriever._mask_future_microbiology_results; null it out
        # of the organism columns so the word only appears in status.
        rows: list[dict] = []
        for r in _strip_hidden(items):
            r = dict(r)
            is_pending = any(
                isinstance(r.get(f), str) and r[f].strip().lower() == "pending"
                for f in _MICRO_ORGANISM_FIELDS
            )
            r["status"] = "pending" if is_pending else "resulted"
            if is_pending:
                for f in _MICRO_ORGANISM_FIELDS:
                    if isinstance(r.get(f), str) and r[f].strip().lower() == "pending":
                        r[f] = None
            rows.append(r)
        # Pending first so reviewers see them at the top of the table.
        rows.sort(key=lambda r: 0 if r.get("status") == "pending" else 1)
        df = pd.DataFrame(rows)
        if "status" in df.columns:
            cols = ["status"] + [c for c in df.columns if c != "status"]
            df = df[cols]
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_procedures(items: list[dict]) -> None:
    if not items:
        return
    with st.expander("Procedures", expanded=False):
        for p in items:
            when = _compact_dttm(p.get("procedure_dttm") or p.get("procedure_date") or "—")
            st.markdown(f"- {p.get('procedure_name', '—')} ({when})")


# Canonical routed note types. Order = render order. Every type listed
# here is wired into AGENT_NOTE_ROUTING for at least one domain agent, so
# absence of a type at render time is reviewer-relevant (it tells the
# reviewer "this type was looked for and not found", not "this type was
# never considered"). Keep in sync with AGENT_NOTE_ROUTING in
# src/icu_pause/config.py.
_NOTE_TYPE_LABELS: dict[str, str] = {
    "progress_note": "Progress Notes",
    "consults_note": "Consult Notes",
    "hp_note": "H&P Notes",
    "nursing_note": "Nursing Notes",
    "plan_of_care_note": "Plan of Care Notes",
    "case_management_note": "Case Management Notes",
    "social_work_note": "Social Work Notes",
    "therapy_note": "Therapy Notes",
}


def _format_window_caption(window: dict[str, Any] | None) -> str | None:
    """Render the lookback window as a human-readable banner string.

    Returns None if window metadata is missing — the banner is then
    omitted entirely rather than rendered with a misleading default.
    """
    if not window:
        return None
    ref = window.get("reference_dttm")
    lookback = window.get("lookback_hours")
    if not ref or lookback is None:
        return None
    ref_display = _compact_dttm(ref)
    ref_dt = _parse_iso_to_display(ref)
    if ref_dt is None:
        return f"**Notes window:** ends {ref_display} (last {lookback}h)"
    start = ref_dt.replace(microsecond=0) - timedelta(hours=int(lookback))
    start_display = _format_compact_display(start)
    return (
        f"**Notes window:** {start_display} → {ref_display} "
        f"(last {lookback}h; anchored at ICU→ward transfer note time, "
        f"America/Chicago)"
    )


def _classify_absence(reason: dict[str, Any] | None) -> str:
    """Bucket an absent note type into one of two display states.

    Reason dict (optional, parsed from .log sibling by prepare_cases.py):
        {"scanned_total": int, "excluded_leakage": int}

    Returns:
      - "excluded_by_window": notes existed in source but all were dropped
        by the leakage guard or lookback boundary — clinically interesting
        because it affects brief completeness.
      - "no_source_notes": no notes of this type exist for this admission,
        OR no log detail is available (treated as no-source: the base rate
        for any given patient is that most note types don't exist, so the
        muted default is the safer rendering when the signal is ambiguous).
    """
    reason = reason or {}
    scanned = int(reason.get("scanned_total", 0) or 0)
    return "excluded_by_window" if scanned > 0 else "no_source_notes"


def _render_notes(
    notes: dict[str, list[dict]],
    notes_window: dict[str, Any] | None = None,
    absence_reasons: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Render per-note-type clinical notes in full (no display-side truncation).

    Empty note types are split into two visual states:
      - excluded_by_window: notes existed but were dropped by the routing
        window — full-weight row, since this affects brief completeness.
      - no_source_notes: no notes of this type were authored — collapsed
        into a single muted line at the bottom, since this is the boring
        baseline for most patients.
    """
    window_caption = _format_window_caption(notes_window)
    if window_caption:
        st.markdown(window_caption)
        st.caption(
            "All clinical-note rows below were filtered against this "
            "window. Note types with content available but excluded by "
            "the window show an explicit row; types not present in the "
            "source are summarized at the bottom."
        )

    # Two-gate description (admissibility + relevance). Mirrors the
    # actual filter logic in retriever._load_notes_for_hospitalization:
    # Gate 1 = strict < on BOTH creation_dttm AND revision_dttm; Gate 2
    # = creation_dttm ≥ ref − notes_lookback_hours, hp_note exempt via
    # PER_ADMISSION_STABLE_NOTE_TYPES. Surface the gates so reviewers
    # don't have to reverse-engineer note inclusion criteria.
    st.caption(
        "**Admissibility (all note types)**: creation AND revision time "
        "strictly before transfer (no future revisions leak). "
        "**Relevance (all types except H&P)**: creation time within the "
        "routing window of transfer. H&P is admission-stable and exempt "
        "from the relevance window."
    )

    absence_reasons = absence_reasons or {}
    no_source_labels: list[str] = []

    for note_type, label in _NOTE_TYPE_LABELS.items():
        note_list = notes.get(note_type) or []
        if note_list:
            with st.expander(
                f"Clinical Notes — {label} ({len(note_list)} notes)",
                expanded=False,
            ):
                for note in note_list:
                    text = note.get("note_text", "") or ""
                    st.caption(_format_note_caption(note))
                    # No char truncation: reviewer must see exactly what the
                    # producing agent saw (parity rule). Note count caps still
                    # apply upstream via AGENT_MAX_NOTES_PER_TYPE.
                    _render_note_body(text)
                    st.divider()
            continue

        reason = absence_reasons.get(note_type)
        if _classify_absence(reason) == "excluded_by_window":
            scanned = int((reason or {}).get("scanned_total", 0) or 0)
            noun = "note" if scanned == 1 else "notes"
            st.markdown(
                f"**Clinical Notes — {label}** — "
                f"{scanned} {noun} excluded (outside routing window)"
            )
        else:
            no_source_labels.append(label)

    # Surface any note types present in the bundle that aren't in the canonical
    # routed list (forward compatibility — e.g., a new note type ships before
    # this label dict is updated). Without this fall-through they'd render
    # nowhere even though the agent received them.
    for note_type, note_list in notes.items():
        if note_type in _NOTE_TYPE_LABELS or not note_list:
            continue
        label = note_type.replace("_", " ").title()
        with st.expander(
            f"Clinical Notes — {label} ({len(note_list)} notes)", expanded=False
        ):
            for note in note_list:
                text = note.get("note_text", "") or ""
                st.caption(_format_note_caption(note))
                _render_note_body(text)
                st.divider()

    if no_source_labels:
        st.markdown(
            "<div style='color: rgba(150,150,150,0.75); font-size: 0.85em; "
            "margin-top: 0.75em;'>Not present in source: "
            f"{', '.join(no_source_labels)}</div>",
            unsafe_allow_html=True,
        )
