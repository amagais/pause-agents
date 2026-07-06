"""Pharmacy Agent: medications, renal/hepatic labs → U_unprescribing, P, S sections.

Pharmacy owns the U_unprescribing section, which lists every drug-related
handoff item — including the inpatient antibiotic history that the 48h
structured-data window cannot surface. The scribe runs ahead of pharmacy
in the workflow and extracts ``AdmissionAntibioticCourse`` rows from
chart text; pharmacy renders those rows as a labeled pin block at the
top of its user_message (via the :meth:`_format_scribe_pins` override).

The block flows through the same prompt as the structured drug data,
so the YAML system prompt is responsible for telling the LLM to merge
the two streams (scribe pin = pre-48h chart history; structured meds.*
= 48h window). The pin block is rendered deterministically — sort,
labels, and token-cap all live in this module so behavior is
test-pinnable without round-tripping the LLM. See
docs/admission_antibiotics_design.md for the design.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from icu_pause.agents.base import BaseDomainAgent


_NOTE_TYPE_LABELS: dict[str, str] = {
    "hp_note": "H&P",
    "progress_note": "progress note",
    "consults_note": "consult note",
    "case_management_note": "case mgmt note",
    "social_work_note": "social work note",
    "plan_of_care_note": "plan of care",
    "nursing_note": "nursing note",
    "therapy_note": "therapy note",
}


def _short_date(value: Any, tz: ZoneInfo | None = None) -> str:
    """Render a datetime/ISO-string as ``M/D`` for inline pin labels.

    Returns ``"?"`` when the value is None/empty or can't be parsed.
    Used for the ``[progress note, 3/5]`` provenance suffix on each
    rendered course line — keeps lines short while still anchoring
    each course to a specific note's date.
    """
    if value is None or value == "":
        return "?"
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value)
        try:
            normalized = s.replace(" ", "T", 1) if "T" not in s[:11] else s
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            return s
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if tz is not None:
        dt = dt.astimezone(tz)
    return f"{dt.month}/{dt.day}"


def _build_note_label_map(state: dict[str, Any]) -> dict[str, str]:
    """Map ``note_id`` → human-friendly label like ``"progress note, 3/5"``.

    Reads the merged ``patient_context_text['notes']`` rather than
    pharmacy's own routed slice, because the scribe's source_note_id
    can come from notes pharmacy doesn't see (e.g. case_management_note,
    social_work_note). The merged context is the union across all
    agent routings, so every source_note_id the scribe could have used
    resolves here.

    Returns an empty dict when notes are unavailable (e.g. notes_enabled
    is False) — callers should fall back to the raw note_id.
    """
    notes_block = state.get("patient_context_text", {}).get("notes")
    if not notes_block:
        return {}
    label_map: dict[str, str] = {}
    # notes_block is dict[note_type → list[note_row_dict]] when the agent_role
    # filter was applied at retrieve-time; the merge-all branch produces the
    # same shape. Tolerate the legacy flat-list shape too.
    if isinstance(notes_block, dict):
        for note_type, rows in notes_block.items():
            if not rows:
                continue
            type_label = _NOTE_TYPE_LABELS.get(note_type, note_type)
            for row in rows:
                nid = str(row.get("note_id", ""))
                if not nid:
                    continue
                d = _short_date(
                    row.get("creation_dttm") or row.get("note_dttm")
                )
                label_map[nid] = f"{type_label}, {d}"
    elif isinstance(notes_block, list):
        for row in notes_block:
            nid = str(row.get("note_id", ""))
            if not nid:
                continue
            note_type = row.get("note_type") or row.get("note_type_key") or ""
            type_label = _NOTE_TYPE_LABELS.get(note_type, note_type or "note")
            d = _short_date(row.get("creation_dttm") or row.get("note_dttm"))
            label_map[nid] = f"{type_label}, {d}"
    return label_map


def _course_sort_key(course: dict[str, Any]) -> tuple[int, str]:
    """Two-pass stable sort key.

    Returned tuple sorts ascending; we want:
      - status='ongoing_outside_window' BEFORE 'completed'
      - within each status, more-recent end_date_parsed FIRST

    To get descending by end_date_parsed via ascending sort, we negate
    the string by mapping None → '' (which sorts last) and reversing
    the comparison via the outer ``reverse=True`` at the call site.
    Returning ``(status_rank, end_date_parsed_or_empty)`` and sorting
    with ``reverse=True`` yields: ongoing first, then completed; within
    each, end_date descending (None last).

    ISO-8601 dates (YYYY-MM-DD) are lexicographically sortable, which
    is why end_date_parsed must be ISO — chart literals like '3/5' vs
    '3/15' would sort wrong. See AdmissionAntibioticCourse.end_date_parsed.
    """
    status = course.get("status", "")
    status_rank = 1 if status == "ongoing_outside_window" else 0
    return (status_rank, course.get("end_date_parsed") or "")


def _render_course_line(
    course: dict[str, Any], label_map: dict[str, str]
) -> str:
    """Render one AdmissionAntibioticCourse as a single pin-block bullet.

    Format:
      - cefepime (3/3 -> 3/5) for Klebsiella UTI [progress note, 3/5]
      - cefepime (3/3 -> ongoing) for HCAP [H&P, 3/3]                 # ongoing
      - cefepime (3/3 -> 3/5) for Klebsiella UTI — de-escalation [...] # notes

    Every field guards against None — the schema permits absent
    start_date_phrase, indication, and notes, and the renderer must
    not silently drop a course just because a verbatim chart slot
    was empty.
    """
    drug = course.get("drug", "?")
    start = course.get("start_date_phrase") or "?"
    if course.get("status") == "ongoing_outside_window":
        end = "ongoing"
    else:
        end = course.get("end_date_phrase") or "?"

    indication = course.get("indication")
    notes = course.get("notes")
    source_note_id = str(course.get("source_note_id", ""))
    label = label_map.get(source_note_id, source_note_id or "?")

    head = f"{drug} ({start} -> {end})"
    if indication:
        head = f"{head} for {indication}"
    if notes:
        head = f"{head} -- {notes}"
    return f"  - {head} [{label}]"


def _apply_token_cap(
    lines: list[str], token_cap: int, chars_per_token: float
) -> tuple[list[str], int]:
    """Truncate the rendered line list to fit ``token_cap`` tokens.

    Returns ``(kept_lines, dropped_count)``. Drops the LAST lines first
    (i.e. oldest after the stable sort already ran), so the most
    clinically relevant courses survive truncation. The char budget
    is computed as ``token_cap * chars_per_token`` — a rough heuristic
    that's intentionally coarse (we're capping prompt tokens, not
    decoding tokens, and the LLM's tokenizer is provider-specific).

    Always keeps at least one line, even if it busts the budget on
    its own — losing every course to a misconfigured cap is worse
    than going slightly over.
    """
    char_budget = int(token_cap * chars_per_token)
    running = 0
    kept: list[str] = []
    for line in lines:
        if kept and (running + len(line) + 1) > char_budget:
            break
        kept.append(line)
        running += len(line) + 1
    dropped = len(lines) - len(kept)
    return kept, dropped


class PharmacyAgent(BaseDomainAgent):
    @property
    def agent_name(self) -> str:
        return "pharmacy"

    @property
    def required_context_keys(self) -> list[str]:
        return ["meds", "labs", "microbiology", "notes"]

    @property
    def target_sections(self) -> list[str]:
        return ["U_unprescribing", "S"]

    def _format_scribe_pins(self, state: dict[str, Any]) -> str:
        """Render the scribe-extracted admission_antibiotics pin block.

        Reads ``state['scribe_extraction']`` (populated upstream by the
        scribe node — workflow edge ``scribe -> pharmacy``). Emits
        content only when the scribe extracted AND validated at least
        one course; otherwise returns "" and pharmacy proceeds as
        before (the structured meds.* streams remain its sole input).

        See docs/admission_antibiotics_design.md §3 for the full
        information flow and ownership boundary with meds.states.
        """
        extraction = state.get("scribe_extraction") or {}
        if not extraction.get("admission_antibiotics_validated"):
            return ""
        courses = extraction.get("admission_antibiotics") or []
        if not courses:
            return ""

        ordered = sorted(courses, key=_course_sort_key, reverse=True)
        label_map = _build_note_label_map(state)
        lines = [_render_course_line(c, label_map) for c in ordered]

        token_cap = self.settings.scribe_pin_token_cap_admission_abx
        chars_per_token = self.settings.scribe_pin_chars_per_token
        kept, dropped = _apply_token_cap(lines, token_cap, chars_per_token)
        body = "\n".join(kept)
        overflow = ""
        if dropped:
            overflow = (
                f"\n  ... and {dropped} older "
                f"{'course' if dropped == 1 else 'courses'} "
                f"truncated by token budget."
            )

        n = len(ordered)
        return (
            "## ADMISSION ANTIBIOTIC HISTORY (extracted by scribe -- "
            f"validated; per-source substring + patient-identity "
            f"guarded; n={n})\n\n"
            f"{body}{overflow}\n\n"
            "USE THIS LIST AS-IS for the inpatient antibiotic history "
            "line in U_unprescribing. These courses are OUTSIDE the 48h "
            "structured-data window and ARE NOT in meds.continuous or "
            "meds.intermittent. Do NOT discard them when summarizing "
            "antibiotic exposure. If a drug listed here also appears in "
            "meds.states (currently active within the 48h window), "
            "prefer the structured row and skip the scribe line for "
            "that drug -- the scribe and structured streams are "
            "complementary, not duplicative.\n"
        )
