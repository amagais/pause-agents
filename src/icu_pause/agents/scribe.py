"""Scribe Agent: phase-0 structured chart extraction (PMH, allergies, home meds, code status).

Mirrors the role of a real-world clinical scribe: read the chart, quote
verbatim, don't synthesize. Four fields are extracted in a single LLM
call:

- PMH and allergies are pinned into the intensivist's Section I input
  (PMH as the patient-history verbatim quote; allergies as a labeled
  first line). See ``IntensivistAgent._format_scribe_pmh_block`` and
  ``_format_scribe_allergies_block``.
- Home medications are pinned into the pharmacy agent's input as the
  OUTPATIENT anchor for the "Changes to home meds" diff in
  U_unprescribing. Pharmacy compares the pin against the inpatient
  ``meds.states.active`` / ``recently_stopped`` buckets to identify
  held / restarted / dose-changed regimens. See
  ``PharmacyAgent._format_scribe_pins``.
- Code status is pinned into the case_manager's input as the
  authoritative "Code Status:" line for Section C. Multi-note rule
  is MOST-RECENT-WINS (code status changes chronologically across
  family meetings), distinct from the additive/spine+append rules
  used for the other three fields. See
  ``CaseManagerAgent._format_scribe_pins``.

All three values bypass the long-note attention failure documented in
docs/pmh_hallucination_investigation.md by surfacing the verbatim chart
quote at the top of the consuming agent's prompt instead of expecting
the consuming agent to locate it inside long SOAP-structured notes.

The scribe is a standalone agent (not a BaseDomainAgent subclass) because
it emits a different schema (``ScribeExtraction``), runs at a different
pipeline phase (between data retrieval and the domain-agent fan-out), and
doesn't participate in QA / deliberation. Modeled on the existing
ResidentAgent pattern.

Self-validates each field before emitting: every comma/semicolon-separated
clause in the LLM-extracted value must substring-match at least one
routed note, and every source note_id must resolve to the target
patient_id. On validation failure, the field is dropped (independently)
and the consuming agent falls back to a labeled "not documented / chart
review required" string for that field only — the other fields can still
ship if they validate.

See ``docs/pmh_structured_extraction_design.md`` and
``docs/home_meds_design_pass.md`` for the design rationale.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Optional

import yaml

from icu_pause.config import Settings
from icu_pause.data.context import format_local_dttm
from icu_pause.llm.provider import BaseLLM, create_llm
from icu_pause.schemas.icu_pause import (
    SubspecialtyConsult,
    AdmissionAntibioticCourse,
    RenalContext,
    ScribeExtraction,
    ScribeExtractionLLM,
)

logger = logging.getLogger(__name__)


# Conditions are most commonly comma-separated; some notes use semicolons
# or " and " as separators. Split on any of these, then strip whitespace
# and drop empties. Used for per-clause substring validation against
# routed notes — each clause must appear in at least one note body.
_CLAUSE_SPLIT_RE = re.compile(r"[,;]|\s+and\s+", re.IGNORECASE)


def _normalize_for_validator(s: str) -> str:
    """NFKC + whitespace collapse + curly-quote/dash/NBSP normalization.

    Chart exports from EHRs routinely contain NBSP (U+00A0), curly
    quotes (U+2018/2019/201C/201D), en-dash (U+2013) and em-dash
    (U+2014) for standard ASCII characters. Without this normalization,
    source_quote 'cefepime (3/3 – 3/5)' (en-dash from a Word-paste)
    won't substring-match body 'cefepime (3/3 - 3/5)' (hyphen).

    Used by:
      - _validate_admission_antibiotics (runtime validator, this module —
        drug-in-quote + notes-anchor checks, ported here 2026-06-08 from the
        removed AdmissionAntibioticCourse schema model_validators).
    """
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(str.maketrans({
        "‘": "'", "’": "'",  # curly single quotes
        "“": '"', "”": '"',  # curly double quotes
        "–": "-", "—": "-",  # en-dash, em-dash
        " ": " ",                  # NBSP
    }))
    s = " ".join(s.split())
    return s.lower()


# Word-boundary regex builder for active-consults validation. Anchored at
# both ends with ``\b`` so "Heme" doesn't match "hemoptysis", "Pulm"
# doesn't match "pulmonary edema", and "Renal" doesn't match "renal
# failure" — exactly the regex false-positive set the round-2 audit
# surfaced. Compiled per-service rather than per-call because the same
# services are checked across multiple notes.
def _word_boundary_pattern(service: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(service)}\b", re.IGNORECASE)


# Note-type alias table for the lenient state_citation parser. Maps
# common chart abbreviations + full names + internal codenames to the
# canonical internal note_type. INTENTIONAL leniency: a future dev
# tightening this into a single regex breaks the model's ability to use
# natural chart-citation phrasing ("H&P 5/27", "History and Physical
# 5/27") and creates false rejections. Update both this table AND
# AGENT_NOTE_ROUTING (config.py) when adding a new routed note type.
_NOTE_TYPE_ALIASES: dict[str, str] = {
    # H&P variants
    "h&p": "hp_note",
    "hp": "hp_note",
    "h p": "hp_note",
    "history and physical": "hp_note",
    "history & physical": "hp_note",
    "admission h&p": "hp_note",
    "hp_note": "hp_note",
    "hp note": "hp_note",
    # Progress note variants
    "progress note": "progress_note",
    "progress_note": "progress_note",
    "pn": "progress_note",
    "daily progress": "progress_note",
    "daily progress note": "progress_note",
    # Consult note variants
    "consult note": "consults_note",
    "consult_note": "consults_note",
    "consults_note": "consults_note",
    "consults note": "consults_note",
    "consult": "consults_note",
    # Social work
    "social work note": "social_work_note",
    "social_work_note": "social_work_note",
    "sw note": "social_work_note",
    "social work": "social_work_note",
    # Case management
    "case management note": "case_management_note",
    "case_management_note": "case_management_note",
    "cm note": "case_management_note",
    "case management": "case_management_note",
}


def _parse_state_citation(
    citation: str,
    last_note_dttm: str,
) -> Optional[tuple[str, str]]:
    """Parse a state_citation string into (canonical_note_type, ISO_date_str).

    Lenient on format — accepts natural model variations like ``H&P 5/27``,
    ``(per History and Physical 2024-05-27)``, ``progress_note 5/27/24``,
    ``(per H&P, 5/27)``. Strict matching against routed-note metadata is
    the caller's responsibility (see ``_validate_subspecialty_consults``); this
    parser's only job is robust structural extraction.

    Year inference: when the date phrase lacks a year (``5/27``), uses
    the year from ``last_note_dttm``. When ``last_note_dttm`` is also
    missing or malformed, returns None — better a clean parse failure
    than a wrong year.

    Returns ``(canonical_note_type, ISO_date_str)`` on success, ``None``
    on parse failure. Date format is YYYY-MM-DD regardless of input.

    INTENTIONAL leniency. A future dev tempted to tighten this into a
    single canonical regex: don't. The alias table + multi-format date
    parsing are the load-bearing things that let the model's natural
    citation phrasing pass the validator. Tightening produces false
    rejections, not safety.
    """
    if not citation:
        return None

    # Normalize: lowercase, strip surrounding parens / "per " prefix /
    # comma+semicolon separators, collapse whitespace.
    s = citation.strip().lstrip("(").rstrip(")").strip().lower()
    s = re.sub(r"^per\s+", "", s)
    s = re.sub(r"[,;]", " ", s)
    s = " ".join(s.split())

    # Find a date pattern. Three accepted shapes (longest first to avoid
    # MM/DD swallowing the MM part of MM/DD/YYYY):
    #   YYYY-MM-DD  (ISO)
    #   MM/DD/YYYY or MM/DD/YY
    #   MM/DD or M/D (year inferred from last_note_dttm)
    from datetime import date as _date

    date_str: Optional[str] = None
    date_match: Optional[re.Match[str]] = None

    iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s)
    if iso:
        y, m, d = iso.groups()
        try:
            date_str = _date(int(y), int(m), int(d)).isoformat()
            date_match = iso
        except ValueError:
            pass

    if date_str is None:
        slash_long = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", s)
        if slash_long:
            m, d, y = slash_long.groups()
            if len(y) == 2:
                y = "20" + y
            try:
                date_str = _date(int(y), int(m), int(d)).isoformat()
                date_match = slash_long
            except ValueError:
                pass

    if date_str is None:
        slash_short = re.search(r"\b(\d{1,2})/(\d{1,2})\b", s)
        if slash_short:
            m, d = slash_short.groups()
            inferred_year: Optional[int] = None
            ydm = re.match(r"(\d{4})-", last_note_dttm or "")
            if ydm:
                inferred_year = int(ydm.group(1))
            if inferred_year is not None:
                try:
                    date_str = _date(inferred_year, int(m), int(d)).isoformat()
                    date_match = slash_short
                except ValueError:
                    pass

    if date_str is None or date_match is None:
        return None

    # Note-type phrase is everything left of the date phrase.
    type_phrase = s[: date_match.start()].strip()
    if not type_phrase:
        return None

    # Exact alias lookup first; then collapse separators/punctuation and
    # retry. Keeps the alias table small while tolerating spelling drift.
    canonical = _NOTE_TYPE_ALIASES.get(type_phrase)
    if canonical is None:
        type_canonical = re.sub(r"[\s_&\-]+", " ", type_phrase).strip()
        canonical = _NOTE_TYPE_ALIASES.get(type_canonical)
        if canonical is None:
            for alias, target in _NOTE_TYPE_ALIASES.items():
                alias_canonical = re.sub(r"[\s_&\-]+", " ", alias).strip()
                if alias_canonical == type_canonical:
                    canonical = target
                    break

    if canonical is None:
        return None

    return (canonical, date_str)


class ScribeAgent:
    """Phase-0 scribe agent. Extracts PMH; pluggable for future fields."""

    agent_name = "scribe"

    def __init__(self, settings: Settings):
        agent_max = settings.agent_max_tokens.get(
            self.agent_name, settings.llm_max_tokens
        )
        self.llm: BaseLLM = create_llm(settings, max_tokens_override=agent_max)
        self.settings = settings
        self.system_prompt, self.prompt_version = self._load_prompt()

    def _load_prompt(self) -> tuple[str, str]:
        path = Path(self.settings.prompts_dir) / "scribe.yaml"
        if not path.exists():
            logger.warning(f"Prompt file not found: {path}. Using default prompt.")
            return self._default_prompt(), "unknown"
        with open(path) as f:
            data = yaml.safe_load(f)
        return data.get("system_prompt", self._default_prompt()), data.get("version", "unknown")

    @staticmethod
    def _default_prompt() -> str:
        return (
            "You are a clinical scribe. Extract PMH, allergies, home "
            "medications, code status, and active subspecialty consultants "
            "verbatim from the clinical notes provided. Quote verbatim — do "
            "not expand abbreviations, reorder, or paraphrase. Emit the full "
            "ScribeExtractionLLM JSON; downstream validators will drop any "
            "field that fails substring + patient-identity guards."
        )

    def _format_notes_for_extraction(
        self, notes_by_type: dict[str, list[dict]]
    ) -> tuple[str, dict[str, dict]]:
        """Render routed notes for the LLM and build a lookup of all
        routed notes by note_id (used by the post-extraction validator).
        """
        parts: list[str] = []
        # Source priority order — H&P first so the LLM sees it at the top.
        # consults_note tail-attached because it's only consumed by
        # subspecialty_consults extraction (the other four fields don't read it),
        # and placing it last keeps the canonical PMH/allergies/home-meds
        # ordering unchanged for those extractions.
        order = [
            "hp_note", "progress_note", "social_work_note",
            "case_management_note", "consults_note",
        ]
        note_lookup: dict[str, dict] = {}

        for note_type in order:
            rows = notes_by_type.get(note_type) or []
            if not rows:
                continue
            parts.append(f"\n## NOTE TYPE: {note_type} ({len(rows)} notes)")
            for row in rows:
                nid = str(row.get("note_id", "?"))
                # Inject _note_type into the lookup row so the
                # downstream state_citation existence check in
                # _validate_subspecialty_consults can index notes by
                # (note_type, date). dict() copy is defensive — caller
                # may pass DataFrame rows or other non-mutable mappings.
                row_with_type = dict(row)
                row_with_type["_note_type"] = note_type
                note_lookup[nid] = row_with_type
                ts = format_local_dttm(
                    row.get("creation_dttm") or row.get("note_dttm")
                )
                body = row.get("note_text") or row.get("text") or ""
                parts.append(
                    f"\n### note_id={nid}  creation_dttm={ts}\n"
                    f"{body}\n"
                )
        return "\n".join(parts), note_lookup

    @staticmethod
    def _audit_hp_note_staleness(
        notes_by_type: dict[str, list[dict]],
        reference_dttm_iso: Optional[str],
        stale_threshold_days: float = 30.0,
    ) -> Optional[dict]:
        """Compute per-hp_note ages vs reference_dttm and return a trace
        event dict. Returns None when no audit is possible (no hp_notes
        routed, no reference_dttm, or no parseable creation_dttm on any
        routed H&P). Info-level event when all H&Ps within threshold;
        warn-level event when any exceeds threshold.

        Background: ``hp_note`` is exempt from ``notes_lookback_hours``
        per ``PER_ADMISSION_STABLE_NOTE_TYPES`` (config.py) —
        clinical-reviewer decision 2026-05-26. The exemption is needed
        (without it, any ICU stay > 48h silently drops the H&P) but
        leaves a staleness surface: a vent-dependent patient with an
        11-month-old H&P sees it routed regardless. This audit makes
        the surface visible to pilot + Phase-2 reviewers and de-risks
        the time-sensitive-modifier rule by surfacing the cases where
        intensivist confirmation against routed notes is most load-
        bearing. Trace-only; does NOT change routing or mutate the
        brief.
        """
        hp_rows = notes_by_type.get("hp_note") or []
        if not hp_rows or not reference_dttm_iso:
            return None

        try:
            ref_dt = datetime.fromisoformat(str(reference_dttm_iso).strip())
        except (TypeError, ValueError):
            return None

        per_note_ages: list[dict] = []
        for row in hp_rows:
            nid = str(row.get("note_id", "?"))
            creation_raw = row.get("creation_dttm") or row.get("note_dttm")
            if not creation_raw:
                continue
            try:
                created_dt = datetime.fromisoformat(str(creation_raw).strip())
            except (TypeError, ValueError):
                continue
            # Normalize tz: if one side is aware and the other naive,
            # strip tz from both. Cohort exports vary between naive
            # (local) and aware (UTC) timestamps.
            if (ref_dt.tzinfo is None) != (created_dt.tzinfo is None):
                ref_cmp = ref_dt.replace(tzinfo=None) if ref_dt.tzinfo else ref_dt
                created_cmp = (
                    created_dt.replace(tzinfo=None)
                    if created_dt.tzinfo else created_dt
                )
            else:
                ref_cmp = ref_dt
                created_cmp = created_dt
            age_days = (ref_cmp - created_cmp).total_seconds() / 86400.0
            per_note_ages.append({"note_id": nid, "age_days": round(age_days, 1)})

        if not per_note_ages:
            return None

        max_age = max(n["age_days"] for n in per_note_ages)
        is_stale = max_age > stale_threshold_days
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "hp_note_age_audit",
            "node": "scribe",
            "level": "warn" if is_stale else "info",
            "message": (
                f"H&P staleness: max_age_days={max_age:.1f} "
                f"({'STALE >' if is_stale else 'within '}"
                f"{stale_threshold_days:.0f}d) across "
                f"{len(per_note_ages)} routed hp_note(s)"
            ),
            "data": {
                "per_note_ages": per_note_ages,
                "max_age_days": max_age,
                "stale_threshold_days": stale_threshold_days,
                "is_stale": is_stale,
                "reference_dttm": str(reference_dttm_iso),
            },
        }

    @staticmethod
    def _validate_field(
        field_name: str,
        value: Optional[str],
        sources: list[str],
        note_lookup: dict[str, dict],
        target_patient_id: Optional[str],
    ) -> tuple[bool, Optional[str]]:
        """Run the substring + patient-identity guards for one extracted field.

        Returns (validated, reason_if_dropped). Validated is True only when
        every clause in ``value`` appears as a substring of at least one
        routed note's body AND every source note_id resolves to the target
        patient_id.

        ``field_name`` (e.g. "PMH", "allergies") is folded into the
        dropped-reason string for audit clarity; the validation logic
        itself is field-agnostic — PMH and allergies share the same
        comma/semicolon/'and' clause splitter and the same identity guard.
        """
        if not value:
            # Legitimate absence — not a rejection, just no extraction.
            return False, None

        # Patient-identity guard FIRST. A chart-routing error is more
        # severe than a substring miss — bail loudly.
        if target_patient_id is not None:
            for nid in sources:
                src = note_lookup.get(nid)
                if src is None:
                    return False, (
                        f"{field_name}_source note_id={nid} not in routed notes"
                    )
                src_pid = str(src.get("patient_id", ""))
                if src_pid and src_pid != str(target_patient_id):
                    return False, (
                        f"{field_name}_source note_id={nid} belongs to "
                        f"patient_id={src_pid}, expected {target_patient_id}"
                    )

        # Substring guard: every clause must appear in at least one routed
        # note. Concatenate all routed note bodies once; lowercase both
        # sides so trivial casing differences ("HTN" vs "htn") pass but
        # content differences fail.
        all_bodies = " ".join(
            (n.get("note_text") or n.get("text") or "")
            for n in note_lookup.values()
        ).lower()

        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(value) if c.strip()]
        for clause in clauses:
            if clause.lower() not in all_bodies:
                return False, (
                    f"{field_name} clause not found in any routed note: {clause!r}"
                )

        return True, None

    # Backwards-compat alias for the previous PMH-only signature. The test
    # suite and any prior callers reference _validate_pmh by name; the
    # implementation is the field-generic _validate_field above.
    @classmethod
    def _validate_pmh(
        cls,
        pmh: Optional[str],
        pmh_sources: list[str],
        note_lookup: dict[str, dict],
        target_patient_id: Optional[str],
    ) -> tuple[bool, Optional[str]]:
        return cls._validate_field(
            "PMH", pmh, pmh_sources, note_lookup, target_patient_id
        )

    @staticmethod
    def _validate_subspecialty_consults(
        consults: Optional[list[SubspecialtyConsult]],
        note_lookup: dict[str, dict],
        target_patient_id: Optional[str],
    ) -> tuple[
        Optional[list[SubspecialtyConsult]], bool, Optional[str]
    ]:
        """State-aware per-entry validator for the SubspecialtyConsult list.

        Field name is ``subspecialty_consults`` and the method name retains
        ``_subspecialty_consults`` for backward compatibility through the
        turn-4 field rename to ``subspecialty_consults``. Entries can
        carry ANY state value per the 2026-05-29 Section A revision.

        **Active entries** (state == ``Active``): existing per-service
        word-boundary substring guard against the body of
        ``source_note_id``. Corpus-wide substring matching is too
        permissive for consults — a passing mention of "renal failure"
        in any note would rubber-stamp a fabricated Nephrology consult.
        Word-boundary matching via ``\\b...\\b`` avoids the round-2
        false-positive set (Heme/hemoptysis, Pulm/pulmonary edema,
        Renal/renal failure, ID/ID number).

        **Non-Active entries** (state in Planned / Declined /
        AssessedNotNeeded): two guards.

        1. ``state_quote`` must (leniently) substring-match the body of
           ``source_note_id``. Normalization via
           ``_normalize_for_validator`` (NFKC + whitespace collapse +
           lowercase + curly-quote / dash / NBSP fold). Defensive
           leniency tolerates EHR-export character variants without
           weakening the hallucination guard — strictly: did this
           phrase actually appear in this specific note?

        2. ``state_citation`` must parse via ``_parse_state_citation``
           to a ``(note_type, ISO_date)`` tuple matching one of the
           routed notes' ``(_note_type, creation_dttm[:10])`` tuples.
           Parser is lenient on format (alias table for note_type,
           multi-format date parsing); citation EXISTENCE matching is
           strict against the routed-note metadata index.

        Structural integrity of state ↔ (quote, citation) is enforced
        HERE as a graceful per-entry drop (ported 2026-06-08 from the
        removed ``SubspecialtyConsult._check_state_evidence_consistency``
        model_validator, which used to RAISE and fail the whole
        ``ScribeExtractionLLM`` parse); the content correctness checks
        (substring, citation) follow.

        **KNOWN UNENFORCED LIMITATIONS** — surfaced for the npj Digital
        Medicine safety story; honest scope statement preempts the
        "your validator doesn't actually prove safety" pushback:

        - **Contextual misquote.** Substring match passes when the
          model picks a quote that exists verbatim in the source but
          in the wrong clinical context (e.g., ``PT`` as prothrombin
          time inside a coagulation paragraph, picked as evidence for
          a physical-therapy planning intent). This validator is a
          hallucination guard, not a clinical-accuracy guard. The
          model is responsible for picking contextually meaningful
          quotes; the convention rules in scribe.yaml +
          therapist.yaml carry that load.
        - **Wrong-source attribution.** When ``state_quote`` exists
          verbatim in multiple routed notes, the model can cite any
          of them and both substring + citation-existence guards
          will pass. Semantic matching between the quote's clinical
          intent and the cited note's content is not cheaply
          enforceable here.

        **MULTI-SOURCE PLANNING (DESIGN, NOT BUG).** Multiple routed
        notes containing planning language for the same service
        produce multiple ``SubspecialtyConsult`` entries with the
        same ``service`` but different ``state_quote`` +
        ``state_citation`` pairs. The intensivist's Section A render
        step dedupes/aggregates per service. Do NOT collapse these at
        the validator layer — every quote+citation pair is an
        independent provenance record that the receiving clinician
        may want to inspect.

        **Drop-with-audit semantics.** Individual entry failures drop
        just that entry; the field as a whole still validates if at
        least one entry survives. Patient-identity violations are
        always fatal (the entire field is invalidated) — a chart-
        routing error must surface loudly, not get partially absorbed.

        Returns ``(filtered_consults, validated, reason_if_dropped)``.
        ``filtered_consults`` is None when the input was None, an
        empty list when every entry was rejected, or a non-empty list
        when some survived. ``validated`` is True iff at least one
        entry passed AND no patient-identity violations occurred.
        """
        if not consults:
            return None, False, None

        # Patient-identity guard FIRST. Any cross-patient leak fails the
        # whole field — same posture as _validate_field. A single
        # mis-attributed source_note_id signals a routing bug that we do
        # NOT want to silently filter around.
        if target_patient_id is not None:
            for c in consults:
                src = note_lookup.get(str(c.source_note_id))
                if src is None:
                    return None, False, (
                        f"subspecialty_consults source note_id={c.source_note_id} "
                        f"(service={c.service!r}, state={c.state}) "
                        f"not in routed notes"
                    )
                src_pid = str(src.get("patient_id", ""))
                if src_pid and src_pid != str(target_patient_id):
                    return None, False, (
                        f"subspecialty_consults source note_id={c.source_note_id} "
                        f"(service={c.service!r}, state={c.state}) belongs "
                        f"to patient_id={src_pid}, expected {target_patient_id}"
                    )

        # Build (note_type, ISO_date) -> set[note_id] index for the
        # non-Active state_citation existence check. _note_type is
        # injected by _format_notes_for_extraction.
        routed_index: dict[tuple[str, str], set[str]] = {}
        for nid, row in note_lookup.items():
            nt = row.get("_note_type")
            if not nt:
                continue
            dttm = row.get("creation_dttm") or row.get("note_dttm")
            if not dttm:
                continue
            # YYYY-MM-DD prefix from ISO timestamp; cheap and avoids
            # bringing a tz-aware datetime parser into the validator.
            date_str = str(dttm)[:10]
            routed_index.setdefault((nt, date_str), set()).add(nid)

        # Per-entry validation. Active vs non-Active dispatch.
        kept: list[SubspecialtyConsult] = []
        rejected: list[str] = []
        for c in consults:
            # Structural integrity — ported 2026-06-08 from the removed schema
            # model_validator ``_check_state_evidence_consistency`` (which used
            # to RAISE and fail the whole ScribeExtractionLLM parse). Active
            # entries must NOT carry state_quote/state_citation; non-Active
            # entries MUST carry both. Malformed entries drop individually.
            if c.state == "Active" and (
                c.state_quote is not None or c.state_citation is not None
            ):
                rejected.append(
                    f"{c.service!r}@{c.source_note_id} (state=Active): "
                    f"Active entries must not carry state_quote/state_citation "
                    f"(structural integrity)"
                )
                continue
            if c.state != "Active" and (
                c.state_quote is None or c.state_citation is None
            ):
                rejected.append(
                    f"{c.service!r}@{c.source_note_id} (state={c.state}): "
                    f"non-Active entries require both state_quote and "
                    f"state_citation (structural integrity)"
                )
                continue

            src = note_lookup.get(str(c.source_note_id), {})
            body = src.get("note_text") or src.get("text") or ""

            if c.state == "Active":
                if _word_boundary_pattern(c.service).search(body):
                    kept.append(c)
                else:
                    rejected.append(
                        f"{c.service!r}@{c.source_note_id} "
                        f"(state=Active, source_type={c.source_type}): "
                        f"service word-boundary not in source body"
                    )
                continue

            # Non-Active: quote substring (lenient) + citation existence.
            quote_norm = _normalize_for_validator(c.state_quote or "")
            body_norm = _normalize_for_validator(body)
            if not quote_norm or quote_norm not in body_norm:
                rejected.append(
                    f"{c.service!r}@{c.source_note_id} "
                    f"(state={c.state}): state_quote not in source body "
                    f"after NFKC + whitespace + case normalization"
                )
                continue

            parsed = _parse_state_citation(
                c.state_citation or "", c.last_note_dttm
            )
            if parsed is None:
                rejected.append(
                    f"{c.service!r}@{c.source_note_id} "
                    f"(state={c.state}): state_citation="
                    f"{c.state_citation!r} could not be parsed to "
                    f"(note_type, date) tuple"
                )
                continue

            note_type, date_str = parsed
            if (note_type, date_str) not in routed_index:
                rejected.append(
                    f"{c.service!r}@{c.source_note_id} "
                    f"(state={c.state}): citation ({note_type}, "
                    f"{date_str}) does not match any routed note's "
                    f"(_note_type, creation_dttm[:10])"
                )
                continue

            kept.append(c)

        if not kept:
            return [], False, (
                "subspecialty_consults: no entry passed all validator guards "
                "— rejected: " + "; ".join(rejected)
            )

        if rejected:
            return kept, True, (
                f"subspecialty_consults: kept {len(kept)} entry(ies), "
                f"dropped {len(rejected)}: " + "; ".join(rejected)
            )

        return kept, True, None

    @staticmethod
    def _validate_admission_antibiotics(
        courses: Optional[list[AdmissionAntibioticCourse]],
        note_lookup: dict[str, dict],
        target_patient_id: Optional[str],
    ) -> tuple[
        Optional[list[AdmissionAntibioticCourse]],
        bool,
        Optional[str],
        list[str],  # coverage_log — kept-but-flagged signals for trace event
    ]:
        """Per-source substring + patient-identity + specificity guard.

        Per-course validation (cheap rejections first):
          1. Patient-identity guard: source_note_id resolves to target.
          2. source_quote >= 15 chars after normalization (prevents
             trivial anchors gaming the substring check).
          3. Drug name (or canonical alias) appears in source_quote.
          4. source_quote is a substring of source_note_id's body
             (NFKC + whitespace + curly-quote/dash normalized).

        Individual course failures drop just that course; the field as
        a whole validates if at least one course survives.

        Returns a 4-tuple; the 4th element ``coverage_log`` carries
        kept-but-flagged signals (currently: phrase-present-but-parsed-
        None dates — LLM emitted a date phrase it couldn't normalize to
        ISO). scribe.run() unpacks it into the trace event. Other field
        validators stay on the 3-tuple shape; this is the first field
        with a soft-signal outcome.
        """
        if not courses:
            return None, False, None, []

        # Patient-identity guard FIRST (any cross-patient leak fails the
        # whole field — chart-routing error must surface loudly, same
        # posture as _validate_subspecialty_consults).
        if target_patient_id is not None:
            for c in courses:
                src = note_lookup.get(str(c.source_note_id))
                if src is None:
                    return None, False, (
                        f"admission_antibiotics source note_id="
                        f"{c.source_note_id} (drug={c.drug!r}) not in "
                        f"routed notes"
                    ), []
                src_pid = str(src.get("patient_id", ""))
                if src_pid and src_pid != str(target_patient_id):
                    return None, False, (
                        f"admission_antibiotics source note_id="
                        f"{c.source_note_id} (drug={c.drug!r}) belongs "
                        f"to patient_id={src_pid}, expected "
                        f"{target_patient_id}"
                    ), []

        # Lazy import to keep the canonicalization module's dependency
        # graph minimal (no runtime dependency on icu_pause.agents.*).
        from icu_pause.tools.drug_canonicalization import drug_appears_in_text

        kept: list[AdmissionAntibioticCourse] = []
        rejected: list[str] = []
        coverage_log: list[str] = []

        for c in courses:
            quote_norm = _normalize_for_validator(c.source_quote)

            # Specificity floor — post-normalization to defeat NBSP-
            # padded "the" or similar trivial anchors.
            if len(quote_norm) < 15:
                rejected.append(
                    f"{c.drug!r}: source_quote too short "
                    f"({len(quote_norm)} chars after normalization, "
                    f"min 15)"
                )
                continue

            # Drug name (or any canonical alias) MUST appear in source_quote.
            # This is now the SOLE enforcement point — the schema's
            # _drug_name_in_source_quote model_validator was removed
            # 2026-06-08 (it failed the whole parse). Drops just this course.
            if not drug_appears_in_text(c.drug, quote_norm):
                rejected.append(
                    f"{c.drug!r}: drug name (or canonical alias) not "
                    f"in source_quote"
                )
                continue

            # Body substring check (per-source — NOT corpus-wide).
            src = note_lookup.get(str(c.source_note_id), {})
            body_norm = _normalize_for_validator(
                src.get("note_text") or src.get("text") or ""
            )
            if quote_norm not in body_norm:
                rejected.append(
                    f"{c.drug!r}@{c.source_note_id}: source_quote not "
                    f"found in source-note body"
                )
                continue

            # Notes-anchor — ported 2026-06-08 from the removed schema
            # model_validator ``_notes_must_be_in_source_quote`` (which RAISED
            # and failed the whole ScribeExtractionLLM parse). A populated
            # ``notes`` annotation must be substring-anchored in source_quote,
            # else an unanchored phrase could render to the receiving team.
            # Minimal-harm port: CLEAR the unanchored notes but KEEP the
            # otherwise-valid course — don't discard good antibiotic data over
            # a bad annotation (the same anti-pattern this whole fix removes).
            if c.notes and _normalize_for_validator(c.notes) not in quote_norm:
                coverage_log.append(
                    f"{c.drug!r}: notes={c.notes!r} not anchored in "
                    f"source_quote — cleared (course kept)"
                )
                c.notes = None

            # Completeness signal — ported 2026-06-08 from the removed schema
            # model_validator ``_completed_courses_must_have_end_date``,
            # downgraded from hard-raise to a soft coverage flag per reviewer
            # sign-off (completeness nit, not a safety guard; active-med truth
            # comes from medication_admin structured data, not this narrative).
            if getattr(c, "status", None) == "completed" and not c.end_date_phrase:
                coverage_log.append(
                    f"{c.drug!r}: status='completed' but no end_date_phrase "
                    f"(completeness gap)"
                )

            # Coverage signal: phrase present but ISO parsed missing.
            # Course is KEPT (phrase is still useful clinical info) but
            # the log gets logged to the trace event so the eval suite
            # can see which date phrasings the LLM stumbles on.
            for date_field in ("start_date", "end_date"):
                phrase = getattr(c, f"{date_field}_phrase")
                parsed = getattr(c, f"{date_field}_parsed")
                if phrase and not parsed:
                    coverage_log.append(
                        f"{c.drug!r} {date_field}_phrase={phrase!r} "
                        f"could not be normalized to ISO date"
                    )

            kept.append(c)

        if not kept:
            return [], False, (
                "admission_antibiotics: no course passed all validator "
                "guards — rejected: " + "; ".join(rejected)
            ), coverage_log
        if rejected:
            return kept, True, (
                f"admission_antibiotics: kept {len(kept)}, dropped "
                f"{len(rejected)}: " + "; ".join(rejected)
            ), coverage_log
        return kept, True, None, coverage_log

    # Subfields whose chart-quoted text is validated against routed notes.
    # baseline_source_quote is also checked but plays a special role
    # (it's the chart anchor for baseline_creatinine, not an independent
    # subfield) so it's handled separately in _validate_renal_context.
    _RENAL_CONTEXT_QUOTED_SUBFIELDS: ClassVar[tuple[str, ...]] = (
        "baseline_creatinine",
        "baseline_creatinine_date",
        "kdigo_stage",
        "urine_output_pattern",
        "nephrology_status",
        "rrt_indications_documented",
    )

    @classmethod
    def _validate_renal_context(
        cls,
        rc: Optional[RenalContext],
        sources: list[str],
        note_lookup: dict[str, dict],
        target_patient_id: Optional[str],
    ) -> tuple[
        Optional[RenalContext],  # filtered renal_context (subfields dropped)
        bool,                    # validated
        Optional[str],           # dropped_reason (whole-field drop)
        list[str],               # partial_drops (per-subfield audit)
    ]:
        """Per-subfield substring + per-source patient-identity guard.

        Distinct from ``_validate_field`` in that subfields drop
        INDEPENDENTLY rather than the whole field dropping on any clause
        miss. This matches the design-doc §4.3.3 partial-population
        requirement: a chart may anchor baseline_creatinine but not
        kdigo_stage, or document UOP but not nephrology consult status.
        Dropping the entire bundle on any one clause miss would lose
        receiver-relevant content.

        Returns ``(filtered, validated, dropped_reason, partial_drops)``.

        - ``filtered`` is None when the input was None OR every subfield
          dropped. Otherwise it's a new RenalContext with failing
          subfields nulled out.
        - ``validated`` is True when at least one subfield passed AND
          identity guards held. A patient-identity violation is always
          fatal — the whole bundle drops, never partial.
        - ``dropped_reason`` is set when the WHOLE bundle drops (no
          subfield survived OR identity guard tripped). None when
          partial drops occurred.
        - ``partial_drops`` lists per-subfield rejections in
          ``"<subfield>: <reason>"`` format. Empty when no partial drops.
        """
        if rc is None:
            return None, False, None, []

        # Patient-identity guard FIRST — same posture as other field
        # validators. Cross-patient leak fails the whole bundle.
        if target_patient_id is not None:
            for nid in sources:
                src = note_lookup.get(nid)
                if src is None:
                    return None, False, (
                        f"renal_context_source note_id={nid} not in "
                        f"routed notes"
                    ), []
                src_pid = str(src.get("patient_id", ""))
                if src_pid and src_pid != str(target_patient_id):
                    return None, False, (
                        f"renal_context_source note_id={nid} belongs to "
                        f"patient_id={src_pid}, expected "
                        f"{target_patient_id}"
                    ), []

        # Corpus body (concatenation of all routed notes) — same shape
        # as _validate_field. Lowercased once.
        all_bodies = " ".join(
            (n.get("note_text") or n.get("text") or "")
            for n in note_lookup.values()
        ).lower()

        # baseline_source_quote consistency check: when
        # baseline_creatinine is populated, baseline_source_quote MUST
        # be populated AND must substring-match a routed note. Without
        # the source_quote anchor, baseline_creatinine itself is too
        # short to validate reliably (a chart-extracted "1.4" is
        # almost certain to appear in any note's body by chance).
        baseline_ok = True
        if rc.baseline_creatinine is not None:
            if rc.baseline_source_quote is None:
                baseline_ok = False
                # baseline_creatinine drop is tracked below in the
                # subfield loop alongside other subfield drops.
            else:
                sq_lower = rc.baseline_source_quote.lower()
                if sq_lower not in all_bodies:
                    baseline_ok = False

        # Per-subfield validation: each quoted subfield must
        # substring-match at least one routed note. Lazy subfield-by-
        # subfield drop — no clause-splitter for renal context
        # (subfields are already small structured anchors, not
        # comma-separated lists).
        partial_drops: list[str] = []
        field_kwargs: dict[str, Optional[str]] = {}
        any_passed = False

        for subfield in cls._RENAL_CONTEXT_QUOTED_SUBFIELDS:
            value = getattr(rc, subfield)
            if value is None:
                field_kwargs[subfield] = None
                continue
            # baseline_creatinine and baseline_creatinine_date drop
            # together when the baseline_source_quote check failed
            # (they share the source anchor).
            if subfield in ("baseline_creatinine", "baseline_creatinine_date"):
                if not baseline_ok:
                    field_kwargs[subfield] = None
                    if subfield == "baseline_creatinine":
                        if rc.baseline_source_quote is None:
                            partial_drops.append(
                                "baseline_creatinine: baseline_source_quote "
                                "missing (required to anchor the value)"
                            )
                        else:
                            partial_drops.append(
                                "baseline_creatinine: baseline_source_quote "
                                "not found in any routed note"
                            )
                    continue
            if value.lower() not in all_bodies:
                partial_drops.append(
                    f"{subfield}: clause not found in any routed note "
                    f"({value!r})"
                )
                field_kwargs[subfield] = None
                continue
            field_kwargs[subfield] = value
            any_passed = True

        # baseline_source_quote follows baseline_creatinine —
        # preserved when the baseline pair survived, otherwise nulled.
        field_kwargs["baseline_source_quote"] = (
            rc.baseline_source_quote
            if (baseline_ok and field_kwargs.get("baseline_creatinine"))
            else None
        )

        if not any_passed:
            return None, False, (
                "renal_context: no subfield passed substring validation. "
                + ("Partial drops: " + "; ".join(partial_drops)
                   if partial_drops else
                   "LLM emitted RenalContext with all-None subfields.")
            ), partial_drops

        filtered = RenalContext(**field_kwargs)
        return filtered, True, None, partial_drops

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the scribe.

        Reads ``agent_context_text[scribe].notes`` for the routed notes and
        ``patient_context_text.demographics.patient_id`` for the identity
        check. Returns a ``scribe_extraction`` dict in state.
        """
        t0 = datetime.now(timezone.utc)
        trace_events: list[dict] = []

        agent_contexts = state.get("agent_context_text", {}) or {}
        scribe_ctx = agent_contexts.get(self.agent_name, {}) or {}
        notes_by_type = scribe_ctx.get("notes") or {}

        patient_ctx = state.get("patient_context_text", {}) or {}
        demographics = patient_ctx.get("demographics") or {}
        target_patient_id = demographics.get("patient_id")

        notes_text, note_lookup = self._format_notes_for_extraction(notes_by_type)

        trace_events.append({
            "timestamp": t0.isoformat(),
            "type": "agent_input",
            "node": self.agent_name,
            "level": "info",
            "message": (
                f"Routed notes: " + ", ".join(
                    f"{nt}({len(rows or [])})"
                    for nt, rows in notes_by_type.items() if rows
                ) or "Routed notes: NONE"
            ),
            "data": {
                "note_types": {
                    nt: len(rows or [])
                    for nt, rows in notes_by_type.items() if rows
                },
                "target_patient_id": target_patient_id,
            },
        })

        hp_audit = self._audit_hp_note_staleness(
            notes_by_type, state.get("reference_dttm")
        )
        if hp_audit is not None:
            trace_events.append(hp_audit)

        # Empty input — emit an empty extraction and let downstream agents fall back.
        if not note_lookup:
            extraction = ScribeExtraction(
                pmh=None,
                pmh_sources=[],
                pmh_validated=False,
                pmh_dropped_reason="no routed notes",
                allergies=None,
                allergies_sources=[],
                allergies_validated=False,
                allergies_dropped_reason="no routed notes",
                home_meds=None,
                home_meds_sources=[],
                home_meds_validated=False,
                home_meds_dropped_reason="no routed notes",
                code_status=None,
                code_status_sources=[],
                code_status_validated=False,
                code_status_dropped_reason="no routed notes",
                subspecialty_consults=None,
                subspecialty_consults_validated=False,
                subspecialty_consults_dropped_reason="no routed notes",
                admission_antibiotics=None,
                admission_antibiotics_validated=False,
                admission_antibiotics_dropped_reason="no routed notes",
                renal_context=None,
                renal_context_sources=[],
                renal_context_validated=False,
                renal_context_dropped_reason="no routed notes",
                renal_context_partial_drops=[],
            )
            return {
                "scribe_extraction": extraction.model_dump(),
                "pipeline_metrics": [{
                    "agent": self.agent_name,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0.0,
                    "model": "(skipped — no notes)",
                }],
                "trace_events": trace_events,
            }

        user_message = (
            "Extract PMH, allergies, home medications, code status, "
            "active subspecialty consultants, admission antibiotics, "
            "and renal context (baseline Cr + KDIGO + UOP + nephrology "
            "status + RRT indications) from the routed clinical notes "
            "below. Follow the spine+append rule (PMH), the additive-"
            "union rules (allergies, home meds), the most-recent-wins "
            "rule (code status), the (a)∪(c) extraction + sign-off "
            "exclusion rule (active consults), the most-recent-"
            "comprehensive-wins rule (admission antibiotics), the "
            "per-subfield source-priority rules for renal context "
            "(MOST-RECENT-WINS / AUTHORITATIVE-WINS / ACCUMULATIVE — "
            "see your system prompt §RENAL CONTEXT), and the "
            "verbatim-quoting hard rules from your system prompt. "
            "Return JSON only.\n\n"
            f"{notes_text}"
        )

        try:
            llm_out = self.llm.invoke(
                system=self.system_prompt,
                user=user_message,
                response_format=ScribeExtractionLLM,
            )
        except Exception as e:
            logger.error(f"Scribe LLM call failed: {e}")
            extraction = ScribeExtraction(
                pmh=None,
                pmh_sources=[],
                pmh_validated=False,
                pmh_dropped_reason=f"LLM call failed: {e}",
                allergies=None,
                allergies_sources=[],
                allergies_validated=False,
                allergies_dropped_reason=f"LLM call failed: {e}",
                home_meds=None,
                home_meds_sources=[],
                home_meds_validated=False,
                home_meds_dropped_reason=f"LLM call failed: {e}",
                code_status=None,
                code_status_sources=[],
                code_status_validated=False,
                code_status_dropped_reason=f"LLM call failed: {e}",
                subspecialty_consults=None,
                subspecialty_consults_validated=False,
                subspecialty_consults_dropped_reason=f"LLM call failed: {e}",
                admission_antibiotics=None,
                admission_antibiotics_validated=False,
                admission_antibiotics_dropped_reason=f"LLM call failed: {e}",
                renal_context=None,
                renal_context_sources=[],
                renal_context_validated=False,
                renal_context_dropped_reason=f"LLM call failed: {e}",
                renal_context_partial_drops=[],
            )
            usage = self.llm.last_usage
            metrics = {
                "agent": self.agent_name,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "latency_ms": round((datetime.now(timezone.utc) - t0).total_seconds() * 1000, 1),
                "model": usage.model,
            }
            trace_events.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "agent_output",
                "node": self.agent_name,
                "level": "error",
                "message": f"LLM call failed: {e}",
                "data": {"metrics": metrics},
            })
            return {
                "scribe_extraction": extraction.model_dump(),
                "pipeline_metrics": [metrics],
                "trace_events": trace_events,
            }

        pmh_ok, pmh_drop = ScribeAgent._validate_field(
            "PMH", llm_out.pmh, llm_out.pmh_sources,
            note_lookup, target_patient_id,
        )
        allergies_ok, allergies_drop = ScribeAgent._validate_field(
            "allergies", llm_out.allergies, llm_out.allergies_sources,
            note_lookup, target_patient_id,
        )
        home_meds_ok, home_meds_drop = ScribeAgent._validate_field(
            "home_meds", llm_out.home_meds, llm_out.home_meds_sources,
            note_lookup, target_patient_id,
        )
        code_status_ok, code_status_drop = ScribeAgent._validate_field(
            "code_status", llm_out.code_status, llm_out.code_status_sources,
            note_lookup, target_patient_id,
        )
        consults_kept, consults_ok, consults_drop = (
            ScribeAgent._validate_subspecialty_consults(
                llm_out.subspecialty_consults, note_lookup, target_patient_id,
            )
        )
        (
            abx_kept, abx_ok, abx_drop, abx_coverage_log,
        ) = ScribeAgent._validate_admission_antibiotics(
            llm_out.admission_antibiotics, note_lookup, target_patient_id,
        )
        (
            rc_kept, rc_ok, rc_drop, rc_partial_drops,
        ) = ScribeAgent._validate_renal_context(
            llm_out.renal_context,
            llm_out.renal_context_sources,
            note_lookup, target_patient_id,
        )

        if not pmh_ok and llm_out.pmh and pmh_drop:
            logger.warning(
                "Scribe PMH dropped by validator: %s | LLM emitted: %r",
                pmh_drop, llm_out.pmh,
            )
        if not allergies_ok and llm_out.allergies and allergies_drop:
            logger.warning(
                "Scribe allergies dropped by validator: %s | LLM emitted: %r",
                allergies_drop, llm_out.allergies,
            )
        if not home_meds_ok and llm_out.home_meds and home_meds_drop:
            logger.warning(
                "Scribe home_meds dropped by validator: %s | LLM emitted: %r",
                home_meds_drop, llm_out.home_meds,
            )
        if not code_status_ok and llm_out.code_status and code_status_drop:
            logger.warning(
                "Scribe code_status dropped by validator: %s | LLM emitted: %r",
                code_status_drop, llm_out.code_status,
            )
        if consults_drop and llm_out.subspecialty_consults:
            log_fn = logger.warning if not consults_ok else logger.info
            log_fn(
                "Scribe subspecialty_consults validator: %s | LLM emitted %d "
                "consult(s)",
                consults_drop, len(llm_out.subspecialty_consults),
            )
        if abx_drop and llm_out.admission_antibiotics:
            log_fn = logger.warning if not abx_ok else logger.info
            log_fn(
                "Scribe admission_antibiotics validator: %s | LLM "
                "emitted %d course(s)",
                abx_drop, len(llm_out.admission_antibiotics),
            )
        if abx_coverage_log:
            logger.info(
                "Scribe admission_antibiotics coverage_log (%d): %s",
                len(abx_coverage_log), "; ".join(abx_coverage_log),
            )
        if rc_drop and llm_out.renal_context:
            log_fn = logger.warning if not rc_ok else logger.info
            log_fn(
                "Scribe renal_context validator: %s",
                rc_drop,
            )
        if rc_partial_drops:
            logger.info(
                "Scribe renal_context partial drops (%d): %s",
                len(rc_partial_drops), "; ".join(rc_partial_drops),
            )

        extraction = ScribeExtraction(
            pmh=llm_out.pmh if pmh_ok else None,
            pmh_sources=llm_out.pmh_sources if pmh_ok else [],
            pmh_validated=pmh_ok,
            pmh_dropped_reason=None if pmh_ok else pmh_drop,
            allergies=llm_out.allergies if allergies_ok else None,
            allergies_sources=llm_out.allergies_sources if allergies_ok else [],
            allergies_validated=allergies_ok,
            allergies_dropped_reason=None if allergies_ok else allergies_drop,
            home_meds=llm_out.home_meds if home_meds_ok else None,
            home_meds_sources=(
                llm_out.home_meds_sources if home_meds_ok else []
            ),
            home_meds_validated=home_meds_ok,
            home_meds_dropped_reason=None if home_meds_ok else home_meds_drop,
            code_status=llm_out.code_status if code_status_ok else None,
            code_status_sources=(
                llm_out.code_status_sources if code_status_ok else []
            ),
            code_status_validated=code_status_ok,
            code_status_dropped_reason=(
                None if code_status_ok else code_status_drop
            ),
            subspecialty_consults=consults_kept if consults_ok else None,
            subspecialty_consults_validated=consults_ok,
            subspecialty_consults_dropped_reason=consults_drop,
            admission_antibiotics=abx_kept if abx_ok else None,
            admission_antibiotics_validated=abx_ok,
            admission_antibiotics_dropped_reason=abx_drop,
            renal_context=rc_kept,
            renal_context_sources=(
                llm_out.renal_context_sources if rc_ok else []
            ),
            renal_context_validated=rc_ok,
            renal_context_dropped_reason=rc_drop,
            renal_context_partial_drops=rc_partial_drops,
        )

        usage = self.llm.last_usage
        elapsed_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000
        metrics = {
            "agent": self.agent_name,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "latency_ms": round(elapsed_ms, 1),
            "model": usage.model,
            "prompt_version": self.prompt_version,
        }

        trace_events.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "agent_output",
            "node": self.agent_name,
            "level": "info",
            "message": (
                f"PMH validated={extraction.pmh_validated} "
                f"sources={len(extraction.pmh_sources)} "
                f"dropped_reason={extraction.pmh_dropped_reason!r} | "
                f"allergies validated={extraction.allergies_validated} "
                f"sources={len(extraction.allergies_sources)} "
                f"dropped_reason={extraction.allergies_dropped_reason!r} | "
                f"home_meds validated={extraction.home_meds_validated} "
                f"sources={len(extraction.home_meds_sources)} "
                f"dropped_reason={extraction.home_meds_dropped_reason!r} | "
                f"code_status validated={extraction.code_status_validated} "
                f"sources={len(extraction.code_status_sources)} "
                f"dropped_reason={extraction.code_status_dropped_reason!r} | "
                f"subspecialty_consults validated={extraction.subspecialty_consults_validated} "
                f"count={len(extraction.subspecialty_consults or [])} "
                f"dropped_reason={extraction.subspecialty_consults_dropped_reason!r} | "
                f"admission_antibiotics validated={extraction.admission_antibiotics_validated} "
                f"count={len(extraction.admission_antibiotics or [])} "
                f"unparsed_dates={len(abx_coverage_log)} "
                f"dropped_reason={extraction.admission_antibiotics_dropped_reason!r} | "
                f"renal_context validated={extraction.renal_context_validated} "
                f"sources={len(extraction.renal_context_sources)} "
                f"partial_drops={len(extraction.renal_context_partial_drops)} "
                f"dropped_reason={extraction.renal_context_dropped_reason!r}"
            ),
            "data": {
                "pmh_validated": extraction.pmh_validated,
                "pmh_sources": extraction.pmh_sources,
                "pmh_dropped_reason": extraction.pmh_dropped_reason,
                "pmh_chars": len(extraction.pmh) if extraction.pmh else 0,
                "allergies_validated": extraction.allergies_validated,
                "allergies_sources": extraction.allergies_sources,
                "allergies_dropped_reason": extraction.allergies_dropped_reason,
                "allergies_chars": (
                    len(extraction.allergies) if extraction.allergies else 0
                ),
                "home_meds_validated": extraction.home_meds_validated,
                "home_meds_sources": extraction.home_meds_sources,
                "home_meds_dropped_reason": extraction.home_meds_dropped_reason,
                "home_meds_chars": (
                    len(extraction.home_meds) if extraction.home_meds else 0
                ),
                "code_status_validated": extraction.code_status_validated,
                "code_status_sources": extraction.code_status_sources,
                "code_status_dropped_reason": (
                    extraction.code_status_dropped_reason
                ),
                "code_status_chars": (
                    len(extraction.code_status)
                    if extraction.code_status else 0
                ),
                "subspecialty_consults_validated": (
                    extraction.subspecialty_consults_validated
                ),
                "subspecialty_consults_count": len(extraction.subspecialty_consults or []),
                "subspecialty_consults_services": [
                    c.service for c in (extraction.subspecialty_consults or [])
                ],
                "subspecialty_consults_dropped_reason": (
                    extraction.subspecialty_consults_dropped_reason
                ),
                "admission_antibiotics_validated": (
                    extraction.admission_antibiotics_validated
                ),
                "admission_antibiotics_count": len(
                    extraction.admission_antibiotics or []
                ),
                "admission_antibiotics_drugs": [
                    c.drug for c in (extraction.admission_antibiotics or [])
                ],
                "admission_antibiotics_unparsed_dates": len(abx_coverage_log),
                "admission_antibiotics_unparsed_phrases": abx_coverage_log,
                "admission_antibiotics_dropped_reason": (
                    extraction.admission_antibiotics_dropped_reason
                ),
                "renal_context_validated": extraction.renal_context_validated,
                "renal_context_sources": extraction.renal_context_sources,
                "renal_context_populated_subfields": (
                    [
                        sf for sf in ScribeAgent._RENAL_CONTEXT_QUOTED_SUBFIELDS
                        if extraction.renal_context is not None
                        and getattr(extraction.renal_context, sf) is not None
                    ]
                    if extraction.renal_context else []
                ),
                "renal_context_partial_drops": (
                    extraction.renal_context_partial_drops
                ),
                "renal_context_dropped_reason": (
                    extraction.renal_context_dropped_reason
                ),
                "metrics": metrics,
            },
        })

        return {
            "scribe_extraction": extraction.model_dump(),
            "pipeline_metrics": [metrics],
            "trace_events": trace_events,
        }
