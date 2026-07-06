"""Intensivist Agent: clinical reasoning layer that harmonizes domain agent outputs.

This agent acts as the "attending physician" in the pipeline. It receives:
- All 6 domain agent JSON outputs (snippets)
- Full structured data bundle
- Direct access to progress notes and consult notes

It resolves cross-domain conflicts, adjudicates uncertain fields, and produces
the harmonized clinical narrative across all ICU-PAUSE sections.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from icu_pause.config import Settings
from icu_pause.data.context import format_local_dttm
from icu_pause.llm.provider import BaseLLM, create_llm
from icu_pause.tools.med_classes import (
    SEDATION_ANALGESIA_PARALYTIC_DRUGS,
    SedationDrugClass,
    classify_drug,
)
from icu_pause.schemas.icu_pause import (
    AgentSnippet,
    AgentSnippetLLM,
    ICUPauseSection,
    IntensivistOutput,
    IntensivistOutputLLM,
    Warning,
    WarningCategory,
    WarningSeverity,
    wrap_llm_intensivist,
    wrap_llm_snippet,
)

logger = logging.getLogger(__name__)

# All ICU-PAUSE section keys the Intensivist adjudicates
ALL_SECTIONS = [s.value for s in ICUPauseSection]


class IntensivistAgent:
    """Clinical reasoning agent that synthesizes domain agent outputs into
    a harmonized ICU-PAUSE handoff brief.

    Unlike domain agents (which read raw patient data), the Intensivist reads
    the domain agents' structured JSON outputs alongside the raw data and notes,
    then produces the final adjudicated content for every ICU-PAUSE section.
    """

    agent_name = "intensivist"

    def __init__(self, settings: Settings):
        self.llm: BaseLLM = create_llm(settings)
        # Intensivist needs a higher token budget for CoT reasoning + 8 sections
        self.llm.max_tokens = max(settings.llm_max_tokens, settings.intensivist_max_tokens)
        self.settings = settings
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        path = Path(self.settings.prompts_dir) / "intensivist.yaml"
        if not path.exists():
            logger.warning(f"Prompt file not found: {path}. Using default prompt.")
            return self._default_prompt()
        with open(path) as f:
            data = yaml.safe_load(f)
        return data.get("system_prompt", self._default_prompt())

    @staticmethod
    def _default_prompt() -> str:
        return (
            "You are an ICU attending physician (intensivist) performing the final "
            "clinical review for an ICU-to-ward transfer handoff note using the "
            "ICU-PAUSE framework.\n\n"
            "You receive outputs from 6 domain agents (Nurse, Respiratory Therapist, "
            "Pharmacist, Dietitian, Case Manager, Physical/Occupational Therapist) "
            "along with the original patient data and clinical notes.\n\n"
            "Your responsibilities:\n"
            "1. HARMONIZE: Merge domain agent contributions into a coherent clinical "
            "narrative for each ICU-PAUSE section.\n"
            "2. RESOLVE CONFLICTS: When agents disagree (e.g., fluid restriction vs. "
            "nutrition goals), use clinical reasoning and source data to adjudicate.\n"
            "3. FILL GAPS: If domain agents missed clinically important information "
            "visible in the notes or data, add it.\n"
            "4. ENSURE SAFETY: Flag any critical clinical concerns, pending actions, "
            "or safety issues that must not be lost in handoff.\n"
            "5. SIGN OFF: Produce the final authoritative content for all 8 ICU-PAUSE "
            "sections.\n\n"
            "Output valid JSON matching the AgentSnippet schema with agent_name "
            "'intensivist' and contributions for ALL 8 sections."
        )

    @staticmethod
    def _format_pending_tests_block(context: dict[str, Any]) -> str:
        """Build a deterministic Section P pending-tests list from masked
        microbiology + lab rows.

        The retriever marks not-yet-resulted cultures with
        ``organism = "pending"`` and not-yet-resulted labs with
        ``lab_value = "pending"`` (see retriever._mask_future_microbiology_results
        and _mask_future_lab_results). LLMs reliably miss these in raw
        JSON, so we pre-extract them and surface the list explicitly to
        the Intensivist's Section P prompt. Returns empty string when
        nothing is pending.
        """
        if not isinstance(context, dict):
            return ""

        pending_marker = "pending"

        cultures: list[str] = []
        for row in context.get("microbiology") or []:
            if not isinstance(row, dict):
                continue
            organism_fields = (
                str(row.get("organism") or "").strip().lower(),
                str(row.get("organism_name") or "").strip().lower(),
                str(row.get("organism_category") or "").strip().lower(),
            )
            if pending_marker not in organism_fields:
                continue
            specimen = (
                row.get("specimen_category")
                or row.get("specimen_type")
                or row.get("specimen")
                or "unknown specimen"
            )
            collected = format_local_dttm(
                row.get("collect_dttm") or row.get("collected_dttm")
            )
            cultures.append(f"  - {specimen} (collected {collected})")

        labs: list[str] = []
        for row in context.get("labs") or []:
            if not isinstance(row, dict):
                continue
            value = str(row.get("lab_value") or "").strip().lower()
            if value != pending_marker:
                continue
            category = (
                row.get("lab_category")
                or row.get("lab_name")
                or "unknown lab"
            )
            collected = format_local_dttm(row.get("lab_collect_dttm"))
            labs.append(f"  - {category} (collected {collected})")

        if not cultures and not labs:
            return ""

        parts = [
            "**PENDING TESTS DETECTED in the structured data — use these for Section P:**"
        ]
        if cultures:
            parts.append("Microbiology cultures collected, not yet resulted:")
            parts.extend(cultures)
        if labs:
            parts.append("Labs collected, not yet resulted:")
            parts.extend(labs)
        parts.append(
            "List each of the above in Section P. Add anything additional "
            "you find in the clinical notes (imaging, consults, etc.)."
        )
        return "\n".join(parts) + "\n"

    @staticmethod
    def _format_med_state_block(
        context: dict[str, Any],
        drug_class_filter: set[SedationDrugClass],
    ) -> str:
        """Pre-extract sedation/analgesia/paralytic state from
        ``context['meds']['states']['records']`` so the intensivist
        renders Section I prose with explicit temporal tags.

        Same pattern as ``_format_pending_tests_block`` and
        ``_format_transfer_exam_block``: pull the signal out of the raw
        JSON the LLM unreliably parses and surface a verbatim labeled
        block at the top of the user message. The Section I prompt rule
        ("SEDATION/ANALGESIA/PARALYTIC STATE TAGGING") references this
        block by name and forbids bare "sedation with {drug}" renderings.

        Filter:
        - ``classify_drug(drug_name)`` ∈ ``drug_class_filter``, AND
        - state ∈ {ACTIVE, ACTIVE_SCHEDULED, ACTIVE_PRN, RECENTLY_STOPPED}
          OR ``trending_to_zero`` is True

        HISTORICAL records drop entirely from the pin — the goal is the
        active-tense rendering backstop, and a HISTORICAL pin would
        regenerate the same noise problem (intensivist dutifully renders
        a propofol bolus from 5 days ago as if it were current). The
        ``RECENTLY_STOPPED`` window is already 48h-bounded by the
        classifier ([med_state.py:86](../tools/med_state.py)), so no
        additional 24h heuristic is needed.

        Returns empty string when no records pass the filter (so callers
        can skip injection cleanly).

        Function signature is intentionally generic on ``drug_class_filter``
        so future tense rules for other med groups (e.g., vasopressor
        wean tense) can reuse this pin pattern — see
        ``tools/med_classes.py`` header comment for the asymmetry note.
        """
        if not isinstance(context, dict):
            return ""
        meds = context.get("meds") or {}
        states = meds.get("states") or {}
        records = states.get("records") or []
        if not records:
            return ""

        active_states = {
            "ACTIVE", "ACTIVE_SCHEDULED", "ACTIVE_PRN", "RECENTLY_STOPPED",
        }

        lines: list[str] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            drug_name = rec.get("drug_name") or ""
            cls = classify_drug(drug_name)
            if cls is None or cls not in drug_class_filter:
                continue
            state = (rec.get("state") or "").upper()
            trending = bool(rec.get("trending_to_zero"))
            if state not in active_states and not trending:
                continue
            last = format_local_dttm(rec.get("last_admin_dttm"))
            qualifiers: list[str] = []
            if trending:
                qualifiers.append("trending-to-zero")
            qualifier_str = (
                f", {'; '.join(qualifiers)}" if qualifiers else ""
            )
            lines.append(
                f"  - {drug_name} [{cls}]: {state} "
                f"(last admin {last}{qualifier_str})"
            )

        if not lines:
            return ""

        return (
            "## ACTIVE / RECENTLY STOPPED SEDATION-ANALGESIA-PARALYTIC AGENTS\n"
            "(pre-extracted from meds.states.records; HISTORICAL drugs "
            "intentionally excluded)\n\n"
            + "\n".join(lines)
            + "\n\n"
            "USE THIS BLOCK AS THE AUTHORITATIVE SOURCE for sedation/"
            "analgesia/paralytic state when writing Section I prose. "
            "Apply the SEDATION/ANALGESIA/PARALYTIC STATE TAGGING rule "
            "from your system prompt: ACTIVE → 'on {drug}'; "
            "RECENTLY_STOPPED → '{drug} (weaned off M/D)' using "
            "last_admin_dttm; trending-to-zero → '{drug} (weaning, last "
            "admin M/D HH:MM)'. Tenseless 'sedation with {drug}' is "
            "forbidden. Drugs NOT in this block are either HISTORICAL "
            "(do not render in Section I prose unless clinically "
            "essential, in which case use 'previously on {drug} (d/c'd "
            "M/D)') or absent from the structured data.\n"
        )

    @staticmethod
    def _format_transfer_exam_block(context: dict[str, Any]) -> str:
        """Wrap the deterministic Phase-3 transfer-exam block for prompt
        injection at the top of Section E.

        The block itself is computed in ``context.py:build_transfer_exam_block``
        and lands on the agent context as ``transfer_exam_block``. We surface
        it to the LLM with a strong anti-duplication preamble — paraphrase
        prevention is the highest-risk failure mode for this design, so the
        instruction is repeated explicitly and the verbatim text is fenced.
        Empty string when the block is absent (e.g. reference_dttm is None).
        """
        if not isinstance(context, dict):
            return ""
        block = context.get("transfer_exam_block") or ""
        if not block:
            return ""
        return (
            "**TRANSFER EXAM BLOCK — copy verbatim into Section E. "
            "Place this BEFORE any narrative observations.**\n"
            "```\n"
            f"{block}\n"
            "```\n"
            "ANTI-DUPLICATION RULE (apply to Section E and every other "
            "section that follows): values that appear in the TRANSFER EXAM "
            "block above MUST NOT appear anywhere else in Section E, "
            "INCLUDING in paraphrased form. Do not write 'blood pressure was "
            "within normal limits' or 'the patient was hemodynamically "
            "stable on vitals' — the structured block is the ONLY place "
            "these values appear. Narrative observations (lines/drains, "
            "wound status, isolation, C-collar, device positioning) go "
            "AFTER the block as freehand text."
        )

    @staticmethod
    def _format_scribe_pmh_block(state: dict[str, Any]) -> str:
        """Build the labeled PMH pin for the top of the intensivist message.

        Reads ``state['scribe_extraction']`` (a ScribeExtraction dict). Only
        emits content when scribe extracted AND validated a PMH; otherwise
        returns the empty string and Section I falls back to the chart-review
        string (see ``_format_agent_outputs`` PMH RULE branching).

        See docs/pmh_structured_extraction_design.md.
        """
        extraction = state.get("scribe_extraction") or {}
        if not extraction.get("pmh_validated"):
            return ""
        pmh = (extraction.get("pmh") or "").strip()
        if not pmh:
            return ""
        sources = extraction.get("pmh_sources") or []
        sources_str = ", ".join(str(s) for s in sources) if sources else "(no source IDs)"
        return (
            "## PMH (extracted by scribe — verbatim from routed notes, "
            f"validated; sources: {sources_str})\n\n"
            f"```\n{pmh}\n```\n\n"
            "This PMH text feeds TWO render slots in Section I, with "
            "different rules per slot:\n"
            "1. **One-liner lead sentence:** SELECT ≤3 load-bearing "
            "conditions and ABBREVIATE per the ONE-LINER PMH SELECTION "
            "RULE in the Section I instructions below. Emit the structured "
            "``one_liner_pmh_selection`` field alongside the lead sentence "
            "with per-entry ``display`` / ``source_clause_anchor`` / "
            "``why_lead`` / ``modifier_confirmation`` (for time-sensitive "
            "modifiers) / ``rank``. The orchestrator runs the in-prose ↔ "
            "structured alignment lint using normalized-form matching.\n"
            "2. **PMH paragraph slot:** Render the verbatim PMH text "
            "above AS-IS, on its own line within Section I, labeled "
            "``Full PMH per chart:``. Do NOT paraphrase, expand "
            "abbreviations, reorder, or re-derive. The scribe quoted it "
            "verbatim from the chart and the orchestrator has substring- + "
            "patient-identity-validated every clause; the paragraph slot "
            "preserves that audit-safe contract.\n"
        )

    @staticmethod
    def _format_scribe_allergies_block(state: dict[str, Any]) -> str:
        """Build the labeled allergies pin for the top of the intensivist message.

        Reads ``state['scribe_extraction']``. Emits content only when scribe
        extracted AND validated an allergies value; otherwise returns ""
        and Section I uses the fallback ("Allergies not documented in
        available notes") via the ALLERGIES RULE branching in
        ``_format_agent_outputs``.

        Parallel to ``_format_scribe_pmh_block`` — allergies and PMH are
        independently validated, so either can pin while the other falls
        back.
        """
        extraction = state.get("scribe_extraction") or {}
        if not extraction.get("allergies_validated"):
            return ""
        allergies = (extraction.get("allergies") or "").strip()
        if not allergies:
            return ""
        sources = extraction.get("allergies_sources") or []
        sources_str = (
            ", ".join(str(s) for s in sources) if sources else "(no source IDs)"
        )
        return (
            "## ALLERGIES (extracted by scribe — verbatim from routed notes, "
            f"validated; sources: {sources_str})\n\n"
            f"```\n{allergies}\n```\n\n"
            "USE THIS ALLERGIES VALUE AS-IS as the FIRST LINE of Section I "
            "content (formatted as 'Allergies: <value>') before the ICU "
            "course prose. The scribe has quoted it verbatim and the "
            "orchestrator has substring- + patient-identity-validated "
            "every item. Do NOT paraphrase 'NKDA' to 'no known drug "
            "allergies' (or vice versa), add reaction descriptors that "
            "aren't in the quote, or re-derive allergies from raw notes.\n"
        )

    @staticmethod
    def _format_scribe_subspecialty_consults_block(state: dict[str, Any]) -> str:
        """Build the labeled active-consults pin for Section A's input.

        Reads ``state['scribe_extraction']``. Emits content only when
        scribe extracted AND validated at least one SubspecialtyConsult;
        otherwise returns "" and Section A falls back to the existing
        therapist (PT/OT/SLP/Wound Care) + case_manager (Palliative /
        Social Work) merge — which does NOT include medical/surgical
        subspecialties, which is the structural gap this pin closes.

        Renders as a compact bulleted list with source attribution per
        consult, so the intensivist can quote service names verbatim
        without re-locating them in long consult notes.
        """
        extraction = state.get("scribe_extraction") or {}
        if not extraction.get("subspecialty_consults_validated"):
            return ""
        consults = extraction.get("subspecialty_consults") or []
        if not consults:
            return ""

        # Order by last_note_dttm descending so the most recently active
        # service appears first; ties broken by service name to keep
        # rendering deterministic. Some ICU patients (post-transplant,
        # multi-organ failure) legitimately have long consultant rosters,
        # so we prefer compact rendering over truncation when the list
        # grows — receiving teams need to see every active service, just
        # not necessarily on its own line.
        ordered = sorted(
            consults,
            key=lambda c: (
                str(c.get("last_note_dttm") or ""),
                c.get("service") or "",
            ),
            reverse=True,
        )
        n = len(ordered)

        # Three rendering modes:
        #   N ≤ 6  → bulleted, one per line, with provenance per service
        #   6 < N ≤ 10 → compact comma-separated, no truncation
        #   N > 10 → compact + truncate after 10 with "+K more" overflow
        # The compact branch loses per-service provenance to gain
        # density. The full structured list still ships to the audit log
        # via the trace event; only the LLM-facing prompt block is
        # compressed.
        if n <= 6:
            lines = []
            for c in ordered:
                service = c.get("service", "?")
                dttm = format_local_dttm(c.get("last_note_dttm"))
                stype = c.get("source_type", "?")
                label = (
                    "consult note" if stype == "consults_note"
                    else "progress note A&P"
                )
                lines.append(f"  - {service} (last seen {dttm}, via {label})")
            body = "\n".join(lines)
        elif n <= 10:
            services_csv = ", ".join(c.get("service", "?") for c in ordered)
            body = f"  {services_csv}"
        else:
            shown = ordered[:10]
            overflow = n - 10
            services_csv = ", ".join(c.get("service", "?") for c in shown)
            body = (
                f"  {services_csv}, "
                f"and {overflow} additional active "
                f"{'service' if overflow == 1 else 'services'} "
                f"(see consult notes for the full roster)"
            )

        return (
            "## ACTIVE SUBSPECIALTY CONSULTS (extracted by scribe — "
            f"validated; per-service word-boundary + patient-identity "
            f"guarded; n={n})\n\n"
            f"{body}\n\n"
            "USE THIS LIST AS-IS for the 'Subspecialty Consultants:' "
            "line in Section A. The scribe identified each service via "
            "either a consult note or a present-tense engagement frame "
            "in the most recent progress-note A&P, and the validator "
            "confirmed each service word-boundary appears in its source "
            "note's body. Do NOT add services from your own reading of "
            "the consult notes — if a service isn't in this list, the "
            "extractor either didn't find it or actively excluded it "
            "(e.g., sign-off detection). Merge with Therapist's "
            "PT/OT/SLP/Wound Care and Case Manager's Palliative / "
            "Social Work lines to form the full Section A — those are "
            "separate lanes, not duplicates.\n"
        )

    @staticmethod
    def _format_scribe_renal_context_block(state: dict[str, Any]) -> str:
        """Build the labeled renal-context pin for the AKI/CKD problem
        in Section S.

        Reads ``state['scribe_extraction']``. Emits content only when
        scribe extracted AND validated renal_context (i.e., at least
        one subfield survived the per-subfield substring + identity
        guard). Each subfield renders verbatim or with the explicit
        "not documented in source" / "no baseline anchor in source"
        fallback.

        See docs/renal_electrolyte_vte_extraction_design.md §4.3.4
        (pin format) + §4.3.6 (intensivist S-synthesis rule). The pin
        format anticipates the v3.1 KDIGO compute downstream:
        ``baseline_creatinine`` + ``baseline_creatinine_date`` feed
        Path A (long-window ratio); the orchestrator joins to
        structured Cr for Path B (48h short-window). The "Baseline
        source" disclaimer line tells the receiver and downstream
        consumers that the baseline anchor is whatever the chart
        documented, which may differ from the KDIGO-canonical lowest-
        3mo baseline.
        """
        extraction = state.get("scribe_extraction") or {}
        if not extraction.get("renal_context_validated"):
            return ""
        rc = extraction.get("renal_context") or {}
        if not rc:
            return ""

        sources = extraction.get("renal_context_sources") or []
        sources_str = (
            ", ".join(str(s) for s in sources)
            if sources else "(no source IDs)"
        )

        # Per-subfield fallback strings — each is explicit so the
        # downstream rule in intensivist.yaml can require the
        # rendered S problem to either quote the verbatim subfield
        # value OR the specific fallback. Substring-checking against
        # these sentinel strings is how the §4.3.7 post-render check
        # detects "AKI problem missing baseline reference."
        def _v(key: str, fallback: str) -> str:
            val = rc.get(key)
            return val if val else fallback

        baseline_cr = _v(
            "baseline_creatinine", "no baseline anchor in source"
        )
        baseline_date = _v(
            "baseline_creatinine_date", "not specified"
        )
        kdigo = _v(
            "kdigo_stage", "not assigned in source"
        )
        uop = _v(
            "urine_output_pattern", "not documented in source"
        )
        nephro = _v(
            "nephrology_status", "not documented in source"
        )
        rrt = _v(
            "rrt_indications_documented", "none documented in source"
        )
        source_quote = rc.get("baseline_source_quote") or "n/a"

        # Per-subfield drop audit — surfaced in the pin so the
        # intensivist sees WHICH subfields the validator rejected (vs.
        # which the LLM simply didn't emit). Mostly diagnostic; the
        # primary signal is the rendered subfield value itself.
        partial_drops = extraction.get("renal_context_partial_drops") or []
        partial_drops_str = (
            "\n\nPartial-drop audit (validator-rejected subfields):\n"
            + "\n".join(f"- {d}" for d in partial_drops)
            if partial_drops else ""
        )

        # Baseline-source disclaimer renders contextually: when a
        # baseline anchor was extracted, the disclaimer flags that the
        # chart-extracted anchor may not match the KDIGO-canonical
        # lowest-3mo baseline. When NO anchor was extracted, the
        # disclaimer says so explicitly rather than rendering the
        # KDIGO-canonical caveat against an empty slot (which would
        # read as noise to the receiver). This was an iter-1
        # implementation finding flagged at the scribe.yaml + intensivist
        # chunk review — see design doc §12 iteration-findings log.
        if rc.get("baseline_creatinine"):
            baseline_source_line = (
                "Baseline source: chart-extracted — may differ from "
                "KDIGO-canonical lowest-3mo baseline."
            )
        else:
            baseline_source_line = (
                "Baseline source: n/a — no baseline anchor extracted."
            )

        return (
            "## RENAL CONTEXT (extracted by scribe — validated; "
            f"per-subfield substring + patient-identity guarded; "
            f"sources: {sources_str})\n\n"
            f"```\n"
            f"Baseline Cr: {baseline_cr}\n"
            f"Anchor date: {baseline_date}\n"
            f"{baseline_source_line}\n"
            f'Source quote: "{source_quote}"\n'
            f"KDIGO stage (chart-documented): {kdigo}\n"
            f"Urine output pattern: {uop}\n"
            f"Nephrology status: {nephro}\n"
            f"RRT indications documented: {rrt}\n"
            f"```{partial_drops_str}\n\n"
            "USE THESE VALUES AS-IS when authoring any #AKI, #CKD, "
            "#AKI on CKD, #Acute kidney injury, or #Renal failure "
            "problem in Section S. The orchestrator will compute "
            "delta vs latest structured Cr and Path A / Path B KDIGO "
            "stages deterministically (per docs/renal_electrolyte_vte_"
            "extraction_design.md §4.3.5, including the v3.1 gating "
            "fix that prevents chronic-ESRD mis-staging). Your job "
            "is to surface the pin's verbatim content + the "
            "deterministic compute output together — see the "
            "RENAL CONTEXT block in your system prompt for the "
            "ordered 7-component render.\n"
        )

    @staticmethod
    def _extract_creatinine_stats(
        context: dict[str, Any],
    ) -> tuple[Any, Any]:
        """Extract (latest_cr, structured_48h_min) from the structured
        labs view.

        ``context["labs"]`` is the serialized labs list (sorted
        newest-first on ``lab_result_dttm`` by the serializer). Selects
        serum creatinine only — rows with ``lab_category == "creatinine"``
        (exact match; urine creatinine / clearance are excluded). The
        value is read from ``lab_value_numeric`` first, falling back to
        ``lab_value`` only when the numeric column is absent — the
        serializer drops the ``lab_value`` string whenever
        ``lab_value_numeric`` is present, so a ``lab_value``-only read
        misses every resulted creatinine. The 48h window math anchors on
        ``lab_result_dttm`` to match the serializer's sort key.
        Returns (None, None) when no creatinine values are available
        — the renal compute then renders the "Delta: not computable"
        fallback.
        """
        if not isinstance(context, dict):
            return None, None
        labs = context.get("labs") or []
        if not isinstance(labs, list):
            return None, None

        creatinine_rows: list[tuple[Any, float]] = []
        for row in labs:
            if not isinstance(row, dict):
                continue
            # Serum creatinine only: exact lab_category match, matching the
            # codebase convention (_resolve_renal_status_for_meds in context.py,
            # med_state.py, reframing.py). A substring match (or a lab_name
            # fallback) would capture urine creatinine / creatinine-clearance
            # components and contaminate the KDIGO delta.
            cat = str(row.get("lab_category") or "").strip().lower()
            if cat != "creatinine":
                continue
            # Numeric-first read: the serializer drops the ``lab_value`` string
            # whenever ``lab_value_numeric`` is set
            # (_drop_lab_value_when_numeric_present in context.py), so reading
            # ``lab_value`` alone misses every resulted creatinine. Mirror the
            # canonical fallback used by citation_index/_first, lab_ranges, etc.
            # Pending rows (lab_value == "pending", numeric null) and any
            # non-numeric value fall through the float() guard and are skipped.
            val_raw = row.get("lab_value_numeric")
            if val_raw is None:
                val_raw = row.get("lab_value")
            if val_raw is None:
                continue
            try:
                val = float(str(val_raw).strip())
            except (TypeError, ValueError):
                continue
            # Window math keys off lab_result_dttm — the same field the
            # serializer sorts labs by (context.py) — so the "newest-first /
            # row[0] is latest" assumption below holds. (Collect-time is the
            # physiologically correct anchor for a 48h KDIGO window; result-time
            # is used here for serialization consistency. Canonical end-state is
            # to sort+window on collect-time, a serializer change tracked
            # separately — see docs follow-up.)
            ts = (
                row.get("lab_result_dttm")
                or row.get("lab_collect_dttm")
                or row.get("collect_dttm")
            )
            creatinine_rows.append((ts, val))

        if not creatinine_rows:
            return None, None

        # Labs are conventionally newest-first per the retriever, so
        # creatinine_rows[0] is latest. structured_48h_min is the min
        # value within the past 48h relative to the latest timestamp.
        # The KDIGO compute treats None for structured_48h_min as
        # "Path B not evaluable" — appropriate when fewer than 2
        # readings exist or readings span > 48h.
        latest_ts, latest_val = creatinine_rows[0]
        from datetime import timedelta
        from icu_pause.data.context import _parse_cite_timestamp

        latest_dt = _parse_cite_timestamp(latest_ts) if latest_ts else None
        structured_48h_min: Any = None
        if latest_dt is not None:
            window_start = latest_dt - timedelta(hours=48)
            in_window: list[float] = [latest_val]
            for ts, val in creatinine_rows[1:]:
                dt = _parse_cite_timestamp(ts) if ts else None
                if dt is None:
                    continue
                if dt >= window_start:
                    in_window.append(val)
                else:
                    # Labs are newest-first; once we hit one out of
                    # window the rest are too.
                    break
            if len(in_window) >= 2:
                structured_48h_min = min(in_window)
        return latest_val, structured_48h_min

    @staticmethod
    def _compute_renal_block(
        state: dict[str, Any],
        context: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Run the orchestrator's Path A/B KDIGO compute and return
        (rendered_block, compute_metadata).

        rendered_block is the text to inject into the intensivist
        prompt (alongside the scribe RENAL CONTEXT pin). It contains
        §4.3.6 ordered render components 3-4 (current Cr + delta +
        KDIGO stages) — see docs/renal_electrolyte_vte_extraction_
        design.md §4.3.5 + §4.3.6.

        compute_metadata is the full dict returned by the orchestrator
        function — info_signals + warns are extracted from it and
        accumulated on ``state['safety_drift_emissions']`` by the
        caller. Empty dict when the scribe didn't extract a valid
        renal_context (then no compute runs and the intensivist
        prompt-rule fallback handles the no-pin case).
        """
        extraction = state.get("scribe_extraction") or {}
        if not extraction.get("renal_context_validated"):
            return "", {}
        rc = extraction.get("renal_context")
        if not rc:
            return "", {}

        from icu_pause.agents.orchestrator import (
            compute_renal_delta_and_kdigo_path_a_b,
        )

        latest_cr, structured_48h_min = (
            IntensivistAgent._extract_creatinine_stats(context)
        )
        chart_documented = rc.get("kdigo_stage")
        result = compute_renal_delta_and_kdigo_path_a_b(
            scribe_renal_context=rc,
            latest_cr=latest_cr,
            structured_48h_min=structured_48h_min,
            chart_documented_kdigo_stage=chart_documented,
        )
        return result.get("rendered_block", ""), result

    def _format_agent_outputs(
        self,
        snippets: list[AgentSnippet],
        pending_tests_block: str = "",
        transfer_exam_block: str = "",
        scribe_pmh_block: str = "",
        scribe_allergies_block: str = "",
        scribe_subspecialty_consults_block: str = "",
        scribe_renal_context_block: str = "",
        renal_compute_block: str = "",
        med_state_block: str = "",
    ) -> str:
        """Format domain agent outputs as section-ownership directives.

        Instead of generic JSON, formats as explicit pass-through instructions
        per section, so the Intensivist knows exactly what to use for each section.

        ``scribe_pmh_block`` (when non-empty) is prepended above the
        section directives. It carries the validated PMH text extracted by
        the scribe agent, pinned at the top of the input so the intensivist
        doesn't have to locate it inside long SOAP-structured notes (see
        docs/pmh_structured_extraction_design.md). ``scribe_allergies_block``
        is accepted for signature compatibility but is currently disabled
        upstream (rendering dropped; extraction still runs).
        """
        # Collect all agent contributions by section
        section_contributions: dict[str, list[tuple[str, str, float]]] = {}
        all_warnings = []
        for snippet in snippets:
            for sec in snippet.sections:
                if sec.content and sec.content != "Not enough information from structured data.":
                    key = sec.section
                    if key not in section_contributions:
                        section_contributions[key] = []
                    section_contributions[key].append(
                        (snippet.agent_name, sec.content, sec.confidence)
                    )
            all_warnings.extend(snippet.warnings)

        # Format as section-ownership directives
        parts = []

        # Phase-0 scribe pins (PMH + active consults, when each is
        # extracted + validated). Placed BEFORE the section directives so
        # the labeled headers sit at the top of the user message — avoids
        # the long-note attention failure described in
        # docs/pmh_hallucination_investigation.md. Pins are independent;
        # any subset may fire on a given case. (Allergies pin is currently
        # disabled — extraction runs but rendering is off.)
        if scribe_pmh_block:
            parts.append(scribe_pmh_block)
        if scribe_allergies_block:
            parts.append(scribe_allergies_block)
        if scribe_subspecialty_consults_block:
            parts.append(scribe_subspecialty_consults_block)
        if scribe_renal_context_block:
            parts.append(scribe_renal_context_block)
        if renal_compute_block:
            parts.append(renal_compute_block)
        # Med-state pin (sedation/analgesia/paralytic by default; signature
        # is generic so future tense rules can reuse). Section I rule
        # references this block by name.
        if med_state_block:
            parts.append(med_state_block)

        parts.append("## SECTION OWNERSHIP — AGENT OUTPUTS AND YOUR TASKS\n")

        # Sections I and P: explicit instruction that Intensivist must write these
        parts.append("### Section I — YOU MUST WRITE THIS YOURSELF")
        parts.append("No agent contributes to section I. Write the ICU course narrative")
        parts.append("directly from the PATIENT DATA AND CLINICAL NOTES above.")
        parts.append("Apply the Section I rules from your system prompt:")
        if scribe_pmh_block:
            parts.append(
                "- PMH RULE: Use the scribe-extracted PMH at the top of this "
                "message as-is. The scribe has quoted it verbatim from the "
                "routed notes and the orchestrator has substring + "
                "patient-identity validated every clause. Do NOT paraphrase "
                "it, expand abbreviations, or re-derive PMH from raw notes."
            )
        else:
            parts.append(
                "- PMH RULE (scribe pin EMPTY — intensivist fallback): The "
                "scribe produced no validated PMH. Do NOT default to 'chart "
                "review required'. COMPOSE the PMH one-liner from the patient's "
                "history already in the NOTES above:\n"
                "  1. SOURCE: the H&P (hp_note) history is AUTHORITATIVE; use "
                "progress-note openers to supplement, or when the H&P yields "
                "nothing.\n"
                "  2. RECOGNIZE the bare demographic opener even with NO "
                "'PMH'/'PMHx'/'Hx notable for' label — e.g. '[age]yo [sex] with "
                "[conditions] who presents / presenting with / admitted for / "
                "p/w / now with ...'. The conditions between the opener's "
                "'with' and that presentation pivot ARE the PMH.\n"
                "  3. Do NOT pull the ACUTE presentation/complaint into PMH: in "
                "'75yo M with septic shock who presents…', septic shock is the "
                "complaint, not history. ACUTE-ON-CHRONIC: a chronic condition "
                "in the opener IS PMH even when it is also the acute problem "
                "(e.g. CKD in 'acute-on-chronic kidney injury').\n"
                "  4. NEGATION: 'no significant PMH' / 'previously healthy' / "
                "'unremarkable' → there is NO PMH. Write 'PMH not extracted "
                "from available notes — chart review required' and emit NO "
                "pmh_fallback.\n"
                "  5. Do NOT invent beyond what the notes state; do NOT infer "
                "PMH from age/sex priors, medication regimens, or ICD codes.\n"
                "  6. PMH ONLY — never free-narrate allergies, home meds, or "
                "code status this way; those remain scribe-owned.\n"
                "  7. When you compose a PMH this way, write it into the "
                "Section I PMH slot AND emit the structured 'pmh_fallback' "
                "field: {text: <the PMH conditions>, note_ids: [<ids drawn "
                "from>], note_types: [<'hp_note' and/or 'progress_note'>]}. "
                "Leave pmh_fallback null ONLY when the notes genuinely have no "
                "PMH (negation / previously healthy)."
            )
        if med_state_block:
            parts.append(
                "- SEDATION/ANALGESIA/PARALYTIC STATE TAGGING RULE: When you "
                "name any sedation, analgesia, or paralytic drug in Section I "
                "prose, you MUST render its temporal state from the ACTIVE / "
                "RECENTLY STOPPED SEDATION-ANALGESIA-PARALYTIC AGENTS pin "
                "block above. Tenseless 'sedation with {drug}' contradicts "
                "current-state clauses and is the failure mode this rule "
                "exists to prevent. See the SEDATION/ANALGESIA/PARALYTIC "
                "STATE TAGGING block in your system prompt for render "
                "templates."
            )
        parts.append("- ADMISSION REASON RULE: Use the ORIGINAL admission/transfer reason,")
        parts.append("  not the current acute issue (these differ for long-stay patients).")
        parts.append("- TRAJECTORY RULE: Preserve hospital admit → MICU escalation → course")
        parts.append("  sequence when notes describe it.")
        parts.append("- MULTIPLE MICU ADMISSIONS RULE: When notes mention 'previously")
        parts.append("  admitted to MICU' or 're-admitted to MICU', the patient had MORE")
        parts.append("  THAN ONE MICU stay. Quote EACH escalation reason in chronological")
        parts.append("  order. Do NOT collapse into a single transfer-to-MICU statement.")
        parts.append("Format (single MICU admit, prior ward): '[Age]yo [sex] with PMH")
        parts.append("[verbatim], initially admitted [date] for [orig reason], transferred")
        parts.append("to MICU [date] for [ICU reason]. ICU course c/b [...]. Currently [...].'")
        parts.append("Format (direct MICU admit): '[Age]yo [sex] with PMH [verbatim],")
        parts.append("admitted to MICU [date] for [reason]. ICU course c/b [...].")
        parts.append("Currently [...].'")
        parts.append("Format (multiple MICU admits): '[Age]yo [sex] with PMH [verbatim],")
        parts.append("initially admitted [date] for [orig], admitted to MICU [date1] for")
        parts.append("[reason1], transitioned to ward, re-admitted to MICU [date2] for")
        parts.append("[reason2]. ICU course c/b [...]. Currently [...].'\n")

        parts.append("### Section P — YOU MUST WRITE THIS YOURSELF")
        parts.append("No agent contributes to section P. Search the source data and notes")
        parts.append("for PENDING items (labs, imaging, procedures, consults not yet completed).")
        parts.append("If nothing is pending, write 'None'.")
        parts.append("Do NOT put patient history or current treatments here.\n")
        if pending_tests_block:
            parts.append(pending_tests_block)

        # Other sections: pass-through from agents
        parts.append("### AGENT OUTPUTS — USE AS PRIMARY SOURCE")
        parts.append("For sections below, USE the agent content verbatim. Only modify for factual conflicts.\n")

        for section_key, contributions in sorted(section_contributions.items()):
            parts.append(f"### Section {section_key}")
            # Phase 3: inject the deterministic transfer-exam block at the
            # top of Section E. The block is the authoritative source for
            # all vitals / neuro / respiratory-support values; agent
            # outputs below the block are for narrative observations only.
            if section_key == "E" and transfer_exam_block:
                parts.append(transfer_exam_block)
                parts.append("")
            for agent_name, content, confidence in contributions:
                parts.append(
                    f"**{agent_name}** (confidence {confidence}):\n"
                    f"```\n{content}\n```"
                )
            parts.append("")

        # Defensive: if no agent contributed to Section E (rare edge case
        # where nurse/respiratory/therapist all produced empty E content),
        # still surface the deterministic block — it's the most important
        # part of Section E and must not be suppressed by upstream silence.
        if (
            transfer_exam_block
            and "E" not in section_contributions
        ):
            parts.append("### Section E")
            parts.append(transfer_exam_block)
            parts.append("")

        if all_warnings:
            parts.append(
                f"### Agent Warnings\n"
                + chr(10).join(f"- {w.message}" for w in all_warnings)
            )

        return "\n".join(parts)

    def _format_qa_issues(self, qa_issues: list[str]) -> str:
        if not qa_issues:
            return "No QA issues identified."
        return "QA Issues to address:\n" + "\n".join(f"- {issue}" for issue in qa_issues)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the Intensivist agent.

        Reads domain agent snippets, QA issues, structured data, and notes,
        then produces harmonized content for all ICU-PAUSE sections.
        """
        snippets: list[AgentSnippet] = state.get("agent_snippets", [])
        revised: list[AgentSnippet] = state.get("revised_snippets", [])
        qa_issues: list[str] = state.get("qa_issues", [])

        # Use revised snippets where available
        revised_agents = {s.agent_name for s in revised}
        effective_snippets = [
            s for s in snippets if s.agent_name not in revised_agents
        ] + list(revised)

        # Build compact clinical summary for the Intensivist.
        # Agent outputs already contain the clinical detail — the Intensivist
        # only needs authoritative structured data for adjudication and gap-filling.
        # This reduces input from ~100k tokens (raw JSON) to ~5-10k tokens.
        agent_contexts = state.get("agent_context_text", {})
        context = agent_contexts.get(
            self.agent_name, state.get("patient_context_text", {})
        )

        summary_parts = []
        for key, value in context.items():
            # Skip diagnoses — ICD billing codes are not clinically useful;
            # diagnoses should come from clinical notes and agent synthesis
            if key == "diagnoses":
                continue
            if not value:
                continue

            if isinstance(value, str):
                # GPT-5.4 has plenty of context window; pass full notes
                # so the Intensivist sees the same text the domain agents saw.
                # Reviewer/agent parity rule.
                summary_parts.append(f"## {key.upper()}\n{value}")
            elif isinstance(value, dict):
                compact = json.dumps(value, indent=1, default=str)
                # Truncate oversized sections (e.g., large med lists)
                if len(compact) > 3000:
                    compact = compact[:3000] + "\n... (truncated for context window)"
                summary_parts.append(f"## {key.upper()}\n{compact}")
            elif isinstance(value, list):
                # Keep most recent entries for time-series data
                if len(value) > 30:
                    value = value[-30:]
                compact = json.dumps(value, indent=1, default=str)
                if len(compact) > 3000:
                    compact = compact[:3000] + "\n... (truncated for context window)"
                summary_parts.append(f"## {key.upper()}\n{compact}")

        data_sections = "\n\n".join(summary_parts)

        # Build the user message. Pre-extract pending tests from the
        # structured-data view so Section P always sees a concrete list
        # rather than relying on the LLM to spot masked rows in raw JSON.
        pending_tests_block = self._format_pending_tests_block(context)
        transfer_exam_block_text = self._format_transfer_exam_block(context)
        scribe_pmh_block = self._format_scribe_pmh_block(state)
        # Allergies rendering disabled: extraction + validation still run in
        # the scribe agent (state['scribe_extraction']['allergies'*]). Only
        # the brief-template wiring is dropped — EHR banner + CPOE already
        # carry allergy-safety. To re-enable, restore:
        #   scribe_allergies_block = self._format_scribe_allergies_block(state)
        scribe_allergies_block = ""
        scribe_subspecialty_consults_block = self._format_scribe_subspecialty_consults_block(state)
        scribe_renal_context_block = self._format_scribe_renal_context_block(state)
        # Renal Path A/B KDIGO compute — deterministic, runs alongside
        # the scribe RENAL CONTEXT pin. The compute reads the scribe's
        # chart-extracted baseline + structured Cr (latest + 48h-min)
        # and produces the §4.3.6 components 3-4 (current Cr + delta +
        # KDIGO stages). Output feeds into the intensivist input
        # alongside the scribe pin. Side effect: info_signals + warns
        # accumulate on state["safety_drift_emissions"] for the drift-
        # metric module to ingest (next chunk).
        renal_compute_block, renal_compute_meta = (
            self._compute_renal_block(state, context)
        )
        if renal_compute_meta:
            emissions = state.setdefault("safety_drift_emissions", {})
            for key, val in renal_compute_meta.get("info_signals", {}).items():
                if val:
                    emissions.setdefault("info_signals", {})[key] = True
            for key, val in renal_compute_meta.get("warns", {}).items():
                if val:
                    emissions.setdefault("warns", {})[key] = True
        # Sedation/analgesia/paralytic state pin for Section I tense-tagging.
        # Generic signature so future tense rules (e.g., vasopressor wean)
        # can reuse this pin pattern; called this round with the
        # sedation/analgesia/paralytic class set.
        med_state_block = self._format_med_state_block(
            context,
            drug_class_filter={
                "true_sedative",
                "arousal_preserving",
                "analgesic",
                "dissociative",
                "paralytic",
            },
        )
        agent_outputs_text = self._format_agent_outputs(
            effective_snippets,
            pending_tests_block=pending_tests_block,
            transfer_exam_block=transfer_exam_block_text,
            scribe_pmh_block=scribe_pmh_block,
            scribe_allergies_block=scribe_allergies_block,
            scribe_subspecialty_consults_block=scribe_subspecialty_consults_block,
            scribe_renal_context_block=scribe_renal_context_block,
            renal_compute_block=renal_compute_block,
            med_state_block=med_state_block,
        )
        qa_text = self._format_qa_issues(qa_issues)

        # Include risk score if available
        risk_score = state.get("risk_score")
        if risk_score and risk_score.get("available"):
            risk_text = (
                f"## RISK PREDICTION (72h)\n\n"
                f"Model: {risk_score.get('model', 'unknown')}\n"
                f"Readmission risk: {risk_score.get('risk_72h_readmission', 'N/A')}\n"
                f"Mortality risk: {risk_score.get('risk_72h_mortality', 'N/A')}\n\n"
                f"Factor this risk assessment into your clinical narrative, "
                f"especially in the U_uncertainty and S sections.\n\n"
            )
        else:
            risk_text = ""

        section_descriptions = "\n".join(
            f"  - {s.value}: {s.name}" for s in ICUPauseSection
        )

        # Include Resident pre-brief if available
        resident_brief = state.get("resident_pre_brief")
        resident_text = ""
        if resident_brief and isinstance(resident_brief, dict):
            brief_parts = ["## RESIDENT PRE-SYNTHESIS BRIEF\n"]
            brief_parts.append(
                "The senior resident has reviewed all domain agent outputs and "
                "prepared this analysis. Use it as your starting point.\n"
            )
            conflicts = resident_brief.get("cross_domain_conflicts", [])
            if conflicts:
                brief_parts.append(f"### Cross-Domain Conflicts ({len(conflicts)})")
                for c in conflicts:
                    sev = c.get("severity", "clinical")
                    brief_parts.append(
                        f"- [{sev.upper()}] {c.get('domain_a', '?')} vs "
                        f"{c.get('domain_b', '?')}: {c.get('conflict_description', '')}"
                    )
                brief_parts.append("")

            gaps = resident_brief.get("critical_gaps", [])
            if gaps:
                brief_parts.append(f"### Critical Gaps ({len(gaps)})")
                for g in gaps:
                    brief_parts.append(
                        f"- Section {g.get('icu_pause_field', '?')} "
                        f"({g.get('gap_type', '?')})"
                        + (f": {g.get('note', '')}" if g.get("note") else "")
                    )
                brief_parts.append("")

            redundancies = resident_brief.get("redundancies", [])
            if redundancies:
                brief_parts.append(f"### Redundancy Clusters ({len(redundancies)})")
                for r in redundancies:
                    members = ", ".join(r.get("cluster", []))
                    brief_parts.append(f"- CONSOLIDATE: [{members}]")
                    brief_parts.append(f"  → Proposed header: {r.get('proposed_header', '')}")
                    brief_parts.append(f"  → Rationale: {r.get('rationale', '')}")
                    sublines = r.get("preserve_as_sublines", [])
                    if sublines:
                        brief_parts.append(f"  → Preserve as sub-bullets: {sublines}")
                brief_parts.append("")

            narrative = resident_brief.get("pre_brief_narrative", {})
            if narrative:
                brief_parts.append("### Pre-Brief Narrative")
                brief_parts.append(f"**Theme**: {narrative.get('dominant_clinical_theme', '')}")
                deps = narrative.get("inter_domain_dependencies", [])
                if deps:
                    brief_parts.append("**Dependencies**:")
                    for d in deps:
                        brief_parts.append(f"  - {d}")
                todos = narrative.get("priority_todo_items", [])
                if todos:
                    brief_parts.append("**Priority To-Dos**:")
                    for t in todos:
                        brief_parts.append(f"  - {t}")
                brief_parts.append("")

            confidence = resident_brief.get("resident_confidence", "moderate")
            brief_parts.append(f"Resident confidence: {confidence}\n")
            resident_text = "\n".join(brief_parts) + "\n"

        user_message = (
            f"## PATIENT DATA AND CLINICAL NOTES\n\n{data_sections}\n\n"
            f"## QA VALIDATION RESULTS\n\n{qa_text}\n\n"
            f"{resident_text}"
            f"{risk_text}"
            f"{agent_outputs_text}\n\n"
            f"## YOUR TASK\n\n"
            f"Complete BOTH phases as described in your instructions.\n"
            f"CRITICAL: For Phase 2, START with the agent outputs above as your "
            f"primary source for each section. COPY agent content verbatim where "
            f"it exists. Only rewrite if there is a factual conflict.\n\n"
            f"**Phase 1**: Produce the reasoning_log (conflicts, gaps, safety_checks, "
            f"competing_risks_check, confidence_notes) by explicitly reasoning through "
            f"the domain agent outputs and source data.\n\n"
            f"safety_checks holds CHECK entries (one per safety domain in the SAFETY "
            f"CHECKS rule).\n\n"
            f"competing_risks_check holds one TYPED entry per active in-scope therapy as "
            f"defined by the COMPETING-RISKS SCOPE CHECK rule in the system prompt — emit "
            f"entries for therapies that resolve to conflict_status='no' as well, so the "
            f"reasoning_log records the audit trail for every in-scope therapy. Indication "
            f"grounding is BINARY: either the literal hedge phrase \"indication not "
            f"documented in available notes\" with source_note_id and source_quote both "
            f"null, OR a non-hedge indication string with both source_note_id AND "
            f"source_quote populated. source_quote MUST be a verbatim contiguous phrase "
            f"from source_note_id's body — no ellipsis, no paraphrase, no contextual "
            f"reconstruction. The downstream validator substring-checks every non-hedge "
            f"citation against the routed note body and rewrites un-grounded indications "
            f"to the hedge phrase.\n\n"
            f"**Phase 2**: Using your reasoning log, generate the 8 ICU-PAUSE sections.\n\n"
            f"ICU-PAUSE sections to complete:\n{section_descriptions}\n\n"
            f"Respond with a JSON object matching this schema:\n"
            f'{{\n'
            f'  "agent_name": "intensivist",\n'
            f'  "reasoning_log": {{\n'
            f'    "conflicts": ["CONFLICT: [agent_a] says X vs [agent_b] says Y. Source data shows Z. RESOLUTION: ..."],\n'
            f'    "gaps": ["GAP: [data_key] shows X but no agent addressed this. Relevant to section(s): Y"],\n'
            f'    "safety_checks": [\n'
            f'      "CHECK: high_risk_meds — Y/N, details",\n'
            f'      "CHECK: code_status — Y/N, details",\n'
            f'      "CHECK: pending_cultures — Y/N, details",\n'
            f'      "CHECK: infusion_plans — Y/N, details"\n'
            f'    ],\n'
            f'    "competing_risks_check": [\n'
            f'      {{\n'
            f'        "drug": "<drug name verbatim from pharmacy>",\n'
            f'        "indication": "<verbatim phrase from source OR the literal string \\"indication not documented in available notes\\">",\n'
            f'        "source_note_id": "<note_id from your routed notes; null when indication is the hedge>",\n'
            f'        "source_quote": "<verbatim contiguous span from that note that grounds the indication; null when indication is the hedge>",\n'
            f'        "conflict_status": "<yes|no|unclear>",\n'
            f'        "conflict_condition": "<the conflicting active condition (yes) or brief unclear-reason; null when conflict_status is no>",\n'
            f'        "risk_of_continuing": "<one-line qualitative consequence; null when conflict_status is no>",\n'
            f'        "risk_of_holding_or_reducing": "<one-line qualitative consequence; null when conflict_status is no>"\n'
            f'      }}\n'
            f'    ],\n'
            f'    "confidence_notes": [\n'
            f'      "I: N agents contributed, data richness note",\n'
            f'      "C: note"\n'
            f'    ]\n'
            f'  }},\n'
            f'  "sections": [\n'
            f'    {{\n'
            f'      "section": "<section_key>",\n'
            f'      "content": "<harmonized clinical text>",\n'
            f'      "confidence": 0.0-1.0,\n'
            f'      "data_sources_used": ["<agent_name>", "<data_key>"]\n'
            f'    }}\n'
            f'  ],\n'
            f'  "warnings": ["<any critical safety concerns or unresolved conflicts>"]\n'
            f'}}'
        )

        # Try parsing as IntensivistOutputLLM (wire schema, with reasoning_log), fall back
        # to AgentSnippetLLM wrapped into an IntensivistOutput with an empty reasoning_log.
        # Downstream state holds IntensivistOutput so `reasoning_log` survives for
        # diagnostics and audit (CLUSTER_OMITTED entries, gap analysis, etc.).
        try:
            llm_output = self.llm.invoke(
                system=self.system_prompt,
                user=user_message,
                response_format=IntensivistOutputLLM,
            )
            intensivist_output = wrap_llm_intensivist(
                llm_output,
                category=WarningCategory.SAFETY_FLAG,
                severity=WarningSeverity.CLINICAL,
            )
            logger.info(
                f"Intensivist produced {len(intensivist_output.sections)} sections "
                f"with reasoning log ({len(intensivist_output.reasoning_log.conflicts)} conflicts, "
                f"{len(intensivist_output.reasoning_log.gaps)} gaps)"
            )
        except Exception as e:
            logger.warning(f"IntensivistOutputLLM parse failed, trying AgentSnippetLLM: {e}")
            try:
                fallback_llm = self.llm.invoke(
                    system=self.system_prompt,
                    user=user_message,
                    response_format=AgentSnippetLLM,
                )
                fallback = wrap_llm_snippet(
                    fallback_llm,
                    category=WarningCategory.SAFETY_FLAG,
                    severity=WarningSeverity.CLINICAL,
                    source_agent="intensivist",
                )
                intensivist_output = IntensivistOutput(
                    agent_name=fallback.agent_name,
                    sections=fallback.sections,
                    warnings=fallback.warnings,
                )
                logger.info(
                    f"Intensivist produced {len(intensivist_output.sections)} sections (no reasoning log)"
                )
            except Exception as e2:
                logger.error(f"Intensivist agent failed: {e2}")
                intensivist_output = IntensivistOutput(
                    agent_name=self.agent_name,
                    sections=[],
                    warnings=[
                        Warning(
                            category=WarningCategory.QA_PROCESS,
                            severity=WarningSeverity.INFO,
                            message=f"Intensivist execution failed: {str(e2)}",
                            source_agent="intensivist",
                        )
                    ],
                )

        snippet = intensivist_output
        reasoning_log = intensivist_output.reasoning_log

        usage = self.llm.last_usage
        metrics = {
            "agent": self.agent_name,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "latency_ms": round(usage.latency_ms, 1),
            "model": usage.model,
        }

        # Build trace events
        trace_events = []
        if reasoning_log:
            from datetime import datetime, timezone
            trace_events.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "reasoning",
                "node": "intensivist",
                "level": "info",
                "message": (
                    f"Clinical reasoning: {len(reasoning_log.conflicts)} conflicts, "
                    f"{len(reasoning_log.gaps)} gaps, "
                    f"{sum(1 for v in reasoning_log.safety_checks if isinstance(v, str) and '— N' in v)} safety concerns"
                ),
                "data": {
                    "conflicts": reasoning_log.conflicts,
                    "gaps": reasoning_log.gaps,
                    "safety_checks": reasoning_log.safety_checks,
                    # competing_risks_check is the model's RAW emission at
                    # this point — the orchestrator's post-pass validator
                    # has not yet run. The orchestrator emits a separate
                    # competing_risks_validated trace event with both raw
                    # and final entries (the validator may have mutated
                    # this list in place). Persisting the raw here is the
                    # belt-and-suspenders audit anchor: even if the
                    # orchestrator doesn't run (e.g., truncated pipeline),
                    # the model's structured reasoning survives in the
                    # trace. Field was omitted in v1.8.0 — extractor saw
                    # 0 entries despite post-pass firing with real values.
                    "competing_risks_check": [
                        e.model_dump() if hasattr(e, "model_dump") else dict(e)
                        for e in (reasoning_log.competing_risks_check or [])
                    ],
                    "confidence_notes": reasoning_log.confidence_notes,
                },
            })

        # Trace: log intensivist output with full section content
        from datetime import datetime, timezone
        trace_events.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "agent_output",
            "node": "intensivist",
            "level": "info",
            "message": f"Intensivist produced {len(snippet.sections)} sections: {', '.join(s.section for s in snippet.sections)}",
            "data": {
                "sections": {
                    s.section: {
                        "content": s.content,
                        "confidence": s.confidence,
                        "data_sources_used": s.data_sources_used,
                    }
                    for s in snippet.sections
                },
                "warnings": [w.model_dump() for w in snippet.warnings],
                "metrics": metrics,
            },
        })

        return {
            "intensivist_output": snippet,
            "pipeline_metrics": [metrics],
            "trace_events": trace_events,
        }
