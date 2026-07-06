"""QA/Consistency Agent: validates agent outputs for numeric fidelity and cross-agent consistency."""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from math import ceil
from pathlib import Path
from typing import Any

import yaml

from icu_pause.config import Settings
from icu_pause.llm.provider import BaseLLM, create_llm
from icu_pause.schemas.icu_pause import AgentSnippet, ICUPauseSection
from icu_pause.tools.med_classes import (
    QA_FIRE_SEVERITY,
    classify_drug,
)

logger = logging.getLogger(__name__)


# Top-level labels rendered inside the pharmacy U_unprescribing block,
# used to bound the Antibiotics line extraction. Kept module-level (not
# inside _extract_antibiotics_line) so any future field renamings stay
# visible at one site.  Anchored to MULTILINE start because pharmacy's
# rendered block uses a leading newline before each label.
_OTHER_U_FIELD_RE = re.compile(
    r"^\s*(Changes to home meds|Anticoagulation):",
    flags=re.MULTILINE | re.IGNORECASE,
)


# Empty-bracket variants pharmacy may legitimately render. Iteration-2
# of the design spec used r"n/?a\b|none\b" which matched only "n/a", "na",
# "none" — clinically reasonable phrasings like "No planned antibiotics
# this admission", "Off antibiotics x4 days" would have been classified
# as non-empty and falsely tripped the HIGH contract check when pharmacy
# did the right thing. See docs/admission_antibiotics_design.md §5.5.
_EMPTY_BRACKET_PATTERN = re.compile(
    r"^("
    r"n/?a\b"
    r"|none\b"
    r"|no\s+(antibiotic|antimicrobial|abx|planned)"
    r"|off\s+antibiotic"
    r")",
    re.IGNORECASE,
)


# Tokens that indicate the patient is awake / interactive / off sedation at
# transfer. Used by ``_check_sedation_tense_conflict`` to detect the
# contradiction between an ACTIVE fire-eligible sedation drug in
# meds.states and a current-state clause that says the patient is awake.
# RASS 0/+1 and GCS 14/15 are the standard awake-charting thresholds;
# extubation to HFNC/NC/RA implies sedation off by clinical convention.
_AWAKE_TOKEN_PATTERN = re.compile(
    r"\b("
    r"awake"
    r"|alert\s+and\s+(oriented|interactive)"
    r"|a&ox?[34]"
    r"|interactive"
    r"|following\s+commands?"
    r"|off\s+sedation"
    r"|rass\s*[:=]?\s*[0+]"
    r"|gcs\s*[:=]?\s*1[45]"
    r"|extubated\s+to\s+(hfnc|nc|ra|room\s+air|nasal\s+cannula)"
    r")\b",
    re.IGNORECASE,
)


class QAAgent:
    """Step 3: Cross-checks agent outputs for consistency and numeric fidelity."""

    def __init__(self, settings: Settings):
        self.llm: BaseLLM = create_llm(settings)
        self.numeric_tolerance = settings.numeric_tolerance
        self.ensemble_passes = settings.qa_ensemble_passes
        self.drug_interaction_enabled = settings.drug_interaction_enabled
        self.drug_interaction_allow_network = settings.drug_interaction_allow_network
        self.drug_interaction_timeout = settings.drug_interaction_timeout_seconds
        self.device_dwell_enabled = settings.device_dwell_enabled
        self.lab_range_check_enabled = settings.lab_range_check_enabled
        self._load_prompt(settings)
        # Per-run parse-failure tracking. Reset at the top of run() so
        # successive QA invocations on the same instance don't accumulate
        # across patients.
        self._parse_failures: list[dict[str, Any]] = []

    def _load_prompt(self, settings: Settings) -> None:
        path = Path(settings.prompts_dir) / "qa.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            self.system_prompt = data.get("system_prompt", "")
        else:
            self.system_prompt = (
                "You are a clinical QA reviewer. Identify contradictions "
                "or inconsistencies across agent outputs. Return a JSON list of issues."
            )

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute QA checks on agent snippets.

        Ordering is explicit and deliberate:
        1. Deterministic checks first (section coverage, numeric fidelity)
           — fast, zero-cost, and produce clean audit logs.
        2. LLM-based contradiction detection ONLY if deterministic checks
           pass — avoids wasting an LLM call when basic issues exist, and
           prevents the LLM from conflating deterministic failures with
           semantic contradictions.
        """
        snippets: list[AgentSnippet] = state.get("agent_snippets", [])
        patient_data: dict[str, Any] = state.get("patient_context_text", {})
        issues: list[str] = []
        scope_issues: list[str] = []
        self._parse_failures = []

        # --- Phase 1: Deterministic checks (fast, zero-cost) ---
        issues.extend(self._check_section_coverage(snippets))
        issues.extend(self._check_numeric_fidelity(snippets, patient_data))
        drug_interaction_issues = self._check_drug_interactions(snippets, patient_data)
        issues.extend(drug_interaction_issues)
        issues.extend(self._check_line_dwell_time(snippets, patient_data, state))
        issues.extend(self._check_lab_reference_ranges(snippets, patient_data))
        issues.extend(self._check_infection_grounding(snippets, patient_data))
        issues.extend(
            self._check_antibiotic_pin_contract(
                snippets, state.get("scribe_extraction"),
            )
        )
        issues.extend(
            self._check_sedation_tense_conflict(snippets, patient_data)
        )

        deterministic_failed = len(issues) > 0
        ensemble_unique_issues = 0
        ensemble_passes_attempted = 0
        ensemble_passes_succeeded = 0
        if deterministic_failed:
            logger.info(
                f"QA: {len(issues)} deterministic issues found — skipping LLM check"
            )
        else:
            # --- Phase 2: LLM-based check (only when deterministic passes) ---
            if len(snippets) > 1:
                (
                    consistency_issues,
                    ensemble_passes_attempted,
                    ensemble_passes_succeeded,
                ) = self._check_consistency_llm(snippets)
                ensemble_unique_issues = len(consistency_issues)
                # Separate scope violations from clinical issues
                for issue in consistency_issues:
                    if self._is_scope_issue(issue):
                        scope_issues.append(issue)
                    else:
                        issues.append(issue)

        qa_passed = len(issues) == 0
        logger.info(f"QA: {len(issues)} clinical issues, {len(scope_issues)} scope issues. Passed: {qa_passed}")

        # "Degraded" means every attempted pass exhausted its retry without
        # ever returning parsed JSON. A pass that fails initially but
        # succeeds on retry is NOT degraded — the consistency check ran. The
        # flag exists so the run is observably less-validated when all passes
        # die in the parser, not as a noisier "did anything fail" signal.
        parse_failure_count = len(self._parse_failures)
        qa_ensemble_degraded = (
            ensemble_passes_attempted > 0
            and ensemble_passes_succeeded == 0
        )

        usage = self.llm.last_usage
        metrics: dict[str, Any] = {
            "agent": "qa",
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "latency_ms": round(usage.latency_ms, 1),
            "model": usage.model,
            "deterministic_issues": len(issues) if deterministic_failed else 0,
            "llm_check_skipped": deterministic_failed,
            "qa_ensemble_passes": self.ensemble_passes,
            "qa_ensemble_unique_issues": ensemble_unique_issues,
            "qa_scope_issues": len(scope_issues),
            "qa_parse_failures": parse_failure_count,
            "qa_ensemble_degraded": qa_ensemble_degraded,
        }
        if self._parse_failures:
            # Truncated snippets are dev-only diagnostics — they MUST NOT
            # surface in clinician warnings (which is the whole point of
            # this change). They ride in pipeline metrics for the audit log.
            metrics["qa_parse_failure_details"] = list(self._parse_failures)

        return {
            "qa_issues": issues,
            "qa_scope_issues": scope_issues,
            "qa_passed": qa_passed,
            "qa_ensemble_degraded": qa_ensemble_degraded,
            "pipeline_metrics": [metrics],
        }

    @staticmethod
    def _is_scope_issue(issue: str) -> bool:
        """Classify whether a QA issue is a scope violation (system-internal)
        vs a clinical issue (physician-facing)."""
        lower = issue.lower()
        scope_keywords = [
            "scope", "out of scope", "outside", "domain",
            "not their area", "belongs to", "should be handled by",
            "not within", "outside expertise", "scope violation",
        ]
        return any(kw in lower for kw in scope_keywords)

    # Sections authored by the Intensivist, not domain agents.
    # QA should not flag these as "missing agent contribution" since
    # no domain agent is assigned to them.
    INTENSIVIST_OWNED_SECTIONS = {"I", "P", "U_uncertainty"}

    def _check_section_coverage(self, snippets: list[AgentSnippet]) -> list[str]:
        """Ensure ICU-PAUSE sections have at least one non-empty contribution.

        Sections in INTENSIVIST_OWNED_SECTIONS are excluded from this check
        because the Intensivist writes them directly (not domain agents).
        """
        covered = set()
        for snippet in snippets:
            for section in snippet.sections:
                if section.content and section.content != "Not enough information from structured data.":
                    covered.add(section.section)

        # Only check sections that domain agents are responsible for
        agent_sections = {s.value for s in ICUPauseSection} - self.INTENSIVIST_OWNED_SECTIONS
        missing = agent_sections - covered
        return [f"Section '{s}' has no substantive agent contribution" for s in sorted(missing)]

    @staticmethod
    def _extract_antibiotics_line(u_content: str) -> str:
        """Slice the ``Antibiotics:`` block out of pharmacy's
        U_unprescribing content.

        Returns everything from ``Antibiotics:`` until the next top-level
        U_unprescribing label (``Changes to home meds:`` /
        ``Anticoagulation:``) or end-of-content. Returns ``""`` when no
        ``Antibiotics:`` label is found.

        Explicit I/O pairs are unit-tested in
        ``tests/test_qa_antibiotic_pin_contract.py`` — see
        docs/admission_antibiotics_design.md §5.5.
        """
        m = re.search(r"Antibiotics:", u_content, flags=re.IGNORECASE)
        if not m:
            return ""
        rest = u_content[m.start():]
        # pos=1 so the regex doesn't re-match the Antibiotics: header itself
        # via the broader _OTHER_U_FIELD_RE alternation (Antibiotics is not
        # listed there, but starting search at offset 1 keeps the slice
        # logic robust to future label expansions).
        next_match = _OTHER_U_FIELD_RE.search(rest, pos=1)
        if next_match:
            return rest[:next_match.start()]
        return rest

    @staticmethod
    def _antibiotics_block_is_empty(block: str) -> bool:
        """True when pharmacy's Antibiotics block conveys no antibiotic
        content (no drug AND no History: sub-line referring to past
        courses).

        A History: sub-line is the bridge between pharmacy's structured-
        data view and the scribe's admission_antibiotics pin block — if
        pharmacy lists a History: line, the block is NOT empty for
        contract-check purposes even when the [x] bracket is blank.

        Explicit I/O pairs are unit-tested in
        ``tests/test_qa_antibiotic_pin_contract.py`` (9 cases including
        the four iteration-3 expanded variants).
        """
        if not block:
            return True
        body = re.sub(
            r"^Antibiotics:\s*", "", block, count=1, flags=re.IGNORECASE,
        )
        body_lower = body.lower().strip()
        if re.search(r"\bhistory:", body_lower):
            return False
        bracket_match = re.search(
            r"\[\s*[x ]\s*\]\s*(.+?)(?:\n|$)", body, flags=re.IGNORECASE,
        )
        if not bracket_match:
            return True
        bracket_content = bracket_match.group(1).strip().lower()
        if _EMPTY_BRACKET_PATTERN.match(bracket_content):
            return True
        return False

    def _check_antibiotic_pin_contract(
        self,
        snippets: list[AgentSnippet],
        scribe_extraction: dict[str, Any] | None,
    ) -> list[str]:
        """Scribe -> pharmacy contract verification.

        Fires ONLY when (a) scribe extracted >=1 validated admission
        course AND (b) pharmacy rendered a non-empty Antibiotics line
        AND (c) >=1 scribe-extracted drug name is absent from that line.

        Out of scope this PR:
        - Empty Antibiotics line + scribe has entries -> no flag
          (deferred until graded WARNING severity is a real QA concept).
        - Drug-name alias vs canonical paraphrasing -> no flag.
        - Free-text antimicrobial-keyword note scan -> permanently out
          of scope for runtime QA (alert-fatigue risk; belongs in eval
          suite).
        """
        if not scribe_extraction or not scribe_extraction.get(
            "admission_antibiotics_validated"
        ):
            return []
        courses = scribe_extraction.get("admission_antibiotics") or []
        if not courses:
            return []

        pharmacy_snippet = next(
            (s for s in snippets if s.agent_name == "pharmacy"), None,
        )
        if pharmacy_snippet is None:
            return []
        u_section = next(
            (
                s for s in pharmacy_snippet.sections
                if s.section == "U_unprescribing"
            ),
            None,
        )
        if u_section is None or not u_section.content:
            return []

        abx_block = self._extract_antibiotics_line(u_section.content)
        if self._antibiotics_block_is_empty(abx_block):
            return []

        from icu_pause.tools.drug_canonicalization import drug_appears_in_text

        issues: list[str] = []
        for course in courses:
            drug = (course.get("drug") or "").strip()
            if not drug:
                continue
            if drug_appears_in_text(drug, abx_block):
                continue
            issues.append(
                f"[ANTIBIOTIC_PIN_CONTRACT/HIGH] Scribe extracted "
                f"admission course '{drug}' (source: note_id="
                f"{course.get('source_note_id', '?')}) but it is absent "
                f"from the U_unprescribing Antibiotics line (which "
                f"pharmacy rendered non-empty). Contract violation -- "
                f"regen required."
            )
        return issues

    @staticmethod
    def _check_sedation_tense_conflict(
        snippets: list[AgentSnippet],
        patient_data: dict[str, Any],
    ) -> list[str]:
        """Preventive backstop for the Section I sedation-tense bug.

        Fires when (a) ``meds.states.records`` contains an ACTIVE drug
        whose class is fire-eligible (true_sedative / dissociative /
        paralytic per ``QA_FIRE_SEVERITY``), AND (b) any domain agent
        snippet content contains an awake/interactive/off-sedation token
        (Section E exam, Section S problems). The contradiction is what
        a clinician would catch by eye — pin block + Section I prompt
        rule are the primary fixes; this check feeds a structured
        warning into the intensivist's ``qa_text`` input so they apply
        the SEDATION/ANALGESIA/PARALYTIC STATE TAGGING rule
        deliberately.

        Class-aware severity (``QA_FIRE_SEVERITY``):
        - paralytic + awake → HIGH (paralyzed-and-charted-awake is a
          documentation or clinical emergency)
        - true_sedative (propofol/midazolam/lorazepam) + awake → MEDIUM
        - dissociative (ketamine) + awake → LOW (could be analgesia dose)
        - arousal_preserving (dexmedetomidine) + awake → no flag
          (clinically routine — dex preserves arousal by design)
        - analgesic (fentanyl/hydromorphone/morphine gtt) + awake → no
          flag (analgesia patients can and should be awake)

        Workflow note: this check runs at the qa_check node, which is
        BEFORE the intensivist writes Section I. So we can't match on
        bare-active rendering in Section I prose — the issue is emitted
        as a warning the intensivist must reflect when synthesizing
        Section I, not a post-hoc Section I scan. If a post-intensivist
        validation pass is added later, this method should be extended
        with the bare-active regex described in the original spec.
        """
        meds = (patient_data or {}).get("meds") or {}
        states = meds.get("states") or {}
        records = states.get("records") or []
        if not records:
            return []

        active_states = {"ACTIVE", "ACTIVE_SCHEDULED", "ACTIVE_PRN"}

        firing_drugs: list[tuple[str, str, str]] = []  # (drug, class, severity)
        for rec in records:
            if not isinstance(rec, dict):
                continue
            state = (rec.get("state") or "").upper()
            if state not in active_states:
                continue
            drug_name = rec.get("drug_name") or ""
            cls = classify_drug(drug_name)
            if cls is None:
                continue
            severity = QA_FIRE_SEVERITY.get(cls, "no_flag")
            if severity == "no_flag":
                continue
            firing_drugs.append((drug_name, cls, severity))

        if not firing_drugs:
            return []

        # Combine all snippet content into one searchable blob — the
        # awake signal can land in Section E (nurse exam, respiratory
        # device transition) or Section S (nurse problems, pharmacy
        # state lines). We don't care which section it came from for the
        # purpose of flagging the contradiction.
        all_content = " \n ".join(
            sec.content
            for snip in snippets
            for sec in snip.sections
            if sec.content
        )
        awake_hits = _AWAKE_TOKEN_PATTERN.findall(all_content)
        if not awake_hits:
            return []
        # findall returns tuples for grouped patterns — flatten to a
        # representative single-string list for the issue message.
        awake_examples: list[str] = []
        for h in awake_hits:
            if isinstance(h, tuple):
                awake_examples.append(next((x for x in h if x), ""))
            else:
                awake_examples.append(h)
        awake_examples = [e for e in awake_examples if e][:3]

        issues: list[str] = []
        for drug, cls, severity in firing_drugs:
            issues.append(
                f"[SECTION_I_SEDATION_TENSE_CONFLICT/{severity.upper()}] "
                f"Active {drug} ({cls}) at transfer per meds.states.records, "
                f"but domain agent content contains awake/interactive "
                f"tokens ({', '.join(awake_examples) or 'see Section E/S'}). "
                f"Section I prose MUST render {drug}'s temporal state "
                f"explicitly per the SEDATION/ANALGESIA/PARALYTIC STATE "
                f"TAGGING rule — do NOT write tenseless 'sedation with "
                f"{drug}' or bare 'on {drug}' alongside an awake-patient "
                f"clause. Use the ACTIVE / RECENTLY STOPPED SEDATION-"
                f"ANALGESIA-PARALYTIC AGENTS pin block as the "
                f"authoritative state source."
            )
        return issues

    @staticmethod
    def _extract_numbers_from_json(obj: Any) -> set[float]:
        """Recursively extract all numeric values from a nested JSON-like structure."""
        numbers: set[float] = set()
        if isinstance(obj, bool):
            return numbers
        if isinstance(obj, (int, float)):
            numbers.add(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                numbers.update(QAAgent._extract_numbers_from_json(v))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                numbers.update(QAAgent._extract_numbers_from_json(item))
        return numbers

    def _check_numeric_fidelity(
        self, snippets: list[AgentSnippet], patient_data: dict[str, Any]
    ) -> list[str]:
        """Check that numbers in agent outputs roughly match source data."""
        issues = []
        # Extract all numbers from source data
        source_numbers: set[float] = set()
        for value in patient_data.values():
            source_numbers.update(self._extract_numbers_from_json(value))

        # Extract numbers from agent outputs and check they appear in source
        for snippet in snippets:
            for section in snippet.sections:
                agent_numbers = [
                    float(m)
                    for m in re.findall(r"[-+]?\d*\.?\d+", section.content)
                    if m
                ]
                for num in agent_numbers:
                    # Check if this number is close to any source number
                    if num == 0:
                        continue
                    found = any(
                        abs(num - src) / max(abs(src), 1e-9) <= self.numeric_tolerance
                        for src in source_numbers
                        if src != 0
                    )
                    # Only flag large clinically meaningful numbers that don't match
                    if not found and abs(num) > 1:
                        # Could be a derived value — don't flag aggressively
                        pass

        return issues

    def _check_drug_interactions(
        self, snippets: list[AgentSnippet], patient_data: dict[str, Any]
    ) -> list[str]:
        """Deterministic drug interaction check (hybrid static + openFDA).

        1. Extract active medications from CLIF data.
        2. Check pairs against the clinician-reviewed static ICU table; fall
           back to openFDA drug-label API (moderate-severity only) when
           network is allowed.
        3. For high-severity interactions (static only), check if the
           pharmacy agent's output acknowledges them. Flag unacknowledged
           ones as QA issues.
        """
        if not self.drug_interaction_enabled:
            return []

        meds_data = patient_data.get("meds", {})
        if not meds_data:
            return []

        from icu_pause.tools.drug_interactions import check_interactions

        reference_dttm = (patient_data.get("demographics") or {}).get(
            "reference_dttm"
        )

        result = check_interactions(
            meds_data,
            allow_network=self.drug_interaction_allow_network,
            timeout=self.drug_interaction_timeout,
            reference_dttm=reference_dttm,
        )

        if not result.api_available:
            logger.warning(
                "Drug interaction API unavailable: %s", result.error_message
            )
            return []

        if not result.interactions:
            return []

        # Collect pharmacy agent output text for cross-referencing
        pharmacy_text = ""
        for snippet in snippets:
            if snippet.agent_name == "pharmacy":
                pharmacy_text = " ".join(
                    s.content.lower() for s in snippet.sections if s.content
                )
                break

        issues: list[str] = []
        for ix in result.interactions:
            # Only severity='high' gates QA. 'high' is produced exclusively by
            # the clinician-reviewed static table, so every flag here traces
            # to an audited entry.
            if ix.severity == "high":
                drug_a_lower = ix.drug_a.lower()
                drug_b_lower = ix.drug_b.lower()
                mentioned = (
                    drug_a_lower in pharmacy_text and drug_b_lower in pharmacy_text
                ) or "interaction" in pharmacy_text
                if not mentioned:
                    issues.append(
                        f"Drug interaction not addressed by pharmacy agent: "
                        f"{ix.drug_a} + {ix.drug_b} — {ix.description} "
                        f"(severity: {ix.severity}, source: {ix.source})"
                    )

        if result.unresolved_drugs:
            logger.info(
                "Drug interaction check: %d drugs could not be resolved to RxCUI: %s",
                len(result.unresolved_drugs),
                ", ".join(result.unresolved_drugs),
            )

        logger.info(
            "Drug interaction check: %d drugs checked, %d interactions found, "
            "%d high-severity unacknowledged",
            result.checked_drug_count,
            len(result.interactions),
            len(issues),
        )

        return issues

    def _check_line_dwell_time(
        self,
        snippets: list[AgentSnippet],
        patient_data: dict[str, Any],
        state: dict[str, Any],
    ) -> list[str]:
        """Deterministic device dwell-time check.

        Scans procedures for device-insertion events and flags those exceeding
        clinical threshold durations.  Cross-references with agent outputs —
        surfaces unmentioned warnings and always surfaces critical flags.
        """
        if not self.device_dwell_enabled:
            return []

        procedures = patient_data.get("procedures", [])
        if not procedures:
            return []

        reference_dttm = state.get("reference_dttm")
        if not reference_dttm:
            return []

        demographics = patient_data.get("demographics", {})
        icu_admission_dttm = demographics.get("icu_admission_dttm")

        from icu_pause.tools.device_dwell import check_device_dwell

        result = check_device_dwell(procedures, reference_dttm, icu_admission_dttm)

        if not result.flags:
            return []

        # Collect all agent text for cross-referencing
        all_agent_text = " ".join(
            s.content.lower()
            for snippet in snippets
            for s in snippet.sections
            if s.content
        )

        issues: list[str] = []
        for flag in result.flags:
            device_label = flag.device_type.replace("_", " ")
            device_mentioned = device_label in all_agent_text
            # Surface unmentioned warnings and always surface critical
            if not device_mentioned or flag.severity == "critical":
                severity_tag = flag.severity.upper()
                issues.append(
                    f"[DEVICE_DWELL/{severity_tag}] "
                    f"{device_label.title()} in place {flag.dwell_days} days "
                    f"(threshold: {flag.threshold_days}d). "
                    f"{flag.recommended_action}"
                )

        logger.info(
            "Device dwell check: %d devices checked, %d flags, %d surfaced",
            result.devices_checked,
            len(result.flags),
            len(issues),
        )
        return issues

    def _check_lab_reference_ranges(
        self, snippets: list[AgentSnippet], patient_data: dict[str, Any]
    ) -> list[str]:
        """Deterministic lab reference range validation.

        Compares most recent lab values against standard reference ranges.
        Flags critical values unconditionally and flags mischaracterizations
        where an agent describes an abnormal value as normal.
        """
        if not self.lab_range_check_enabled:
            return []

        labs = patient_data.get("labs", [])
        if not labs:
            return []

        # Collect all agent text for mismatch detection
        all_agent_text = " ".join(
            s.content
            for snippet in snippets
            for s in snippet.sections
            if s.content
        )

        from icu_pause.tools.lab_ranges import check_lab_ranges

        # PR 3: pass deterministic clinical context (chronic ESRD/cirrhosis/
        # COPD/AFib/trach flags) so check_lab_ranges can populate
        # patient-context-aware reframings on out-of-range values. None
        # at sites that don't run the retriever or pre-PR-1 cases.
        #
        # patient_data round-trips clinical_context through serialize_to_json,
        # which calls .to_dict() — so we receive a dict, not the dataclass.
        # Rehydrate before passing to reframing code that uses attribute
        # access (otherwise: AttributeError on .has_esrd_dialysis when any
        # chronic-condition flag is set).
        clinical_context = patient_data.get("clinical_context")
        if isinstance(clinical_context, dict):
            from icu_pause.safety.clinical_context import PatientClinicalContext
            clinical_context = PatientClinicalContext.from_dict(clinical_context)

        result = check_lab_ranges(labs, all_agent_text, clinical_context=clinical_context)

        if not result.flags:
            return []

        issues: list[str] = []
        for flag in result.flags:
            if flag.status.startswith("critical"):
                # Critical values: severity floor — never softened. The
                # reframe layer either provides an action-guiding context
                # phrase (e.g. "common in ESRD between HD sessions; confirm
                # session schedule and trend") or returns None, in which
                # case the bare critical alarm stands alone. "At baseline"
                # wording is intentionally absent here — see
                # safety/reframing.py header.
                msg = (
                    f"[LAB_RANGE/CRITICAL] {flag.lab_name} = {flag.value} {flag.unit} "
                    f"is critically {flag.status.replace('critical_', '')} "
                    f"(reference: {flag.reference_range})"
                )
                if flag.reframed_text:
                    msg = f"{msg} — context: {flag.reframed_text}"
                issues.append(msg)
            elif flag.reframed_text:
                # Non-critical with reframing. Tier ("CHRONIC" or "REVIEW")
                # controls the tag and whether we append a "confirm
                # consistent with prior" prompt. CHRONIC = system has
                # patient-priors or a self-verifying mechanism+action
                # phrase. REVIEW = chronic context applies but no patient-
                # priors anchor — clinician should verify against prior
                # values before dismissing.
                #
                # Reference range trails as "vs. general reference X" — it
                # still matters for renally-cleared drug dosing and the
                # audit trail, but doesn't lead the eye. ``status`` on the
                # flag is unchanged (reframing is text-only).
                tier = flag.reframed_tier or "CHRONIC"
                if tier == "REVIEW":
                    issues.append(
                        f"[LAB_RANGE/REVIEW] {flag.lab_name} {flag.value} {flag.unit} "
                        f"— {flag.reframed_text}; confirm consistent with prior "
                        f"(vs. general reference {flag.reference_range})"
                    )
                else:
                    issues.append(
                        f"[LAB_RANGE/CHRONIC] {flag.lab_name} {flag.value} {flag.unit} "
                        f"— {flag.reframed_text} "
                        f"(vs. general reference {flag.reference_range})"
                    )
            elif flag.mismatch:
                issues.append(
                    f"[LAB_RANGE/MISMATCH] {flag.lab_name} = {flag.value} {flag.unit} "
                    f"is {flag.status} (reference: {flag.reference_range}) "
                    f"but agent described as: {flag.agent_characterization}"
                )

        logger.info(
            "Lab range check: %d labs checked, %d flags, %d surfaced",
            result.labs_checked,
            len(result.flags),
            len(issues),
        )
        return issues

    # -- Antimicrobial classes for infection grounding check --
    _ANTIMICROBIAL_KEYWORDS: set[str] = {
        "vancomycin", "vanco", "meropenem", "piperacillin", "tazobactam",
        "pip-tazo", "zosyn", "cefepime", "ceftriaxone", "cefazolin",
        "ciprofloxacin", "cipro", "levofloxacin", "levo", "metronidazole",
        "flagyl", "azithromycin", "doxycycline", "linezolid", "daptomycin",
        "ampicillin", "unasyn", "sulfamethoxazole", "trimethoprim",
        "tobramycin", "gentamicin", "amikacin", "fluconazole",
        "micafungin", "oseltamivir", "tamiflu", "acyclovir",
        "caspofungin", "colistin", "polymyxin",
    }

    _INFECTION_KEYWORDS: list[str] = [
        "infection", "pneumonia", "sepsis", "bacteremia", "cellulitis",
        "uti", "urinary tract", "meningitis", "endocarditis", "osteomyelitis",
        "abscess", "empyema", "peritonitis", "influenza", "covid",
        "coronavirus", "hap", "vap", "clostridium", "c. diff", "cdiff",
        "mrsa", "vre", "pseudomonas", "staph", "strep", "e. coli",
        "klebsiella", "acinetobacter", "stenotrophomonas", "candida",
        "fungal", "viral", "bacterial",
    ]

    def _check_infection_grounding(
        self, snippets: list[AgentSnippet], patient_data: dict[str, Any]
    ) -> list[str]:
        """Deterministic infection grounding cross-check.

        If an antimicrobial-class drug is active in structured med data, at
        least one agent (expected: pharmacy) should mention an associated
        infection.  Flags ungrounded antimicrobials.
        """
        meds = patient_data.get("meds", {})
        continuous = meds.get("continuous", []) if isinstance(meds, dict) else []
        intermittent = meds.get("intermittent", []) if isinstance(meds, dict) else []

        # Find active antimicrobials
        active_abx: list[str] = []
        for med in continuous + intermittent:
            if not isinstance(med, dict):
                continue
            name = str(
                med.get("med_category", med.get("medication_name", ""))
            ).strip().lower()
            if any(kw in name for kw in self._ANTIMICROBIAL_KEYWORDS):
                active_abx.append(name)

        if not active_abx:
            return []

        # Collect all agent text
        all_agent_text = " ".join(
            s.content.lower()
            for snippet in snippets
            for s in snippet.sections
            if s.content
        )

        # Check if any agent mentions an infection
        infection_mentioned = any(
            kw in all_agent_text for kw in self._INFECTION_KEYWORDS
        )

        issues: list[str] = []
        if not infection_mentioned:
            abx_list = ", ".join(sorted(set(active_abx)))
            issues.append(
                f"[INFECTION_GROUNDING/WARNING] Active antimicrobials ({abx_list}) "
                f"but no agent mentions an associated infection. Verify infectious "
                f"indication is documented."
            )

        return issues

    def _check_consistency_llm(
        self, snippets: list[AgentSnippet]
    ) -> tuple[list[str], int, int]:
        """Use LLM to check for contradictions across agent outputs.

        When ensemble_passes > 1, runs the check N times with shuffled agent
        orderings (positional debiasing) and surfaces issues that appear in
        a majority of passes (>= ceil(N/2)).

        Returns ``(issues, passes_attempted, passes_succeeded)``. A pass
        "succeeds" if at least one of (initial call, strict-retry) returned
        parsed JSON. Caller uses ``passes_succeeded == 0`` to detect a
        fully-degraded ensemble.
        """
        n_passes = max(1, self.ensemble_passes)

        if n_passes == 1:
            issues, succeeded = self._single_consistency_pass(snippets)
            return issues, 1, (1 if succeeded else 0)

        # --- Ensemble mode ---
        # Collect all non-empty (snippet, section) pairs for shuffling
        items = [
            (s, sec)
            for s in snippets
            for sec in s.sections
            if sec.content and sec.content != "Not enough information from structured data."
        ]
        if not items:
            return [], 0, 0

        all_pass_issues: list[list[str]] = []
        passes_succeeded = 0
        for i in range(n_passes):
            shuffled = random.sample(items, len(items))
            all_content = "\n\n".join(
                f"[{s.agent_name}] {sec.section}: {sec.content}"
                for s, sec in shuffled
            )
            pass_issues, succeeded = self._run_llm_check(
                all_content, attempt_label=f"pass{i + 1}"
            )
            all_pass_issues.append(pass_issues)
            if succeeded:
                passes_succeeded += 1
            logger.info(f"QA ensemble pass {i + 1}/{n_passes}: {len(pass_issues)} issues")

        # Majority vote: surface issues appearing in >= ceil(N/2) passes
        return self._majority_vote(all_pass_issues, n_passes), n_passes, passes_succeeded

    def _single_consistency_pass(
        self, snippets: list[AgentSnippet]
    ) -> tuple[list[str], bool]:
        """Original single-pass consistency check.

        Returns ``(issues, succeeded)`` — see ``_run_llm_check``.
        """
        all_content = "\n\n".join(
            f"[{s.agent_name}] {sec.section}: {sec.content}"
            for s in snippets
            for sec in s.sections
            if sec.content and sec.content != "Not enough information from structured data."
        )
        if not all_content.strip():
            return [], True  # nothing to check is not a parse failure
        return self._run_llm_check(all_content, attempt_label="single")

    _STRICT_JSON_SUFFIX = (
        "\n\nRespond with valid JSON only. No markdown code fences. "
        "No preamble. No trailing text. The response must be a JSON array."
    )

    def _run_llm_check(
        self, all_content: str, *, attempt_label: str = "single"
    ) -> tuple[list[str], bool]:
        """Run a single LLM consistency check and return parsed issues.

        Parse failures (malformed JSON in the LLM response) are infrastructure
        noise, not clinical findings. They are NEVER returned as strings to
        ``qa_issues`` — that previously leaked into clinician-visible warnings.
        Instead: catch JSONDecodeError explicitly, retry once with a strict-
        JSON suffix, log to ``self._parse_failures``, and return [] on double
        failure. Non-parse exceptions (LLM transport, etc.) are caught the
        same way so they don't leak either.

        Returns ``(issues, succeeded)``. ``succeeded`` is True iff at least
        one attempt parsed JSON successfully — the caller uses this to
        decide whether the whole ensemble degraded.
        """
        from icu_pause.llm.provider import _strip_code_fences

        base_user = (
            f"Review these agent outputs for contradictions:\n\n{all_content}"
        )

        def _attempt(user_msg: str) -> list[str]:
            response = self.llm.invoke(
                system=self.system_prompt,
                user=user_msg,
            )
            cleaned = _strip_code_fences(response)
            match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
            if not match:
                return []
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            return []

        for attempt_no, user_msg in enumerate(
            (base_user, base_user + self._STRICT_JSON_SUFFIX), start=1
        ):
            try:
                return _attempt(user_msg), True
            except json.JSONDecodeError as e:
                self._parse_failures.append({
                    "agent": "qa",
                    "attempt_label": attempt_label,
                    "attempt_no": attempt_no,
                    "error_type": "JSONDecodeError",
                    "error_snippet": (str(e) or "")[:200],
                })
                logger.warning(
                    f"QA LLM parse failed (attempt {attempt_no}, {attempt_label}): {e}"
                )
                continue
            except Exception as e:
                # Transport, auth, rate-limit, etc. Same defense: log to
                # metrics, do NOT leak the raw exception string into clinician
                # warnings. One attempt is enough — retrying a 429 or 500 here
                # would slow the pipeline without changing the outcome class.
                self._parse_failures.append({
                    "agent": "qa",
                    "attempt_label": attempt_label,
                    "attempt_no": attempt_no,
                    "error_type": type(e).__name__,
                    "error_snippet": (str(e) or "")[:200],
                })
                logger.warning(
                    f"QA LLM call failed (attempt {attempt_no}, {attempt_label}): {e}"
                )
                return [], False

        # Both attempts failed JSON parse — return empty so the caller's
        # majority vote ignores this pass cleanly.
        return [], False

    @staticmethod
    def _normalize_issue(text: str) -> str:
        """Normalize an issue string for deduplication."""
        return re.sub(r'\s+', ' ', text.lower().strip())

    @staticmethod
    def _majority_vote(all_pass_issues: list[list[str]], n_passes: int) -> list[str]:
        """Surface issues that appear in >= ceil(N/2) passes.

        Uses normalized string matching for deduplication. Returns the
        original (non-normalized) text of the first occurrence.
        """
        threshold = ceil(n_passes / 2)

        # Map normalized → (original text, count across passes)
        norm_to_original: dict[str, str] = {}
        norm_counts: Counter[str] = Counter()

        for pass_issues in all_pass_issues:
            # Deduplicate within a single pass
            seen_this_pass: set[str] = set()
            for issue in pass_issues:
                norm = QAAgent._normalize_issue(issue)
                if norm not in seen_this_pass:
                    seen_this_pass.add(norm)
                    norm_counts[norm] += 1
                    if norm not in norm_to_original:
                        norm_to_original[norm] = issue

        results = []
        for norm, count in norm_counts.items():
            if count >= threshold:
                label = "unanimous" if count == n_passes else f"majority ({count}/{n_passes})"
                logger.info(f"QA ensemble {label}: {norm_to_original[norm]}")
                results.append(norm_to_original[norm])
            else:
                logger.debug(f"QA ensemble minority ({count}/{n_passes}), dropped: {norm_to_original[norm]}")

        return results
