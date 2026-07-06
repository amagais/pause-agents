"""Pydantic models for ICU-PAUSE output and agent communication."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ICUPauseSection(str, Enum):
    """The 8 sections of the ICU-PAUSE mnemonic.

    Note: 'U' appears twice in ICU-PAUSE with different meanings,
    so we disambiguate with U_unprescribing and U_uncertainty.
    """

    I = "I"  # ICU Admission Reason & Brief ICU Course
    C = "C"  # Code Status / DPOA / Goals of Care / ACP Note
    U_UNPRESCRIBING = "U_unprescribing"  # Unprescribing & High-Risk Meds
    P = "P"  # Pending Tests at Time of Transfer
    A = "A"  # Active Consultants (including Rehab)
    U_UNCERTAINTY = "U_uncertainty"  # Uncertainty Measure / Diagnostic Pause
    S = "S"  # Summary of Major Problems & To-Do's
    E = "E"  # Exam at Transfer, Lines/Drains/Airways & Data Review


class SectionContribution(BaseModel):
    """A single agent's contribution to one ICU-PAUSE section."""

    section: str  # ICUPauseSection value
    content: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
        description="0.0-0.3 sparse data, 0.4-0.7 moderate, 0.8-1.0 rich data",
    )
    data_sources_used: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_content(cls, values: Any) -> Any:
        """Coerce list-typed content to string (LLMs sometimes return arrays)."""
        if isinstance(values, dict):
            content = values.get("content")
            if isinstance(content, list):
                values["content"] = "\n".join(str(item) for item in content)
        return values


class WarningCategory(str, Enum):
    """Routing tag for a Warning. Determines whether it reaches the clinician
    panel or only the audit log / dev-mode view.
    """

    # Clinician-facing
    SAFETY_FLAG = "safety_flag"
    CROSS_DOMAIN_CONFLICT = "cross_domain_conflict"
    DATA_GAP = "data_gap"
    DETERMINISTIC_OVERRIDE = "deterministic_override"
    # Audit-only
    EDITORIAL_REVISION = "editorial_revision"
    SELF_CRITIQUE = "self_critique"
    QA_PROCESS = "qa_process"


class WarningSeverity(str, Enum):
    """Four-tier severity, parallel to ConflictSeverity with INFO added."""

    SAFETY_CRITICAL = "safety_critical"
    CLINICAL = "clinical"
    LOGISTICAL = "logistical"
    INFO = "info"


CLINICIAN_FACING_CATEGORIES: frozenset[WarningCategory] = frozenset({
    WarningCategory.SAFETY_FLAG,
    WarningCategory.CROSS_DOMAIN_CONFLICT,
    WarningCategory.DATA_GAP,
    WarningCategory.DETERMINISTIC_OVERRIDE,
})

SEVERITY_ORDER: dict[WarningSeverity, int] = {
    WarningSeverity.SAFETY_CRITICAL: 0,
    WarningSeverity.CLINICAL: 1,
    WarningSeverity.LOGISTICAL: 2,
    WarningSeverity.INFO: 3,
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
# Pipeline / LLM-as-judge infrastructure noise. These strings describe
# system-internal failures, not clinical findings, and must not reach the
# clinician-facing warning panel. Patterns are deliberately specific (no bare
# "timeout") so clinical phrasings ("sedation timeout", "vent timeout") don't
# get swallowed by the audit bucket.
_INFRASTRUCTURE_ERROR = re.compile(
    r"(could not be completed"
    r"|qa (?:llm |ensemble |consistency )*check (?:failed|could not)"
    r"|json ?decode(?: error)?"
    r"|jsondecodeerror"
    r"|expecting value(?: starting at)?"
    r"|expecting property name"
    r"|expecting ['\",] delimiter"
    r"|(?:llm|tool|api|http|model)[- ]call (?:timeout|timed out|failed)"
    r"|(?:llm|tool|api|http|model) (?:timeout|timed out)"
    r"|(?:deliberation|revise|revision) (?:failed|aborted|timed out|did not complete)"
    r"|(?:citation )?preservation (?:drop|mismatch)"
    r"|parse (?:failure|error|failed)"
    r"|ensemble degraded)",
    re.IGNORECASE,
)


def classify_legacy_warning(message: str, *, source_agent: str = "legacy") -> "Warning":
    """Heuristic classifier for free-text warning strings.

    Used in two places: the AgentSnippet legacy-string coercion validator (so
    on-disk output.json from prior pipeline runs loads cleanly), and the
    scripts/backfill_warnings.py one-time migration. Single source of truth
    keeps pre- and post-refactor cases comparable across annotation
    iterations.
    """
    text = message.strip()
    lower = text.lower()

    if text.startswith("CITATION_DROPPED:"):
        return Warning(
            category=WarningCategory.QA_PROCESS,
            severity=WarningSeverity.INFO,
            message=text,
            source_agent=source_agent,
        )
    if text.startswith("CITATION:"):
        # Fabricated cite tag — evidence of hallucination, route as safety.
        return Warning(
            category=WarningCategory.SAFETY_FLAG,
            severity=WarningSeverity.CLINICAL,
            message=text,
            source_agent=source_agent,
        )
    if "execution failed" in lower or "agent failed" in lower:
        return Warning(
            category=WarningCategory.QA_PROCESS,
            severity=WarningSeverity.INFO,
            message=text,
            source_agent=source_agent,
        )
    # Infrastructure-error strings (JSON parse failures, LLM/tool timeouts,
    # deliberation aborts, citation preservation drops) route to QA_PROCESS
    # so they only appear in the dev-mode audit log, never on the clinician
    # panel. This must come BEFORE _EDITORIAL_VERBS — "revision failed" would
    # otherwise be classified as an editorial revision — and before the
    # _CONFLICT_HINTS heuristic, which would catch "preservation mismatch".
    if _INFRASTRUCTURE_ERROR.search(text):
        return Warning(
            category=WarningCategory.QA_PROCESS,
            severity=WarningSeverity.INFO,
            message=text,
            source_agent=source_agent,
        )
    if _NO_ISSUE.search(text):
        return Warning(
            category=WarningCategory.SELF_CRITIQUE,
            severity=WarningSeverity.INFO,
            message=text,
            source_agent=source_agent,
        )
    if _EDITORIAL_VERBS.search(text):
        return Warning(
            category=WarningCategory.EDITORIAL_REVISION,
            severity=WarningSeverity.INFO,
            message=text,
            source_agent=source_agent,
        )
    if _DATA_GAP_HINTS.search(text):
        return Warning(
            category=WarningCategory.DATA_GAP,
            severity=WarningSeverity.CLINICAL,
            message=text,
            source_agent=source_agent,
        )
    if _CONFLICT_HINTS.search(text):
        return Warning(
            category=WarningCategory.CROSS_DOMAIN_CONFLICT,
            severity=WarningSeverity.CLINICAL,
            message=text,
            source_agent=source_agent,
        )
    if _SAFETY_HINTS.search(text):
        return Warning(
            category=WarningCategory.SAFETY_FLAG,
            severity=WarningSeverity.CLINICAL,
            message=text,
            source_agent=source_agent,
        )
    # Conservative default: treat as a clinician-facing safety flag rather
    # than dropping it. Better to over-show than to silently filter a real
    # issue we couldn't pattern-match.
    return Warning(
        category=WarningCategory.SAFETY_FLAG,
        severity=WarningSeverity.CLINICAL,
        message=text,
        source_agent=source_agent,
    )


class Warning(BaseModel):
    """Structured warning emitted by an agent or pipeline stage.

    Categories route the warning either to the clinician-facing panel (see
    CLINICIAN_FACING_CATEGORIES) or to the audit log only. Severity orders
    items within the clinician panel.
    """

    category: WarningCategory
    severity: WarningSeverity
    message: str
    source_agent: str
    source_section: Optional[str] = None
    cite: Optional[str] = None


def _coerce_warning_list(value: Any) -> Any:
    """Validator helper: accept str | dict | Warning entries, coerce to dicts.

    Legacy on-disk output.json files have warnings: list[str]. Apply the
    classifier so they load with proper categories.
    """
    if not isinstance(value, list):
        return value
    out = []
    for item in value:
        if isinstance(item, str):
            out.append(classify_legacy_warning(item).model_dump())
        else:
            out.append(item)
    return out


class AgentSnippet(BaseModel):
    """Output produced by a single domain agent.

    The internal representation carries structured Warning objects. When LLMs
    emit warnings they go through AgentSnippetLLM (warnings as strings) so
    we don't ask small models to categorize their own output — the wrapper
    at the producer boundary assigns category and severity based on the
    emitting pipeline stage.
    """

    agent_name: str
    sections: list[SectionContribution]
    warnings: list[Warning] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_warnings(cls, values: Any) -> Any:
        if isinstance(values, dict) and "warnings" in values:
            values["warnings"] = _coerce_warning_list(values["warnings"])
        return values


class AgentSnippetLLM(BaseModel):
    """Wire schema for LLM-facing structured output.

    Mirrors AgentSnippet but keeps warnings as plain strings so the JSON
    schema sent to the model doesn't force per-warning categorization.
    Converted to AgentSnippet by `wrap_llm_snippet()`.
    """

    agent_name: str
    sections: list[SectionContribution]
    warnings: list[str] = Field(default_factory=list)


def wrap_llm_snippet(
    llm_snippet: "AgentSnippetLLM",
    *,
    category: WarningCategory,
    severity: WarningSeverity,
    source_agent: str | None = None,
) -> AgentSnippet:
    """Convert an LLM wire snippet to an internal AgentSnippet.

    Each free-text warning is wrapped as Warning(category=..., severity=...,
    message=str, source_agent=...). The category/severity reflect the
    emitting pipeline stage (e.g. self-critique pass uses EDITORIAL_REVISION
    + INFO; first-pass agent output uses SAFETY_FLAG + CLINICAL).
    """
    agent = source_agent or llm_snippet.agent_name
    wrapped = [
        Warning(
            category=category,
            severity=severity,
            message=msg,
            source_agent=agent,
        )
        for msg in llm_snippet.warnings
    ]
    return AgentSnippet(
        agent_name=llm_snippet.agent_name,
        sections=llm_snippet.sections,
        warnings=wrapped,
    )


class InterpreterOutput(BaseModel):
    """Output from an interpreter agent (CR-DSF mode).

    Produces per-domain clinical summaries from a single data modality.
    """

    agent_name: str  # "structured_interpreter" or "note_interpreter"
    domain_summaries: dict[str, str] = Field(
        description="Mapping of domain agent name to clinical summary text"
    )
    warnings: list[Warning] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_warnings(cls, values: Any) -> Any:
        if isinstance(values, dict) and "warnings" in values:
            values["warnings"] = _coerce_warning_list(values["warnings"])
        return values


class SalienceOutput(BaseModel):
    """Output from the S2 structured-salience selector (compression sub-study).

    Same shape as InterpreterOutput, but the model legitimately emits each
    domain's *selected view* as a structured object (field→value), not a single
    string — so we coerce non-string values to a readable "field: value" view
    before validation. Robust on local models where structured-output
    enforcement is off and the model returns whatever shape fits the task.
    """

    agent_name: str = "structured_salience"
    domain_summaries: dict[str, str] = Field(
        description="Mapping of domain agent name to the selected-view text"
    )
    warnings: list[Warning] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_views(cls, data: Any) -> Any:
        import json as _json

        def _render(val: Any) -> str:
            if val is None:
                return "Not documented"
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                return "\n".join(f"{k}: {_render(v)}" for k, v in val.items())
            if isinstance(val, list):
                return "\n".join(_render(v) for v in val)
            return str(val)

        if isinstance(data, dict) and isinstance(data.get("domain_summaries"), dict):
            data["domain_summaries"] = {
                d: _render(v) for d, v in data["domain_summaries"].items()
            }
        if isinstance(data, dict) and "warnings" in data:
            data["warnings"] = _coerce_warning_list(data["warnings"])
        return data


class ExtractorOutput(BaseModel):
    """Output from the structured extractor agent (CR-DSF+ mode).

    Extracts predefined discrete fields per domain agent from clinical notes.
    These serve as factual anchors that prevent information loss during
    narrative summarization.
    """

    agent_name: str = "structured_extractor"
    # dict[str, Any] (not dict[str, dict[str, str]]) so BOTH shapes validate: the
    # multi-domain nested form {domain: {field: value}} AND the per-domain FLAT
    # form {field: value} that the per-domain extractor prompt actually requests.
    # Local models (structured-output enforcement off) emit flat on some cases and
    # nested on others; typing this strictly caused intermittent dict_type
    # validation failures → dropped cases (a harness artifact, not a method
    # property). run() disambiguates the two shapes after validation.
    extraction_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Either {domain: {field: value}} (nested) or {field: value} (flat per-domain)",
    )
    warnings: list[Warning] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_extraction_values(cls, data: Any) -> Any:
        """Coerce non-string extraction values to JSON strings (both nested and
        flat shapes), and legacy string warnings to Warning objects.
        """
        import json as _json

        def _to_str(v: Any) -> str:
            if v is None:
                return ""
            return v if isinstance(v, str) else _json.dumps(v, default=str)

        if isinstance(data, dict) and isinstance(data.get("extraction_fields"), dict):
            fields = data["extraction_fields"]
            for key, val in list(fields.items()):
                if isinstance(val, dict):
                    # Nested: {domain: {field: value}} — coerce inner values.
                    for k2, v2 in list(val.items()):
                        val[k2] = _to_str(v2)
                else:
                    # Flat: {field: value} — coerce the value itself.
                    fields[key] = _to_str(val)
        if isinstance(data, dict) and "warnings" in data:
            data["warnings"] = _coerce_warning_list(data["warnings"])
        return data


# ---------------------------------------------------------------------------
# Resident Agent pre-synthesis schemas
# ---------------------------------------------------------------------------


class ConflictSeverity(str, Enum):
    """Severity of a cross-domain conflict flagged by the Resident."""

    SAFETY_CRITICAL = "safety_critical"  # Drug interaction, vent status mismatch
    CLINICAL = "clinical"  # Care plan contradiction
    LOGISTICAL = "logistical"  # Timing/workflow disagreement


class CrossDomainConflict(BaseModel):
    """A contradiction between two domain agents' outputs."""

    domain_a: str
    domain_b: str
    conflict_description: str
    severity: ConflictSeverity = ConflictSeverity.CLINICAL
    relevant_sections: list[str] = Field(default_factory=list)


class CriticalGap(BaseModel):
    """A required ICU-PAUSE field where no domain agent provided content."""

    icu_pause_field: str
    gap_type: Literal["explicitly_unavailable", "silently_absent"]
    responsible_domain: Optional[str] = None
    note: Optional[str] = None


class RedundancyCluster(BaseModel):
    """A cluster of domain-agent S-section headers describing the same clinical entity."""

    cluster: list[str]  # e.g. ["nurse: #Airway clearance", "respiratory: #Tracheostomy care"]
    proposed_header: str  # e.g. "#Tracheostomy management"
    rationale: str  # e.g. "All describe the same airway device management"
    preserve_as_sublines: list[str] = Field(default_factory=list)
        # e.g. ["tobramycin for PsA tracheobronchitis (pharmacy)", "suctioning q4h (nurse)"]


class PreBriefNarrative(BaseModel):
    """Concise cross-domain narrative drafted by the Resident."""

    dominant_clinical_theme: str
    inter_domain_dependencies: list[str] = Field(default_factory=list)
    priority_todo_items: list[str] = Field(default_factory=list)


class ResidentPreBrief(BaseModel):
    """Structured pre-synthesis output from the Resident agent.

    The Resident reviews all domain agent outputs and produces a ~2-3k token
    brief that the Intensivist uses as its starting point. The Resident does
    NOT have access to raw structured data or notes — only agent outputs.
    """

    cross_domain_conflicts: list[CrossDomainConflict] = Field(default_factory=list)
    critical_gaps: list[CriticalGap] = Field(default_factory=list)
    redundancies: list[RedundancyCluster] = Field(default_factory=list)
    pre_brief_narrative: PreBriefNarrative
    self_critique_passed: bool = True
    self_critique_flags: list[str] = Field(default_factory=list)
    resident_confidence: Literal["high", "moderate", "low"] = "moderate"


# ---------------------------------------------------------------------------
# Intensivist Agent schemas
# ---------------------------------------------------------------------------


class CompetingRisksEntry(BaseModel):
    """One Phase-1 SCOPE CHECK entry per active in-scope therapy.

    Pydantic-enforced binary indication grounding (Schema-Slot
    Confabulation Grounding pattern, see
    docs/competing_risks_indication_grounding_v1.8_design.md):
      - indication == HEDGE_PHRASE  → source_note_id and source_quote
        MUST be None; the model is hedging because no chart text
        documents the indication;
      - indication is any other string → source_note_id and source_quote
        MUST be populated; downstream post-pass validator
        (orchestrator._validate_competing_risks_grounding) verifies the
        citation resolves to a routed note's body via substring match.

    conflict_status drives arms population:
      - "no"  → conflict_condition, risk_of_continuing, and
        risk_of_holding_or_reducing MUST be None;
      - "yes" / "unclear" → all three MUST be populated.
    """

    HEDGE_PHRASE: ClassVar[str] = "indication not documented in available notes"

    drug: str = Field(
        description="The active in-scope therapy this entry concerns. "
        "Verbatim from pharmacy's S section or the U_unprescribing "
        "Therapeutic IV line.",
    )

    indication: str = Field(
        description=(
            "Either the literal hedge phrase "
            "'indication not documented in available notes' (when no "
            "indication is documented in available notes), OR the "
            "indication string verbatim from the source. Non-hedge "
            "values REQUIRE source_note_id and source_quote to be "
            "populated."
        ),
    )

    source_note_id: Optional[str] = Field(
        default=None,
        description="The note_id whose body source_quote came from. "
        "MUST be in the case's routed-note set. None ONLY when "
        "indication is the hedge phrase.",
    )
    source_quote: Optional[str] = Field(
        default=None,
        description="Contiguous verbatim source span (no ellipsis, no "
        "truncation) that grounds the non-hedge indication. MUST "
        "substring-match source_note_id's body (post-normalization). "
        "None ONLY when indication is the hedge phrase.",
    )

    conflict_status: Literal["yes", "no", "unclear"] = Field(
        description="Phase-1 scope-check decision. 'unclear' is a "
        "first-class option for ambiguous tensions where the "
        "conflicting condition is documented but its severity / "
        "activity is uncertain.",
    )
    conflict_condition: Optional[str] = Field(
        default=None,
        description="The conflicting active condition (yes) or the "
        "brief reason the conflict is ambiguous (unclear). "
        "MUST be None when conflict_status == 'no'.",
    )

    risk_of_continuing: Optional[str] = Field(
        default=None,
        description="One-line clinical consequence of continuing the "
        "named therapy in the named conflicting condition. Required "
        "when conflict_status in {'yes', 'unclear'}; MUST be None when "
        "conflict_status == 'no'. Direct-consequence rule applies — "
        "must not introduce new clinical entities. Qualitative-only "
        "rule applies — no unsupported quantitative predictions.",
    )
    risk_of_holding_or_reducing: Optional[str] = Field(
        default=None,
        description="One-line clinical consequence of discontinuing or "
        "reducing the named therapy given the named indication. Same "
        "population rule and two clinical rules as risk_of_continuing.",
    )

    @model_validator(mode="after")
    def _hedge_xor_citation(self) -> "CompetingRisksEntry":
        # Note: ellipsis / truncation detection happens in the post-pass
        # validator (orchestrator.validate_competing_risks_grounding), NOT
        # here. Raising at Pydantic-validation time would propagate up
        # through the intensivist's response_format parse and trigger the
        # AgentSnippetLLM fallback, losing the ENTIRE reasoning_log
        # (conflicts, gaps, safety_checks, other valid competing_risks
        # entries) for one bad citation. Post-pass handles truncation
        # gracefully: rewrite the single bad entry to hedge + emit
        # INDICATION_QUOTE_TRUNCATED WARN.
        if self.indication == self.HEDGE_PHRASE:
            if self.source_note_id is not None or self.source_quote is not None:
                raise ValueError(
                    f"indication is the hedge phrase but source_note_id "
                    f"({self.source_note_id!r}) or source_quote "
                    f"({self.source_quote!r}) is populated — these must "
                    f"be None when hedging"
                )
            return self
        # Non-hedge indication
        if not self.source_note_id or not self.source_quote:
            raise ValueError(
                f"non-hedge indication ({self.indication!r}) requires "
                f"both source_note_id and source_quote to be populated"
            )
        return self

    @model_validator(mode="after")
    def _arms_match_conflict_status(self) -> "CompetingRisksEntry":
        """conflict_status determines whether arms / conflict_condition
        are populated. 'no' means no competing-risk tension — arms are
        semantically incoherent and must be None. 'yes' / 'unclear' both
        require all three populated."""
        if self.conflict_status == "no":
            if self.conflict_condition is not None:
                raise ValueError(
                    "conflict_status='no' but conflict_condition is "
                    f"populated ({self.conflict_condition!r}); no "
                    "conflict means no condition to name"
                )
            if (
                self.risk_of_continuing is not None
                or self.risk_of_holding_or_reducing is not None
            ):
                raise ValueError(
                    "conflict_status='no' but arms are populated; arms "
                    "are semantically incoherent without a conflict"
                )
        else:  # yes or unclear
            if not self.conflict_condition:
                raise ValueError(
                    f"conflict_status={self.conflict_status!r} requires "
                    f"conflict_condition to be populated"
                )
            if (
                not self.risk_of_continuing
                or not self.risk_of_holding_or_reducing
            ):
                raise ValueError(
                    f"conflict_status={self.conflict_status!r} requires "
                    f"both risk_of_continuing and risk_of_holding_or_reducing "
                    f"to be populated"
                )
        return self


class ReasoningLog(BaseModel):
    """Chain-of-thought reasoning log from the Intensivist agent.

    Captures explicit clinical reasoning before section generation:
    conflict adjudication, gap detection, safety checks, and the
    competing-risks scope check.
    """

    conflicts: list[str] = Field(
        default_factory=list,
        description="Cross-agent conflicts detected and resolutions. "
        "Format: 'CONFLICT: [agent_a] says X vs [agent_b] says Y → RESOLUTION: Z'",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Clinically significant findings in data that no agent mentioned. "
        "Format: 'GAP: [data_key] shows X but no agent addressed this'",
    )
    safety_checks: list[str] = Field(
        default_factory=list,
        description="Results of safety verification checks. "
        "Format: 'CHECK: [item] — [result]' (e.g., 'CHECK: code_status — FULL CODE documented'). "
        "COMPETING_RISKS entries that historically lived here in v1.7 are now in the "
        "dedicated competing_risks_check field below.",
    )
    competing_risks_check: list[CompetingRisksEntry] = Field(
        default_factory=list,
        description="Phase-1 SCOPE CHECK: one entry per active in-scope "
        "therapy in this patient (anticoagulants, antiplatelets, pressors, "
        "high-dose immunosuppression, deep sedation, NMB, rate-control / "
        "anti-arrhythmics, diuretics in borderline renal, beta-blockers in "
        "cardiogenic shock / decompensated HF). Includes therapies that "
        "resolve to conflict_status='no' so the auditable miss is "
        "preserved. See CompetingRisksEntry for the binary "
        "indication-grounding contract.",
    )
    confidence_notes: list[str] = Field(
        default_factory=list,
        description="Per-section notes on data completeness and agent agreement. "
        "Format: '[section_key]: [note]' (e.g., 'S: high confidence, all agents agree')",
    )


class FocusProblemEntry(BaseModel):
    """One pinned focus problem for the receiving floor team.

    Emitted by the Intensivist agent to identify the leading problems
    that anchor the S section problem list and drive U_uncertainty
    working-diagnosis selection. The orchestrator pins these to the
    top of S in rank order; the remaining problems fall back to the
    existing organ-system reorder + bottom-anchor pass.

    Citation grounding rule (enforced post-render, not at schema
    parse): ``why_focus`` SHOULD contain ≥1 inline citation matching
    ``CITE_PATTERN`` — any registered source_type, including the
    routed-note types added by the 2026-05-29 Path 1 commit. The
    orchestrator validates that each cite tag in why_focus resolves
    against ``metadata.citation_index`` after rendering; unresolved
    tags emit ``FOCUS_RATIONALE_CITE_UNRESOLVED`` qa_process WARN.
    The schema does NOT enforce cite presence — keeping it permissive
    avoids the AgentSnippetLLM fallback path losing the entire
    intensivist output on one missing tag.

    Match-key normalization happens at orchestrator pin-merge time
    (lowercase, alphanumeric, substring scan against rendered
    ``#Problem`` headers). Multi-header matches emit
    ``FOCUS_PROBLEM_HEADER_AMBIGUOUS`` qa_process WARN and skip the
    pin — fail-on-ambiguity, never silently drop.

    The ``intent="goals_of_care_primary"`` override exists for
    palliative / GOC-dominant phenotypes where leading with goals of
    care is clinically correct. Without the override, the bottom-
    anchor pass (BOTTOM_ANCHORED_CATEGORIES in orchestrator.py)
    relocates goals_of_care / disposition / code_status pins to the
    tail. The override is per-entry, not per-section: pinning GOC as
    primary does not promote disposition or code_status too.
    """

    problem_match_key: str = Field(
        ...,
        min_length=1,
        description="Short canonical name (e.g., 'AKI', 'hepatic "
        "encephalopathy') used at orchestrator pin-merge to "
        "substring-match this pin against rendered #Problem headers "
        "after normalization (lowercase, alphanumeric). Carried "
        "verbatim for the audit trail; normalization is not in the "
        "schema layer.",
    )
    why_focus: str = Field(
        ...,
        min_length=1,
        description="One-sentence rationale for why this is a leading "
        "problem for the receiving floor team's first-24h decisions. "
        "SHOULD contain ≥1 inline citation in the existing "
        "(source_type M-DD HH:MM) format covering any registered "
        "source type (lab/vital/med/resp/assess/code/proc/exam-* or "
        "note types). Cite resolution is validated post-render at "
        "the orchestrator.",
    )
    intent: Optional[Literal["goals_of_care_primary"]] = Field(
        default=None,
        description="Optional bottom-anchor override. Set to "
        "'goals_of_care_primary' ONLY when the patient phenotype "
        "warrants leading with goals of care / hospice / palliative "
        "transition (receiving-team next-24h action is the GOC "
        "conversation, not organ-system management). Default None "
        "preserves the standard bottom-anchor for goals_of_care / "
        "disposition / code_status / access.",
    )
    rank: int = Field(
        ...,
        ge=1,
        description="1-based priority rank within the focus_problems "
        "list. rank=1 is the dominant problem. Orchestrator emits "
        "pins in rank order at the top of the S section.",
    )


class ModifierConfirmation(BaseModel):
    """Per-modifier confirmation pin for the time-sensitive-modifier rule.

    The intensivist emits one of these for EACH time-sensitive modifier
    rendered in a OneLinerPMHEntry.display (e.g., 'on paclitaxel',
    'chronic vent-dependent', 'with active GIB', 'recurrent', 'naive',
    'requiring [intervention]'). Time-safe modifiers (s/p, c/b, with
    [anatomic site], staging) do NOT need confirmation entries — they
    are stable across reference windows by definition.

    The orchestrator validates that ``confirmation_quote`` substring-
    matches the body of the routed note identified by
    ``confirmed_in_note_id``, and that the note's creation_dttm sits
    within the reference window. Failure on either check emits
    ``ONE_LINER_MODIFIER_UNCONFIRMED`` qa_process WARN and drops the
    modifier from the rendered one-liner — the bare condition still
    renders, but without the unconfirmed time-sensitive qualifier.

    Substring matching uses ``normalize_for_validator`` from
    tools.text_normalize so curly quotes / dashes / NBSP variants
    between chart exports and LLM emissions don't false-reject.
    """

    modifier_text: str = Field(
        ...,
        min_length=1,
        description="Verbatim time-sensitive modifier as it appears in "
        "the OneLinerPMHEntry.display (e.g., 'on paclitaxel', "
        "'chronic vent-dependent', 'with active GIB'). Used by the "
        "orchestrator to locate the modifier within display for the "
        "drop-on-failure render path.",
    )
    confirmed_in_note_id: str = Field(
        ...,
        min_length=1,
        description="note_id of the routed note in which the modifier "
        "is confirmed. Must resolve against metadata.citation_index "
        "AND the note's creation_dttm must sit within the reference "
        "window. Unresolved or out-of-window note_ids fail validation.",
    )
    confirmation_quote: str = Field(
        ...,
        min_length=1,
        description="Verbatim substring of the cited note's body that "
        "establishes the modifier as currently true. Substring match "
        "runs on normalize_for_validator output for both sides to "
        "tolerate punctuation / whitespace drift. Truncation markers "
        "('...' / '…') in the quote fail validation.",
    )


class OneLinerPMHEntry(BaseModel):
    """One PMH condition selected for the Section I one-liner lead sentence.

    Emitted by the Intensivist agent to identify the load-bearing PMH
    conditions that anchor the first sentence of Section I. The
    orchestrator does NOT pin these in a post-pass (unlike
    focus_problems); the intensivist authors the lead sentence directly
    and emits the structured pin alongside it for audit and lint.

    Selection criterion (operationalized in intensivist.yaml):
    a condition is load-bearing if it (a) shapes the acute-problem
    differential, (b) changes management or treatment decisions for the
    acute problem, or (c) shapes prognosis or goals-of-care framing.
    Tiebreaker when >3 are load-bearing: temporal + physiological
    proximity to the acute problem. NEVER selection basis: chart order,
    alphabetical, organ system, chronicity, comprehensiveness.

    Modifier rule: ``display`` carries time-safe modifiers freely
    (s/p, c/b, with [anatomic site], staging) and may carry time-
    sensitive modifiers (on [drug], requiring [intervention], with
    active [condition], recurrent / naive / chronic [organ-failure]
    dependent) ONLY when accompanied by a ``modifier_confirmation``
    entry per modifier. The orchestrator drops any time-sensitive
    modifier whose confirmation fails validation.

    In-prose ↔ structured alignment guard runs at the orchestrator:
    each entry's ``display`` must have a normalized-form match in the
    rendered Section I lead sentence (via expand_pmh_abbreviations +
    normalize_for_pmh_match in tools.text_normalize), and vice versa.
    Mismatches emit ``ONE_LINER_PMH_ALIGNMENT_MISMATCH`` qa_process
    WARN. Normalization tolerates abbreviation pairs (BrCa ↔ breast
    cancer, s/p ↔ status post, c/b ↔ complicated by, mets ↔
    metastases) so abbreviation rewrites in the lead sentence don't
    false-fail the lint.

    ``source_clause_anchor`` is a verbatim substring of scribe.pmh
    that this entry abbreviates from. Orchestrator validates substring
    presence in scribe.pmh; failure emits
    ``ONE_LINER_PMH_ANCHOR_UNRESOLVED`` qa_process WARN. The anchor
    binds the abbreviated display back to the verbatim audit-safe
    paragraph slot so reviewers can trace any one-liner condition back
    to the chart clause it derives from.

    Cap: ≤3 entries (modal 1–2). Empty list is permitted — same
    convention as focus_problems — when scribe PMH is null or the
    intensivist legitimately has nothing to lead with.
    """

    display: str = Field(
        ...,
        min_length=1,
        description="Abbreviated form rendered in the Section I one-"
        "liner lead sentence (e.g., 'metastatic breast cancer (liver "
        "mets)', 'laryngeal Ca s/p trach (chronic vent-dependent)'). "
        "≤1 modifier phrase per entry. Time-sensitive modifiers MUST "
        "be accompanied by a modifier_confirmation entry.",
    )
    source_clause_anchor: str = Field(
        ...,
        min_length=1,
        description="Verbatim contiguous substring of scribe.pmh that "
        "this entry abbreviates. Orchestrator validates substring "
        "presence post-render. Binds the abbreviated display to the "
        "audit-safe verbatim paragraph slot.",
    )
    why_lead: str = Field(
        ...,
        min_length=1,
        description="One-sentence rationale for why this condition is "
        "load-bearing for the one-liner per criterion (a)/(b)/(c). "
        "SHOULD reference the acute admission problem this PMH item "
        "contextualizes (e.g., 'HFpEF shapes the hyponatremia "
        "differential via volume + diuretic exposure').",
    )
    modifier_confirmation: Optional[list[ModifierConfirmation]] = Field(
        default=None,
        description="REQUIRED if display contains any time-sensitive "
        "modifier (on [drug], requiring [intervention], with active "
        "[condition], recurrent, naive, chronic [organ-failure] "
        "dependent). One ModifierConfirmation per time-sensitive "
        "modifier. None / empty list is correct for entries whose "
        "display carries only time-safe modifiers (s/p, c/b, with "
        "[anatomic site], staging).",
    )
    rank: int = Field(
        ...,
        ge=1,
        description="1-based rank within the one_liner_pmh_selection "
        "list. rank=1 is the most load-bearing condition. Renders "
        "left-to-right in the lead sentence in rank order.",
    )


class PMHFallback(BaseModel):
    """Provenance for an intensivist-COMPOSED PMH one-liner.

    Emitted by the intensivist ONLY when the scribe PMH pin is empty AND the
    intensivist composed PMH from the H&P (authoritative) / progress-note
    opener already in its routed context. None whenever the scribe pin is
    present (scribe stays primary, verbatim-grounded) or when the notes
    genuinely have no PMH (negation / previously healthy).

    NOTE-level provenance only — the post-hoc GroundingEvaluator does the
    per-claim safety work, so clause-level verbatim spans are intentionally
    NOT required here (forcing the intensivist to emit verbatim spans would
    fight its synthesis job). Surfaced to the run manifest (#12) and the
    GroundingEvaluator via ``pmh_source``.
    """

    pmh_source: str = Field(
        default="intensivist_fallback",
        description="Provenance tag — always 'intensivist_fallback' for this "
        "field. Distinguishes intensivist-composed PMH from the scribe pin.",
    )
    note_ids: list[str] = Field(
        default_factory=list,
        description="note_id(s) the PMH was composed from (H&P primary, "
        "progress-note opener secondary).",
    )
    note_types: list[str] = Field(
        default_factory=list,
        description="Note type(s) drawn from: 'hp_note' and/or 'progress_note' "
        "— the hp-vs-progress rollup for fire-rate measurement.",
    )
    text: str = Field(
        description="The composed PMH one-liner text — chronic conditions "
        "only, between the demographic opener's 'with' and the presentation "
        "pivot. Excludes the acute presenting complaint.",
    )


class IntensivistOutput(BaseModel):
    """Output from the Intensivist agent with chain-of-thought reasoning.

    Extends AgentSnippet with a reasoning_log that captures the explicit
    clinical reasoning performed before section generation.
    """

    agent_name: str = "intensivist"
    reasoning_log: ReasoningLog = Field(default_factory=ReasoningLog)
    sections: list[SectionContribution] = Field(default_factory=list)
    focus_problems: list[FocusProblemEntry] = Field(
        default_factory=list,
        description="Ranked priority pins driving S-section problem "
        "ordering and U_uncertainty working-diagnosis selection. "
        "Empty list is permitted — Stage-2 reviewers grade "
        "empty-vs-emitted as a distinct instrumentation signal from "
        "incorrect-pick (see Ship 2 design memo).",
    )
    one_liner_pmh_selection: list[OneLinerPMHEntry] = Field(
        default_factory=list,
        description="Ranked PMH conditions selected for the Section I "
        "one-liner lead sentence. Cap ≤3, modal 1–2. Empty list is "
        "permitted (scribe.pmh null or legitimately nothing to lead "
        "with). Renders left-to-right in the lead sentence in rank "
        "order. Orchestrator runs the in-prose ↔ structured alignment "
        "lint and the modifier_confirmation completeness lint.",
    )
    pmh_fallback: Optional[PMHFallback] = Field(
        default=None,
        description="Set ONLY when the scribe PMH pin is empty AND the "
        "intensivist composed PMH from the H&P/progress-note opener. None "
        "when the scribe pin is present (scribe stays primary) or when the "
        "notes genuinely have no PMH (negation / previously healthy, in which "
        "case Section I keeps the 'chart review required' string).",
    )
    warnings: list[Warning] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_warnings(cls, values: Any) -> Any:
        if isinstance(values, dict) and "warnings" in values:
            values["warnings"] = _coerce_warning_list(values["warnings"])
        return values


class IntensivistOutputLLM(BaseModel):
    """Wire schema for the Intensivist's LLM call (warnings as strings)."""

    agent_name: str = "intensivist"
    reasoning_log: ReasoningLog = Field(default_factory=ReasoningLog)
    sections: list[SectionContribution] = Field(default_factory=list)
    focus_problems: list[FocusProblemEntry] = Field(default_factory=list)
    one_liner_pmh_selection: list[OneLinerPMHEntry] = Field(default_factory=list)
    pmh_fallback: Optional[PMHFallback] = Field(default=None)
    warnings: list[str] = Field(default_factory=list)


def wrap_llm_intensivist(
    llm_output: "IntensivistOutputLLM",
    *,
    category: WarningCategory,
    severity: WarningSeverity,
) -> IntensivistOutput:
    wrapped = [
        Warning(
            category=category,
            severity=severity,
            message=msg,
            source_agent="intensivist",
        )
        for msg in llm_output.warnings
    ]
    return IntensivistOutput(
        agent_name=llm_output.agent_name,
        reasoning_log=llm_output.reasoning_log,
        sections=llm_output.sections,
        focus_problems=llm_output.focus_problems,
        one_liner_pmh_selection=llm_output.one_liner_pmh_selection,
        pmh_fallback=llm_output.pmh_fallback,
        warnings=wrapped,
    )


# ---------------------------------------------------------------------------
# Scribe Agent schemas
# ---------------------------------------------------------------------------


class SubspecialtyConsult(BaseModel):
    """One subspecialty consultant entry for Section A.

    Carries the consultant's engagement state across four discrete values
    (Active / Planned / Declined / AssessedNotNeeded) per the 2026-05-29
    Section A revision. The state vocabulary mirrors the therapist's
    PT/OT/SLP/Wound Care convention so the intensivist renders both lanes
    with a consistent first-word state-phrase scanning anchor.

    For ``state == "Active"``, structured so the per-source word-boundary
    substring guard can attribute each service to a specific source note
    rather than a corpus-wide substring match. Tighter than the PMH /
    allergies / home_meds / code_status validator pattern (which only
    requires each clause to appear *somewhere* in the routed corpus)
    because consults are easy to confabulate from passing mentions
    ("renal failure" → false-positive Nephrology).

    For ``state in ("Planned", "Declined", "AssessedNotNeeded")``,
    ``state_quote`` + ``state_citation`` are validator-required: quote
    must substring-match the cited source note's body, citation must
    reference a routed note that actually exists. Closes the
    hallucination vector for non-active states where the model is
    summarizing intent rather than mirroring chart metadata.

    ``service`` is the canonical service name (e.g. "Infectious Disease",
    "Nephrology") quoted from a chart literal. The scribe is instructed to
    canonicalize light abbreviation variants (ID → Infectious Disease)
    only when the canonical form also appears as a substring of the source
    note — see scribe.yaml's verbatim hard rules.

    ``source_type`` distinguishes the extraction paths:
    - ``consults_note``: state=Active, identified by structured
      consult-note metadata (note_type/service field) for a note inside
      the lookback window. Primary signal for daily-rounding services.
    - ``progress_note_ap``: state=Active, identified by present-tense
      engagement phrasing ("following", "managing", "recommending",
      "per [service]") inside the most recent progress-note A&P. Picks
      up services that round every 2–3 days and don't have a consult-note
      in the 48–72h window.
    - ``state_attribution``: state in (Planned, Declined,
      AssessedNotNeeded); identified by a quoted phrase in a routed
      note that attributes the consult to one of the non-active states
      (planning / declining / sign-off). Requires ``state_quote`` +
      ``state_citation``. Named ``state_attribution`` rather than
      ``planning_quote`` because the value covers refusals and
      closeouts too, not just planning.
    """

    service: str = Field(
        description=(
            "Canonical subspecialty service name (e.g. 'Infectious "
            "Disease', 'Nephrology', 'Hematology/Oncology'). Must "
            "word-boundary match in the body of ``source_note_id``."
        ),
    )
    state: Literal["Active", "Planned", "Declined", "AssessedNotNeeded"] = Field(
        default="Active",
        description=(
            "Engagement state. Default 'Active' preserves the pre-"
            "2026-05-29 extraction semantics (every entry represented "
            "an active consultant). Non-Active values require "
            "``state_quote`` + ``state_citation`` (validator-enforced). "
            "Renders in Section A as the first-word scanning anchor per "
            "the convention in scribe.yaml + therapist.yaml."
        ),
    )
    state_quote: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim quote of the planning / declining / sign-off "
            "language from ``source_note_id``. REQUIRED when state != "
            "Active; FORBIDDEN when state == Active. Validator enforces "
            "substring match against the cited source note's body."
        ),
    )
    state_citation: Optional[str] = Field(
        default=None,
        description=(
            "Citation in canonical form `<note_type> <YYYY-MM-DD>` "
            "(e.g. `H&P 2024-05-27`, `progress_note 2024-05-29`). "
            "REQUIRED when state != Active; FORBIDDEN when state == "
            "Active. The cited note must exist in routed notes "
            "(validator-enforced existence). Distinct from "
            "``source_note_id`` (internal id reference); this is the "
            "human-readable citation rendered in Section A."
        ),
    )
    last_note_dttm: str = Field(
        description=(
            "ISO timestamp of the most recent note attributing this "
            "service to the patient. For state == Active: consult-note "
            "or progress-note A&P creation_dttm. For state != Active: "
            "the cited note's creation_dttm."
        ),
    )
    source_note_id: str = Field(
        description=(
            "note_id of the routed note that establishes this entry. "
            "Used by the per-service word-boundary substring guard "
            "(Active) and by the ``state_quote`` substring guard "
            "(non-Active)."
        ),
    )
    source_type: Literal["consults_note", "progress_note_ap", "state_attribution"] = Field(
        description=(
            "Which extraction path identified this entry. "
            "``state_attribution`` is used when state != Active "
            "(covers Planned, Declined, AssessedNotNeeded)."
        ),
    )

    # NOTE (2026-06-08, scribe-parse-wipeout fix): the
    # ``_check_state_evidence_consistency`` model_validator was REMOVED here
    # and ported to a graceful per-entry drop in
    # ``ScribeAgent._validate_subspecialty_consults``. Raising at Pydantic
    # parse time failed the ENTIRE ``ScribeExtractionLLM`` object — taking
    # PMH/allergies/home_meds/code_status down as collateral — when a single
    # consult was malformed. The structural state ↔ (quote, citation)
    # integrity check now drops only the offending entry.
    # See memory: project_icu_pause_scribe_parse_wipeout.


# Class-level ``ActiveConsult`` alias was removed concurrent with the
# 2026-05-29 ``active_consults`` → ``subspecialty_consults`` field rename.
# Field-level aliases on the renamed fields (``alias="active_consults"``,
# etc.) provide JSON-level backward-compat for pre-rename serialized
# briefs; class-level identity import paths must use ``SubspecialtyConsult``.


class AdmissionAntibioticCourse(BaseModel):
    """One inpatient antibiotic course given THIS ADMISSION and OUTSIDE
    the 48h structured-data window.

    Mirrors the ActiveConsult per-source attribution pattern (NOT the
    corpus-wide PMH/allergies validator pattern). ``source_quote`` is
    the verbatim chart literal; the LLM normalizes dates and drug
    names; the runtime validator substring-checks source_quote against
    the body of ``source_note_id`` only.

    Scope is schema-enforced INPATIENT-DURING-THIS-ADMISSION and
    OUTSIDE 48h. Active drugs (within 48h) are owned by ``meds.states``
    in structured data and MUST NOT appear here. Pre-admission
    outpatient courses are EXCLUDED (patient-reported outpatient
    antibiotic history is low-fidelity content; out of scope).

    See docs/admission_antibiotics_design.md.
    """

    drug: str = Field(
        description=(
            "Canonical antibiotic name (e.g. 'cefepime', 'cefazolin', "
            "'piperacillin-tazobactam'). LLM may canonicalize from chart "
            "literals ('Maxipime' -> 'cefepime', 'Zosyn' -> "
            "'piperacillin-tazobactam') ONLY when the canonical or brand "
            "form is substring-present in source_quote. The scribe's "
            "runtime admission-antibiotics validator enforces this — a "
            "course whose drug name is absent from source_quote is dropped."
        ),
    )

    start_date_phrase: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim chart literal for start date — '3/3', '3/3/24', "
            "'March 3'. Rendered to clinicians as-is. None when not "
            "documented in source — do NOT invent."
        ),
    )
    start_date_parsed: Optional[str] = Field(
        default=None,
        description=(
            "ISO-8601 (YYYY-MM-DD) normalized date inferred by LLM "
            "using reference_dttm context for year disambiguation. "
            "None when phrase is None OR phrase is genuinely "
            "unparseable. Used by the pin-block renderer for "
            "chronological sort — NEVER displayed to clinicians."
        ),
    )
    end_date_phrase: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim chart literal for end date. Same format "
            "conventions as start_date_phrase. None when ongoing "
            "(use status='ongoing_outside_window') or not documented."
        ),
    )
    end_date_parsed: Optional[str] = Field(
        default=None,
        description=(
            "ISO-8601 normalized end date. Same semantics as "
            "start_date_parsed."
        ),
    )

    indication: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim MEDICAL REASON the drug was given (e.g. "
            "'Klebsiella UTI', 'HCAP', 'empiric for sepsis'). NOT a "
            "catch-all narrative slot — switch/intolerance/dose-change "
            "context goes in `notes`. None when not documented in "
            "source — do NOT fill in plausible-sounding indications."
        ),
    )

    notes: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim chart context about the COURSE itself — switch "
            "reason, intolerance, dose change, organism speciation "
            "update. NOT the medical indication (that's `indication`). "
            "Example: 'Red man syndrome — switched to daptomycin'. "
            "notes MUST be a substring of source_quote (transitively "
            "anchoring it to chart text); the scribe's runtime validator "
            "CLEARS an unanchored notes value (the course is kept)."
        ),
    )

    status: Literal["completed", "ongoing_outside_window"] = Field(
        description=(
            "Ownership boundary. 'completed' SHOULD have end_date_phrase "
            "populated; a missing end date on a completed course is "
            "surfaced as a runtime coverage flag (no longer a hard reject). "
            "'ongoing_outside_window' = course started >48h ago AND "
            "still active in the structured-data window (rare — "
            "pharmacy will see the recent doses; this entry captures "
            "the pre-window start). Active courses fully within the "
            "48h window MUST NOT appear here — they're owned by "
            "meds.states. NO DEFAULT: LLM omission = Pydantic "
            "validation failure."
        ),
    )

    source_quote: str = Field(
        description=(
            "Verbatim chart literal anchoring this course (e.g. "
            "'cefepime (3/3 - 3/5) for Klebsiella UTI'). Runtime "
            "validator substring-matches source_quote against the "
            "body of source_note_id ONLY (per-source, not corpus-wide). "
            "Schema enforces: (a) drug name appears in source_quote, "
            "(b) notes (if present) is a substring of source_quote."
        ),
    )
    source_note_id: str = Field(
        description=(
            "note_id of the routed note whose body source_quote came "
            "from. Used by the per-source substring guard and the "
            "patient-identity guard in _validate_admission_antibiotics."
        ),
    )

    # NOTE (2026-06-08, scribe-parse-wipeout fix): three model_validators were
    # REMOVED here and handled gracefully in
    # ``ScribeAgent._validate_admission_antibiotics``. Raising at Pydantic
    # parse time failed the ENTIRE ``ScribeExtractionLLM`` object — silently
    # dropping a perfectly good PMH (and allergies/home_meds/code_status) —
    # whenever ONE antibiotic course was malformed (e.g. drug='amphotericin B'
    # vs note 's/p amphotericin'). Replacements, per-course (drop/flag the
    # offending course only, never the whole extraction):
    #   * _drug_name_in_source_quote      -> already enforced (drug_appears_in_text)
    #   * _notes_must_be_in_source_quote  -> ported: unanchored `notes` is cleared
    #   * _completed_courses_must_have_end_date -> downgraded to coverage_log flag
    # See memory: project_icu_pause_scribe_parse_wipeout.


class RenalContext(BaseModel):
    """Renal context extracted by scribe — baseline + KDIGO + UOP +
    nephrology status + RRT indications.

    Designed for partial population: any subfield can be None
    independently. The whole field is dropped only when EVERY subfield
    fails validation; the per-subfield drop audit lives on
    ``ScribeExtraction.renal_context_partial_drops``.

    Source priority is per-subfield per ``renal_electrolyte_vte_extraction_design.md``
    §4.0.1: baseline_creatinine + baseline_creatinine_date → AUTHORITATIVE-WINS
    over {H&P, nephrology consult}; kdigo_stage / urine_output_pattern /
    nephrology_status → MOST-RECENT-WINS; rrt_indications_documented →
    ACCUMULATIVE union across the admission.

    The ``kdigo_stage`` field carries chart text only — the LLM does NOT
    compute KDIGO. Deterministic Path A / Path B computation lives in
    ``orchestrator._compute_renal_delta_and_kdigo_path_a_b`` per §4.3.5 of
    the design doc, including the v3.1 ``latest_cr ≥ 4.0`` gating fix that
    prevents chronic-ESRD mis-staging.
    """

    baseline_creatinine: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim baseline-Cr anchor quoted from a routed note: "
            "'1.4', '1.4 (per OSH records)', '1.5-1.7 baseline range'. "
            "None if no chart anchor exists. AUTHORITATIVE-WINS source "
            "priority: prefer H&P or nephrology consult over progress "
            "notes; within the authoritative set, most recent wins."
        ),
    )
    baseline_creatinine_date: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim date attached to the baseline-Cr reference: "
            "'2024-01-15 outpatient labs', 'per H&P, no date specified'. "
            "None when date is not stated. Must come from the same source "
            "note as baseline_creatinine."
        ),
    )
    kdigo_stage: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim chart-documented KDIGO stage: 'KDIGO Stage 2', "
            "'AKI Stage 3 per nephrology'. None when no chart assignment. "
            "DO NOT compute — code owns the KDIGO computation via Path A "
            "(long-window ratio against chart_baseline) and Path B (48h "
            "short-window against structured Cr). MOST-RECENT-WINS within "
            "the routed notes."
        ),
    )
    urine_output_pattern: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim UOP pattern characterization: 'non-oliguric, UOP "
            "1500/24h', 'oliguric, UOP 200/24h', 'anuric x12h'. None "
            "when no narrative discussion. MOST-RECENT-WINS. NOTE: v3.1 "
            "renders this as documentation only — code does NOT yet "
            "compute UOP-based KDIGO (deferred to R11 per design doc "
            "§11.6); the AKI #Problem render is Cr-based with UOP "
            "narrative appended."
        ),
    )
    nephrology_status: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim nephrology consult / engagement status: "
            "'nephrology consulted 5-24, recs continue current "
            "management', 'no nephrology involvement', 'RRT being "
            "considered per renal'. None when not addressed in source. "
            "MOST-RECENT-WINS — consults engage and disengage; the "
            "latest documented state is the active one."
        ),
    )
    rrt_indications_documented: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim AEIOU RRT triggers when documented anywhere in "
            "the admission: 'refractory hyperK', 'refractory acidosis', "
            "'refractory volume', 'uremic symptoms', 'intoxication'. "
            "None when none documented. ACCUMULATIVE union across all "
            "routed notes — any indication ever documented is "
            "receiver-relevant for RRT-planning decisions."
        ),
    )
    baseline_source_quote: Optional[str] = Field(
        default=None,
        description=(
            "Contiguous source span containing baseline_creatinine for "
            "substring validation. None when baseline_creatinine is "
            "None. Required (non-None) when baseline_creatinine is "
            "non-None — used by ``ScribeAgent._validate_renal_context`` "
            "to anchor the baseline against routed-note text."
        ),
    )


class ScribeExtractionLLM(BaseModel):
    """LLM-facing slim schema for the scribe agent.

    The scribe's LLM call returns ONLY these two fields. The full
    ``ScribeExtraction`` (with validator metadata) is constructed in code
    after the orchestrator-side substring + patient-identity guard runs.

    ``populate_by_name=True`` lets callers use either the Python field
    name (``subspecialty_consults``) or the JSON alias
    (``active_consults``) when constructing or loading. Backward-compat
    for pre-2026-05-29 serialized briefs / trace files. Alias removal is
    deferred until the pilot corpus is migrated; field aliases stay
    separately from the class-level alias (which was removed in the
    same revision).
    """

    model_config = ConfigDict(populate_by_name=True)

    pmh: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim PMH paragraph or None. See scribe.yaml for the "
            "spine+append rule and verbatim-quoting hard rules."
        ),
    )
    pmh_sources: list[str] = Field(
        default_factory=list,
        description="note_id values that contributed to pmh.",
    )
    allergies: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim allergy list (e.g. 'PCN - rash, Sulfa - hives' or "
            "'NKDA') or None. See scribe.yaml for the verbatim-quoting "
            "hard rules — do not paraphrase 'no known drug allergies' to "
            "'NKDA' or vice versa; quote the chart literal."
        ),
    )
    allergies_sources: list[str] = Field(
        default_factory=list,
        description="note_id values that contributed to allergies.",
    )
    home_meds: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim outpatient medication list quoted from the chart "
            "(typically the H&P Home Medications section), or None when "
            "no such section is present in any routed note. The pharmacy "
            "agent uses this as the OUTPATIENT anchor for the 'Changes to "
            "home meds' diff in U_unprescribing. See scribe.yaml for the "
            "additive multi-note rule and verbatim hard rules."
        ),
    )
    home_meds_sources: list[str] = Field(
        default_factory=list,
        description="note_id values that contributed to home_meds.",
    )
    code_status: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim code-status string quoted from the most recent "
            "note that documents it (e.g. 'Full Code', 'DNR/DNI', "
            "'Comfort Care'), or None when no routed note carries an "
            "explicit code-status statement. The case_manager agent "
            "uses this as the authoritative source for the 'Code "
            "Status:' line in Section C. Multi-note rule is "
            "MOST-RECENT-WINS (unlike PMH/allergies/home_meds) "
            "because code status changes chronologically — admission "
            "Full Code → family meeting DNR/DNI is a real transition, "
            "not a documentation conflict."
        ),
    )
    code_status_sources: list[str] = Field(
        default_factory=list,
        description="note_id values that contributed to code_status.",
    )
    subspecialty_consults: Optional[list[SubspecialtyConsult]] = Field(
        default=None,
        alias="active_consults",
        description=(
            "Subspecialty consultants identified by either (a) "
            "consult-note metadata in the lookback window, (b) "
            "present-tense engagement frames in the most recent "
            "progress-note A&P, or (c) a state-attribution quote in any "
            "routed note (for non-Active states). Null when no consult "
            "evidence of any state is found. The intensivist uses this "
            "list as the source for Section A's 'Subspecialty "
            "Consultants:' sub-block (merged with therapist's "
            "PT/OT/SLP and case_manager's palliative/social work). See "
            "scribe.yaml for the extraction rules and sign-off "
            "exclusion patterns. Renamed from ``active_consults`` in "
            "the 2026-05-29 Section A revision; ``alias=\"active_consults\"`` "
            "preserves backward-compat deserialization of pre-rename "
            "serialized briefs / trace files. Field alias removal is "
            "deferred until the pilot brief corpus is migrated or "
            "replayed; class-level alias was removed concurrent with "
            "this rename."
        ),
    )
    admission_antibiotics: Optional[list[AdmissionAntibioticCourse]] = Field(
        default=None,
        description=(
            "Inpatient antibiotic courses given THIS ADMISSION and "
            "OUTSIDE the 48h structured-data window (i.e. earlier-in-"
            "admission completed courses that pharmacy's structured "
            "med tables don't carry). None when no historical courses "
            "are identified — NOT a signal of 'no antibiotics' (active "
            "drugs are owned by meds.states, not scribe). See "
            "scribe.yaml ADMISSION ANTIBIOTICS RULE for the source-"
            "priority order (ID consult > pharmacy consult > most-"
            "recent comprehensive A&P > H&P > other consults) and the "
            "most-recent-comprehensive-wins multi-note rule."
        ),
    )
    renal_context: Optional[RenalContext] = Field(
        default=None,
        description=(
            "Renal-context bundle: baseline creatinine + date, "
            "chart-documented KDIGO stage (DO NOT compute), urine "
            "output pattern, nephrology consult status, documented "
            "AEIOU RRT indications. Each subfield is independently "
            "drop-able; the scribe emits the field if ANY subfield is "
            "populatable from routed notes. Source priority is "
            "per-subfield (see RenalContext docstring + design doc "
            "§4.0.1). The intensivist pins this for the AKI/CKD "
            "#Problem in S; the orchestrator computes KDIGO Path A / "
            "Path B + delta against structured Cr."
        ),
    )
    renal_context_sources: list[str] = Field(
        default_factory=list,
        description=(
            "note_id values that contributed to any populated "
            "renal_context subfield. Used by the patient-identity guard "
            "before the field is pinned to the intensivist's S input."
        ),
    )


class ScribeExtraction(BaseModel):
    """Structured patient-context extraction from the Scribe agent.

    Phase-0 step in the pipeline. The scribe reads canonical chart sources
    (H&P primary; progress + social work + case management secondary) and
    extracts fields the downstream intensivist would otherwise need to
    locate in long SOAP notes — where attention degrades and confabulation
    risk is high.

    The ``pmh`` field is a verbatim quote (no abbreviation expansion, no
    reordering, no paraphrasing). ``pmh_sources`` lists the note_ids that
    contributed for the orchestrator-side substring + patient-identity
    guard. ``pmh_validated`` is set by the scribe's own self-validator
    after substring-checking each clause against the routed notes; the
    intensivist only pins PMH when this is True.

    See docs/pmh_structured_extraction_design.md.

    ``populate_by_name=True`` enables both the Python field name
    (``subspecialty_consults`` / ``subspecialty_consults_validated`` /
    ``subspecialty_consults_dropped_reason``) and the legacy JSON aliases
    (``active_consults*``) for input. Backward-compat for pre-2026-05-29
    serialized data; alias removal deferred until corpus migration.
    """

    model_config = ConfigDict(populate_by_name=True)

    pmh: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim PMH paragraph quoted from routed notes, or None when "
            "no condition-enumerating sentence is present in any routed "
            "note. Multi-note rule: longest/most-recent paragraph as the "
            "spine, with conditions named in other routed notes appended "
            "if missing from the spine."
        ),
    )
    pmh_sources: list[str] = Field(
        default_factory=list,
        description=(
            "note_id values for every routed note that contributed a "
            "condition to the PMH string. Used by the substring + "
            "patient-identity guard before the PMH is pinned to the "
            "intensivist's Section I input."
        ),
    )
    pmh_validated: bool = Field(
        default=False,
        description=(
            "True only when every condition clause in pmh has been "
            "substring-matched against at least one routed note AND every "
            "pmh_sources note_id resolves to the target patient_id. False "
            "drops the field and the intensivist falls back to the "
            "'chart review required' string."
        ),
    )
    pmh_dropped_reason: Optional[str] = Field(
        default=None,
        description=(
            "When pmh_validated is False because validation rejected an "
            "LLM-emitted PMH (rather than the LLM legitimately finding "
            "nothing), this records WHY for audit. None when the LLM "
            "emitted no PMH in the first place."
        ),
    )
    allergies: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim allergy list (e.g. 'PCN - rash, Sulfa - hives' or "
            "'NKDA') quoted from a routed note's Allergies section, or "
            "None when no Allergies header is present in any routed note. "
            "Safety-critical field — every clinical handoff framework "
            "(SBAR, I-PASS) treats allergies as standard-of-care content."
        ),
    )
    allergies_sources: list[str] = Field(
        default_factory=list,
        description=(
            "note_id values for every routed note that contributed to "
            "allergies. Used by the substring + patient-identity guard "
            "before the allergies value is pinned to Section I."
        ),
    )
    allergies_validated: bool = Field(
        default=False,
        description=(
            "True only when every comma/semicolon-separated allergy clause "
            "has been substring-matched against at least one routed note "
            "AND every allergies_sources note_id resolves to the target "
            "patient_id. False drops the field and Section I falls back "
            "to 'Allergies not documented in available notes'."
        ),
    )
    allergies_dropped_reason: Optional[str] = Field(
        default=None,
        description=(
            "When allergies_validated is False because validation rejected "
            "an LLM-emitted allergies string (not because the LLM found "
            "nothing), records WHY for audit. None when the LLM emitted no "
            "allergies value in the first place."
        ),
    )
    home_meds: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim outpatient medication list quoted from a routed "
            "note's Home Medications section (typically the H&P), or None "
            "when no such section is present in any routed note. Consumed "
            "by the pharmacy agent as the OUTPATIENT anchor for the "
            "'Changes to home meds' diff against the inpatient med admin "
            "tables (meds.states)."
        ),
    )
    home_meds_sources: list[str] = Field(
        default_factory=list,
        description=(
            "note_id values for every routed note that contributed to "
            "home_meds. Used by the substring + patient-identity guard "
            "before the home_meds value is pinned to pharmacy's input."
        ),
    )
    home_meds_validated: bool = Field(
        default=False,
        description=(
            "True only when every comma/semicolon-separated medication "
            "clause in home_meds has been substring-matched against at "
            "least one routed note AND every home_meds_sources note_id "
            "resolves to the target patient_id. False drops the field "
            "and pharmacy falls back to 'home medications not documented "
            "in available notes — chart review required'."
        ),
    )
    home_meds_dropped_reason: Optional[str] = Field(
        default=None,
        description=(
            "When home_meds_validated is False because validation "
            "rejected an LLM-emitted home_meds string (not because the "
            "LLM found nothing), records WHY for audit. None when the "
            "LLM emitted no home_meds value in the first place."
        ),
    )
    code_status: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim code-status string quoted from the most recent "
            "routed note that documents it. Consumed by case_manager "
            "as the authoritative source for the 'Code Status:' line "
            "in Section C. None when no routed note has an explicit "
            "code-status statement — the case_manager then falls back "
            "to the structured clif_code_status parquet (if any) or "
            "to 'Not documented'."
        ),
    )
    code_status_sources: list[str] = Field(
        default_factory=list,
        description=(
            "note_id values for every routed note that contributed to "
            "code_status. Used by the substring + patient-identity "
            "guard before the code_status value is pinned to "
            "case_manager's input."
        ),
    )
    code_status_validated: bool = Field(
        default=False,
        description=(
            "True only when the verbatim code_status string has been "
            "substring-matched against at least one routed note AND "
            "every code_status_sources note_id resolves to the target "
            "patient_id. False drops the field and case_manager falls "
            "back to its existing parquet/notes reconciliation."
        ),
    )
    code_status_dropped_reason: Optional[str] = Field(
        default=None,
        description=(
            "When code_status_validated is False because validation "
            "rejected an LLM-emitted value (not because the LLM found "
            "nothing), records WHY for audit. None when the LLM "
            "emitted no code_status value in the first place."
        ),
    )
    subspecialty_consults: Optional[list[SubspecialtyConsult]] = Field(
        default=None,
        alias="active_consults",
        description=(
            "Subspecialty consultants quoted from routed notes, or None "
            "when no consult-note metadata, progress-note A&P engagement "
            "frame, or state-attribution quote is present in any routed "
            "note. Consumed by the intensivist as the source for "
            "Section A's 'Subspecialty Consultants:' sub-block. Distinct "
            "from PT / OT / SLP / Wound Care (therapist's lane) and "
            "Palliative / Social Work (case_manager's lane). Renamed "
            "from ``active_consults`` in the 2026-05-29 Section A "
            "revision; alias preserves backward-compat deserialization."
        ),
    )
    subspecialty_consults_validated: bool = Field(
        default=False,
        alias="active_consults_validated",
        description=(
            "True only when every emitted consult passed the validator "
            "guards (per-service word-boundary for Active; lenient "
            "quote substring + citation existence for non-Active) AND "
            "every source_note_id resolves to the target patient_id. "
            "False drops the field and Section A falls back to the "
            "existing therapist + case_manager merge (which won't "
            "include ID / Nephrology / etc — that's the structural gap "
            "this field exists to close). Renamed from "
            "``active_consults_validated`` in the 2026-05-29 revision; "
            "alias preserves backward-compat."
        ),
    )
    subspecialty_consults_dropped_reason: Optional[str] = Field(
        default=None,
        alias="active_consults_dropped_reason",
        description=(
            "When subspecialty_consults_validated is False because "
            "validation rejected an LLM-emitted list (not because the "
            "LLM found nothing), records WHY for audit. None when the "
            "LLM emitted an empty / null list in the first place. "
            "Renamed from ``active_consults_dropped_reason`` in the "
            "2026-05-29 revision; alias preserves backward-compat."
        ),
    )
    admission_antibiotics: Optional[list[AdmissionAntibioticCourse]] = Field(
        default=None,
        description=(
            "Validated list of inpatient antibiotic courses given THIS "
            "ADMISSION and OUTSIDE the 48h structured-data window. None "
            "when the LLM emitted nothing OR every course was rejected "
            "by the runtime validator. Distinct from active drugs "
            "(owned by meds.states in structured data). Consumed by "
            "pharmacy's _format_scribe_pins hook (Phase 3 PR)."
        ),
    )
    admission_antibiotics_validated: bool = Field(
        default=False,
        description=(
            "True only when at least one course passed all validator "
            "guards: per-source source_quote substring-match against "
            "source_note_id's body, >=15-char specificity floor, drug "
            "name appearing in source_quote, AND every source_note_id "
            "resolving to the target patient_id. False drops the field "
            "(pharmacy's pin block falls back to empty — same posture "
            "as the other scribe fields)."
        ),
    )
    admission_antibiotics_dropped_reason: Optional[str] = Field(
        default=None,
        description=(
            "When admission_antibiotics_validated is False because "
            "validation rejected the LLM emission (rather than the LLM "
            "legitimately finding nothing), records WHY for audit. "
            "Includes a per-course rejection list when partial. None "
            "when the LLM emitted no admission_antibiotics list in the "
            "first place."
        ),
    )
    renal_context: Optional[RenalContext] = Field(
        default=None,
        description=(
            "Validated renal context (baseline Cr + KDIGO + UOP + "
            "nephrology status + RRT indications). Individual subfields "
            "may be None even when the bundle is present — partial "
            "population is the design (any subfield can validate or "
            "drop independently). None ONLY when every subfield failed "
            "validation; see renal_context_partial_drops for per-"
            "subfield audit when bundle is partially populated."
        ),
    )
    renal_context_sources: list[str] = Field(
        default_factory=list,
        description=(
            "note_id values for every routed note that contributed to "
            "any populated renal_context subfield. Used by the "
            "patient-identity guard before the field is pinned to "
            "intensivist's S input."
        ),
    )
    renal_context_validated: bool = Field(
        default=False,
        description=(
            "True when at least one renal_context subfield passed its "
            "substring guard AND every renal_context_sources note_id "
            "resolves to the target patient_id. False drops the entire "
            "bundle; intensivist's pin block then falls back to the "
            "'no baseline anchor in source' fallback string."
        ),
    )
    renal_context_dropped_reason: Optional[str] = Field(
        default=None,
        description=(
            "When renal_context_validated is False because validation "
            "rejected the LLM emission (rather than the LLM legitimately "
            "finding nothing), records WHY for audit. None when the LLM "
            "emitted no renal_context in the first place."
        ),
    )
    renal_context_partial_drops: list[str] = Field(
        default_factory=list,
        description=(
            "Per-subfield drop audit when the renal_context bundle is "
            "partially populated. Each entry names the dropped subfield "
            "and the reason ('kdigo_stage: clause not found in routed "
            "notes', 'baseline_creatinine: source_quote missing'). "
            "Empty when no partial drops occurred."
        ),
    )


class CitationRow(BaseModel):
    """One source row backing a citation tag.

    A single tag (e.g. ``"(vital 7-20 06:00)"``) can resolve to multiple
    rows when several measurements share a timestamp/bucket — e.g. an exam
    summary that cites BP, HR, MAP, SpO2 at the same moment maps to four
    distinct vital rows under one tag. ``CitationEntry.rows`` holds them
    all so the tooltip can render every sibling, not just the first.

    ``time`` is the per-row timestamp (ISO format). Defaults to None so
    legacy on-disk output.json files (pre-Phase-3) still validate; new
    code populates it so the tooltip can show per-vital timestamps when
    they differ from the tag's anchor (the new exam-* source types
    aggregate values across a window and can have rows at different
    moments).
    """

    label: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    time: Optional[str] = None


class CitationEntry(BaseModel):
    """One resolved source record for a citation tag in the rendered brief.

    Keyed in ``ICUPauseOutput.metadata['citation_index']`` by the raw tag
    string (e.g. ``"(vital 1-12 07:00)"``).  Renderers use this to show a
    hoverable tooltip instead of the inline parenthetical.

    ``tier`` is two-valued for now:
      * ``decision_critical`` — tag resolves to a real row in ``cite_registry``.
        Under the current default ``citation_mode='decision_critical'`` this is
        every resolved tag by construction.
      * ``unverified`` — tag appears in the text but does not resolve.  Render
        with a warning affordance so the reviewer notices.

    ``rows`` is the authoritative list of source rows that backed this tag.
    For a tag with N siblings (multiple vitals at the same bucket, say),
    ``rows`` has N entries; older consumers that only read ``label``/``value``/
    ``unit`` continue to work — those fields mirror ``rows[0]`` for backwards
    compatibility with on-disk output.json files.
    """

    source_type: Literal[
        "lab", "vital", "med", "resp", "assess", "code", "proc",
        # Phase 3: deterministic transfer-exam sub-blocks. Each tag
        # resolves to multiple rows spanning a 4 h window before
        # reference_dttm — per-row timestamps can differ from the tag
        # anchor.
        "exam-vitals", "exam-neuro", "exam-resp",
        # Routed-note source types — one per AGENT_NOTE_ROUTING key.
        # Tag shape "(<note_type> M-DD HH:MM)" with format parity to
        # structured types. cite_registry rows hold the routed-note row
        # by reference; trim → _trim_note in citation_index.py.
        "progress_note", "hp_note", "consults_note", "plan_of_care_note",
        "nursing_note", "case_management_note",
        "social_work_note", "therapy_note",
    ]
    time: str  # ISO local timestamp of the source record
    label: Optional[str] = None  # mirrors rows[0].label — kept for back-compat
    value: Optional[str] = None  # mirrors rows[0].value — kept for back-compat
    unit: Optional[str] = None   # mirrors rows[0].unit — kept for back-compat
    rows: list[CitationRow] = Field(default_factory=list)
    tier: Literal["decision_critical", "unverified"] = "decision_critical"


class ICUPauseOutput(BaseModel):
    """The complete ICU-PAUSE handoff brief."""

    hospitalization_id: str
    generated_at: str
    sections: dict[str, str]  # section key -> merged text
    todo_checklist: list[dict[str, str]] = Field(
        default_factory=list,
        description="Each item: {'bucket': 'pre_transfer'|'ward_ongoing'|'discharge', 'text': str}",
    )
    warnings: list[Warning] = Field(default_factory=list)
    qa_issues: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_warnings(cls, values: Any) -> Any:
        if isinstance(values, dict) and "warnings" in values:
            values["warnings"] = _coerce_warning_list(values["warnings"])
        return values
    section_confidences: dict[str, float] = Field(
        default_factory=dict,
        description="Per-section confidence scores (0.0-1.0) from agent contributions",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
