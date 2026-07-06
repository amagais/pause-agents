"""LangGraph state definition for the ICU-PAUSE workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional

from typing_extensions import TypedDict

from icu_pause.schemas.icu_pause import AgentSnippet, IntensivistOutput


class GraphState(TypedDict):
    """Central state flowing through the LangGraph workflow.

    Fields set at different stages:
    - hospitalization_id: set at invocation
    - lookback_hours: set at invocation (default 48; None = entire stay)
    - reference_dttm: set at invocation (ISO string or None for prospective)
    - notes_lookback_hours: set at invocation (overrides settings default)
    - patient_context_text: set by data_retrieval node (shared structured data)
    - agent_context_text: set by data_retrieval node (per-agent context with
      agent-specific notes routed in)
    - agent_snippets: accumulated by domain agents (fan-out → fan-in)
    - qa_issues / qa_passed: set by QA agent
    - icu_pause_output: set by section merger
    """

    # Input
    hospitalization_id: str
    lookback_hours: Optional[int]
    reference_dttm: Optional[str]  # ISO-8601 string; None = prospective / auto-detect
    notes_lookback_hours: Optional[int]  # Override for notes-specific lookback

    # Set by data retrieval (deterministic) -- values are JSON-serializable structures
    patient_context_text: dict[str, Any]

    # Per-agent serialized context (agent_role -> dict with agent-specific notes)
    agent_context_text: dict[str, dict[str, Any]]

    # Fusion mode: "early_fusion" (default) or "cr_dsf" (Cross-Referenced Dual-Stream)
    fusion_mode: str

    # Resolved compression axes (compression sub-study). base.py composes the
    # agent message from these: structured_axis ∈ {s0,s1,s2}, notes_axis ∈
    # {n0,n1,n2}. Derived from fusion_mode | explicit axes via Settings.resolved_cell.
    structured_axis: str
    notes_axis: str

    # Set by interpreter agents (s1 / n1)
    structured_summaries: dict[str, str]  # agent_name -> structured data summary
    note_summaries: dict[str, str]  # agent_name -> clinical note summary

    # Set by the structured salience selector (s2). Substitutive per-agent view of
    # the SAME tiered tables s0 produces; read by base.py in place of raw tables.
    structured_views: dict[str, str]  # agent_name -> salience-selected view

    # Set by structured extractor (CR-DSF+ mode only)
    extraction_fields: dict[str, dict[str, str]]  # agent_name -> {field: value}

    # Set by per-domain extractors (hybrid_v1 mode only). Each per-domain
    # extractor writes its own domain's slice; the reducer merges them.
    # Read by the Stage E anchor-override wrapper applied to domain agents
    # in the hybrid_v1 workflow branch.
    per_domain_extractions: Annotated[
        dict[str, dict[str, str]],
        lambda a, b: {**(a or {}), **(b or {})},
    ]

    # Accumulated by domain agents via operator.add reducer
    agent_snippets: Annotated[list[AgentSnippet], operator.add]

    # Accumulated timing/token metrics from each node
    pipeline_metrics: Annotated[list[dict], operator.add]

    # Set by risk predictor (runs in parallel with domain agents)
    risk_score: Optional[dict[str, Any]]  # 72h readmission/mortality risk

    # Set by QA agent
    qa_issues: list[str]  # Clinical issues (physician-facing)
    qa_scope_issues: list[str]  # Scope violations (system-internal, not shown to physician)
    qa_passed: bool

    # Set by deliberation node (when enabled)
    revised_snippets: Annotated[list[AgentSnippet], operator.add]
    deliberation_log: Annotated[list[dict], operator.add]

    # Set by Resident agent (pre-synthesis brief for Intensivist)
    resident_pre_brief: Optional[dict[str, Any]]

    # Set by Scribe agent (phase-0 structured extraction of PMH from
    # canonical chart sources). Read by the intensivist to pin a labeled
    # PMH header above its Section I prose, avoiding long-note attention
    # failure. ScribeExtraction model — see schemas/icu_pause.py.
    scribe_extraction: Optional[dict[str, Any]]

    # Set by Intensivist agent (harmonized clinical narrative).
    # Uses IntensivistOutput (superset of AgentSnippet) so reasoning_log
    # flows through to downstream consumers / diagnostics.
    intensivist_output: Optional[IntensivistOutput]

    # Set by section merger
    icu_pause_output: dict[str, Any]

    # Citation registry: cite string → list of source row dicts.
    # Populated by data_retrieval when citation_mode != "off".
    cite_registry: dict[str, list[dict[str, Any]]]

    # Run trace (accumulated debug events)
    trace_events: Annotated[list[dict], operator.add]

    # Per-agent physician-note floor metadata. Populated by data_retrieval
    # for the floor-eligible agents (intensivist, respiratory, pharmacy,
    # dietitian). Surfaced to ICUPauseOutput.metadata.physician_floor by
    # the section merger.
    physician_floor: dict[str, dict[str, Any]]
