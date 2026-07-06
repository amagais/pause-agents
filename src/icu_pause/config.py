from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

# Allowed values for the two independent compression axes (pre-reg compression
# sub-study). The structured axis and notes axis are three representations each
# of the SAME per-agent tiered tables / routed notes; they vary one factor.
STRUCTURED_AXES = ("s0", "s1", "s2")
NOTES_AXES = ("n0", "n1", "n2")


def axes_from_fusion_mode(fusion_mode: str) -> tuple[str, str]:
    """Map a legacy fusion_mode enum to its (structured_axis, notes_axis) cell.

    Single source of truth shared by Settings.resolved_cell and the
    BaseDomainAgent.run state-fallback (when a hand-built state carries only
    fusion_mode and not the resolved axes). Anchor-override is handled
    separately (it is orthogonal to the representation grid).
    """
    if fusion_mode == "hybrid_v1":
        return ("s0", "n0")  # + Stage-E anchor (orthogonal toggle)
    if fusion_mode in ("cr_dsf", "cr_dsf_plus"):
        return ("s1", "n1")
    return ("s0", "n0")  # early_fusion + unknown


@dataclass(frozen=True)
class ResolvedCell:
    """The fully-resolved compression behavior for a run.

    Centralizes the (fusion_mode | explicit axes) → behavior mapping so base.py
    and workflow.py read booleans instead of re-deriving the fiddly anchor /
    extractor logic. See Settings.resolved_cell.
    """

    structured_axis: str  # s0 | s1 | s2
    notes_axis: str  # n0 | n1 | n2
    run_extractors: bool  # per-domain N2 extractors run (for prompt anchors OR Stage-E)
    apply_anchor_override: bool  # Stage-E output reconciliation against extracted anchors


# ---------------------------------------------------------------------------
# Note-to-agent routing map
# Keys = agent role names, values = list of note-type keys (matching
# NOTE_FILE_MAP keys below).  This is the single source of truth for which
# agents see which note types.
# ---------------------------------------------------------------------------
# hp_note is routed to every clinical agent (not only the scribe / pharmacy /
# case_manager) following the 2026-05-26 clinical-reviewer decision in
# docs/hp_note_routing_question.md. Per-agent rationale for the H&P narrative
# is documented inline; the scribe still produces structured PMH / allergies /
# home-meds / code-status pins, and those tracks remain orthogonal — the raw
# H&P gives each agent narrative the scribe extraction does not capture
# (HPI chronology, exam baseline, PLOF, social hx detail, A&P reasoning).
AGENT_NOTE_ROUTING: dict[str, list[str]] = {
    # nurse needs baseline mental status (delirium / CAM-ICU anchor), baseline
    # skin exam (HAPI reference point), baseline mobility / pain, and
    # language / interpreter needs from the admission H&P.
    "nurse": ["nursing_note", "progress_note", "hp_note"],
    # respiratory needs home O2, home BiPAP/CPAP settings, prior intubation /
    # trach history, pack-years, and baseline exercise tolerance.
    "respiratory": ["progress_note", "consults_note", "hp_note"],
    # pharmacy: H&P Home Medications is the OUTPATIENT anchor for the
    # U_unprescribing diff; predates the 2026-05-26 expansion. See
    # docs/home_meds_design_pass.md.
    "pharmacy": ["progress_note", "consults_note", "plan_of_care_note", "hp_note"],
    # dietitian needs weight change, oral intake decline, dysphagia history,
    # nutrition baseline — all narrative content typically only present in
    # the admission H&P.
    "dietitian": ["consults_note", "plan_of_care_note", "progress_note", "hp_note"],
    # case_manager: H&P code-status / GOC narrative is the fallback when no
    # later GOC discussion is documented; predates the 2026-05-26 expansion.
    "case_manager": ["case_management_note", "social_work_note", "progress_note", "hp_note"],
    # therapist (PT/OT/SLP) needs prior level of function, mobility,
    # equipment, social / living context — PLOF is rarely captured anywhere
    # else.
    "therapist": ["therapy_note", "progress_note", "hp_note"],
    # intensivist needs HPI chronology, admitting A&P reasoning, and exam
    # baseline before sedation / lines — the elements that put the structured
    # data in clinical context.
    "intensivist": ["progress_note", "consults_note", "hp_note"],
    # Scribe: phase-0 structured extraction of PMH, allergies, home meds,
    # code status, and active subspecialty consultants. Routed to the
    # canonical PMH/allergies/code-status/home-meds sources (H&P primary,
    # progress + social work + case management secondary) PLUS consults_note
    # for the subspecialty_consults extraction's primary signal
    # (consult-note metadata). See docs/pmh_structured_extraction_design.md.
    "scribe": [
        "hp_note", "progress_note", "social_work_note",
        "case_management_note", "consults_note",
    ],
}

# Note types that document a per-admission stable fact and are exempt from
# the ``notes_lookback_hours`` filter. These are written once at admission
# (or once per goals-of-care decision) and remain valid for the rest of the
# stay; gating them on a 48h recency window silently drops them for any
# patient whose ICU stay exceeds the lookback before the reference_dttm.
#
# Currently only ``hp_note`` is wired through the pipeline as a note type.
# Future members anticipated by the 2026-05-26 clinical-reviewer decision —
# code-status orders, advance directives, admission imaging reads — should be
# added here once the corresponding note_type keys exist in
# ``DEFAULT_NOTE_FILE_MAP`` and ``AGENT_NOTE_ROUTING``. Do NOT pre-add
# entries whose underlying source schema is not yet defined.
#
# The per-type cap (``AGENT_MAX_NOTES_PER_TYPE``) and the leakage guard
# (``creation_dttm < reference_dttm``) still apply — exemption is from the
# recency floor only.
PER_ADMISSION_STABLE_NOTE_TYPES: frozenset[str] = frozenset({"hp_note"})


# Stable identifier for the current note-routing configuration. Bump when
# ``AGENT_NOTE_ROUTING`` changes in a way that affects what agents see, so
# downstream A/B comparisons across runs can group inputs by config. Bump
# the date suffix; do NOT recycle a version label.
#
# History:
#   hp_v1_baseline      — pre-2026-05-26. hp_note routed only to pharmacy,
#                         case_manager, scribe.
#   hp_v2_2026-05-26    — clinical-reviewer decision in
#                         docs/hp_note_routing_question.md. hp_note added
#                         to intensivist, nurse, respiratory, dietitian,
#                         therapist. Per-admission-stable lookback exemption
#                         introduced (PER_ADMISSION_STABLE_NOTE_TYPES).
NOTE_ROUTING_VERSION: str = "hp_v2_2026-05-26"

# Maximum notes PER note type retained per agent (most recent kept, oldest
# dropped). Calibrated empirically from the post-deduplication note count
# distribution observed across 426 MICU transfer-eligible encounters in the
# 48h pre-transfer window. See `docs/note_cap_empirical_distribution.csv`
# and the methods note in `docs/ARCHITECTURE.md`.
#
# Default cap = p95 of the empirical distribution per type, except
# progress_note which is set at p90 (covers ≥90% of patients) because it
# routes to four agents and the 90→95 increment costs ~3k aggregate tokens
# per run for marginal coverage gain on 5% of patients.
#
# case_management_note retained at 3 pending non-MICU validation (zero
# observed in MICU cohort; placeholder for surgical/cardiothoracic).
AGENT_MAX_NOTES_PER_TYPE: dict[str, int] = {
    "nursing_note":         2,   # p95 (MICU empirical)
    "progress_note":        5,   # p90 (routes to 4 agents; token-cost tradeoff)
    "consults_note":        3,   # p95
    "plan_of_care_note":    1,   # p95
    "case_management_note": 3,   # placeholder, pending non-MICU validation
    "social_work_note":     1,   # p95
    "therapy_note":         3,   # p95
    # H&P is stable across an admission — agents rarely need more than one
    # version, and revisions are typically formatting/typo fixes. Cap at 2
    # to allow both an initial H&P and a documented re-admit H&P when
    # multiple MICU stays exist. This cap applies uniformly across all
    # agents that receive ``hp_note`` (intensivist, nurse, respiratory,
    # pharmacy, dietitian, case_manager, therapist, scribe — expanded
    # 2026-05-26), so the post-decision token-budget upper bound stays
    # bounded at ≤2 × ~2,200 tokens per agent.
    "hp_note":              2,
}

# ---------------------------------------------------------------------------
# Per-model context windows (in tokens). Used to compute the warn threshold
# at TOKEN_WARN_THRESHOLD_PCT of the configured model's window so the alarm
# fires when a prompt is genuinely close to the model's limit, not against
# a one-size-fits-all 32k assumption.
#
# Add a model when adopting it; values come from the model card. Keys are
# matched case-insensitively as substrings against the configured model
# name, so "google/gemma-4-31B-it" matches the "gemma-4" key, and
# "qwen2.5:32b-32k" matches "qwen2.5".
#
# Order is significant when keys could substring-overlap. Put the more
# specific key first so it wins. Current keys do not overlap.
# ---------------------------------------------------------------------------
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Layer 1 / orchestration / reasoning (cloud)
    "gpt-5.4":           1_050_000,    # confirmed by user 2026-04-30
    "claude-opus-4-7":   1_000_000,
    # Layer 2 local models (deployed on Northwestern H100 cluster)
    "gemma-4":             256_000,    # google/gemma-4-31B-it
    "qwen3.6":             256_000,    # Qwen/Qwen3.6-27B
    "deepseek-r1-32b":     128_000,
    "medgemma":            128_000,    # google/medgemma-1.5-4b-it
    # Legacy / older models still referenced
    "qwen2.5":              32_000,
    # Conservative fallback when model name isn't recognized
    "_default":             32_000,
}

# Warn fires at this fraction of the model's context window. 0.85 leaves
# enough headroom to handle a few-thousand-token output reservation without
# tripping a noisy alarm on every long prompt.
TOKEN_WARN_THRESHOLD_PCT: float = 0.85


def get_token_warn_threshold(model_name: str | None) -> int:
    """Return the warn threshold for a given model.

    Falls back to ~85% of 32k if the model isn't recognized — preserves the
    legacy assumption as a safety net so unconfigured models trigger
    warnings rather than silently overflow.
    """
    name = (model_name or "").lower()
    for key, ctx in MODEL_CONTEXT_WINDOWS.items():
        if key == "_default":
            continue
        if key in name:
            return int(ctx * TOKEN_WARN_THRESHOLD_PCT)
    return int(MODEL_CONTEXT_WINDOWS["_default"] * TOKEN_WARN_THRESHOLD_PCT)


# Legacy global threshold (~85% of 32k). Kept for backward compatibility
# with call sites that don't yet pass a model name. New code should call
# get_token_warn_threshold(model_name) instead.
TOKEN_WARN_THRESHOLD: int = 28_000

# Default mapping from note-type key → CSV filename.
# The filename can be overridden via the ``note_file_map`` setting so that
# study-specific suffixes (e.g. ``_2024``) are not hard-coded.
DEFAULT_NOTE_FILE_MAP: dict[str, str] = {
    "nursing_note": "nursing_note_2024.csv",
    "progress_note": "progress_note_2024.csv",
    "consults_note": "consults_note_2024.csv",
    "plan_of_care_note": "plan_of_care_note_2024.csv",
    "case_management_note": "case_management_note_2024.csv",
    "social_work_note": "social_work_note.csv",
    "therapy_note": "therapy_note_2024.csv",
    # hp_note added 2026-05-26 alongside the H&P routing expansion. Was
    # previously discovered only by the loader's keyword fallback (substring
    # "hp"), which silently misses files named "history_and_physical_*.csv"
    # or "h_and_p_*.csv" at sites that follow a different naming convention.
    "hp_note": "hp_note_2024.csv",
}


class Settings(BaseSettings):
    """Application configuration loaded from .env and environment variables."""

    # Data paths
    clif_data_dir: str = Field(description="Path to CLIF-formatted Parquet directory")
    notes_data_dir: str = Field(
        default="",
        description="Path to note CSV directory. Falls back to clif_data_dir if empty.",
    )
    timezone: str = "America/Chicago"

    # Notes configuration
    notes_lookback_hours: int = Field(
        default=48,
        description="Hours of note history to include (from reference_dttm). Configurable.",
    )
    note_file_map: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_NOTE_FILE_MAP),
        description="Mapping of note-type key to CSV filename. Override for different studies.",
    )

    # LLM configuration
    llm_provider: str = "local"  # "local" | "openai" | "anthropic"
    llm_model: str = "llama3.1:8b"  # Default for Ollama; override for other providers
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    intensivist_max_tokens: int = 8192  # Higher budget for CoT reasoning + 8 sections
    llm_context_window: int = 32768  # Ollama num_ctx — context window for input tokens

    # API keys (fall back to standard env vars if ICUPAUSE_ prefixed ones are empty)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    local_llm_url: str = "http://localhost:11434/v1"
    local_llm_backend: str = "ollama"  # "ollama" or "vllm"

    # Azure OpenAI settings
    azure_api_key: str = ""
    azure_endpoint: str = ""  # e.g., https://your-resource.openai.azure.com/
    azure_api_version: str = "2024-12-01-preview"

    @model_validator(mode="after")
    def _fallback_api_keys(self) -> "Settings":
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.azure_api_key:
            self.azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        if not self.azure_endpoint:
            self.azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        # Configure LangSmith tracing
        if self.use_langsmith:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ.setdefault("LANGCHAIN_PROJECT", self.langsmith_project)
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return self

    @property
    def resolved_notes_data_dir(self) -> str:
        """Return notes_data_dir if set, otherwise fall back to clif_data_dir."""
        return self.notes_data_dir if self.notes_data_dir else self.clif_data_dir

    # Per-agent output token budgets (overrides llm_max_tokens for specific agents)
    agent_max_tokens: dict[str, int] = Field(
        default_factory=lambda: {
            "nurse": 2048,
            "respiratory": 3072,
            "pharmacy": 3072,
            "dietitian": 1536,
            "case_manager": 1536,
            "therapist": 1536,
            "qa": 4096,
            "intensivist": 8192,
            "resident": 4096,
        },
        description="Per-agent max output tokens. Agents not listed fall back to llm_max_tokens.",
    )

    # QA thresholds
    numeric_tolerance: float = 0.05  # ±5%
    qa_ensemble_passes: int = 1  # Number of QA LLM passes with shuffled orderings (1 = no ensemble)
    agent_self_critique: bool = True  # Each domain agent critiques its own output before emitting

    # Data modality toggles (for ablation experiments)
    structured_data_enabled: bool = True  # Toggle structured CLIF data input
    notes_enabled: bool = True  # Toggle clinical notes input

    # Risk predictor settings
    risk_predictor_enabled: bool = False  # Toggle Aim 1 risk prediction model

    # Fusion mode (LEGACY single-enum interface; kept as back-compat aliases).
    fusion_mode: str = "early_fusion"
    # Choices: "early_fusion" | "cr_dsf" | "cr_dsf_plus" | "hybrid_v1" | "hybrid_v1_no_anchor"
    # hybrid_v1: per-domain extractors on raw notes + Stage E anchor override
    # hybrid_v1_no_anchor: same as hybrid_v1 but with use_anchor_override=False
    #   (pre-reg §1.7 mechanism ablation on GPT-5.4)

    # Independent compression axes (pre-reg compression sub-study). When EITHER
    # is set non-empty, the run is a "clean axis cell" and these win over
    # fusion_mode (see resolved_cell). Empty → derive the cell from fusion_mode.
    #   structured_axis: s0 (raw tiered tables) | s1 (LLM summary) | s2 (LLM salience-selected view)
    #   notes_axis:      n0 (raw routed) | n1 (LLM summary) | n2 (per-domain extracted anchors)
    # N2 is SUBSTITUTIVE: the agent reads the extracted anchors IN PLACE OF raw
    # notes (decision 2026-06-07). Stage-E anchor-override is an orthogonal toggle
    # (use_anchor_override), not part of the N2 cell; it is OFF for all clean cells.
    structured_axis: str = ""
    notes_axis: str = ""

    # Stage E anchor-override gate (hybrid_v1 / hybrid_v1_no_anchor; pre-reg §1.7).
    # True under "hybrid_v1"; main.py forces this False when
    # --fusion-mode=hybrid_v1_no_anchor is selected. Ignored for clean axis cells.
    use_anchor_override: bool = True

    @model_validator(mode="after")
    def _validate_axes(self) -> "Settings":
        """Validate explicit axis values and guard against fusion_mode conflict."""
        if self.structured_axis and self.structured_axis not in STRUCTURED_AXES:
            raise ValueError(
                f"structured_axis={self.structured_axis!r} not in {STRUCTURED_AXES}"
            )
        if self.notes_axis and self.notes_axis not in NOTES_AXES:
            raise ValueError(f"notes_axis={self.notes_axis!r} not in {NOTES_AXES}")
        return self

    def resolved_cell(self) -> ResolvedCell:
        """Resolve (fusion_mode | explicit axes | use_anchor_override) → behavior.

        Single source of truth for what each run actually does. base.py keys its
        message composition on structured_axis/notes_axis; workflow.py keys its
        node wiring on run_extractors/apply_anchor_override.
        """
        if self.structured_axis or self.notes_axis:
            # Clean axis cell: explicit axes win; Stage-E always OFF.
            s = self.structured_axis or "s0"
            n = self.notes_axis or "n0"
            return ResolvedCell(
                structured_axis=s,
                notes_axis=n,
                run_extractors=(n == "n2"),
                apply_anchor_override=False,
            )
        # Legacy fusion_mode aliases.
        fm = self.fusion_mode
        s, n = axes_from_fusion_mode(fm)
        if fm == "hybrid_v1":
            # Production anchor path = (s0,n0) + Stage-E. hybrid_v1_no_anchor is
            # collapsed to fusion_mode=hybrid_v1 + use_anchor_override=False in
            # main.py: extractors still run (cost-matched H5 pair), anchor off.
            return ResolvedCell(
                structured_axis=s,
                notes_axis=n,
                run_extractors=True,
                apply_anchor_override=self.use_anchor_override,
            )
        # cr_dsf / cr_dsf_plus → (s1,n1); early_fusion / unknown → (s0,n0).
        return ResolvedCell(
            structured_axis=s,
            notes_axis=n,
            run_extractors=False,
            apply_anchor_override=False,
        )

    # Citation mode — controls deterministic source-tag injection and prompting
    # "off": no cite fields injected, no citation prompts, no provenance checks
    # "decision_critical": cite fields on all rows, prompt asks for cites on
    #     decision-critical values only (vent settings, doses, labs, code status)
    # "all": cite fields on all rows, prompt asks for cites on every value
    citation_mode: str = "decision_critical"

    # Resident pre-synthesis
    resident_enabled: bool = True  # Toggle Resident pre-synthesis agent before Intensivist

    # Scribe-pin propagation (admission_antibiotics → pharmacy and future siblings).
    # Token cap is a rough char/token heuristic budget for the rendered pin block
    # injected into the pharmacy user_message. Truncation drops oldest courses
    # first (stable sort retained; see PharmacyAgent._format_scribe_pins).
    scribe_pin_token_cap_admission_abx: int = 600
    scribe_pin_chars_per_token: float = 3.0

    # Drug interaction checking (hybrid: static ICU table + openFDA labels)
    drug_interaction_enabled: bool = True
    # When True, allow openFDA drug-label API calls to broaden coverage beyond
    # the static table. Set to False for evaluation / reproducibility runs —
    # the static table still runs and is the only source of severity='high'.
    drug_interaction_allow_network: bool = True
    drug_interaction_timeout_seconds: float = 5.0
    # Deprecated: kept as no-ops for config-file backward compatibility.
    rxnav_base_url: str = "https://rxnav.nlm.nih.gov"
    rxnav_timeout_seconds: float = 5.0

    # Device dwell time checking (deterministic, from procedures data)
    device_dwell_enabled: bool = True

    # Lab reference range checking (deterministic, local reference ranges)
    lab_range_check_enabled: bool = True

    # Data caching (skips Parquet loading + serialization on cache hit)
    data_cache_enabled: bool = False  # Toggle data caching for batch runs
    data_cache_dir: str = ""  # Directory for cache files; empty = no caching

    # Deliberation settings
    deliberation_enabled: bool = False  # Toggle QA-triggered agent discussion
    max_deliberation_rounds: int = 1  # Max revision rounds per conflict

    # Observability / tracing
    use_langsmith: bool = Field(
        default=False,
        description=(
            "Enable LangSmith cloud tracing (safe for de-identified MIMIC data). "
            "Set to False (default) for real CLIF/PHI data — use LangGraph Studio local instead."
        ),
    )
    langsmith_project: str = "icu-pause-dev"

    # Evaluation LLM (defaults to pipeline LLM if empty)
    eval_llm_provider: str = ""
    eval_llm_model: str = ""

    # Per-evaluator LLM overrides (fall back to eval_llm_* → pipeline LLM)
    grounding_llm_provider: str = ""
    grounding_llm_model: str = ""
    pdsqi9_llm_provider: str = ""
    pdsqi9_llm_model: str = ""
    hqi_llm_provider: str = ""
    hqi_llm_model: str = ""

    # Prompt directory
    prompts_dir: str = str(Path(__file__).resolve().parent.parent.parent / "config" / "prompts")

    model_config = {
        "env_prefix": "ICUPAUSE_",
        "env_file": ".env",
        "extra": "ignore",
    }
