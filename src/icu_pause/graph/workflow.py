"""LangGraph StateGraph definition for the ICU-PAUSE agentic workflow.

Graph topology — Early Fusion (default):
    START
      |
    [data_retrieval]                      Step 1: deterministic, no LLM
      |
    ┌─┬─┬─┬─┬─┬────────────────┐
    N R P D C T  [risk_pred]?            Step 2: 6 domain agents (+ risk if enabled)
    └─┴─┴─┴─┴─┴────────────────┘
      |
    [qa_check]                            Step 3: QA/Consistency validation
      |
    [intensivist]                         Step 4: Clinical reasoning (sees risk score)
      |
    [merge_and_render]                    Step 5: SectionMerger → ICU-PAUSE
      |
    END

Graph topology — CR-DSF (Cross-Referenced Dual-Stream Fusion):
    START
      |
    [data_retrieval]                      Step 1: deterministic, no LLM
      |
    ┌──────────────────────┐
    [structured_interpreter]              Step 2: Interpreter agents (parallel)
    [note_interpreter]
    └──────────────────────┘
      |
    ┌─┬─┬─┬─┬─┬────────────────┐
    N R P D C T  [risk_pred]?            Step 3: Domain agents (receive summaries)
    └─┴─┴─┴─┴─┴────────────────┘
      |
    [qa_check] → [intensivist] → [merge_and_render] → END

Graph topology (deliberation enabled):
    ... same as above through qa_check ...
      |
    [qa_check] ──┬── (qa_passed) ────→ [intensivist] → ...
                 └── (qa_failed) ────→ [deliberation] → [intensivist] → ...
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from langgraph.graph import END, START, StateGraph

from icu_pause.agents.anchor_override import (
    AnchorBinding,
    apply_anchor_override,
)
from icu_pause.agents.case_manager import CaseManagerAgent
from icu_pause.agents.deliberation import DeliberationNode
from icu_pause.agents.dietitian import DietitianAgent
from icu_pause.agents.extractors import build_per_domain_extractors
from icu_pause.agents.intensivist import IntensivistAgent
from icu_pause.agents.interpreter import (
    NoteInterpreterAgent, StructuredInterpreterAgent, StructuredExtractorAgent,
    StructuredSalienceSelectorAgent,
)
from icu_pause.agents.nurse import NurseAgent
from icu_pause.agents.orchestrator import SectionMerger
from icu_pause.agents.pharmacy import PharmacyAgent
from icu_pause.agents.qa import QAAgent
from icu_pause.agents.resident import ResidentAgent
from icu_pause.agents.respiratory import RespiratoryAgent
from icu_pause.agents.scribe import ScribeAgent
from icu_pause.agents.therapist import TherapistAgent
from icu_pause.config import AGENT_NOTE_ROUTING, Settings
from icu_pause.data.context import PatientContext, serialize_to_json
from icu_pause.data.retriever import DataRetriever
from icu_pause.schemas.graph_state import GraphState
from icu_pause.tracing import RunTrace

logger = logging.getLogger(__name__)

# --- Anchor bindings per domain (hybrid_v1 Stage E) ---
# Empty for v1.0 — Stage E reconciliation runs but no fields are declared yet.
# v1.1 work: declare per-agent bindings once pre-reg §1.5 schema expansion is
# signed off by Saki. Each binding maps an extractor anchor path to an
# agent-output path. See pre-reg §1.3 Stage E + tests/test_anchor_override.py.
ANCHOR_BINDINGS_BY_DOMAIN: dict[str, list[AnchorBinding]] = {
    "nurse": [],
    "respiratory": [],
    "pharmacy": [],
    "dietitian": [],
    "case_manager": [],
    "therapist": [],
}


def _wrap_agent_with_stage_e(agent, settings: Settings, domain_name: str):
    """Wrap a domain agent's ``run`` so Stage E reconciles its output against
    the per-domain extracted anchors when hybrid_v1 is active.

    The wrapper is applied ONLY in the hybrid_v1 workflow branch — the agent
    class itself is untouched, per the production-finalized rule. When
    ``settings.use_anchor_override`` is False (the ``hybrid_v1_no_anchor``
    ablation per pre-reg §1.7), the wrapper short-circuits and the agent's
    output passes through unchanged.

    Stage E is a no-op until ``ANCHOR_BINDINGS_BY_DOMAIN[domain_name]`` is
    populated; the wiring is exercised today so the LangGraph edges are
    correct, and the reconciliation activates once bindings land.
    """
    original_run = agent.run

    def wrapped_run(state):
        result = original_run(state)
        if not settings.use_anchor_override:
            return result
        bindings = ANCHOR_BINDINGS_BY_DOMAIN.get(domain_name, [])
        if not bindings:
            return result  # no-op until v1.1 bindings land
        extractions = (
            state.get("per_domain_extractions", {}) or {}
        ).get(domain_name, {})
        if not extractions:
            return result
        snippets = list(result.get("agent_snippets", []))
        if not snippets:
            return result
        new_trace_events: list[dict] = list(result.get("trace_events", []))
        new_snippets = []
        for snip in snippets:
            try:
                snip_dict = snip.model_dump() if hasattr(snip, "model_dump") else dict(snip)
                reconc = apply_anchor_override(snip_dict, extractions, bindings)
                if reconc.events:
                    # Rebuild snippet with reconciled values
                    new_snip = type(snip)(**reconc.corrected_output)
                    new_snippets.append(new_snip)
                    new_trace_events.extend(ev.to_trace() for ev in reconc.events)
                else:
                    new_snippets.append(snip)
            except Exception as e:
                logger.warning(
                    f"Stage E reconciliation failed for {domain_name}: {e}; "
                    "preserving agent output unchanged"
                )
                new_snippets.append(snip)
        merged = dict(result)
        merged["agent_snippets"] = new_snippets
        if len(new_trace_events) != len(result.get("trace_events", [])):
            merged["trace_events"] = new_trace_events
        return merged

    return wrapped_run


# All domain agent names (excluding intensivist — it runs after QA)
DOMAIN_AGENTS = ["nurse", "respiratory", "pharmacy", "dietitian", "case_manager", "therapist"]


def _parse_reference_dttm(value: str | None) -> datetime | None:
    """Parse an ISO-8601 reference_dttm string into a datetime.

    Tolerates leading/trailing whitespace and CRLF line endings (common when
    the value is piped from a shell loop reading a Windows-style CSV) and
    accepts both "T" and " " separators between date and time.
    """
    if not value:
        return None
    from datetime import datetime as _dt

    s = value.strip()
    if not s:
        return None
    # fromisoformat in 3.11+ accepts "YYYY-MM-DD HH:MM:SS+HH:MM" natively, but
    # fall back to swapping the first space for "T" if a stricter parser is
    # ever in play.
    candidates = [s]
    if " " in s and "T" not in s[:11]:
        candidates.append(s.replace(" ", "T", 1))
    for cand in candidates:
        try:
            return _dt.fromisoformat(cand)
        except (ValueError, TypeError):
            continue
    logger.warning(f"Could not parse reference_dttm {value!r}; using auto-detect")
    return None


def build_graph(settings: Settings) -> Any:
    """Build and compile the ICU-PAUSE LangGraph workflow.

    Args:
        settings: Application configuration.

    Returns:
        Compiled LangGraph that can be invoked with initial state.
    """
    builder = StateGraph(GraphState)

    # --- Step 1: Data Retrieval (deterministic) ---
    retriever = DataRetriever(settings)

    def _has_data(v: Any) -> bool:
        """Check if a serialized context value contains actual data."""
        if isinstance(v, dict):
            return any(bool(sub) for sub in v.values())
        return bool(v)

    # --- Data cache helpers ---
    import hashlib as _hashlib
    import json as _json
    from pathlib import Path as _Path

    def _cache_key(hosp_id: str, lookback: int | None, ref_dttm: str | None,
                   notes_lookback: int | None) -> str:
        """Deterministic cache key from retrieval parameters."""
        raw = f"{hosp_id}|{lookback}|{ref_dttm}|{notes_lookback}"
        return _hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _try_load_cache(hosp_id: str, key: str) -> dict | None:
        """Try to load cached serialized context from disk."""
        if not settings.data_cache_enabled or not settings.data_cache_dir:
            return None
        cache_path = _Path(settings.data_cache_dir) / f"{hosp_id}_{key}.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = _json.load(f)
                logger.info("Cache HIT for %s (key=%s)", hosp_id, key)
                return data
            except Exception as e:
                logger.warning("Cache read failed for %s: %s", hosp_id, e)
        return None

    def _save_cache(hosp_id: str, key: str, data: dict) -> None:
        """Save serialized context to cache."""
        if not settings.data_cache_enabled or not settings.data_cache_dir:
            return
        cache_dir = _Path(settings.data_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{hosp_id}_{key}.json"
        try:
            with open(cache_path, "w") as f:
                _json.dump(data, f, separators=(",", ":"), default=str)
            logger.info("Cache WRITE for %s (key=%s)", hosp_id, key)
        except Exception as e:
            logger.warning("Cache write failed for %s: %s", hosp_id, e)

    def data_retrieval_node(state: GraphState) -> dict:
        hosp_id = state["hospitalization_id"]
        lookback_hours = state.get("lookback_hours", 48)
        reference_dttm = _parse_reference_dttm(state.get("reference_dttm"))
        notes_lookback_hours = state.get("notes_lookback_hours")

        # --- Check cache ---
        cache_k = _cache_key(hosp_id, lookback_hours,
                             state.get("reference_dttm"), notes_lookback_hours)
        cached = _try_load_cache(hosp_id, cache_k)
        if cached is not None:
            trace_events = [{
                "timestamp": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "type": "cache_hit",
                "node": "data_retrieval",
                "level": "info",
                "message": f"Loaded from cache for {hosp_id}",
                "data": {"cache_key": cache_k},
            }]
            return {
                "patient_context_text": cached["patient_context_text"],
                "agent_context_text": cached["agent_context_text"],
                "cite_registry": {},  # not cached — citations re-injected on non-cached runs
                "fusion_mode": settings.fusion_mode,
                "structured_summaries": {},
                "note_summaries": {},
                "extraction_fields": {},
                "trace_events": trace_events,
                "physician_floor": cached.get("physician_floor", {}),
            }

        trace = RunTrace(hosp_id)
        window_desc = f"last {lookback_hours}h" if lookback_hours is not None else "entire stay"
        trace.log("pipeline_start", "data_retrieval",
                  message=f"Loading data for {hosp_id} ({window_desc})",
                  data={"lookback_hours": lookback_hours, "fusion_mode": settings.fusion_mode})

        # Clear retriever load_log before each run
        retriever.load_log.clear()

        # Trace: log data paths for debugging
        trace.log("config", "data_retrieval",
                  message=f"CLIF dir: {retriever.data_dir}, Notes dir: {retriever.notes_data_dir}",
                  data={"clif_data_dir": retriever.data_dir,
                        "notes_data_dir": retriever.notes_data_dir,
                        "notes_enabled": retriever.notes_enabled,
                        "structured_data_enabled": retriever.structured_data_enabled,
                        "note_file_map": retriever.note_file_map})

        ctx: PatientContext = retriever.retrieve(
            hosp_id,
            lookback_hours=lookback_hours,
            reference_dttm=reference_dttm,
            notes_lookback_hours=notes_lookback_hours,
        )

        # Build cite_registry when citations are enabled
        from zoneinfo import ZoneInfo
        cite_registry: dict[str, list[dict]] = {}
        cite_kwargs: dict[str, Any] = {}
        if settings.citation_mode != "off":
            cite_kwargs["cite_registry"] = cite_registry
            cite_kwargs["display_tz"] = ZoneInfo(settings.timezone)

        # Shared context (without agent-specific notes — agents read from agent_context_text)
        context_text = serialize_to_json(ctx, lookback_hours=lookback_hours, **cite_kwargs)
        available = sum(1 for v in context_text.values() if _has_data(v))

        # Trace: log which data domains loaded
        trace_events = []
        for key, value in context_text.items():
            has = _has_data(value)
            if has:
                # Estimate row count for structured data
                if isinstance(value, dict):
                    row_info = {k: len(v) if isinstance(v, list) else 1
                                for k, v in value.items() if v}
                    evt = trace.log_data_loaded("data_retrieval", key,
                                                sum(row_info.values()))
                elif isinstance(value, list):
                    evt = trace.log_data_loaded("data_retrieval", key, len(value))
                else:
                    evt = trace.log("data_loaded", "data_retrieval",
                                    message=f"Loaded {key}", data={"table": key})
                trace_events.append(evt)

        trace.log("data_summary", "data_retrieval",
                  message=f"{available} data domains available",
                  data={"available_domains": [k for k, v in context_text.items() if _has_data(v)],
                        "missing_domains": [k for k, v in context_text.items() if not _has_data(v)]})

        # Per-agent contexts: same structured data, but with agent-specific notes.
        # Filter to only the data keys each agent needs to avoid bloating
        # context with irrelevant data (e.g., 2500 billing procedure rows).
        from icu_pause.config import get_token_warn_threshold

        # Map agent → required data keys (must match required_context_keys in agent classes)
        _AGENT_DATA_KEYS: dict[str, list[str]] = {
            # Nurse gets ``meds`` so the classified med_state view is
            # available (states.records is the authoritative source for
            # infusion activity — see ``_strip_meds_raw_admin_rows`` below).
            "nurse": ["vitals", "assessments", "demographics", "adt", "sofa", "notes", "meds"],
            "respiratory": ["respiratory", "vitals", "labs", "position", "microbiology", "notes"],
            "pharmacy": ["meds", "labs", "microbiology", "notes"],
            # dietitian: meds removed — Northwestern's CLIF medication_admin
            # tables do not represent nutrition administration. See
            # docs/clif_data_gaps_investigation.md and DietitianAgent.required_context_keys.
            "dietitian": ["labs", "vitals", "assessments", "notes"],
            "case_manager": ["code_status", "adt", "demographics", "notes"],
            "therapist": ["assessments", "position", "notes"],
            "intensivist": [
                "demographics", "adt", "vitals", "labs", "meds", "respiratory",
                "assessments", "code_status", "microbiology", "sofa",
                "crrt", "ecmo", "position", "notes",
                "transfer_exam_block",
            ],
            # Scribe needs notes + demographics only — single-task PMH
            # extraction. Excluding the rest keeps the extractor's context
            # focused (lost-in-the-middle mitigation) and tokens cheap.
            "scribe": ["demographics", "notes"],
        }

        agent_context_text: dict[str, dict[str, Any]] = {}
        physician_floor: dict[str, dict[str, Any]] = {}
        for agent_role in AGENT_NOTE_ROUTING:
            agent_ctx = serialize_to_json(
                ctx, lookback_hours=lookback_hours, agent_role=agent_role,
                **cite_kwargs,
            )
            # Lift physician-note floor metadata out of the agent-visible
            # slice so the reviewer source-data union (which iterates over
            # agent_context_text keys as domain names) doesn't pick it up,
            # and so it surfaces under metadata.physician_floor on the
            # final ICUPauseOutput.  Underscore-prefixed key from
            # serialize_to_json is plumbing, not data.
            floor_meta = agent_ctx.pop("_physician_floor", None)
            if floor_meta is not None:
                physician_floor[agent_role] = floor_meta

            # Filter to only the keys this agent needs
            allowed_keys = set(_AGENT_DATA_KEYS.get(agent_role, agent_ctx.keys()))
            allowed_keys.add("demographics")  # always include demographics
            allowed_keys.add("notes")  # always include routed notes
            agent_ctx = {k: v for k, v in agent_ctx.items() if k in allowed_keys}

            # Nurse-only: strip raw admin rows from the meds bucket so the
            # only medication view is the classified state. The med_state
            # classifier is the authoritative source for infusion activity
            # (vasopressors, sedation, paralytics, insulin gtt); raw admin
            # rows here used to drive ad-hoc "is it still running?"
            # inference that disagreed with pharmacy and produced spurious
            # cross-domain conflicts (iter-0: norepi RECENTLY_STOPPED
            # miscalled active). Pharmacy/intensivist keep the raw rows
            # because they reason over doses; nurse does not.
            if agent_role == "nurse" and isinstance(agent_ctx.get("meds"), dict):
                meds_bucket = agent_ctx["meds"]
                agent_ctx["meds"] = {
                    "states": meds_bucket.get("states") or {},
                }

            est_tokens = len(str(agent_ctx)) // 4
            # Threshold is per-model: 85% of the configured LLM's context window.
            # Falls back to ~28k if the model name isn't in MODEL_CONTEXT_WINDOWS,
            # so unconfigured models still trigger a warning rather than silently
            # overflow.
            threshold = get_token_warn_threshold(settings.llm_model)
            if est_tokens > threshold:
                logger.warning(
                    "agent=%s estimated_tokens=%d exceeds threshold=%d (model=%s)",
                    agent_role, est_tokens, threshold, settings.llm_model,
                )
            agent_context_text[agent_role] = agent_ctx

            # Trace: log note routing per agent
            notes = agent_ctx.get("notes")
            if notes and isinstance(notes, dict):
                note_types = {k: len(v) if isinstance(v, list) else 1
                              for k, v in notes.items() if v}
                evt = trace.log_note_routing(agent_role, note_types)
            else:
                evt = trace.log_note_routing(agent_role, {})
            trace_events.append(evt)

        # Include retriever's detailed debug log (parquet loads, ID matching, etc.)
        retriever_log = list(retriever.load_log)  # copy before it might be cleared
        trace_events.extend(retriever_log)

        # Include RunTrace events (pipeline_start, data_summary)
        trace_events.extend(trace.events)

        logger.info(
            f"Data retrieval trace: {len(trace_events)} events total "
            f"({len(retriever_log)} from retriever)"
        )

        # Trace: log floor activation per agent
        from datetime import timezone as _tz
        for role, meta in physician_floor.items():
            trace_events.append({
                "timestamp": datetime.now(_tz.utc).isoformat(),
                "type": "physician_floor",
                "node": f"data_retrieval/{role}",
                "level": "info",
                "message": (
                    f"physician_floor[{role}] applied={meta.get('floor_applied')} "
                    f"reason={meta.get('reason')} "
                    f"specialty={meta.get('floor_specialty')}"
                ),
                "data": meta,
            })

        # --- Save to cache for subsequent model runs ---
        _save_cache(hosp_id, cache_k, {
            "patient_context_text": context_text,
            "agent_context_text": agent_context_text,
            "physician_floor": physician_floor,
        })

        _cell = settings.resolved_cell()
        return {
            "patient_context_text": context_text,
            "agent_context_text": agent_context_text,
            "cite_registry": cite_registry,
            "fusion_mode": settings.fusion_mode,
            "structured_axis": _cell.structured_axis,
            "notes_axis": _cell.notes_axis,
            "structured_summaries": {},
            "note_summaries": {},
            "structured_views": {},
            "extraction_fields": {},
            "trace_events": trace_events,
            "physician_floor": physician_floor,
        }

    builder.add_node("data_retrieval", data_retrieval_node)

    # --- Step 2: Domain Agents (parallel) ---
    agent_instances = {
        "nurse": NurseAgent(settings),
        "respiratory": RespiratoryAgent(settings),
        "pharmacy": PharmacyAgent(settings),
        "dietitian": DietitianAgent(settings),
        "case_manager": CaseManagerAgent(settings),
        "therapist": TherapistAgent(settings),
    }
    # Compression sub-study: node wiring is keyed on the resolved cell — two
    # independent axes (structured_axis ∈ {s0,s1,s2}, notes_axis ∈ {n0,n1,n2})
    # plus the orthogonal Stage-E anchor-override toggle. See Settings.resolved_cell.
    cell = settings.resolved_cell()

    # Stage-E anchor-override wrap (orthogonal toggle; ON only for the production
    # hybrid_v1 path). Wrapping happens at build time only — agent classes are
    # untouched; unwrapped agents are byte-identical to today.
    for name, agent in agent_instances.items():
        if cell.apply_anchor_override:
            builder.add_node(name, _wrap_agent_with_stage_e(agent, settings, name))
        else:
            builder.add_node(name, agent.run)

    # --- Step 2a: Structured-axis producer (S1 summary | S2 salience view) ---
    if cell.structured_axis == "s1":
        builder.add_node("structured_interpreter", StructuredInterpreterAgent(settings).run)
    elif cell.structured_axis == "s2":
        builder.add_node("structured_salience", StructuredSalienceSelectorAgent(settings).run)

    # --- Step 2a': Notes-axis producer (N1 summary) ---
    if cell.notes_axis == "n1":
        builder.add_node("note_interpreter", NoteInterpreterAgent(settings).run)

    # --- Step 2a'': Per-domain extractors (6 parallel) — N2 prompt anchors AND/OR
    # the Stage-E anchor-override source. Run whenever either needs them.
    per_domain_extractor_instances: dict[str, Any] = {}
    if cell.run_extractors:
        per_domain_extractor_instances = build_per_domain_extractors(settings)
        for domain_name, extractor in per_domain_extractor_instances.items():
            builder.add_node(f"extractor_{domain_name}", extractor.run)

    # --- Step 2b: Risk Predictor (parallel with domain agents, optional) ---
    if settings.risk_predictor_enabled:
        def risk_predictor(state: GraphState) -> dict:
            """Risk prediction model — runs in parallel with domain agents.

            When the Aim 1 Transformer model is integrated, this node will:
            1. Call the model with patient time-series data from patient_context_text
            2. Return a 72h readmission/death risk score
            3. The Intensivist agent can factor this into its clinical narrative

            Currently returns a placeholder score structure.
            """
            logger.info("Risk predictor: placeholder (Aim 1 model not yet integrated)")
            return {
                "risk_score": {
                    "model": "placeholder",
                    "risk_72h_readmission": None,
                    "risk_72h_mortality": None,
                    "available": False,
                }
            }

        builder.add_node("risk_predictor", risk_predictor)

    # --- Step 3: QA Agent ---
    qa_agent = QAAgent(settings)
    builder.add_node("qa_check", qa_agent.run)

    # --- Step 3b: Deliberation (conditional, when enabled) ---
    if settings.deliberation_enabled:
        deliberation = DeliberationNode(settings, agent_instances)
        builder.add_node("deliberation", deliberation.run)

    # --- Step 3c: Resident Agent (pre-synthesis, when enabled) ---
    if settings.resident_enabled:
        resident = ResidentAgent(settings)
        builder.add_node("resident", resident.run)

    # --- Step 3d: Scribe Agent (phase-0 structured PMH extraction) ---
    # Runs in parallel with the existing fan-out; its output reaches the
    # intensivist via a separate edge so the intensivist waits for both
    # (qa/deliberation/resident path) AND scribe before running.
    scribe = ScribeAgent(settings)
    builder.add_node("scribe", scribe.run)

    # --- Step 4: Intensivist Agent (clinical reasoning & harmonization) ---
    intensivist = IntensivistAgent(settings)
    builder.add_node("intensivist", intensivist.run)

    # --- Step 5: Section Merger ---
    merger = SectionMerger(settings)
    builder.add_node("merge_and_render", merger.run)

    # --- Edges ---

    # START -> data_retrieval
    builder.add_edge(START, "data_retrieval")

    # data_retrieval -> {axis producers that exist for this cell} -> domain agents.
    # Each domain agent fans in from every producer it consumes (structured-axis
    # producer, notes-axis producer, and/or its per-domain extractor); if a cell
    # has no producers (early_fusion = s0,n0,no-anchor) the agent reads
    # data_retrieval directly.
    struct_producer = None
    if cell.structured_axis == "s1":
        struct_producer = "structured_interpreter"
    elif cell.structured_axis == "s2":
        struct_producer = "structured_salience"
    notes_producer = "note_interpreter" if cell.notes_axis == "n1" else None

    if struct_producer:
        builder.add_edge("data_retrieval", struct_producer)
    if notes_producer:
        builder.add_edge("data_retrieval", notes_producer)
    if cell.run_extractors:
        for domain_name in per_domain_extractor_instances:
            builder.add_edge("data_retrieval", f"extractor_{domain_name}")

    for agent_name in DOMAIN_AGENTS:
        producers = []
        if struct_producer:
            producers.append(struct_producer)
        if notes_producer:
            producers.append(notes_producer)
        if cell.run_extractors and agent_name in per_domain_extractor_instances:
            producers.append(f"extractor_{agent_name}")
        if producers:
            for p in producers:
                builder.add_edge(p, agent_name)
        else:
            builder.add_edge("data_retrieval", agent_name)
    if settings.risk_predictor_enabled:
        builder.add_edge("data_retrieval", "risk_predictor")

    # all 6 domain agents (+ risk_predictor if enabled) -> qa_check (fan-in)
    for agent_name in DOMAIN_AGENTS:
        builder.add_edge(agent_name, "qa_check")
    if settings.risk_predictor_enabled:
        builder.add_edge("risk_predictor", "qa_check")

    # qa_check -> deliberation (conditional) -> resident (if enabled) -> intensivist
    # Determine the node that comes right before intensivist
    pre_intensivist = "resident" if settings.resident_enabled else "intensivist"

    if settings.deliberation_enabled:
        def route_after_qa(state: GraphState) -> str:
            if state.get("qa_passed", True):
                logger.info("QA passed — skipping deliberation")
                return pre_intensivist
            logger.info("QA found issues — routing to deliberation")
            return "deliberation"

        builder.add_conditional_edges(
            "qa_check",
            route_after_qa,
            {"deliberation": "deliberation", pre_intensivist: pre_intensivist},
        )
        builder.add_edge("deliberation", pre_intensivist)
    else:
        builder.add_edge("qa_check", pre_intensivist)

    if settings.resident_enabled:
        builder.add_edge("resident", "intensivist")

    # Scribe: parallel branch off data_retrieval, joins at intensivist AND
    # pharmacy. LangGraph waits for all incoming edges, so:
    #   - intensivist runs after both the qa/resident path AND scribe complete
    #   - pharmacy runs after both data_retrieval AND scribe complete
    # The scribe→pharmacy edge feeds the admission_antibiotics pin block
    # rendered by PharmacyAgent._format_scribe_pins; see
    # docs/admission_antibiotics_design.md.
    builder.add_edge("data_retrieval", "scribe")
    builder.add_edge("scribe", "pharmacy")
    builder.add_edge("scribe", "intensivist")

    # intensivist -> merge_and_render -> END
    builder.add_edge("intensivist", "merge_and_render")
    builder.add_edge("merge_and_render", END)

    return builder.compile()
