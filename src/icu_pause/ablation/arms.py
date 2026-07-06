"""Arm dispatcher for the decomposition ablation.

``build_arm(arm, settings)`` returns an ``Arm`` whose ``run_case(...)`` produces
an ``icu_pause_output`` dict for one patient. The runner retrieves the shared
bundle ONCE per case (for arm-independent ground truth) and passes it in; the
monolith arms consume it directly, the graph-based arms retrieve internally via
the same deterministic ``DataRetriever`` (identical inputs → identical bundle).

All graph-based arms run early_fusion — enforced by the runner setting
ICUPAUSE_FUSION_MODE=early_fusion before Settings() is constructed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from icu_pause.data.context import CITE_PATTERN, serialize_to_json
from icu_pause.data.retriever import DataRetriever
from icu_pause.graph.workflow import _parse_reference_dttm, build_graph
from icu_pause.ablation.monolith import MonolithAgent

logger = logging.getLogger(__name__)


def _load_full_brief(full_dir: str, hosp_id: str) -> dict:
    """Load an EXISTING production full-pipeline brief as the 'full' arm output.

    Reuses the frozen, already-generated full brief (read-only) instead of
    re-running the pipeline. Citation markers `(source M-DD HH:MM)` are stripped
    so they don't spuriously satisfy the numeric-fidelity ±5% match. Tolerates
    both save shapes: top-level `sections` and nested `pipeline_output.sections`.
    """
    path = Path(full_dir) / f"{hosp_id}.brief.json"
    if not path.exists():
        raise FileNotFoundError(f"no full brief for {hosp_id} at {path}")
    brief = json.loads(path.read_text())
    secs = brief.get("sections")
    if not isinstance(secs, dict):
        po = brief.get("pipeline_output") or {}
        secs = po.get("sections") if isinstance(po, dict) else None
    if not isinstance(secs, dict):
        raise ValueError(f"no sections found in {path}")
    stripped = {k: CITE_PATTERN.sub("", str(v)) for k, v in secs.items()}
    return {
        "hospitalization_id": hosp_id,
        "generated_at": brief.get("generated_at", ""),
        "sections": stripped,
        "todo_checklist": [],
        "warnings": [],
        "qa_issues": [],
        "section_confidences": {},
        "metadata": {"arm": "full", "ingested_from": str(path),
                     "citations_stripped": True},
    }

ARM_KEYS = [
    "full",
    "monolith_best_effort",
    "monolith_templated",
    "monolith_guided",
    "extract_only",
    "no_intensivist",
    "no_qa",
]

# Arms scored on the PRIMARY numeric-fidelity endpoint ONLY (excluded from the
# secondary PDSQI-9 / prioritization comparison). no_intensivist drops sections
# I and P (sole-authored by the intensivist), which confounds the secondary
# metrics but NOT numeric fidelity (the checked numerics live in domain sections).
PRIMARY_ONLY_ARMS = {"no_intensivist"}


@dataclass
class Arm:
    key: str
    run_case: Callable[..., dict[str, Any]]
    primary_only: bool = False


def retrieve_bundle(settings, hosp_id: str, reference_dttm: str,
                    lookback_hours: int | None,
                    notes_lookback_hours: int | None) -> dict[str, Any]:
    """The shared serialized bundle = full structured data + union of all notes.

    Identical to what the pipeline's data_retrieval node feeds every arm
    (workflow.py builds it with the same ``serialize_to_json(ctx)`` call). Used
    as the monolith input AND as the arm-independent ground-truth source.
    """
    retriever = DataRetriever(settings)
    ref = _parse_reference_dttm(reference_dttm)
    ctx = retriever.retrieve(
        hosp_id,
        reference_dttm=ref,
        lookback_hours=lookback_hours,
        notes_lookback_hours=notes_lookback_hours,
    )
    return serialize_to_json(ctx, lookback_hours=lookback_hours)


def _initial_state(hosp_id, reference_dttm, lookback_hours, notes_lookback_hours):
    """Initial GraphState — mirrors main.py / eval.batch."""
    return {
        "hospitalization_id": hosp_id,
        "lookback_hours": lookback_hours,
        "reference_dttm": reference_dttm,
        "notes_lookback_hours": notes_lookback_hours,
        "patient_context_text": {},
        "agent_context_text": {},
        "cite_registry": {},
        "agent_snippets": [],
        "pipeline_metrics": [],
        "fusion_mode": "early_fusion",
        "structured_summaries": {},
        "note_summaries": {},
        "extraction_fields": {},
        "risk_score": None,
        "qa_issues": [],
        "qa_passed": False,
        "revised_snippets": [],
        "deliberation_log": [],
        "intensivist_output": None,
        "icu_pause_output": {},
        "trace_events": [],
    }


def build_arm(arm: str, settings, full_from_existing: str | None = None,
              temperature: float = 0.0) -> Arm:
    if arm not in ARM_KEYS:
        raise ValueError(f"unknown arm {arm!r}; valid: {ARM_KEYS}")

    # --- Arms 2-4: monolith baselines (no graph) ---
    #   best_effort : free-form    | templated : 8-section labels
    #   guided      : 8 sections + DISTILLED per-section instructions (instruction-matched)
    if arm in ("monolith_best_effort", "monolith_templated", "monolith_guided"):
        mode = arm[len("monolith_"):]  # best_effort | templated | guided
        agent = MonolithAgent(settings, mode, temperature=temperature)

        def run_case(hosp_id, reference_dttm, lookback_hours,
                     notes_lookback_hours, bundle=None):
            if bundle is None:
                bundle = retrieve_bundle(settings, hosp_id, reference_dttm,
                                         lookback_hours, notes_lookback_hours)
            state = {"hospitalization_id": hosp_id, "patient_context_text": bundle}
            return agent.run(state)["icu_pause_output"]

        return Arm(arm, run_case)

    # --- Arm 1: full pipeline ---
    if arm == "full":
        # Reuse mode: load the EXISTING production full brief (citations stripped)
        # instead of regenerating the pipeline. Lets a multi-model ablation reuse
        # already-generated full briefs and only generate the monolith arms.
        if full_from_existing:
            def run_case(hosp_id, reference_dttm, lookback_hours,
                         notes_lookback_hours, bundle=None):
                return _load_full_brief(full_from_existing, hosp_id)

            return Arm(arm, run_case)

        if getattr(settings, "fusion_mode", None) != "early_fusion":
            logger.warning("full arm expects early_fusion, got fusion_mode=%s",
                           getattr(settings, "fusion_mode", None))
        graph = build_graph(settings)

        def run_case(hosp_id, reference_dttm, lookback_hours,
                     notes_lookback_hours, bundle=None):
            result = graph.invoke(_initial_state(
                hosp_id, reference_dttm, lookback_hours, notes_lookback_hours))
            return result.get("icu_pause_output", {})

        return Arm(arm, run_case)

    # --- Arms 4-6: implemented after the arms 1-3 Gemma smoke (build order) ---
    raise NotImplementedError(
        f"arm {arm!r} is not yet wired — build after the arms 1-3 smoke. "
        "extract_only: data_retrieval → scribe → mechanical synthesis; "
        "no_intensivist / no_qa: build_ablation_graph with skip flags."
    )
