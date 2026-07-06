"""Batch evaluation harness for ICU-PAUSE notes.

Runs the pipeline for N hospitalization IDs and optionally scores each
generated note with the custom QA rubric, PDSQI-9, grounding check,
data utilization, golden case comparison, and prompt versioning.

Usage:
    python -m icu_pause.eval.batch \
        --ids cases.txt \
        --output-dir /tmp/eval_results \
        --rubric --pdsqi9 --grounding --data-utilization \
        --golden-dir golden_cases/ \
        --lookback-hours 48
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from icu_pause.config import Settings
from icu_pause.graph.workflow import build_graph
from icu_pause.rendering.formatter import render_icu_pause_text

logger = logging.getLogger(__name__)


def _build_evaluator_source(
    patient_ctx: dict[str, Any],
    agent_contexts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build evaluator source context with merged notes from all agents.

    The shared ``patient_context_text`` has empty notes because
    ``serialize_to_json()`` is called without an ``agent_role``. This merges
    all per-agent routed notes into a single deduplicated dict so evaluators
    (PDSQI-9, grounding, data utilization) see the same information the
    agents saw.
    """
    merged_notes: dict[str, list] = {}
    for agent_ctx in agent_contexts.values():
        for note_type, notes in (agent_ctx.get("notes") or {}).items():
            if not isinstance(notes, list):
                continue
            if note_type not in merged_notes:
                merged_notes[note_type] = []
            existing_ids = {
                n.get("note_id")
                for n in merged_notes[note_type]
                if isinstance(n, dict)
            }
            for note in notes:
                if isinstance(note, dict) and note.get("note_id") not in existing_ids:
                    merged_notes[note_type].append(note)
                    existing_ids.add(note.get("note_id"))

    evaluator_ctx = dict(patient_ctx)  # shallow copy
    if merged_notes:
        evaluator_ctx["notes"] = merged_notes

    # Remove data keys that NO agent receives — the evaluator should only
    # score against data the agents actually had access to. Build the union
    # of all per-agent context keys to determine what was visible.
    agent_visible_keys: set[str] = set()
    for agent_ctx in agent_contexts.values():
        agent_visible_keys.update(agent_ctx.keys())
    # Always keep notes (merged above) and demographics (always visible)
    agent_visible_keys.add("notes")
    agent_visible_keys.add("demographics")
    keys_to_remove = [
        k for k in evaluator_ctx
        if k not in agent_visible_keys
    ]
    for k in keys_to_remove:
        del evaluator_ctx[k]

    return evaluator_ctx


def _load_ids(path: str) -> list[str]:
    """Read hospitalization IDs from a text file (one per line)."""
    ids = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                ids.append(stripped)
    return ids


def _load_reference_note(reference_dir: str, hosp_id: str) -> str | None:
    """Load a human-written reference note for a hospitalization ID.

    Looks for ``<reference_dir>/<hosp_id>.txt`` or ``<hosp_id>.json``.
    """
    base = Path(reference_dir)
    for ext in (".txt", ".json"):
        path = base / f"{hosp_id}{ext}"
        if path.exists():
            return path.read_text()
    return None


def run_batch(
    ids: list[str],
    settings: Settings,
    output_dir: Path,
    run_rubric: bool = False,
    run_pdsqi9: bool = False,
    run_grounding: bool = False,
    run_data_utilization: bool = False,
    reference_dir: str | None = None,
    golden_dir: str | None = None,
    snapshot_dir: str | None = None,
    lookback_hours: int | None = 48,
) -> Path:
    """Run the full pipeline + optional evaluations for each hospitalization ID.

    Args:
        ids: List of hospitalization IDs to process.
        settings: Application settings.
        output_dir: Directory to write per-case JSON and summary CSV.
        run_rubric: Whether to score with the custom QA rubric.
        run_pdsqi9: Whether to score with the PDSQI-9 instrument.
        run_grounding: Whether to run per-agent grounding/hallucination check.
        run_data_utilization: Whether to run per-agent data utilization check.
        reference_dir: Directory of human-written notes for PDSQI-9 comparison.
        golden_dir: Directory of golden case JSON files for regression testing.
        snapshot_dir: Directory to save pipeline snapshots for prompt versioning.
        lookback_hours: Lookback window (None = entire stay).

    Returns:
        Path to the summary CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = build_graph(settings)

    # --- Prompt versioning: hash all prompts at run start ---
    from icu_pause.eval.prompt_versioning import hash_all_prompts, get_all_prompt_versions
    prompt_hashes = hash_all_prompts(settings.prompts_dir)
    prompt_versions = get_all_prompt_versions(settings.prompts_dir)
    logger.info(f"Prompt hashes: {prompt_hashes}")
    logger.info(f"Prompt versions: {prompt_versions}")

    # Lazy-init evaluators only if needed
    rubric_evaluator = None
    pdsqi9_evaluator = None
    grounding_evaluator = None
    data_util_evaluator = None
    golden_evaluator = None
    golden_cases = None

    if run_rubric:
        from icu_pause.eval.rubric import ICUPauseRubricEvaluator
        rubric_evaluator = ICUPauseRubricEvaluator(settings)
    if run_pdsqi9:
        from icu_pause.eval.pdsqi9 import PDSQI9Evaluator
        pdsqi9_evaluator = PDSQI9Evaluator(settings)
    if run_grounding:
        from icu_pause.eval.grounding import GroundingEvaluator
        grounding_evaluator = GroundingEvaluator(settings)
    if run_data_utilization:
        from icu_pause.eval.data_utilization import DataUtilizationEvaluator
        data_util_evaluator = DataUtilizationEvaluator(settings)
    if golden_dir:
        from icu_pause.eval.golden_cases import GoldenCaseEvaluator
        golden_evaluator = GoldenCaseEvaluator(settings)
        golden_cases = GoldenCaseEvaluator.load_golden_cases(golden_dir)

    summary_rows: list[dict[str, Any]] = []
    golden_outputs: dict[str, dict[str, str]] = {}  # For golden case batch eval

    for i, hosp_id in enumerate(ids, 1):
        logger.info(f"[{i}/{len(ids)}] Processing {hosp_id}")
        case_result: dict[str, Any] = {
            "hospitalization_id": hosp_id,
            "prompt_hashes": prompt_hashes,
        }

        # --- Run pipeline ---
        t0 = time.perf_counter()
        try:
            result = graph.invoke(
                {
                    "hospitalization_id": hosp_id,
                    "lookback_hours": lookback_hours,
                    "reference_dttm": None,
                    "notes_lookback_hours": None,
                    "patient_context_text": {},
                    "agent_context_text": {},
                    "cite_registry": {},
                    "agent_snippets": [],
                    "pipeline_metrics": [],
                    "risk_score": None,
                    "qa_issues": [],
                    "qa_scope_issues": [],
                    "qa_passed": False,
                    "revised_snippets": [],
                    "deliberation_log": [],
                    "intensivist_output": None,
                    "icu_pause_output": {},
                }
            )
            pipeline_output = result.get("icu_pause_output", {})
            pipeline_ok = True
        except Exception as e:
            logger.error(f"Pipeline failed for {hosp_id}: {e}")
            pipeline_output = {"error": str(e)}
            pipeline_ok = False
            result = {}
        pipeline_elapsed_ms = (time.perf_counter() - t0) * 1000

        case_result["pipeline_output"] = pipeline_output
        case_result["pipeline_elapsed_ms"] = round(pipeline_elapsed_ms, 1)

        # Save intermediate states for debugging
        if pipeline_ok:
            agent_snippets = result.get("agent_snippets", [])
            case_result["agent_snippets"] = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in agent_snippets
            ]
            intensivist_out = result.get("intensivist_output")
            if intensivist_out:
                case_result["intensivist_output"] = (
                    intensivist_out.model_dump()
                    if hasattr(intensivist_out, "model_dump")
                    else intensivist_out
                )
            resident_brief = result.get("resident_pre_brief")
            if resident_brief:
                case_result["resident_pre_brief"] = resident_brief

        # Extract pipeline metrics from output metadata
        meta = pipeline_output.get("metadata", {}) if pipeline_ok else {}
        pipeline_metrics = {
            "total_latency_ms": meta.get("total_latency_ms", 0),
            "total_input_tokens": meta.get("total_input_tokens", 0),
            "total_output_tokens": meta.get("total_output_tokens", 0),
        }
        case_result["pipeline_metrics"] = pipeline_metrics

        # Summary row starts with pipeline info
        row: dict[str, Any] = {
            "hospitalization_id": hosp_id,
            "pipeline_ok": pipeline_ok,
            "pipeline_elapsed_ms": round(pipeline_elapsed_ms, 1),
            "total_latency_ms": pipeline_metrics["total_latency_ms"],
            "total_input_tokens": pipeline_metrics["total_input_tokens"],
            "total_output_tokens": pipeline_metrics["total_output_tokens"],
        }

        # --- Pipeline snapshot (prompt versioning) ---
        if pipeline_ok and snapshot_dir:
            from icu_pause.eval.prompt_versioning import capture_snapshot, save_snapshot

            agent_snippets = result.get("agent_snippets", [])
            revised_snippets = result.get("revised_snippets", [])
            qa_issues = result.get("qa_issues", [])
            qa_passed = result.get("qa_passed", True)

            # Pre-QA snapshot (original domain agent outputs)
            pre_qa = capture_snapshot(
                "pre_qa", agent_snippets, prompt_hashes,
                qa_issues=qa_issues, qa_passed=qa_passed,
                prompt_versions=prompt_versions,
            )
            save_snapshot(pre_qa, snapshot_dir, hosp_id)
            case_result["snapshot_pre_qa"] = pre_qa.model_dump()

            # Post-QA snapshot (revised if deliberation ran)
            if revised_snippets:
                from icu_pause.eval.prompt_versioning import compute_deliberation_delta

                # Build effective post-QA snippets
                revised_agents = {s.agent_name for s in revised_snippets}
                effective = [
                    s for s in agent_snippets if s.agent_name not in revised_agents
                ] + list(revised_snippets)
                post_qa = capture_snapshot(
                    "post_qa", effective, prompt_hashes,
                    qa_issues=qa_issues, qa_passed=qa_passed,
                    prompt_versions=prompt_versions,
                )
                save_snapshot(post_qa, snapshot_dir, hosp_id)
                case_result["snapshot_post_qa"] = post_qa.model_dump()

                # Deliberation delta
                delta = compute_deliberation_delta(pre_qa, post_qa)
                case_result["deliberation_delta"] = delta.model_dump()
                row["delib_agents_revised"] = len(delta.agents_revised)
                row["delib_sections_changed"] = delta.sections_changed
                row["delib_change_rate"] = delta.change_rate

            # Post-Intensivist snapshot
            intensivist_out = result.get("intensivist_output")
            if intensivist_out:
                post_int = capture_snapshot(
                    "post_intensivist", [intensivist_out], prompt_hashes,
                    prompt_versions=prompt_versions,
                )
                save_snapshot(post_int, snapshot_dir, hosp_id)

        # --- Custom rubric evaluation ---
        if run_rubric and rubric_evaluator and pipeline_ok:
            try:
                patient_data = result.get("patient_context_text", {})
                rubric_eval = rubric_evaluator.evaluate(patient_data, pipeline_output)
                case_result["rubric_evaluation"] = rubric_eval.model_dump()
                row["rubric_overall"] = rubric_eval.overall_score
                for s in rubric_eval.scores:
                    row[f"rubric_{s.attribute}"] = s.score
            except Exception as e:
                logger.error(f"Rubric evaluation failed for {hosp_id}: {e}")
                case_result["rubric_evaluation"] = {"error": str(e)}
                row["rubric_overall"] = None

        # Build evaluator source context with merged notes (once, reused below)
        agent_contexts = result.get("agent_context_text", {})
        evaluator_source = _build_evaluator_source(
            result.get("patient_context_text", {}), agent_contexts,
        )

        # --- PDSQI-9 evaluation of generated note ---
        if run_pdsqi9 and pdsqi9_evaluator and pipeline_ok:
            try:
                summary_text = render_icu_pause_text(pipeline_output)
                source_text = json.dumps(evaluator_source, indent=2, default=str)
                pdsqi9_eval = pdsqi9_evaluator.evaluate(source_text, summary_text)
                case_result["pdsqi9_evaluation"] = pdsqi9_eval.model_dump()
                row["pdsqi9_total"] = pdsqi9_eval.total_score
                row["pdsqi9_cited"] = pdsqi9_eval.scores.cited
                row["pdsqi9_accurate"] = pdsqi9_eval.scores.accurate
                row["pdsqi9_thorough"] = pdsqi9_eval.scores.thorough
                row["pdsqi9_useful"] = pdsqi9_eval.scores.useful
                row["pdsqi9_organized"] = pdsqi9_eval.scores.organized
                row["pdsqi9_comprehensible"] = pdsqi9_eval.scores.comprehensible
                row["pdsqi9_succinct"] = pdsqi9_eval.scores.succinct
                row["pdsqi9_synthesized"] = pdsqi9_eval.scores.synthesized
                row["pdsqi9_stigmatizing"] = pdsqi9_eval.scores.stigmatizing
            except Exception as e:
                logger.error(f"PDSQI-9 evaluation failed for {hosp_id}: {e}")
                case_result["pdsqi9_evaluation"] = {"error": str(e)}
                row["pdsqi9_total"] = None

            # --- PDSQI-9 on human reference note (for comparison) ---
            if reference_dir:
                ref_note = _load_reference_note(reference_dir, hosp_id)
                if ref_note:
                    try:
                        source_text = json.dumps(evaluator_source, indent=2, default=str)
                        ref_eval = pdsqi9_evaluator.evaluate(source_text, ref_note)
                        case_result["pdsqi9_reference"] = ref_eval.model_dump()
                        row["pdsqi9_reference_total"] = ref_eval.total_score
                        row["pdsqi9_ref_cited"] = ref_eval.scores.cited
                        row["pdsqi9_ref_accurate"] = ref_eval.scores.accurate
                        row["pdsqi9_ref_thorough"] = ref_eval.scores.thorough
                        row["pdsqi9_ref_useful"] = ref_eval.scores.useful
                        row["pdsqi9_ref_organized"] = ref_eval.scores.organized
                        row["pdsqi9_ref_comprehensible"] = ref_eval.scores.comprehensible
                        row["pdsqi9_ref_succinct"] = ref_eval.scores.succinct
                        row["pdsqi9_ref_synthesized"] = ref_eval.scores.synthesized
                        row["pdsqi9_ref_stigmatizing"] = ref_eval.scores.stigmatizing
                    except Exception as e:
                        logger.error(f"PDSQI-9 reference eval failed for {hosp_id}: {e}")
                        case_result["pdsqi9_reference"] = {"error": str(e)}
                        row["pdsqi9_reference_total"] = None
                else:
                    logger.warning(f"No reference note found for {hosp_id}")

        # --- Per-agent grounding / hallucination check ---
        if run_grounding and grounding_evaluator and pipeline_ok:
            try:
                agent_snippets = result.get("agent_snippets", [])
                grounding_eval = grounding_evaluator.evaluate_all_agents(
                    agent_snippets, agent_contexts, evaluator_source,
                )
                case_result["grounding_evaluation"] = grounding_eval.model_dump()
                row["hallucination_rate"] = grounding_eval.overall_hallucination_rate
                for gr in grounding_eval.results:
                    row[f"grounding_{gr.agent_name}_hallucinated"] = gr.hallucinated
                    row[f"grounding_{gr.agent_name}_total"] = gr.total_claims
                    row[f"grounding_{gr.agent_name}_rate"] = round(gr.hallucination_rate, 4)
            except Exception as e:
                logger.error(f"Grounding evaluation failed for {hosp_id}: {e}")
                case_result["grounding_evaluation"] = {"error": str(e)}
                row["hallucination_rate"] = None

        # --- Per-agent data utilization check ---
        if run_data_utilization and data_util_evaluator and pipeline_ok:
            try:
                agent_snippets = result.get("agent_snippets", [])
                util_eval = data_util_evaluator.evaluate_all_agents(
                    agent_snippets, agent_contexts, evaluator_source,
                )
                case_result["data_utilization_evaluation"] = util_eval.model_dump()
                row["data_utilization_mean"] = util_eval.mean_score
                for ur in util_eval.results:
                    row[f"data_util_{ur.agent_name}"] = ur.score
            except Exception as e:
                logger.error(f"Data utilization evaluation failed for {hosp_id}: {e}")
                case_result["data_utilization_evaluation"] = {"error": str(e)}
                row["data_utilization_mean"] = None

        # Collect generated sections for golden case comparison
        if pipeline_ok and golden_dir:
            golden_outputs[hosp_id] = pipeline_output.get("sections", {})

        summary_rows.append(row)

        # Save per-case JSON
        case_path = output_dir / f"{hosp_id}.json"
        with open(case_path, "w") as f:
            json.dump(case_result, f, indent=2, default=str)
        logger.info(f"  Saved {case_path}")

    # --- Golden case batch evaluation (runs once after all cases) ---
    if golden_evaluator and golden_cases and golden_outputs:
        try:
            golden_result = golden_evaluator.evaluate_batch(golden_cases, golden_outputs)
            golden_summary = golden_result.model_dump()

            # Save golden case results
            golden_path = output_dir / "golden_case_results.json"
            with open(golden_path, "w") as f:
                json.dump(golden_summary, f, indent=2, default=str)
            logger.info(f"Golden case results: {golden_path}")

            # Add aggregate golden metrics to all summary rows
            for row in summary_rows:
                row["golden_mean_overall"] = golden_result.mean_overall
                row["golden_critical_check_pass_rate"] = golden_result.critical_check_pass_rate
        except Exception as e:
            logger.error(f"Golden case evaluation failed: {e}")

    # --- Write summary CSV ---
    summary_path = output_dir / "summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        # Ensure all keys from all rows are included
        for r in summary_rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Summary written to {summary_path}")

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluate ICU-PAUSE notes across multiple cases"
    )
    parser.add_argument(
        "--ids",
        required=True,
        help="Text file with one hospitalization_id per line",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write evaluation results",
    )
    parser.add_argument(
        "--rubric",
        action="store_true",
        help="Run custom QA rubric evaluation on generated notes",
    )
    parser.add_argument(
        "--pdsqi9",
        action="store_true",
        help="Run PDSQI-9 evaluation on generated notes",
    )
    parser.add_argument(
        "--grounding",
        action="store_true",
        help="Run per-agent grounding / hallucination check",
    )
    parser.add_argument(
        "--data-utilization",
        action="store_true",
        help="Run per-agent data utilization quality check",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Directory of human-written notes for PDSQI-9 comparison",
    )
    parser.add_argument(
        "--golden-dir",
        default=None,
        help="Directory of golden case JSON files for regression testing",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="Directory to save pipeline snapshots for prompt versioning",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "openai", "local"],
        default=None,
        help="Override LLM provider",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override LLM model name",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=48,
        help="Lookback window in hours (default: 48, 0 for entire stay)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for data cache (skips Parquet loading on cache hit). "
             "Useful when running the same patients across multiple models.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # CLI overrides
    if args.llm_provider:
        os.environ["ICUPAUSE_LLM_PROVIDER"] = args.llm_provider
    if args.llm_model:
        os.environ["ICUPAUSE_LLM_MODEL"] = args.llm_model
    if args.cache_dir:
        os.environ["ICUPAUSE_DATA_CACHE_ENABLED"] = "true"
        os.environ["ICUPAUSE_DATA_CACHE_DIR"] = args.cache_dir

    settings = Settings()
    ids = _load_ids(args.ids)
    lookback = args.lookback_hours if args.lookback_hours > 0 else None

    logger.info(
        f"Batch evaluation: {len(ids)} cases, rubric={args.rubric}, "
        f"pdsqi9={args.pdsqi9}, grounding={args.grounding}, "
        f"data_util={args.data_utilization}, golden={args.golden_dir is not None}"
    )
    logger.info(f"LLM: {settings.llm_provider}/{settings.llm_model}")

    summary_path = run_batch(
        ids=ids,
        settings=settings,
        output_dir=Path(args.output_dir),
        run_rubric=args.rubric,
        run_pdsqi9=args.pdsqi9,
        run_grounding=args.grounding,
        run_data_utilization=args.data_utilization,
        reference_dir=args.reference_dir,
        golden_dir=args.golden_dir,
        snapshot_dir=args.snapshot_dir,
        lookback_hours=lookback,
    )

    print(f"\nBatch complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
