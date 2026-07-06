"""CLI entry point for generating ICU-PAUSE handoff briefs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

from icu_pause.config import Settings
from icu_pause.graph.workflow import build_graph
from icu_pause.rendering.formatter import render_icu_pause_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an ICU-PAUSE handoff brief for a single patient"
    )
    parser.add_argument(
        "--hospitalization-id",
        required=True,
        help="CLIF hospitalization_id to generate the brief for",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=48,
        help="Number of hours of data to use before the current time point "
        "(default: 48). Set to 0 for entire stay.",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "text", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Write JSON output to this file",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "openai", "azure", "local"],
        default=None,
        help="Override LLM provider (default: from settings)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override LLM model name (default: from settings)",
    )
    parser.add_argument(
        "--reference-dttm",
        required=True,
        help="Reference timestamp (ISO-8601) for retrospective mode. Required — "
        "supply the first ICU→ward transfer note time from the cohort CSV. "
        "Notes and structured data after this time are excluded to prevent leakage.",
    )
    parser.add_argument(
        "--notes-lookback-hours",
        type=int,
        default=None,
        help="Hours of note history to include (from reference_dttm). "
        "Default: from settings (48h).",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=[
            "early_fusion",
            "cr_dsf",
            "cr_dsf_plus",
            "hybrid_v1",
            "hybrid_v1_no_anchor",
        ],
        default=None,
        help=(
            "Fusion strategy: early_fusion (default); cr_dsf (legacy dual-stream); "
            "cr_dsf_plus (legacy dual-stream + structured extraction); "
            "hybrid_v1 (Option B: per-domain extractors + Stage E anchor override; "
            "pre-reg §1.3); hybrid_v1_no_anchor (mechanism-ablation variant of "
            "hybrid_v1 with use_anchor_override=False; pre-reg §1.7)."
        ),
    )
    parser.add_argument(
        "--structured-axis",
        choices=["s0", "s1", "s2"],
        default=None,
        help=(
            "Structured-compression axis for the compression sub-study (overrides "
            "--fusion-mode): s0 (raw tiered tables), s1 (LLM summary), s2 (LLM "
            "salience-selected substitutive view). Compose with --notes-axis."
        ),
    )
    parser.add_argument(
        "--notes-axis",
        choices=["n0", "n1", "n2"],
        default=None,
        help=(
            "Notes-compression axis for the compression sub-study (overrides "
            "--fusion-mode): n0 (raw routed), n1 (LLM summary), n2 (per-domain "
            "extracted anchors, substitutive). Compose with --structured-axis."
        ),
    )
    parser.add_argument(
        "--no-anchor-override",
        action="store_true",
        help="Force Stage-E anchor-override OFF (for axis-cell or hybrid_v1 ablation runs).",
    )
    delib_group = parser.add_mutually_exclusive_group()
    delib_group.add_argument(
        "--deliberation",
        action="store_true",
        help="Enable agent deliberation on QA conflicts",
    )
    delib_group.add_argument(
        "--no-deliberation",
        action="store_true",
        help="Disable agent deliberation (default)",
    )
    parser.add_argument(
        "--citation-mode",
        choices=["off", "decision_critical", "all"],
        default=None,
        help="Citation mode: off (no citations), decision_critical (default), "
        "all (cite every value)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Apply CLI overrides to env before loading settings
    if args.llm_provider:
        os.environ["ICUPAUSE_LLM_PROVIDER"] = args.llm_provider
    if args.llm_model:
        os.environ["ICUPAUSE_LLM_MODEL"] = args.llm_model
    # Explicit compression axes (clean-cell mode) take precedence over the legacy
    # --fusion-mode enum; error if both are given and disagree to avoid ambiguity.
    if (args.structured_axis or args.notes_axis) and args.fusion_mode:
        parser.error(
            "Use either --fusion-mode OR --structured-axis/--notes-axis, not both "
            "(they map to different cells; combining them is ambiguous)."
        )
    if args.structured_axis:
        os.environ["ICUPAUSE_STRUCTURED_AXIS"] = args.structured_axis
    if args.notes_axis:
        os.environ["ICUPAUSE_NOTES_AXIS"] = args.notes_axis
    if args.structured_axis or args.notes_axis:
        # Clean axis cells never apply Stage-E anchor-override.
        os.environ["ICUPAUSE_USE_ANCHOR_OVERRIDE"] = "false"
    if args.no_anchor_override:
        os.environ["ICUPAUSE_USE_ANCHOR_OVERRIDE"] = "false"
    if args.fusion_mode:
        # hybrid_v1_no_anchor is the hybrid_v1 workflow branch with the
        # anchor-override flag forced off (pre-reg §1.7 ablation). The
        # workflow.py branch is shared.
        if args.fusion_mode == "hybrid_v1_no_anchor":
            os.environ["ICUPAUSE_FUSION_MODE"] = "hybrid_v1"
            os.environ["ICUPAUSE_USE_ANCHOR_OVERRIDE"] = "false"
        else:
            os.environ["ICUPAUSE_FUSION_MODE"] = args.fusion_mode
            if args.fusion_mode == "hybrid_v1" and not args.no_anchor_override:
                os.environ["ICUPAUSE_USE_ANCHOR_OVERRIDE"] = "true"
    if args.deliberation:
        os.environ["ICUPAUSE_DELIBERATION_ENABLED"] = "true"
    elif args.no_deliberation:
        os.environ["ICUPAUSE_DELIBERATION_ENABLED"] = "false"
    if args.citation_mode:
        os.environ["ICUPAUSE_CITATION_MODE"] = args.citation_mode

    # Load settings
    settings = Settings()
    logger.info(f"Using LLM provider: {settings.llm_provider} ({settings.llm_model})")
    logger.info(f"CLIF data dir: {settings.clif_data_dir}")
    # Build and run the graph
    logger.info(f"Building ICU-PAUSE workflow graph...")
    graph = build_graph(settings)

    # Convert 0 to None (entire stay)
    lookback_hours = args.lookback_hours if args.lookback_hours > 0 else None
    window_desc = f"last {lookback_hours}h" if lookback_hours is not None else "entire stay"
    logger.info(f"Generating brief for hospitalization: {args.hospitalization_id} ({window_desc})")
    wall_clock_started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    result = graph.invoke(
        {
            "hospitalization_id": args.hospitalization_id,
            "lookback_hours": lookback_hours,
            "reference_dttm": args.reference_dttm,
            "notes_lookback_hours": args.notes_lookback_hours,
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
    )
    wall_clock_ms = (time.perf_counter() - t0) * 1000
    wall_clock_completed_at = datetime.now(timezone.utc).isoformat()
    logger.info(
        f"Pipeline wall-clock: {wall_clock_ms / 1000:.1f}s "
        f"(end-to-end, includes data load + safety checks + serialization)"
    )

    # Save run trace
    trace_events = result.get("trace_events", [])
    if trace_events:
        from icu_pause.tracing import RunTrace
        rt = RunTrace(args.hospitalization_id)
        rt.events = trace_events
        trace_path = rt.save()
        if trace_path:
            logger.info(f"Run trace saved to {trace_path}")

    output = result.get("icu_pause_output", {})

    # Inject wall-clock into metadata. total_latency_ms in metadata sums
    # only LLM call latencies; wall_clock_ms is end-to-end (data load,
    # deterministic safety checks, serialization included) — the right
    # number for operational deployability claims.
    if isinstance(output, dict):
        md = output.setdefault("metadata", {})
        md["wall_clock_ms"] = round(wall_clock_ms, 1)
        md["wall_clock_started_at"] = wall_clock_started_at
        md["wall_clock_completed_at"] = wall_clock_completed_at

    # Display output
    if args.output_format in ("json", "both"):
        print(json.dumps(output, indent=2, default=str))

    if args.output_format in ("text", "both"):
        print()
        print(render_icu_pause_text(output))

    # Save to file
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Output written to {args.output_file}")


if __name__ == "__main__":
    main()
