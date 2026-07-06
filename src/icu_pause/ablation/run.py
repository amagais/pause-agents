"""Runner for the decomposition ablation.

Reads a cohort CSV (columns: hospitalization_id, reference_dttm), runs the
selected arms on each case under IDENTICAL data/model/temperature, scores the
PRIMARY numeric-fidelity endpoint per arm against the arm-independent shared
bundle, writes per-case artifacts, and prints the headline full-vs-monolith gap.

Fairness invariants enforced here:
  * fusion_mode = early_fusion for all graph arms (compression sub-study is out
    of scope).
  * temperature = 0 for ALL arms (the full pipeline's domain agents default to
    0.2 otherwise — that would break the identical-temp requirement).
  * ground truth is extracted ONCE per case from the shared bundle and reused
    for every arm, so the denominator is provably identical across arms.

Example (Gemma 4 screening smoke, arms 1-3, 3 depth cases):
    .venv/bin/python -m icu_pause.ablation.run \
        --ids docs/depth_test_cases.csv \
        --arms full,monolith_best_effort,monolith_templated \
        --llm-provider local --llm-model gemma-3-27b-it \
        --out output/ablation/gemma_smoke --limit 3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_cohort(path: str) -> list[dict[str, str]]:
    """Read cohort rows with hospitalization_id + reference_dttm (extra cols ok)."""
    rows: list[dict[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        hid = cols.get("hospitalization_id") or cols.get("hosp_id")
        ref = cols.get("reference_dttm") or cols.get("reference_dttm_utc")
        if not hid or not ref:
            raise SystemExit(
                f"cohort {path} must have hospitalization_id and reference_dttm "
                f"columns; found {reader.fieldnames}"
            )
        for r in reader:
            h = (r.get(hid) or "").strip()
            rd = (r.get(ref) or "").strip()
            if h and not h.startswith("#"):
                rows.append({"hospitalization_id": h, "reference_dttm": rd,
                             "stratum": (r.get(cols.get("stratum", "")) or "").strip()})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Run the decomposition ablation")
    p.add_argument("--ids", required=True, help="Cohort CSV (hospitalization_id, reference_dttm)")
    p.add_argument("--arms", default="all",
                   help="Comma-separated arm keys, or 'all' (default)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--llm-provider", default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--lookback-hours", type=int, default=48)
    p.add_argument("--notes-lookback-hours", type=int, default=None)
    p.add_argument("--tolerance", type=float, default=0.05, help="Relative fidelity tolerance")
    p.add_argument("--limit", type=int, default=None, help="Cap number of cases (smoke)")
    p.add_argument("--full-from-existing", default=None,
                   help="Directory of EXISTING production full briefs (<hid>.brief.json) to "
                        "REUSE as the 'full' arm (citations stripped) instead of regenerating "
                        "the pipeline. The 'full' arm is auto-added when set.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Generation temperature for the monolith arms (default 0.0 for "
                        "determinism; set to match the reused full briefs, e.g. 0.2).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # --- Enforce fairness invariants via env BEFORE Settings() is built ---
    os.environ["ICUPAUSE_FUSION_MODE"] = "early_fusion"
    os.environ["ICUPAUSE_USE_ANCHOR_OVERRIDE"] = "false"
    os.environ["ICUPAUSE_LLM_TEMPERATURE"] = str(args.temperature)
    # Citations off: their (date 1-15 14:30) markers inject numbers my fidelity
    # regex would match spuriously, and would otherwise differ between the
    # ground-truth bundle (runner) and the pipeline's internal bundle. Off keeps
    # GT source and every arm's input byte-consistent.
    os.environ["ICUPAUSE_CITATION_MODE"] = "off"
    if args.llm_provider:
        os.environ["ICUPAUSE_LLM_PROVIDER"] = args.llm_provider
    if args.llm_model:
        os.environ["ICUPAUSE_LLM_MODEL"] = args.llm_model

    from icu_pause.config import Settings
    from icu_pause.ablation.arms import ARM_KEYS, build_arm, retrieve_bundle
    from icu_pause.eval.numeric_fidelity import extract_ground_truth, score_brief

    settings = Settings()
    logger.info("provider=%s model=%s temp=%s fusion=%s",
                settings.llm_provider, settings.llm_model,
                settings.llm_temperature, settings.fusion_mode)

    arm_keys = ARM_KEYS if args.arms == "all" else [a.strip() for a in args.arms.split(",")]
    # Reuse mode: the 'full' arm is sourced from existing briefs — add it so all
    # three arms are scored even when --arms lists only the monoliths to generate.
    if args.full_from_existing and "full" not in arm_keys:
        arm_keys = ["full"] + arm_keys
    lookback = args.lookback_hours if args.lookback_hours > 0 else None

    # Build arms (skip not-yet-wired ones gracefully so an 'all' smoke still runs).
    arms = []
    for k in arm_keys:
        try:
            arms.append(build_arm(
                k, settings,
                full_from_existing=(args.full_from_existing if k == "full" else None),
                temperature=args.temperature,
            ))
        except NotImplementedError as e:
            logger.warning("skipping arm %s: %s", k, e)
    if not arms:
        raise SystemExit("no runnable arms selected")
    logger.info("running arms: %s", [a.key for a in arms])

    cohort = _load_cohort(args.ids)
    if args.limit:
        cohort = cohort[: args.limit]
    logger.info("cohort: %d cases", len(cohort))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # per_arm_scores[arm] = list of overall retention floats (one per case)
    per_arm_scores: dict[str, list[float]] = {a.key: [] for a in arms}
    summary_rows: list[dict] = []

    for i, case in enumerate(cohort, 1):
        hid = case["hospitalization_id"]
        ref = case["reference_dttm"]
        logger.info("[%d/%d] case %s", i, len(cohort), hid)
        try:
            bundle = retrieve_bundle(settings, hid, ref, lookback, args.notes_lookback_hours)
            gt = extract_ground_truth(bundle)
        except Exception as e:  # noqa: BLE001
            logger.error("retrieval/ground-truth failed for %s: %s", hid, e)
            continue

        case_dir = out / hid
        case_dir.mkdir(exist_ok=True)
        (case_dir / "ground_truth.json").write_text(
            json.dumps([v.__dict__ for v in gt.values], indent=2, default=str))

        for arm in arms:
            row = {"hospitalization_id": hid, "arm": arm.key, "stratum": case.get("stratum", "")}
            try:
                brief = arm.run_case(hid, ref, lookback, args.notes_lookback_hours,
                                     bundle=bundle)
                fid = score_brief(brief.get("sections", {}), gt, tolerance=args.tolerance)
                fd = fid.to_dict()
                (case_dir / f"{arm.key}.brief.json").write_text(
                    json.dumps(brief, indent=2, default=str))
                (case_dir / f"{arm.key}.fidelity.json").write_text(
                    json.dumps(fd, indent=2, default=str))
                overall = fd["overall"]["retention"]
                if overall is not None:
                    per_arm_scores[arm.key].append(overall)
                row.update({
                    "ok": True,
                    "overall_retention": overall,
                    "overall_excl_abx": fd["overall_excl_abx"]["retention"],
                    "vitals_current_only": fd["vitals_current_only_sensitivity"]["retention"],
                    **{f"ret_{dt}": fd["by_type"][dt]["retention"] for dt in fd["by_type"]},
                    "in_scope": fd["overall"]["in_scope"],
                })
            except Exception as e:  # noqa: BLE001
                logger.error("arm %s failed on %s: %s", arm.key, hid, e)
                row.update({"ok": False, "error": str(e)})
            summary_rows.append(row)

    # --- Write summary CSV ---
    summary_path = out / "summary.csv"
    if summary_rows:
        fields = sorted({k for r in summary_rows for k in r})
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summary_rows)

    # --- Headline: full vs BOTH monolith baselines on numeric fidelity ---
    def _mean(xs):
        return statistics.mean(xs) if xs else None

    means = {k: _mean(v) for k, v in per_arm_scores.items()}
    headline = {"means": means, "n_by_arm": {k: len(v) for k, v in per_arm_scores.items()}}
    full = means.get("full")
    be = means.get("monolith_best_effort")
    tm = means.get("monolith_templated")
    print("\n" + "=" * 64)
    print("DECOMPOSITION ABLATION — numeric-fidelity (mean overall retention)")
    print("=" * 64)
    for k in (a.key for a in arms):
        m = means.get(k)
        print(f"  {k:24s} {m:.3f}" if m is not None else f"  {k:24s}   n/a")
    if full is not None and be is not None and tm is not None:
        d_be, d_tm = full - be, full - tm
        headline["delta_full_minus_best_effort"] = d_be
        headline["delta_full_minus_templated"] = d_tm
        print("-" * 64)
        print(f"  full − best_effort : {d_be:+.3f}")
        print(f"  full − templated   : {d_tm:+.3f}")
        beats_both = d_be > 0 and d_tm > 0
        marginal = max(abs(d_be), abs(d_tm)) < 0.03 or not beats_both
        verdict = ("MARGINAL — gap small or full does not beat both baselines; "
                   "headline at risk" if marginal else
                   "full beats BOTH baselines — promising; confirm with CIs + Wilcoxon on n=84")
        headline["verdict"] = verdict
        headline["beats_both"] = beats_both
        print(f"  VERDICT: {verdict}")
        print("  NOTE: means only (smoke). 95% CIs + Wilcoxon signed-rank come in the analysis step.")
    print("=" * 64 + "\n")
    (out / "headline.json").write_text(json.dumps(headline, indent=2, default=str))
    logger.info("wrote %s and per-case artifacts under %s", summary_path, out)


if __name__ == "__main__":
    main()
