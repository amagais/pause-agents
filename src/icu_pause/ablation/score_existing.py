"""Score already-generated ICU-PAUSE briefs with the numeric-fidelity metric.

No LLM / GPU: ground truth is recomputed deterministically from each case's CLIF
bundle (same DataRetriever call the pipeline uses), then the saved brief's
section text is scored. Use this to (a) sanity-check the metric on real gpt-5.4
production briefs and (b) get a full-pipeline fidelity reference point WITHOUT a
new generation run.

CAVEATS (print them in any comparison):
  * Full-pipeline only — says nothing about the gpt-5.4 MONOLITH arms.
  * Production briefs were made with hybrid_v1 + citations ON; citation markers
    inject numbers that can INFLATE apparent fidelity. Treat as a generous
    ceiling, not apples-to-apples with the early_fusion/citations-off ablation.

Example:
    .venv/bin/python -m icu_pause.ablation.score_existing \
        --briefs-dir output/hitl/validation_merged_20260610 \
        --ids docs/final_cohort_84.csv \
        --out output/ablation/gpt54_existing_fidelity
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import statistics
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_sections(brief: dict) -> dict:
    """Tolerate both save shapes: top-level ICUPauseOutput, or {pipeline_output}."""
    if isinstance(brief.get("sections"), dict):
        return brief["sections"]
    po = brief.get("pipeline_output")
    if isinstance(po, dict) and isinstance(po.get("sections"), dict):
        return po["sections"]
    return {}


def main() -> None:
    p = argparse.ArgumentParser(description="Score existing briefs with numeric fidelity")
    p.add_argument("--briefs-dir", required=True)
    p.add_argument("--ids", required=True, help="Cohort CSV (hospitalization_id, reference_dttm)")
    p.add_argument("--out", required=True)
    p.add_argument("--glob", default="*.brief.json")
    p.add_argument("--lookback-hours", type=int, default=48)
    p.add_argument("--notes-lookback-hours", type=int, default=None)
    p.add_argument("--tolerance", type=float, default=0.05)
    p.add_argument("--label", default="existing", help="Label for the headline block")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Ground truth must match the bundle every arm sees: citations off so the GT
    # source is byte-consistent with the ablation runner's bundle.
    os.environ["ICUPAUSE_CITATION_MODE"] = "off"

    from icu_pause.config import Settings
    from icu_pause.ablation.arms import retrieve_bundle
    from icu_pause.ablation.run import _load_cohort
    from icu_pause.eval.numeric_fidelity import extract_ground_truth, score_brief

    settings = Settings()
    ref_by_hid = {r["hospitalization_id"]: r["reference_dttm"] for r in _load_cohort(args.ids)}
    lookback = args.lookback_hours if args.lookback_hours > 0 else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(Path(args.briefs_dir) / args.glob)))
    logger.info("found %d briefs in %s", len(files), args.briefs_dir)

    overall_scores: list[float] = []
    byt: dict[str, list[float]] = defaultdict(list)
    ins: dict[str, list[int]] = defaultdict(list)
    vco: list[float] = []
    rows: list[dict] = []

    for fp in files:
        hid = Path(fp).name.replace(".brief.json", "").replace(".json", "")
        ref = ref_by_hid.get(hid)
        if not ref:
            logger.warning("no reference_dttm for %s in cohort CSV — skipping", hid)
            continue
        try:
            brief = json.load(open(fp))
            sections = _extract_sections(brief)
            if not sections:
                logger.warning("no sections in %s — skipping", fp)
                continue
            bundle = retrieve_bundle(settings, hid, ref, lookback, args.notes_lookback_hours)
            gt = extract_ground_truth(bundle)
            fid = score_brief(sections, gt, tolerance=args.tolerance).to_dict()
        except Exception as e:  # noqa: BLE001
            logger.error("scoring failed for %s: %s", hid, e)
            continue

        (out / f"{hid}.fidelity.json").write_text(json.dumps(fid, indent=2, default=str))
        ov = fid["overall"]["retention"]
        if ov is not None:
            overall_scores.append(ov)
        for t, v in fid["by_type"].items():
            if v["retention"] is not None:
                byt[t].append(v["retention"])
            ins[t].append(v["in_scope"])
        s = fid["vitals_current_only_sensitivity"]["retention"]
        if s is not None:
            vco.append(s)
        rows.append({"hospitalization_id": hid, "overall_retention": ov,
                     "overall_excl_abx": fid["overall_excl_abx"]["retention"],
                     "vitals_current_only": s,
                     **{f"ret_{t}": fid["by_type"][t]["retention"] for t in fid["by_type"]}})

    import csv
    if rows:
        fields = sorted({k for r in rows for k in r})
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    def _m(xs):
        return statistics.mean(xs) if xs else None

    print("\n" + "=" * 64)
    print(f"NUMERIC FIDELITY on existing briefs [{args.label}] — n={len(overall_scores)}")
    print("=" * 64)
    om = _m(overall_scores)
    print(f"  overall retention      {om:.3f}" if om is not None else "  overall   n/a")
    if vco:
        print(f"  vitals_current_only    {_m(vco):.3f}")
    print("  per type (retention | mean denominator):")
    for t in byt:
        print(f"    {t:22s} {_m(byt[t]):.3f}  | {_m(ins[t]):.1f}")
    print("=" * 64 + "\n")
    (out / "headline.json").write_text(json.dumps({
        "label": args.label, "n": len(overall_scores),
        "overall_mean": om, "vitals_current_only_mean": _m(vco),
        "by_type_mean": {t: _m(v) for t, v in byt.items()},
        "by_type_mean_in_scope": {t: _m(v) for t, v in ins.items()},
    }, indent=2, default=str))
    logger.info("wrote per-brief fidelity + summary under %s", out)


if __name__ == "__main__":
    main()
