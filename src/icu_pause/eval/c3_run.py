"""C3 batch runner — score existing briefs with the two-sided I-PASS/JC fidelity
harness (deterministic core + DeepSeek hybrid LLM pass), one arm/generator at a time.

Mirrors ablation/score_existing.py's data plumbing (retrieve_bundle gives the
serialized chart both numeric_fidelity and the LLM pass consume; cohort CSV gives
reference_dttm). Runs identically on the full (multiagent) and monolith arms so the
comparison is schema-exogenous and arm-neutral (plan C3).

DETERMINISTIC side (free): numeric-backed elements via numeric_fidelity (recall;
numeric precision is a known TODO — numeric_fidelity conflates omission vs wrong-
value, so those elements contribute to recall only, never to precision).
LLM side (free on local DeepSeek-R1): every element the deterministic pass leaves
PENDING — the 12 prose elements + the 2 deterministic-TODO (labs, vent_dependence)
+ the prose half of the hybrids — gets a full two-sided verdict.

Idempotent: skips a case whose <hid>.c3.json already exists.

Example (HPC):
  # deterministic only (free, no GPU) — sanity check:
  .venv/bin/python -m icu_pause.eval.c3_run --briefs-dir output/hitl/validation_merged_20260610 \
     --ids docs/final_cohort_84.csv --arm full --generator gpt54 \
     --out output/c3/full_gpt54 --no-llm
  # full two-sided with local DeepSeek-R1 (free on HPC vLLM):
  .venv/bin/python -m icu_pause.eval.c3_run --briefs-dir output/hitl/validation_merged_20260610 \
     --ids docs/final_cohort_84.csv --arm full --generator gpt54 \
     --out output/c3/full_gpt54 --llm-provider local --llm-model deepseek-r1
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from icu_pause.eval import c3_definitions as C3
from icu_pause.eval.c3_score import (
    C3Result, score_deterministic, PENDING_LLM, PENDING_DET,
)

logger = logging.getLogger(__name__)


def _extract_sections(brief: dict) -> dict:
    if isinstance(brief.get("sections"), dict):
        return brief["sections"]
    po = brief.get("pipeline_output")
    if isinstance(po, dict) and isinstance(po.get("sections"), dict):
        return po["sections"]
    return {}


def _brief_text(brief: dict, sections: dict) -> str:
    try:
        from icu_pause.rendering.formatter import render_icu_pause_text
        t = render_icu_pause_text(brief)
        if t and t.strip():
            return t
    except Exception:  # noqa: BLE001
        pass
    return "\n\n".join(f"## {k}\n{v}" for k, v in sections.items())


def score_case(sections, ctx, hosp_id, arm, generator, llm=None, batch_size=6) -> C3Result:
    """Deterministic pass, then (optionally) the LLM pass over PENDING elements."""
    res = score_deterministic(sections, ctx, hosp_id, arm, generator)
    if llm is None:
        return res
    target_ids = {v.element_id for v in res.verdicts
                  if v.status in (PENDING_LLM, PENDING_DET)}
    targets = [g for g in C3.GOLD_REGISTRY if g.id in target_ids]
    if not targets:
        return res
    from icu_pause.eval.c3_llm import evaluate_elements
    chart_json = json.dumps(ctx, default=str)
    brief = {"sections": sections}
    llm_verdicts = evaluate_elements(llm, _brief_text(brief, sections), chart_json,
                                     targets, batch_size=batch_size)
    kept = [v for v in res.verdicts if v.element_id not in target_ids]
    return C3Result(hosp_id, arm, generator, kept + llm_verdicts)


def main() -> None:
    p = argparse.ArgumentParser(description="C3 two-sided fidelity over existing briefs")
    p.add_argument("--briefs-dir", default=None,
                   help="dir of <hid>.brief.json files (mode A)")
    p.add_argument("--condition", default=None,
                   help="judge-staged condition slug, e.g. gemma4_multiagent (mode B): "
                        "reads <base>/<condition>/cases/<hid>/output.json — the SAME "
                        "briefs the PDSQI judge saw, for every generator/arm")
    p.add_argument("--base", default="output/judge_calibration",
                   help="root for --condition staged cases")
    p.add_argument("--ids", required=True, help="cohort CSV (hospitalization_id, reference_dttm)")
    p.add_argument("--out", required=True)
    p.add_argument("--arm", required=True, help="full | monolith (label only)")
    p.add_argument("--generator", required=True, help="gpt54 | gemma4 | medgemma | qwen36")
    p.add_argument("--glob", default="*.brief.json")
    p.add_argument("--lookback-hours", type=int, default=48)
    p.add_argument("--notes-lookback-hours", type=int, default=None)
    p.add_argument("--no-llm", action="store_true", help="deterministic only (free, no GPU)")
    p.add_argument("--llm-provider", default="local")
    p.add_argument("--llm-model", default="deepseek-r1")
    p.add_argument("--llm-batch-size", type=int, default=6,
                   help="gold elements per LLM call (smaller = safer for verbose models)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # GT bundle byte-consistent with the ablation runner (citations off), matching
    # score_existing.py so every arm's ground truth is identical.
    os.environ["ICUPAUSE_CITATION_MODE"] = "off"

    from icu_pause.config import Settings
    from icu_pause.ablation.arms import retrieve_bundle
    from icu_pause.ablation.run import _load_cohort

    settings = Settings()
    ref_by_hid = {r["hospitalization_id"]: r["reference_dttm"] for r in _load_cohort(args.ids)}
    lookback = args.lookback_hours if args.lookback_hours > 0 else None

    llm = None
    if not args.no_llm:
        from icu_pause.eval.c3_llm import build_llm
        llm = build_llm(args.llm_provider, args.llm_model)
        logger.info("LLM judge: provider=%s model=%s", args.llm_provider, args.llm_model)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Enumerate (hid, brief_path). Mode A: glob a briefs dir. Mode B: judge-staged
    # cases/<hid>/output.json for a condition (known paths for every generator/arm).
    if args.condition:
        cases = Path(args.base) / args.condition / "cases"
        pairs = [(h, str(cases / h / "output.json"))
                 for h in ref_by_hid if (cases / h / "output.json").exists()]
        logger.info("condition=%s: %d staged cases under %s", args.condition, len(pairs), cases)
    elif args.briefs_dir:
        pairs = [(Path(fp).name.replace(".brief.json", "").replace(".json", ""), fp)
                 for fp in sorted(glob.glob(str(Path(args.briefs_dir) / args.glob)))]
        logger.info("found %d briefs in %s", len(pairs), args.briefs_dir)
    else:
        p.error("pass --condition or --briefs-dir")
    if args.limit:
        pairs = pairs[: args.limit]

    recalls, precisions, fs, covs = [], [], [], []
    omit_counter, fab_counter = Counter(), Counter()
    n_done = n_skip = n_fail = 0

    for hid, fp in pairs:
        dst = out / f"{hid}.c3.json"
        if dst.exists():
            n_skip += 1
            continue
        ref = ref_by_hid.get(hid)
        if not ref:
            logger.warning("no reference_dttm for %s — skip", hid)
            n_fail += 1
            continue
        try:
            brief = json.load(open(fp))
            sections = _extract_sections(brief)
            if not sections:
                logger.warning("no sections in %s — skip", fp)
                n_fail += 1
                continue
            ctx = retrieve_bundle(settings, hid, ref, lookback, args.notes_lookback_hours)
            res = score_case(sections, ctx, hid, args.arm, args.generator, llm=llm,
                             batch_size=args.llm_batch_size)
        except Exception as e:  # noqa: BLE001
            logger.error("C3 failed for %s: %s", hid, e)
            n_fail += 1
            continue

        d = res.to_dict()
        dst.write_text(json.dumps(d, indent=2, default=str))
        n_done += 1
        if d["recall"] is not None:
            recalls.append(d["recall"])
        if d["precision"] is not None:
            precisions.append(d["precision"])
        if d["f_two_sided"] is not None:
            fs.append(d["f_two_sided"])
        if d["coverage_resolved"] is not None:
            covs.append(d["coverage_resolved"])
        omit_counter.update(d["omissions"])
        fab_counter.update(d["fabrications"])

    def _m(xs):
        return statistics.mean(xs) if xs else None

    summary = {
        "arm": args.arm, "generator": args.generator,
        "llm": None if args.no_llm else f"{args.llm_provider}/{args.llm_model}",
        "n_scored": n_done, "n_skipped": n_skip, "n_failed": n_fail,
        "mean_recall": _m(recalls), "mean_precision": _m(precisions),
        "mean_f_two_sided": _m(fs), "mean_coverage_resolved": _m(covs),
        "top_omissions": omit_counter.most_common(12),
        "top_fabrications": fab_counter.most_common(12),
    }
    (out / "headline.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 64)
    print(f"C3 FIDELITY [{args.arm}/{args.generator}] — n={n_done} "
          f"(skip {n_skip}, fail {n_fail})")
    print("=" * 64)
    for k in ("mean_recall", "mean_precision", "mean_f_two_sided", "mean_coverage_resolved"):
        v = summary[k]
        print(f"  {k:24s} {v:.3f}" if v is not None else f"  {k:24s} n/a")
    print(f"  top omissions:    {summary['top_omissions'][:6]}")
    print(f"  top fabrications: {summary['top_fabrications'][:6]}")
    print("=" * 64)
    logger.info("wrote per-case C3 + headline under %s", out)


if __name__ == "__main__":
    main()
