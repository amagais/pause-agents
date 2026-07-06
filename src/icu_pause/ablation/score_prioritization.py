"""Run top-3 prioritization concordance over ablation-run briefs.

Layout expected (from ablation.run): <ablation_out>/<hid>/<arm>.brief.json.
For each case: recover the human transfer note, extract the clinician's top-3
priorities ONCE, then judge each arm's brief for coverage. Aggregates concordance
per arm.

Judge model is pluggable: --judge-provider/--judge-model (default: local, the
Gemma screening judge). Use azure + o3-mini for the publication number — but note
that judging Gemma-generated briefs with a Gemma judge is a weak/biased screen;
the cross-model judge is the trustworthy one.

Example (after the n=25 run, server free):
    .venv/bin/python -m icu_pause.ablation.score_prioritization \
        --ablation-out output/ablation/gemma_n25 \
        --ids docs/validation_cohort_25.csv \
        --judge-provider local --judge-model google/gemma-4-31B-it \
        --out output/ablation/gemma_n25_prioritization
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import statistics
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def _brief_text(brief: dict) -> str:
    secs = brief.get("sections")
    if isinstance(secs, dict):
        return "\n".join(f"{k}: {v}" for k, v in secs.items())
    po = brief.get("pipeline_output") or {}
    secs = po.get("sections") if isinstance(po, dict) else None
    return "\n".join(f"{k}: {v}" for k, v in secs.items()) if isinstance(secs, dict) else ""


def _usage(llm) -> tuple[int, int]:
    """(input_tokens, output_tokens) from the judge's most recent invoke."""
    u = getattr(llm, "last_usage", None)
    return int(getattr(u, "input_tokens", 0) or 0), int(getattr(u, "output_tokens", 0) or 0)


def main() -> None:
    p = argparse.ArgumentParser(description="Top-3 prioritization concordance")
    p.add_argument("--ablation-out", required=True, help="<dir>/<hid>/<arm>.brief.json")
    p.add_argument("--ids", required=True, help="Cohort CSV (hospitalization_id, reference_dttm)")
    p.add_argument("--arms", default="full,monolith_best_effort,monolith_templated")
    p.add_argument("--judge-provider", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    os.environ["ICUPAUSE_CITATION_MODE"] = "off"
    if args.judge_provider:
        os.environ["ICUPAUSE_LLM_PROVIDER"] = args.judge_provider
    if args.judge_model:
        os.environ["ICUPAUSE_LLM_MODEL"] = args.judge_model

    from icu_pause.config import Settings
    from icu_pause.data.retriever import DataRetriever
    from icu_pause.graph.workflow import _parse_reference_dttm
    from icu_pause.llm.provider import create_llm
    from icu_pause.ablation.prioritization import (
        retrieve_reference_note, extract_priorities, judge_coverage,
    )

    settings = Settings()
    retriever = DataRetriever(settings)
    # Generous output budget: a reasoning judge (DeepSeek-R1) spends tokens on
    # <think> before emitting the PROBLEM:/YES-NO lines — too small a budget cuts
    # the answer off. The provider strips <think> from the returned text, so the
    # line-parsers see the clean answer.
    judge = create_llm(settings, temperature_override=0.0, max_tokens_override=12000)
    arms = [a.strip() for a in args.arms.split(",")]
    ref_by_hid = {r["hospitalization_id"]: r["reference_dttm"]
                  for r in csv.DictReader(open(args.ids))}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_arm: dict[str, list[float]] = defaultdict(list)
    rows: list[dict] = []
    tok_in = tok_out = 0  # judge token accounting
    case_dirs = sorted(d for d in glob.glob(str(Path(args.ablation_out) / "*")) if Path(d).is_dir())
    logger.info("scoring prioritization for %d case dirs", len(case_dirs))

    for cdir in case_dirs:
        hid = Path(cdir).name
        ref = ref_by_hid.get(hid)
        if not ref:
            logger.warning("no reference_dttm for %s — skipping", hid)
            continue
        ref_dt = _parse_reference_dttm(ref)
        note = retrieve_reference_note(retriever, hid, ref_dt)
        if not note:
            logger.warning("no transfer note recovered for %s — skipping", hid)
            continue
        priorities = extract_priorities(judge, note)
        di, do = _usage(judge); tok_in += di; tok_out += do
        if not priorities:
            logger.warning("no priorities extracted for %s — skipping", hid)
            continue

        case_rec = {"hospitalization_id": hid, "priorities": priorities, "arms": {}}
        for arm in arms:
            bp = Path(cdir) / f"{arm}.brief.json"
            if not bp.exists():
                continue
            brief = json.load(open(bp))
            covered = judge_coverage(judge, priorities, _brief_text(brief))
            di, do = _usage(judge); tok_in += di; tok_out += do
            matched = sum(1 for c in covered if c)
            conc = matched / len(priorities)
            per_arm[arm].append(conc)
            case_rec["arms"][arm] = {"covered": covered, "matched": matched, "concordance": conc}
            rows.append({"hospitalization_id": hid, "arm": arm,
                         "matched": matched, "of": len(priorities), "concordance": conc})
        (out / f"{hid}.prioritization.json").write_text(json.dumps(case_rec, indent=2, default=str))

    if rows:
        fields = ["hospitalization_id", "arm", "matched", "of", "concordance"]
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    print("\n" + "=" * 64)
    print("TOP-3 PRIORITIZATION CONCORDANCE (recall of clinician priorities)")
    print("=" * 64)
    means = {}
    for arm in arms:
        xs = per_arm.get(arm, [])
        means[arm] = statistics.mean(xs) if xs else None
        print(f"  {arm:24s} {means[arm]:.3f}  (n={len(xs)})" if xs else f"  {arm:24s}  n/a")
    base = means.get("monolith_best_effort")
    full = means.get("full")
    if full is not None and base is not None:
        print("-" * 64)
        print(f"  full - best_effort : {full - base:+.3f}")
    print("-" * 64)
    print(f"  judge tokens: in={tok_in:,}  out={tok_out:,}  total={tok_in + tok_out:,}")
    print("=" * 64 + "\n")
    (out / "headline.json").write_text(json.dumps(
        {"means": means, "n_by_arm": {a: len(per_arm.get(a, [])) for a in arms},
         "judge_tokens": {"input": tok_in, "output": tok_out, "total": tok_in + tok_out}},
        indent=2, default=str))
    logger.info("wrote prioritization results under %s", out)


if __name__ == "__main__":
    main()
