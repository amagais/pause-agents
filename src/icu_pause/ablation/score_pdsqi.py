"""Run PDSQI-9 over ablation-run briefs with a cross-model judge (DeepSeek).

Overall note-quality (organized, succinct, synthesized, useful, accurate, etc.) —
the secondary axis beyond prioritization. Uses the existing PDSQI9Evaluator with
the EVAL llm pointed at a different model than the brief-generator (no self-bias).
Source bundle is truncated (notes-first, structured kept) so the judge doesn't
itself overflow on heavy patients.

Env: pass --judge-provider/--judge-model (sets EVAL llm) and export
ICUPAUSE_LOCAL_LLM_URL / ICUPAUSE_LOCAL_LLM_BACKEND for the vLLM endpoint.

Example (DeepSeek judge on 18931):
    export ICUPAUSE_LOCAL_LLM_URL=http://localhost:18931/v1
    export ICUPAUSE_LOCAL_LLM_BACKEND=vllm
    .venv/bin/python -m icu_pause.ablation.score_pdsqi \
        --ablation-out output/ablation/gemma_n25 --ids docs/validation_cohort_25.csv \
        --judge-provider local --judge-model <deepseek-id> \
        --out output/ablation/gemma_n25_pdsqi
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

# Source-bundle budget for the judge prompt (keep well under the judge's context;
# R1 also needs room to think). Notes truncated first, structured data kept.
MAX_SOURCE_CHARS = 120000


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
    p = argparse.ArgumentParser(description="PDSQI-9 over ablation briefs")
    p.add_argument("--ablation-out", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--arms", default="full,monolith_best_effort,monolith_templated")
    p.add_argument("--judge-provider", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--max-source-chars", type=int, default=MAX_SOURCE_CHARS)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    os.environ["ICUPAUSE_CITATION_MODE"] = "off"
    # PDSQI uses the EVAL llm config — point it at the judge.
    if args.judge_provider:
        os.environ["ICUPAUSE_EVAL_LLM_PROVIDER"] = args.judge_provider
    if args.judge_model:
        os.environ["ICUPAUSE_EVAL_LLM_MODEL"] = args.judge_model

    from icu_pause.config import Settings
    from icu_pause.ablation.arms import retrieve_bundle
    from icu_pause.ablation.monolith import bundle_to_text
    from icu_pause.ablation.run import _load_cohort
    from icu_pause.eval.pdsqi9 import PDSQI9Evaluator

    settings = Settings()
    evaluator = PDSQI9Evaluator(settings)
    arms = [a.strip() for a in args.arms.split(",")]
    ref_by_hid = {r["hospitalization_id"]: r["reference_dttm"] for r in _load_cohort(args.ids)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_arm: dict[str, list[float]] = defaultdict(list)
    attr_by_arm: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rows: list[dict] = []
    tok_in = tok_out = 0  # judge token accounting
    case_dirs = sorted(d for d in glob.glob(str(Path(args.ablation_out) / "*")) if Path(d).is_dir())
    logger.info("scoring PDSQI-9 for %d case dirs", len(case_dirs))

    for cdir in case_dirs:
        hid = Path(cdir).name
        ref = ref_by_hid.get(hid)
        if not ref:
            continue
        try:
            bundle = retrieve_bundle(settings, hid, ref, 48, None)
            source, _ = bundle_to_text(bundle, args.max_source_chars)
        except Exception as e:  # noqa: BLE001
            logger.error("source build failed for %s: %s", hid, e)
            continue

        for arm in arms:
            bp = Path(cdir) / f"{arm}.brief.json"
            if not bp.exists():
                continue
            try:
                ev = evaluator.evaluate(source, _brief_text(json.load(open(bp))))
                total = ev.total_score
                attrs = ev.scores.model_dump() if hasattr(ev.scores, "model_dump") else {}
                di, do = _usage(evaluator.llm); tok_in += di; tok_out += do
            except Exception as e:  # noqa: BLE001
                logger.error("PDSQI eval failed arm=%s case=%s: %s", arm, hid, e)
                continue
            per_arm[arm].append(total)
            for k, v in attrs.items():
                if isinstance(v, (int, float)):
                    attr_by_arm[arm][k].append(v)
            rows.append({"hospitalization_id": hid, "arm": arm, "total_score": total, **attrs})

    if rows:
        fields = sorted({k for r in rows for k in r})
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    def _m(xs):
        return statistics.mean(xs) if xs else None

    print("\n" + "=" * 64)
    print("PDSQI-9 (mean total_score, 1-5; higher = better note quality)")
    print("=" * 64)
    means = {a: _m(per_arm.get(a, [])) for a in arms}
    for a in arms:
        print(f"  {a:24s} {means[a]:.3f}  (n={len(per_arm.get(a, []))})"
              if means[a] is not None else f"  {a:24s}  n/a")
    base, full = means.get("monolith_best_effort"), means.get("full")
    if full is not None and base is not None:
        print("-" * 64)
        print(f"  full - best_effort : {full - base:+.3f}")
    print("-" * 64)
    print(f"  judge tokens: in={tok_in:,}  out={tok_out:,}  total={tok_in + tok_out:,}")
    print("=" * 64 + "\n")
    (out / "headline.json").write_text(json.dumps({
        "means": means,
        "n_by_arm": {a: len(per_arm.get(a, [])) for a in arms},
        "attr_means": {a: {k: _m(v) for k, v in attr_by_arm[a].items()} for a in arms},
        "judge_tokens": {"input": tok_in, "output": tok_out, "total": tok_in + tok_out},
    }, indent=2, default=str))
    logger.info("wrote PDSQI-9 results under %s", out)


if __name__ == "__main__":
    main()
