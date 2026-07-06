"""LLM-as-a-judge calibration runner (judge_selection_plan v2.3, Step 2).

Scores the FROZEN, already-generated briefs with a candidate PDSQI-9 judge
model, k times per case. Never regenerates the briefs and never touches the
LangGraph pipeline — it reads stored inputs from a local `cases/` mirror and
calls PDSQI9Evaluator directly.

Inputs (local files, NOT Azure Blob — keeps this runnable on the HPC venv which
can't import azure). Materialize the mirror on the Mac with:
    review_app/scripts/export_judge_calibration_humanscores.py --dump-cases-dir DIR
then rsync DIR to HPC for the deepseek-r1 run. Each case is:
    {cases_dir}/{hosp_id}/output.json         -> rendered with render_icu_pause_text
    {cases_dir}/{hosp_id}/source_bundle.json  -> the producer-input snapshot

Output (idempotent — existing files are skipped so runs resume):
    {out_dir}/{label}/{hosp_id}.iter{j}.pdsqi9.json
Each file is the PDSQI9Evaluation dump plus hosp_id + iteration. A parse failure
surfaces as an all-zero score (evaluate()'s fallback) — the smoke gate is
"no all-zero fallbacks", not any ICC value.

Decoding: temperature is forced to 0.0; reasoning models (o3/o4/deepseek) ignore
it and the provider auto-skips response_format for them, so scores are parsed
via pdsqi9._parse_scores (which strips <think>/<unused> blocks first). Use
--debug-raw on the 4-anchor smoke to dump raw completions for the by-hand
DeepSeek <think>-integrity audit.

Usage (one model at a time):
    python -m icu_pause.eval.judge_calibration \
        --cases-dir output/judge_calibration/cases \
        --ids output/judge_calibration/anchor_ids.txt \
        --provider azure --model o3-mini --k 5 --debug-raw

    python -m icu_pause.eval.judge_calibration \
        --cases-dir output/judge_calibration/cases \
        --ids output/judge_calibration/all_ids.txt \
        --provider local --model deepseek-r1 --k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from copy import copy
from pathlib import Path

from icu_pause.config import Settings
from icu_pause.eval.pdsqi9 import PDSQI9Evaluator
from icu_pause.rendering.formatter import render_icu_pause_text

logger = logging.getLogger(__name__)


def _load_ids(path: str) -> list[str]:
    ids = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(s)
    return ids


def _build_judge(provider: str, model: str) -> PDSQI9Evaluator:
    """A PDSQI9Evaluator wired to one candidate judge model at temperature 0.0.

    Settings is frozen, so mutate a shallow copy with object.__setattr__ (same
    pattern as create_eval_llm). Setting pdsqi9_llm_* is what PDSQI9Evaluator
    reads to pick its model."""
    settings = copy(Settings())
    object.__setattr__(settings, "pdsqi9_llm_provider", provider)
    object.__setattr__(settings, "pdsqi9_llm_model", model)
    object.__setattr__(settings, "llm_temperature", 0.0)
    return PDSQI9Evaluator(settings)


def _case_inputs(cases_dir: Path, hosp_id: str) -> tuple[str, str]:
    """Return (summary_text, source_text) for one case, or raise if missing."""
    case_dir = cases_dir / hosp_id
    output = json.loads((case_dir / "output.json").read_text())
    source_bundle = json.loads((case_dir / "source_bundle.json").read_text())
    summary = render_icu_pause_text(output)
    source_text = json.dumps(source_bundle, indent=2, default=str)
    return summary, source_text


def _is_all_zero(scores: dict) -> bool:
    likert = ["cited", "accurate", "thorough", "useful",
              "organized", "comprehensible", "succinct", "synthesized"]
    return all(int(scores.get(a, 0)) == 0 for a in likert)


def run(
    cases_dir: Path,
    hosp_ids: list[str],
    provider: str,
    model: str,
    label: str,
    k: int,
    out_dir: Path,
    debug_raw: bool,
    condition: str | None = None,
) -> None:
    model_out = out_dir / label
    model_out.mkdir(parents=True, exist_ok=True)
    judge = _build_judge(provider, model)

    n_done = n_skipped = n_failed = n_allzero = 0
    tok_in = tok_out = 0
    t0 = time.time()
    for hi, hosp_id in enumerate(hosp_ids, 1):
        try:
            summary, source_text = _case_inputs(cases_dir, hosp_id)
        except FileNotFoundError as e:
            print(f"  [{hi}/{len(hosp_ids)}] {hosp_id}: MISSING input ({e}) — skip")
            n_failed += 1
            continue

        if debug_raw:
            # One diagnostic, non-scored invoke to eyeball the raw completion
            # (esp. DeepSeek <think> integrity). Kept separate from the k
            # scored iterations so it never contaminates the scores.
            raw_dir = model_out / "_raw"
            raw_dir.mkdir(exist_ok=True)
            user_msg = (
                f"## SOURCE CLINICAL DATA\n{source_text}\n\n"
                f"## CLINICAL SUMMARY TO EVALUATE\n{summary}\n\n"
                f"Evaluate the summary using the PDSQI-9 rubric. Return the JSON."
            )
            try:
                raw = judge.llm.invoke(system=judge.system_prompt, user=user_msg)
                (raw_dir / f"{hosp_id}.raw.txt").write_text(str(raw))
                print(f"  [{hi}/{len(hosp_ids)}] {hosp_id}: raw saved "
                      f"({len(str(raw))} chars, has <think>={'<think>' in str(raw)})")
            except Exception as e:  # noqa: BLE001
                print(f"  [{hi}/{len(hosp_ids)}] {hosp_id}: raw invoke failed: {e}")

        for j in range(1, k + 1):
            dst = model_out / f"{hosp_id}.iter{j}.pdsqi9.json"
            if dst.exists():
                n_skipped += 1
                continue
            try:
                ev = judge.evaluate(source_text, summary)
            except Exception as e:  # noqa: BLE001
                print(f"  [{hi}/{len(hosp_ids)}] {hosp_id} iter{j}: ERROR {e}")
                n_failed += 1
                continue
            record = ev.model_dump()
            record["hosp_id"] = hosp_id
            record["iteration"] = j
            record["judge_label"] = label
            record["condition"] = condition
            # token usage for cost accounting (azure output_tokens includes o3/o4
            # reasoning tokens, so this is the real billed count)
            u = getattr(judge.llm, "last_usage", None)
            ti = int(getattr(u, "input_tokens", 0) or 0)
            to = int(getattr(u, "output_tokens", 0) or 0)
            record["input_tokens"] = ti
            record["output_tokens"] = to
            tok_in += ti
            tok_out += to
            with open(dst, "w") as f:
                json.dump(record, f, indent=2, default=str)
            if _is_all_zero(record.get("scores", {})):
                n_allzero += 1
                print(f"  [{hi}/{len(hosp_ids)}] {hosp_id} iter{j}: "
                      f"ALL-ZERO scores (parse failure?)")
            n_done += 1

    dt = time.time() - t0
    print(f"\n[{label}] done: {n_done} scored, {n_skipped} skipped(existing), "
          f"{n_failed} failed, {n_allzero} all-zero, {dt:.0f}s")
    if n_done:
        mi, mo = tok_in / n_done, tok_out / n_done
        print(f"[{label}] tokens: {tok_in} in + {tok_out} out over {n_done} calls "
              f"| mean/call: {mi:.0f} in / {mo:.0f} out "
              f"(out incl. reasoning tokens for o3/o4)")
        print(f"[{label}] extrapolated full run (84 cases x k): "
              f"per-iter cost = mean_in*84*k*IN_RATE + mean_out*84*k*OUT_RATE")
    if n_allzero:
        print(f"  WARNING: {n_allzero} all-zero results — inspect "
              f"{model_out}/_raw or rerun with --debug-raw before trusting scores.")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", default=None,
                    help="condition slug (e.g. gpt54_multiagent). Roots cases/ids/out "
                         "under output/judge_calibration/<condition>/. Overridable by "
                         "the explicit path flags below.")
    ap.add_argument("--cases-dir", default=None,
                    help="cases dir: {id}/output.json + source_bundle.json "
                         "(default: output/judge_calibration/<condition>/cases)")
    ap.add_argument("--ids", default=None, help="hosp_id list file "
                    "(default: output/judge_calibration/<condition>/all_ids.txt)")
    ap.add_argument("--provider", required=True,
                    choices=["azure", "openai", "anthropic", "local"])
    ap.add_argument("--model", required=True,
                    help="judge model / deployment name (e.g. o4-mini, mixtral-8x22b, deepseek-r1)")
    ap.add_argument("--label", default=None,
                    help="judge subdir name (default: sanitized model name)")
    ap.add_argument("--k", type=int, default=3, help="iterations per case (median taken downstream); k=3 per 2026-06-17 amendment")
    ap.add_argument("--out-dir", default=None,
                    help="output root (default: output/judge_calibration/<condition>)")
    ap.add_argument("--limit", type=int, default=0, help="cap #cases (smoke); 0 = all")
    ap.add_argument("--debug-raw", action="store_true",
                    help="dump one raw completion per case for by-hand audit")
    args = ap.parse_args()

    # Resolve the condition-rooted layout:
    #   output/judge_calibration/<condition>/{cases/, all_ids.txt, <label>/...}
    base = Path("output/judge_calibration")
    if args.condition:
        base = base / args.condition
    cases_dir = Path(args.cases_dir) if args.cases_dir else base / "cases"
    ids_path = Path(args.ids) if args.ids else base / "all_ids.txt"
    out_dir = Path(args.out_dir) if args.out_dir else base
    if not args.condition and not (args.cases_dir and args.ids):
        ap.error("pass --condition, or both --cases-dir and --ids")
    if not cases_dir.is_dir():
        ap.error(f"cases dir not found: {cases_dir} (run build_judge_cases.py first)")
    if not ids_path.exists():
        ap.error(f"ids file not found: {ids_path}")

    label = args.label or args.model.replace("/", "_")
    hosp_ids = _load_ids(str(ids_path))
    if args.limit:
        hosp_ids = hosp_ids[: args.limit]

    print(f"Judge calibration: condition={args.condition} provider={args.provider} "
          f"model={args.model} label={label} k={args.k} cases={len(hosp_ids)}")
    print(f"  cases={cases_dir}  out={out_dir}/{label}")
    run(
        cases_dir=cases_dir,
        hosp_ids=hosp_ids,
        provider=args.provider,
        model=args.model,
        label=label,
        k=args.k,
        out_dir=out_dir,
        debug_raw=args.debug_raw,
        condition=args.condition,
    )


if __name__ == "__main__":
    main()
