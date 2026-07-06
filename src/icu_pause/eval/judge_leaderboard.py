"""Judge-selection leaderboard / concordance analysis
(judge_selection_plan v2.3, Step 4).

Joins judge scores (output/judge_calibration/{label}/{id}.iter{j}.pdsqi9.json)
to human scores (human_scores.json from Step 1) on hosp_id and ranks the
candidate judges by concordance with the human reference.

Strata are reported SEPARATELY (never pooled): the 4 anchors have a k=5-rater
median reference; the 80 singles have a single-rater reference — different
reference reliabilities, so pooling would mis-weight them.

Reporting roles (per plan):
  - SECONDARY headline = raw judge-human ICC(3,k) on total + 8 attributes
    (construct-alignment on the single gpt-5.4 condition).
  - SUPPLEMENTARY = disattenuated ICC (ceiling 0.867), clearly labeled.
  - ROBUSTNESS headline = cross-judge condition-rank concordance (Kendall's W)
    — requires Phase-3 multi-condition judge runs; this driver prints an N/A
    note until those exist (single condition today).

gpt-5.4 generated the 84 briefs, so its row is flagged self_preference_confound
and excluded from the recommended-judge computation.

Usage:
    python -m icu_pause.eval.judge_leaderboard --phase anchors
    python -m icu_pause.eval.judge_leaderboard --phase full84
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from statistics import median

import numpy as np

from icu_pause.eval import concordance as C

LIKERT_ATTRS = [
    "cited", "accurate", "thorough", "useful",
    "organized", "comprehensible", "succinct", "synthesized",
]
RESERVED_DIRS = {"cases", "_raw"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _discover_judge_labels(out_dir: str) -> list[str]:
    labels = []
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        if not os.path.isdir(path) or name in RESERVED_DIRS:
            continue
        if any(f.endswith(".pdsqi9.json") for f in os.listdir(path)):
            labels.append(name)
    return labels


def _load_judge_scores(out_dir: str, label: str) -> dict[str, dict]:
    """hosp_id -> median-over-iterations score dict (8 Likert ints + stigmatizing
    + total). Median of k iterations matches Croxford's median-of-iterations."""
    label_dir = os.path.join(out_dir, label)
    iters: dict[str, list[dict]] = {}
    for fname in os.listdir(label_dir):
        if not fname.endswith(".pdsqi9.json"):
            continue
        rec = json.loads(open(os.path.join(label_dir, fname)).read())
        hid = rec.get("hosp_id") or fname.split(".iter")[0]
        iters.setdefault(hid, []).append(rec)

    agg: dict[str, dict] = {}
    for hid, recs in iters.items():
        score_sets = [r.get("scores", {}) for r in recs]
        d = {}
        for a in LIKERT_ATTRS:
            vals = [int(s[a]) for s in score_sets if s.get(a) not in (None, 0)]
            d[a] = float(median(vals)) if vals else 0.0
        n = len(score_sets)
        n_stig = sum(1 for s in score_sets if s.get("stigmatizing"))
        d["stigmatizing"] = n_stig * 2 >= n
        totals = [float(r["total_score"]) for r in recs if r.get("total_score")]
        d["total"] = float(median(totals)) if totals else 0.0
        d["n_iter"] = n
        agg[hid] = d
    return agg


# ---------------------------------------------------------------------------
# Metric computation per (judge, stratum)
# ---------------------------------------------------------------------------


def _paired(judge: dict[str, dict], human: dict, hosp_ids: list[str], key: str):
    """Aligned (judge, human-ref) arrays over hosp_ids that have both."""
    j, h = [], []
    for hid in hosp_ids:
        jr = judge.get(hid)
        hr = human.get(hid, {}).get("reference")
        if jr is None or hr is None:
            continue
        if jr.get(key) in (None, 0) and key != "stigmatizing":
            continue
        j.append(jr[key])
        h.append(hr[key])
    return j, h


def _judge_family(label: str) -> tuple[bool, bool]:
    """(self_preference_confound, same_family_as_generator). Generator = gpt-5.4
    (OpenAI). gpt-5* = confound; other OpenAI o-series/gpt = same family;
    deepseek/other = cross-family."""
    low = label.lower()
    confound = "gpt-5" in low or "gpt5" in low
    openai_family = confound or low.startswith(("o1", "o3", "o4", "gpt", "o-"))
    return confound, (openai_family and not confound)


# Fixed consensus panel = the four note generators (locked 2026-06-15). Every
# pipeline condition is judged by a panel containing exactly its own generator
# plus three others, so the self-preference bias is SYMMETRIC across conditions
# and differences out of the cross-condition ranking (mirrors Vishwanath et al.,
# Nat Med 2026, and is balanced across ALL conditions, unlike that paper).
PANEL_DEFAULT = ["gpt-5.4", "gemma-4", "medgemma", "qwen-3.6"]


def _compute_metrics(judge: dict, human: dict, ids: list[str],
                     stratum: str, label: str, flags: dict) -> tuple[dict, list]:
    """Concordance metrics for one judge (or panel) vs the human reference."""
    jt, ht = _paired(judge, human, ids, "total")
    n = len(jt)
    row = {"model": label, "stratum": stratum, "n": n, **flags}
    per_attr: list[dict] = []
    if n < 3:
        row["note"] = f"n={n} too small for ICC"
        return row, per_attr

    icc = C.icc_3k(jt, ht)
    rho, _p = C.spearman(jt, ht)
    row.update({
        "total_icc3k": round(icc.icc, 3),
        "total_icc_lo": round(icc.ci_low, 3),
        "total_icc_hi": round(icc.ci_high, 3),
        "total_disattenuated": round(C.disattenuate(icc.icc), 3),
        "total_spearman": round(rho, 3),
        "total_median_diff": round(C.median_diff(jt, ht), 3),
    })
    kappas, rhos = [], []
    for a in LIKERT_ATTRS:
        ja, ha = _paired(judge, human, ids, a)
        if len(ja) < 3:
            continue
        aicc = C.icc_3k(ja, ha)
        kap = C.weighted_cohen_kappa(
            [int(round(x)) for x in ja], [int(round(x)) for x in ha])
        arho, ap = C.spearman(ja, ha)
        kappas.append(kap)
        rhos.append(arho)
        per_attr.append({
            "model": label, "stratum": stratum, "attribute": a, "n": len(ja),
            "icc3k": round(aicc.icc, 3),
            "icc_lo": round(aicc.ci_low, 3), "icc_hi": round(aicc.ci_high, 3),
            "weighted_kappa": round(kap, 3),
            "spearman_rho": round(arho, 3), "spearman_p": round(ap, 4),
            "median_diff": round(C.median_diff(ja, ha), 3),
        })
    row["mean_weighted_kappa"] = round(float(np.nanmean(kappas)), 3) if kappas else None
    row["mean_attr_spearman"] = round(float(np.nanmean(rhos)), 3) if rhos else None
    js, hs = _paired(judge, human, ids, "stigmatizing")
    if js:
        row["stig_gwet_ac2"] = round(
            C.gwet_ac2([bool(x) for x in js], [bool(x) for x in hs],
                       categories=[False, True]), 3)
    return row, per_attr


def _panel_aggregate(member_scores: dict[str, dict], members: list[str]) -> dict:
    """Consensus across panel members: per-attribute MEDIAN (ordinal analog of
    the paper's rounded-mean), majority vote for the binary stigmatizing item,
    median of member totals. Same operator we use to collapse k iterations, so
    a panel just widens the pool to members x iterations."""
    all_ids: set[str] = set()
    for m in members:
        all_ids |= set(member_scores.get(m, {}))
    out: dict[str, dict] = {}
    for hid in all_ids:
        present = [member_scores[m][hid] for m in members
                   if hid in member_scores.get(m, {})]
        if not present:
            continue
        d = {}
        for a in LIKERT_ATTRS:
            vals = [p[a] for p in present if p.get(a) not in (None, 0)]
            d[a] = float(median(vals)) if vals else 0.0
        n_stig = sum(1 for p in present if p.get("stigmatizing"))
        d["stigmatizing"] = n_stig * 2 >= len(present)
        totals = [p["total"] for p in present if p.get("total")]
        d["total"] = float(median(totals)) if totals else 0.0
        d["n_members"] = len(present)
        out[hid] = d
    return out


def _self_pref_delta(member_scores: dict[str, dict], generator: str,
                     ids: list[str]) -> float | None:
    """Self-preference magnitude for `generator` = median over cases of
    (generator-judge total - median of the OTHER panel members' totals) on
    briefs this generator produced. Tests the 'equally biased' assumption: if
    the four generators' deltas are similar, the symmetric bias differences out
    of the cross-condition ranking; if they diverge, that's a named residual.
    Only computable for the generator whose briefs are in this run."""
    if generator not in member_scores:
        return None
    self_s = member_scores[generator]
    others = [m for m in member_scores if m != generator]
    diffs = []
    for hid in ids:
        if hid not in self_s:
            continue
        ov = [member_scores[o][hid]["total"] for o in others if hid in member_scores[o]]
        if not ov:
            continue
        diffs.append(self_s[hid]["total"] - float(median(ov)))
    return round(float(median(diffs)), 3) if diffs else None


def _panel_internal_agreement(member_scores: dict[str, dict], members: list[str],
                              ids: list[str]) -> float | None:
    """Krippendorff's alpha (interval) across panel members on the total score —
    how much the judges agree with EACH OTHER (the paper's panel-concordance
    diagnostic). High alpha with low human-ICC = judges share a blind spot."""
    common = [h for h in ids if all(h in member_scores.get(m, {}) for m in members)]
    if len(common) < 2:
        return None
    matrix = {m: [member_scores[m][h]["total"] for h in common] for m in members}
    return round(C.krippendorff_alpha(matrix, level="interval"), 3)


def analyze(out_dir: str, human: dict, phase: str,
            panel_members: list[str], generator: str,
            include_panel: bool = True) -> dict:
    anchors = [h for h, e in human.items() if e["is_anchor"] and e["n_raters"] > 0]
    singles = [h for h, e in human.items() if not e["is_anchor"] and e["n_raters"] > 0]
    strata = {"anchors": anchors}
    if phase == "full84":
        strata["singles"] = singles

    labels = _discover_judge_labels(out_dir)
    results = {"labels": labels, "strata": {}, "ceiling": None,
               "panel_members": panel_members, "generator": generator,
               "panel_diag": {}}

    try:
        results["ceiling"] = _human_human_ceiling(human, anchors)
    except Exception as e:  # noqa: BLE001
        results["ceiling"] = {"error": str(e)}

    available = [m for m in panel_members if m in labels] if include_panel else []
    results["panel_missing_members"] = [m for m in panel_members if m not in labels]
    member_scores = {m: _load_judge_scores(out_dir, m) for m in available}

    for stratum, ids in strata.items():
        rows, per_attr = [], []
        # individual judges
        for label in labels:
            judge = _load_judge_scores(out_dir, label)
            confound, same_family = _judge_family(label)
            row, pa = _compute_metrics(
                judge, human, ids, stratum, label,
                {"self_preference_confound": confound,
                 "same_family_as_generator": same_family, "is_panel": False})
            rows.append(row)
            per_attr.extend(pa)

        # fixed consensus panel (balanced self-preference -> not confounded)
        if len(available) >= 2:
            panel = _panel_aggregate(member_scores, available)
            row, pa = _compute_metrics(
                panel, human, ids, stratum, "panel",
                {"self_preference_confound": False, "same_family_as_generator": False,
                 "is_panel": True, "panel_members": "+".join(available)})
            rows.append(row)
            per_attr.extend(pa)
            # safeguard #2: panel without the generator self-member (only the
            # gpt-5.4 condition is human-labeled, and gpt-5.4 is in the panel,
            # so the with-self panel is self-inflated on the calibration set)
            if generator in available:
                wo = [m for m in available if m != generator]
                panel_wo = _panel_aggregate(member_scores, wo)
                row, pa = _compute_metrics(
                    panel_wo, human, ids, stratum, f"panel(-{generator})",
                    {"self_preference_confound": False, "same_family_as_generator": False,
                     "is_panel": True, "panel_members": "+".join(wo)})
                rows.append(row)
                per_attr.extend(pa)
            results["panel_diag"][stratum] = {
                "self_pref_delta": {generator: _self_pref_delta(member_scores, generator, ids)},
                "internal_agreement_krippendorff": _panel_internal_agreement(
                    member_scores, available, ids),
            }

        rows.sort(key=lambda r: r.get("total_icc3k", -9), reverse=True)
        results["strata"][stratum] = {"rows": rows, "per_attr": per_attr}
    return results


def _human_human_ceiling(human: dict, anchors: list[str]) -> dict:
    """ICC(3,k) among anchor raters on total score, if the anchors share a
    common rater set (required for the long-format ICC)."""
    rater_sets = [set(human[h]["rater_scores"].keys()) for h in anchors]
    common = set.intersection(*rater_sets) if rater_sets else set()
    if len(common) < 2 or len(anchors) < 2:
        return {"note": "insufficient shared raters/anchors for ceiling ICC",
                "n_anchors": len(anchors), "n_common_raters": len(common)}
    raters = sorted(common)
    matrix = {r: [human[h]["rater_scores"][r]["total"] for h in anchors] for r in raters}
    icc = C.human_human_icc(matrix)
    return {"icc": round(icc.icc, 3), "ci_low": round(icc.ci_low, 3),
            "ci_high": round(icc.ci_high, 3), "n_anchors": len(anchors),
            "n_raters": len(raters),
            "contains_published_0.867": icc.ci_low <= 0.867 <= icc.ci_high}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write_csvs(results: dict, out_dir: str, phase: str) -> None:
    lb_rows, pa_rows = [], []
    for stratum, data in results["strata"].items():
        lb_rows.extend(data["rows"])
        pa_rows.extend(data["per_attr"])
    if lb_rows:
        cols = sorted({k for r in lb_rows for k in r})
        cols = ["model", "stratum", "n"] + [c for c in cols if c not in ("model", "stratum", "n")]
        path = os.path.join(out_dir, f"leaderboard_{phase}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(lb_rows)
        print(f"Wrote {path}")
    if pa_rows:
        path = os.path.join(out_dir, f"per_attribute_{phase}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pa_rows[0].keys()))
            w.writeheader()
            w.writerows(pa_rows)
        print(f"Wrote {path}")


def _print_summary(results: dict, phase: str, allow_low_power: bool) -> None:
    print("\n" + "=" * 72)
    if phase == "anchors":
        print("PHASE: anchors (n=4) — EXPLORATORY ONLY. Wide CIs expected.")
        print("Judge SELECTION requires --phase full84 (use --allow-low-power to override).")
    print("=" * 72)

    cl = results.get("ceiling") or {}
    if "icc" in cl:
        print(f"\nLocal ceiling (human-human ICC on {cl['n_anchors']} anchors, "
              f"{cl['n_raters']} raters): {cl['icc']} [{cl['ci_low']},{cl['ci_high']}] "
              f"— CI contains published 0.867: {cl['contains_published_0.867']}")
    else:
        print(f"\nLocal ceiling: {cl.get('note', cl)}")

    missing = results.get("panel_missing_members") or []
    print(f"\nFixed consensus panel = {results.get('panel_members')}")
    if missing:
        print(f"  WARNING: panel members not yet run as judges (excluded): {missing}")

    for stratum, data in results["strata"].items():
        print(f"\n--- stratum: {stratum} ---")
        print(f"{'model':<18}{'n':>4}{'ICC3k':>8}{'CI':>16}"
              f"{'wκ':>7}{'ρ':>7}{'mdiff':>7}  flags")
        for r in data["rows"]:
            if "total_icc3k" not in r:
                print(f"{r['model']:<18}{r['n']:>4}   {r.get('note','')}")
                continue
            if r.get("is_panel"):
                flags = ["PANEL(primary)" if r["model"] == "panel" else "PANEL(-self diag)"]
            elif r["self_preference_confound"]:
                flags = ["SELF-PREF(excluded)"]
            elif r["same_family_as_generator"]:
                flags = ["same-family"]
            else:
                flags = ["cross-family"]
            ci = f"[{r['total_icc_lo']},{r['total_icc_hi']}]"
            print(f"{r['model']:<18}{r['n']:>4}{r['total_icc3k']:>8}{ci:>16}"
                  f"{r.get('mean_weighted_kappa','-'):>7}{r.get('total_spearman','-'):>7}"
                  f"{r.get('total_median_diff','-'):>7}  {','.join(flags)}")

        # panel diagnostics
        diag = (results.get("panel_diag") or {}).get(stratum)
        if diag:
            gen = results.get("generator")
            dlt = diag["self_pref_delta"].get(gen)
            print(f"  panel diag: self-pref δ[{gen}]={dlt} "
                  f"(other 3 generators' δ need their Phase-3 briefs); "
                  f"inter-judge agreement (Krippendorff α)={diag['internal_agreement_krippendorff']}")
            print("    → if the four δ's end up similar, the symmetric bias differences "
                  "out of the cross-condition ranking; if they diverge, report as residual.")

        # the panel is the PRIMARY instrument; surface its standing vs best single
        if phase == "full84" or allow_low_power:
            scored = [r for r in data["rows"] if "total_icc3k" in r]
            panel_row = next((r for r in scored if r["model"] == "panel"), None)
            singles = [r for r in scored
                       if not r.get("is_panel") and not r["self_preference_confound"]]
            if panel_row:
                print(f"\n  PRIMARY = consensus panel ({stratum}): "
                      f"ICC3k={panel_row['total_icc3k']} {panel_row.get('panel_members','')}")
                if singles:
                    best = singles[0]
                    print(f"  best single judge: {best['model']} ICC3k={best['total_icc3k']} "
                          f"(reported as a robustness comparator)")
            elif singles:
                best = singles[0]
                print(f"\n  (no panel) best single judge ({stratum}): {best['model']} "
                      f"ICC3k={best['total_icc3k']}")

    print("\nROBUSTNESS headline (cross-judge condition-rank Kendall's W): N/A — "
          "needs Phase-3 multi-condition judge runs (Gemma/Qwen/DeepSeek briefs). "
          "Use concordance.condition_rank_concordance() once those are scored.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["anchors", "full84"], default="anchors")
    ap.add_argument("--out-dir", default="output/judge_calibration")
    ap.add_argument("--human-scores", default=None,
                    help="path to human_scores.json (default: {out_dir}/human_scores.json)")
    ap.add_argument("--allow-low-power", action="store_true",
                    help="permit a recommendation from the n=4 anchors phase")
    ap.add_argument("--panel-members", default=",".join(PANEL_DEFAULT),
                    help="comma-separated judge labels forming the fixed consensus "
                         "panel (default: the four note generators)")
    ap.add_argument("--generator", default="gpt-5.4",
                    help="model that generated the briefs being scored in this run "
                         "(its self-pref δ is estimated; panel(-generator) reported)")
    ap.add_argument("--no-panel", action="store_true", help="skip the consensus panel")
    args = ap.parse_args()

    human_path = args.human_scores or os.path.join(args.out_dir, "human_scores.json")
    human = json.loads(open(human_path).read())

    panel_members = [m.strip() for m in args.panel_members.split(",") if m.strip()]
    results = analyze(args.out_dir, human, args.phase, panel_members,
                      args.generator, include_panel=not args.no_panel)
    _write_csvs(results, args.out_dir, args.phase)
    _print_summary(results, args.phase, args.allow_low_power)


if __name__ == "__main__":
    main()
