"""Judge selection (Phase 1) — pick the primary PDSQI-9 LLM judge by agreement
with the human labels on the gpt-5.4 condition.

Implements the CONSOLIDATED selection protocol (2026-06-15), NOT the old
ICC-headline leaderboard:

  * Selection set    = the 80 single-reviewer briefs (discrimination power).
  * Human target     = reviewer-CENTERED singles (subtract each reviewer's mean
                       offset; leniency is safe to remove per the IRR findings).
  * Anchors (n=4)    = smoke only — AC2 vs the 5-reviewer consensus + the
                       human-human AC2 ceiling (~0.77). NOT for selection.
  * Primary metrics  = Gwet's AC2 (quadratic) + %-within-1 (raw ordinal), and
                       Spearman vs the reviewer-centered target. ICC is NOT used.
  * Gating variance  = reviewer-centered between-brief variance per axis; if ~0,
                       judges are near-tied -> structural tiebreak.
  * Axis weighting   = emphasize organized/accurate/comprehensible + the
                       objective axes (accurate/thorough); de-weight cited
                       (0.606 human ceiling — no judge can beat it).

Why two agreement tracks: AC2/%-within-1 are ordinal (need 1-5 values), so they
run on RAW judge-vs-human and are partly leniency-contaminated. Reviewer-centering
yields continuous residuals, so the leniency-free "does the judge track brief
quality" question is answered by Spearman vs the centered target. Both reported.

Out-of-window cases (deepseek/llama) simply have no judge score and are skipped
in pairing — each judge is scored on the cases it actually covers.

Usage (HPC):
    .venv/bin/python -m icu_pause.eval.judge_selection \
        --scores-dir output/judge_calibration/gpt54_multiagent \
        --human-scores output/judge_calibration/human_scores.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from statistics import median, pstdev

import numpy as np

from icu_pause.eval import concordance as C

LIKERT = ["cited", "accurate", "thorough", "useful",
          "organized", "comprehensible", "succinct", "synthesized"]
# clinicians agree most here (anchor AC2 ~0.82-0.85) + the objective axes
CORE_AXES = ["organized", "accurate", "comprehensible", "thorough"]
# Cited is excluded from the composite that drives ranking (judges can't verify
# citations reliably -> noisiest/most-biased axis); reported as a standalone
# outcome instead. The composite total is the mean of the 7 non-cited Likert.
NONCITED = [a for a in LIKERT if a != "cited"]
RESERVED = {"cases", "_raw"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _discover_judges(scores_dir: str) -> list[str]:
    out = []
    for name in sorted(os.listdir(scores_dir)):
        d = os.path.join(scores_dir, name)
        if not os.path.isdir(d) or name in RESERVED:
            continue
        if glob.glob(os.path.join(d, "*.pdsqi9.json")):
            out.append(name)
    return out


def _load_judge(scores_dir: str, label: str):
    """hosp_id -> {attr: median-over-iters, '_iters': {attr: [per-iter vals]}}.
    Zeros are treated as invalid (parse/fallback) and dropped from the median,
    matching how the data was produced."""
    iters: dict[str, dict[str, list[float]]] = {}
    for f in glob.glob(os.path.join(scores_dir, label, "*.pdsqi9.json")):
        rec = json.load(open(f))
        hid = rec.get("hosp_id") or os.path.basename(f).split(".iter")[0]
        s = rec.get("scores", {})
        d = iters.setdefault(hid, {a: [] for a in LIKERT})
        for a in LIKERT:
            v = s.get(a)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0:
                d[a].append(float(v))
    agg = {}
    for hid, per in iters.items():
        row = {"_iters": per}
        for a in LIKERT:
            row[a] = float(median(per[a])) if per[a] else None
        agg[hid] = row
    return agg


def _split(human: dict):
    anchors = {h: e for h, e in human.items() if e.get("is_anchor") and e.get("n_raters", 0) > 0}
    singles = {h: e for h, e in human.items() if not e.get("is_anchor") and e.get("n_raters", 0) > 0}
    return anchors, singles


# ---------------------------------------------------------------------------
# Reviewer-centering of the singles target
# ---------------------------------------------------------------------------

def _reviewer_centered_singles(singles: dict):
    """Return {hosp: {attr: centered_value}} and the per-reviewer offsets.

    centered = raw - reviewer_mean(attr) + global_mean(attr). Subtracting the
    reviewer's mean removes intrinsic leniency (safe per the IRR findings);
    adding the global mean keeps values on the ~1-5 scale for interpretability.
    """
    # the single's reviewer = the sole key in rater_scores
    rev_of = {h: next(iter(e["rater_scores"])) for h, e in singles.items() if e["rater_scores"]}
    raw = {h: {a: singles[h]["rater_scores"][rev_of[h]][a] for a in LIKERT}
           for h in singles if h in rev_of}
    # per-reviewer mean per attribute
    by_rev: dict[str, dict[str, list[float]]] = {}
    for h, rv in rev_of.items():
        by_rev.setdefault(rv, {a: [] for a in LIKERT})
        for a in LIKERT:
            by_rev[rv][a].append(float(raw[h][a]))
    rev_mean = {rv: {a: float(np.mean(v[a])) for a in LIKERT} for rv, v in by_rev.items()}
    glob_mean = {a: float(np.mean([raw[h][a] for h in raw])) for a in LIKERT}
    centered = {}
    for h in raw:
        rv = rev_of[h]
        centered[h] = {a: float(raw[h][a]) - rev_mean[rv][a] + glob_mean[a] for a in LIKERT}
    return centered, raw, rev_mean, rev_of


# ---------------------------------------------------------------------------
# Per-judge agreement on the singles
# ---------------------------------------------------------------------------

def _pct_within(j, h, tol=1.0):
    return float(np.mean([abs(a - b) <= tol for a, b in zip(j, h)])) if j else float("nan")


def _judge_vs_singles(judge: dict, raw: dict, centered: dict):
    """Per-attribute AC2(quadratic)+%within1 vs RAW human, Spearman vs CENTERED."""
    res = {}
    for a in LIKERT:
        jv, hv_raw, hv_cen = [], [], []
        for h in raw:
            jr = judge.get(h)
            if jr is None or jr.get(a) is None:
                continue
            jv.append(jr[a]); hv_raw.append(float(raw[h][a])); hv_cen.append(centered[h][a])
        n = len(jv)
        if n < 3:
            res[a] = {"n": n}
            continue
        jr_int = [int(round(x)) for x in jv]
        hr_int = [int(round(x)) for x in hv_raw]
        ac2 = C.gwet_ac2(jr_int, hr_int, categories=[1, 2, 3, 4, 5], weights="quadratic")
        within = _pct_within(jv, hv_raw, 1.0)
        rho, _p = C.spearman(jv, hv_cen)
        res[a] = {"n": n, "ac2": round(ac2, 3), "within1": round(within, 3),
                  "spearman_centered": round(rho, 3),
                  "bias": round(float(np.median(np.array(jv) - np.array(hv_raw))), 2)}
    return res


def _agg(per_attr: dict, axes: list[str], key: str):
    vals = [per_attr[a][key] for a in axes
            if a in per_attr and key in per_attr[a] and per_attr[a][key] == per_attr[a][key]]
    return round(float(np.mean(vals)), 3) if vals else float("nan")


# ---------------------------------------------------------------------------
# Single-rater reliability + the implied judge ceiling
# ---------------------------------------------------------------------------

def _two_way_var_components(matrix_by_rater: dict[str, list[float]], n_briefs: int):
    """Balanced two-way (brief x rater) random-ANOVA variance components.
    Returns (var_brief, var_rater, ms_resid). ms_resid = the per-rating noise
    (rater disagreement on the same brief, net of leniency) — this is the piece
    we trust from the anchors because, unlike between-brief variance, it is NOT
    deflated by the anchors being near-identical."""
    raters = list(matrix_by_rater)
    r, b = len(raters), n_briefs
    Y = np.array([matrix_by_rater[rt] for rt in raters], dtype=float)  # r x b
    grand = Y.mean()
    brief_means = Y.mean(axis=0)
    rater_means = Y.mean(axis=1)
    ss_brief = r * float(np.sum((brief_means - grand) ** 2))
    ss_rater = b * float(np.sum((rater_means - grand) ** 2))
    resid = Y - brief_means[None, :] - rater_means[:, None] + grand
    ss_resid = float(np.sum(resid ** 2))
    ms_brief = ss_brief / (b - 1) if b > 1 else 0.0
    ms_rater = ss_rater / (r - 1) if r > 1 else 0.0
    ms_resid = ss_resid / ((b - 1) * (r - 1)) if (b > 1 and r > 1) else 0.0
    var_brief = max(0.0, (ms_brief - ms_resid) / r)
    var_rater = max(0.0, (ms_rater - ms_resid) / b)
    return var_brief, var_rater, max(0.0, ms_resid)


def _anchor_rater_matrix(anchors: dict, attr_or_fn):
    """{rater -> [value per anchor]} over raters who rated ALL anchors.
    attr_or_fn: an attribute name, or a callable score_dict->float (composite)."""
    aids = sorted(anchors)
    common = set.intersection(*[set(anchors[h]["rater_scores"]) for h in aids]) if aids else set()
    raters = sorted(common)
    def val(sd):
        return float(attr_or_fn(sd)) if callable(attr_or_fn) else float(sd[attr_or_fn])
    return {rt: [val(anchors[h]["rater_scores"][rt]) for h in aids] for rt in raters}, len(aids)


def _reliability(obs_var: float, resid_var: float):
    """Single-rater reliability + implied max correlation with a single rating.
    obs_var = observed reviewer-centered between-brief variance of the SINGLES
    (= true-brief variance + per-rating noise, since each single is rated once);
    resid_var = per-rating noise from the anchors. reliability = signal/total."""
    if obs_var <= 0:
        return float("nan"), float("nan")
    rel = max(0.0, min(1.0, (obs_var - resid_var) / obs_var))
    return rel, rel ** 0.5


# ---------------------------------------------------------------------------
# Anchors smoke + ceiling
# ---------------------------------------------------------------------------

def _anchor_ceiling_and_judges(anchors: dict, judges: dict):
    """human-human AC2 ceiling on anchors (per attr, mean over the shared raters'
    pairwise AC2) + each judge's AC2 vs the 5-reviewer consensus reference."""
    anchor_ids = sorted(anchors)
    # ceiling: average pairwise AC2 across reviewers who rated all anchors
    common = set.intersection(*[set(anchors[h]["rater_scores"]) for h in anchor_ids]) if anchor_ids else set()
    ceiling = {}
    raters = sorted(common)
    for a in LIKERT:
        pair_ac2 = []
        for i in range(len(raters)):
            for k in range(i + 1, len(raters)):
                x = [int(round(anchors[h]["rater_scores"][raters[i]][a])) for h in anchor_ids]
                y = [int(round(anchors[h]["rater_scores"][raters[k]][a])) for h in anchor_ids]
                pair_ac2.append(C.gwet_ac2(x, y, categories=[1, 2, 3, 4, 5], weights="quadratic"))
        ceiling[a] = round(float(np.nanmean(pair_ac2)), 3) if pair_ac2 else float("nan")
    # judges vs consensus reference
    jrows = {}
    for label, judge in judges.items():
        per = {}
        for a in LIKERT:
            jv, hv = [], []
            for h in anchor_ids:
                jr = judge.get(h)
                if jr is None or jr.get(a) is None:
                    continue
                jv.append(int(round(jr[a]))); hv.append(int(round(anchors[h]["reference"][a])))
            per[a] = round(C.gwet_ac2(jv, hv, categories=[1, 2, 3, 4, 5], weights="quadratic"), 3) if len(jv) >= 2 else None
        jrows[label] = per
    return ceiling, raters, jrows


# ---------------------------------------------------------------------------
# o4-mini k-variance
# ---------------------------------------------------------------------------

def _k_variance(judges: dict):
    """Per judge: how much the k iterations disagree. Temp-0 locals ~0; o4-mini
    (forced default temp) is the one that may need k bumped 3->5."""
    out = {}
    for label, judge in judges.items():
        disagree_cases = 0
        total_cases = 0
        spreads = []
        for h, row in judge.items():
            per = row.get("_iters", {})
            case_has = False
            case_disagree = False
            for a in LIKERT:
                vals = per.get(a, [])
                if len(vals) >= 2:
                    case_has = True
                    sp = max(vals) - min(vals)
                    spreads.append(sp)
                    if sp >= 1:
                        case_disagree = True
            if case_has:
                total_cases += 1
                if case_disagree:
                    disagree_cases += 1
        out[label] = {
            "cases": total_cases,
            "cases_with_iter_disagreement": disagree_cases,
            "frac_cases_disagree": round(disagree_cases / total_cases, 3) if total_cases else float("nan"),
            "mean_attr_spread": round(float(np.mean(spreads)), 3) if spreads else 0.0,
            "max_attr_spread": round(float(np.max(spreads)), 1) if spreads else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores-dir", default="output/judge_calibration/gpt54_multiagent")
    ap.add_argument("--human-scores", default="output/judge_calibration/human_scores.json")
    args = ap.parse_args()

    human = json.load(open(args.human_scores))
    anchors, singles = _split(human)
    labels = _discover_judges(args.scores_dir)
    judges = {lab: _load_judge(args.scores_dir, lab) for lab in labels}

    centered, raw, rev_mean, rev_of = _reviewer_centered_singles(singles)

    print("=" * 78)
    print(f"JUDGE SELECTION — Phase 1 (gpt-5.4 condition)")
    print(f"singles={len(raw)}  anchors={len(anchors)}  judges={labels}")
    print("=" * 78)

    # --- gating variance -----------------------------------------------------
    print("\n[GATING] reviewer-centered between-brief variance per axis (singles):")
    gv = {a: round(float(pstdev([centered[h][a] for h in centered])) ** 2, 3) for a in LIKERT}
    for a in LIKERT:
        flag = "  <- ~0: little quality signal" if gv[a] < 0.10 else ""
        print(f"   {a:<15} var={gv[a]:.3f}{flag}")
    core_var = float(np.mean([gv[a] for a in CORE_AXES]))
    print(f"   core-axes mean var = {core_var:.3f}"
          + ("  -> LOW: expect near-tied judges, lean on structural tiebreak" if core_var < 0.10 else ""))

    # --- per-judge agreement on the centered singles -------------------------
    print("\n[SELECTION] per-judge agreement on the 80 singles")
    print("  AC2(quad)+%within1 vs RAW human ; Spearman vs REVIEWER-CENTERED human")
    summary = {}
    for label in labels:
        per = _judge_vs_singles(judges[label], raw, centered)
        summary[label] = per
        core_ac2 = _agg(per, CORE_AXES, "ac2")
        core_w1 = _agg(per, CORE_AXES, "within1")
        core_rho = _agg(per, CORE_AXES, "spearman_centered")
        all_ac2 = _agg(per, [a for a in LIKERT if a != "cited"], "ac2")
        n = max((per[a].get("n", 0) for a in LIKERT), default=0)
        print(f"\n  --- {label} (n≈{n} singles covered) ---")
        print(f"    {'attr':<15}{'n':>4}{'AC2':>7}{'%w1':>7}{'ρ_cen':>7}{'bias':>7}")
        for a in LIKERT:
            d = per[a]
            if "ac2" not in d:
                print(f"    {a:<15}{d.get('n',0):>4}   (insufficient)")
                continue
            tag = " *core" if a in CORE_AXES else (" (deweight)" if a == "cited" else "")
            print(f"    {a:<15}{d['n']:>4}{d['ac2']:>7}{d['within1']:>7}"
                  f"{d['spearman_centered']:>7}{d['bias']:>7}{tag}")
        print(f"    CORE-AXES mean : AC2={core_ac2}  %within1={core_w1}  ρ_centered={core_rho}")
        print(f"    ALL-8 (excl cited) mean AC2={all_ac2}")

    # --- single-rater reliability + implied judge ceiling -------------------
    print("\n[RELIABILITY] can we trust a single clinician rating? "
          "(anchor noise + singles spread)")
    print("  resid = per-rating noise (anchors); obs = singles centered var; "
          "rel = signal/total; max_corr = sqrt(rel) = ceiling for any judge")
    print(f"  {'attr':<15}{'anchorICC(1,1)':>15}{'resid_var':>11}{'obs_var':>9}{'rel':>7}{'max_corr':>10}")
    rel_attr, maxcorr_attr = {}, {}
    for a in LIKERT:
        mat, nb = _anchor_rater_matrix(anchors, a)
        vb, vr, ms_resid = _two_way_var_components(mat, nb)
        icc11 = vb / (vb + vr + ms_resid) if (vb + vr + ms_resid) > 0 else float("nan")
        rel, mc = _reliability(gv[a], ms_resid)
        rel_attr[a], maxcorr_attr[a] = rel, mc
        print(f"  {a:<15}{icc11:>15.2f}{ms_resid:>11.3f}{gv[a]:>9.3f}{rel:>7.2f}{mc:>10.2f}")
    core_rel = float(np.nanmean([rel_attr[a] for a in CORE_AXES]))
    core_mc = float(np.nanmean([maxcorr_attr[a] for a in CORE_AXES]))
    print(f"  CORE-AXES mean: reliability={core_rel:.2f}  max achievable corr={core_mc:.2f}")
    print("  NOTE: anchor ICC(1,1) is range-restriction-DEFLATED (near-identical anchors); "
          "the variance-components reliability (anchor noise vs singles spread) is the trustworthy read.")

    # --- composite (excl-cited) reliability + directional sync --------------
    comp_fn = lambda sd: float(np.mean([sd[a] for a in NONCITED]))
    matc, nbc = _anchor_rater_matrix(anchors, comp_fn)
    _, _, ms_resid_c = _two_way_var_components(matc, nbc)
    h_comp = {h: float(np.mean([centered[h][a] for a in NONCITED])) for h in centered}
    obs_var_c = float(pstdev(list(h_comp.values()))) ** 2
    rel_c, mc_c = _reliability(obs_var_c, ms_resid_c)
    print(f"\n[DIRECTIONAL] composite = mean of 7 non-cited Likert (cited reported separately)")
    print(f"  composite single-rater reliability={rel_c:.2f}  max achievable corr={mc_c:.2f} (ceiling)")
    hmean = float(np.mean(list(h_comp.values())))
    for label in labels:
        j_comp = {}
        for h in h_comp:
            jr = judges[label].get(h)
            if jr and all(jr.get(a) is not None for a in NONCITED):
                j_comp[h] = float(np.mean([jr[a] for a in NONCITED]))
        shared = [h for h in h_comp if h in j_comp]
        if len(shared) < 3:
            print(f"    {label:<14} (insufficient overlap)"); continue
        jv = [j_comp[h] for h in shared]; hv = [h_comp[h] for h in shared]
        rho, _p = C.spearman(jv, hv)
        jmean = float(np.mean(jv))
        sign = float(np.mean([(jx - jmean) * (hx - hmean) >= 0 for jx, hx in zip(jv, hv)]))
        frac = (rho / mc_c) if (mc_c and mc_c == mc_c and mc_c > 0) else float("nan")
        print(f"    {label:<14} n={len(shared):>3}  ρ_composite={rho:+.3f}  "
              f"sign-agree={sign:.2f}  ρ/ceiling={frac:+.2f}")
    print("  ρ_composite = does the judge order briefs like the leniency-corrected clinician; "
          "ρ/ceiling puts it against what's achievable vs a single noisy rating, not vs 1.0.")

    # --- anchor smoke + ceiling ---------------------------------------------
    print("\n[ANCHORS] smoke only (n=4) — AC2 vs 5-reviewer consensus; human ceiling")
    ceiling, raters, jrows = _anchor_ceiling_and_judges(anchors, judges)
    core_ceiling = float(np.nanmean([ceiling[a] for a in CORE_AXES]))
    print(f"  human-human AC2 ceiling (raters={raters}): core-axes mean = {core_ceiling:.3f}")
    print(f"    per-axis ceiling: " + ", ".join(f"{a}={ceiling[a]}" for a in CORE_AXES))
    for label in labels:
        core = [jrows[label][a] for a in CORE_AXES if jrows[label].get(a) is not None]
        m = round(float(np.mean(core)), 3) if core else float("nan")
        print(f"    {label:<14} core-axes AC2 vs consensus = {m}  (ceiling {core_ceiling:.2f})")

    # --- k-variance ----------------------------------------------------------
    print("\n[k-VARIANCE] iteration disagreement (k=3). Temp-0 locals ~0; "
          "o4-mini is the one to watch.")
    kv = _k_variance(judges)
    for label in labels:
        d = kv[label]
        rec = ""
        if d["frac_cases_disagree"] == d["frac_cases_disagree"]:
            if d["frac_cases_disagree"] >= 0.25 or d["mean_attr_spread"] >= 0.4:
                rec = "  -> NON-trivial wobble: consider k=5 for THIS judge"
            else:
                rec = "  -> stable at k=3"
        print(f"    {label:<14} cases={d['cases']:>3}  "
              f"frac_cases_with_disagreement={d['frac_cases_disagree']}  "
              f"mean_spread={d['mean_attr_spread']} max_spread={d['max_attr_spread']}{rec}")

    # --- ranking -------------------------------------------------------------
    print("\n[RANKING] by CORE-AXES AC2 on the singles (primary), %within1 + ρ_centered as tiebreak")
    rank = sorted(labels, key=lambda L: (_agg(summary[L], CORE_AXES, "ac2"),
                                         _agg(summary[L], CORE_AXES, "within1")), reverse=True)
    for i, L in enumerate(rank, 1):
        print(f"  {i}. {L}: AC2={_agg(summary[L], CORE_AXES,'ac2')} "
              f"%within1={_agg(summary[L], CORE_AXES,'within1')} "
              f"ρ_centered={_agg(summary[L], CORE_AXES,'spearman_centered')}")
    print("\n  NOTE: if core-axes variance is ~0 (gating) the AC2 gap is noise -> "
          "select on structural criteria (reasoning + clean lineage + coverage), "
          "and remember o4-mini carries a home-field stake on the gpt-5.4 condition.")


if __name__ == "__main__":
    main()
