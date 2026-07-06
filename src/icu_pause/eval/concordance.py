"""Concordance / inter-rater statistics for judge selection
(judge_selection_plan v2.3, Step 3).

Pure functions over aligned numpy arrays / small frames — no I/O, no LLM. Needs
numpy + pandas + scipy + pingouin (the sandbox/HPC analysis env, NOT the bare
Mac .venv). pingouin/scipy are imported lazily so importing this module for the
hand-rolled metrics (kappa, Gwet, Kendall's W) doesn't require them.

Metric roles (per plan):
  - icc_3k          : HEADLINE — ICC(3,k) two-way mixed consistency, judge vs human ref.
  - human_human_icc : local ceiling consistency check vs published 0.867 (n=4 anchors).
  - condition_rank_concordance (Kendall's W) : ROBUSTNESS HEADLINE across judges.
  - weighted_cohen_kappa / spearman / median_diff / wilcoxon : per-attribute secondary.
  - gwet_ac2        : chance-corrected agreement for the rare-event stigmatizing item.
  - krippendorff_alpha : secondary, ordinal/interval.
  - disattenuate    : SUPPLEMENTARY sensitivity only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ICCResult:
    icc: float
    ci_low: float
    ci_high: float
    n: int
    icc_type: str = "ICC3k"


# ---------------------------------------------------------------------------
# ICC (pingouin-backed)
# ---------------------------------------------------------------------------


def _icc_long(ratings_by_rater: dict[str, list[float]]):
    """Build pingouin long-format DataFrame from {rater -> per-target ratings}.

    All raters must rate the same number of targets, in the same target order.
    """
    import pandas as pd

    raters = list(ratings_by_rater)
    n_targets = len(ratings_by_rater[raters[0]])
    rows = []
    for r in raters:
        vals = ratings_by_rater[r]
        if len(vals) != n_targets:
            raise ValueError(f"rater {r} has {len(vals)} ratings, expected {n_targets}")
        for t in range(n_targets):
            rows.append({"target": t, "rater": r, "rating": float(vals[t])})
    return pd.DataFrame(rows), n_targets


def _icc3k_from_long(df, n_targets: int) -> ICCResult:
    import pingouin as pg

    res = pg.intraclass_corr(
        data=df, targets="target", raters="rater", ratings="rating", nan_policy="omit"
    )
    # The 'Type' column labels the six ICC variants, but the spelling depends on
    # the library/version: pingouin's Shrout-Fleiss style ("ICC3k") vs the
    # McGraw-Wong style ("ICC(C,k)"). ICC(3,k) = two-way mixed, consistency,
    # average measures = Shrout-Fleiss "ICC3k" = McGraw-Wong "ICC(C,k)". Match
    # both on a normalized form (strip parens/commas/space, upper-case).
    norm = res["Type"].astype(str).str.upper().str.replace(r"[()\s,]", "", regex=True)
    sub = res[norm.isin({"ICC3K", "ICCCK"})]
    if sub.empty:
        raise KeyError(
            "ICC3k not found in pingouin output; available Type values: "
            f"{res['Type'].astype(str).tolist()}"
        )
    row = sub.iloc[0]
    cols = {c.lower(): c for c in res.columns}
    # ICC point estimate: column literally named ICC across every schema we've seen.
    icc_col = cols.get("icc")
    icc_val = float(row[icc_col]) if icc_col else float("nan")
    # CI bounds: schema varies. pingouin packs both into a single "CI95%" cell as
    # [lo, hi]; other libs split into lower/upper columns under assorted names.
    # Fall back to NaN (CI is uninformative at n=4 anyway) and log the columns.
    ci_low = ci_high = float("nan")
    if "ci95%" in cols:
        ci = row[cols["ci95%"]]
        ci_low, ci_high = float(ci[0]), float(ci[1])
    else:
        lo = next((cols[k] for k in cols if "lower" in k or k in ("ci_l", "cil")), None)
        hi = next((cols[k] for k in cols if "upper" in k or k in ("ci_u", "ciu")), None)
        if lo and hi:
            ci_low, ci_high = float(row[lo]), float(row[hi])
        else:
            logger.warning("ICC CI columns not recognized; columns=%s", list(res.columns))
    return ICCResult(
        icc=icc_val,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n_targets,
        icc_type="ICC3k",
    )


def icc_3k(judge: np.ndarray, human_ref: np.ndarray) -> ICCResult:
    """ICC(3,k) treating (judge, human_ref) as the k=2 fixed raters across cases.

    This is the headline judge-human agreement metric (Croxford: ICC(3,k),
    two-way mixed, consistency, Shrout-Fleiss CI).
    """
    judge = np.asarray(judge, dtype=float)
    human_ref = np.asarray(human_ref, dtype=float)
    if judge.shape != human_ref.shape:
        raise ValueError("judge and human_ref must be the same length")
    df, n = _icc_long({"judge": list(judge), "human": list(human_ref)})
    return _icc3k_from_long(df, n)


def human_human_icc(rater_matrix: dict[str, list[float]]) -> ICCResult:
    """ICC(3,k) among human raters (anchors). rater_matrix: {reviewer_id ->
    per-anchor ratings} with all reviewers covering the same anchors in order.
    Used as a local consistency check against the published 0.867 ceiling —
    NOT as a disattenuation denominator (n=4 is too unstable)."""
    df, n = _icc_long(rater_matrix)
    return _icc3k_from_long(df, n)


# ---------------------------------------------------------------------------
# Pairwise ordinal / correlation metrics
# ---------------------------------------------------------------------------


def weighted_cohen_kappa(a, b, weights: str = "quadratic",
                         categories: list[int] | None = None) -> float:
    """Weighted Cohen's kappa for two raters on an ordinal scale.

    weights: 'quadratic' (default, standard for Likert) | 'linear' | 'identity'.
    categories: the full ordinal range (default 1..5); fixes the matrix size so
    unused categories don't distort weights.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if categories is None:
        categories = list(range(1, 6))
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    O = np.zeros((k, k), dtype=float)
    for x, y in zip(a, b):
        O[cat_idx[int(x)], cat_idx[int(y)]] += 1
    n = O.sum()
    if n == 0:
        return float("nan")
    O /= n

    row = O.sum(axis=1)
    col = O.sum(axis=0)
    E = np.outer(row, col)

    W = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            d = abs(i - j)
            if weights == "quadratic":
                W[i, j] = (d / (k - 1)) ** 2
            elif weights == "linear":
                W[i, j] = d / (k - 1)
            else:  # identity (unweighted)
                W[i, j] = 0.0 if i == j else 1.0

    denom = float((W * E).sum())
    if denom == 0:
        return float("nan")
    return 1.0 - float((W * O).sum()) / denom


def spearman(a, b) -> tuple[float, float]:
    """Spearman rank correlation (rho, p). Uses scipy."""
    from scipy.stats import spearmanr

    rho, p = spearmanr(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return float(rho), float(p)


def median_diff(judge, human) -> float:
    """Median signed difference (judge - human) = bias direction (Croxford's
    median score difference)."""
    judge = np.asarray(judge, dtype=float)
    human = np.asarray(human, dtype=float)
    return float(np.median(judge - human))


def wilcoxon_signed_rank(judge, human) -> tuple[float, float]:
    """Wilcoxon signed-rank (statistic, p) on the paired diffs. Returns
    (nan, nan) if all diffs are zero (scipy raises)."""
    from scipy.stats import wilcoxon

    judge = np.asarray(judge, dtype=float)
    human = np.asarray(human, dtype=float)
    diffs = judge - human
    if np.allclose(diffs, 0):
        return float("nan"), float("nan")
    try:
        stat, p = wilcoxon(judge, human)
        return float(stat), float(p)
    except ValueError:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Chance-corrected agreement for the binary stigmatizing item
# ---------------------------------------------------------------------------


def gwet_ac2(a, b, categories: list | None = None, weights: str = "identity") -> float:
    """Gwet's AC2 (AC1 when weights='identity'). Kappa-paradox-safe; preferred
    for the rare-event stigmatizing item. For a binary item, identity weights
    (=AC1) are correct.
    """
    a = list(a)
    b = list(b)
    if categories is None:
        categories = sorted(set(a) | set(b))
    cat_idx = {c: i for i, c in enumerate(categories)}
    q = len(categories)
    if q < 2:
        return float("nan")

    O = np.zeros((q, q), dtype=float)
    for x, y in zip(a, b):
        O[cat_idx[x], cat_idx[y]] += 1
    n = O.sum()
    if n == 0:
        return float("nan")
    P = O / n

    W = np.zeros((q, q), dtype=float)
    for i in range(q):
        for j in range(q):
            if weights == "quadratic" and q > 1:
                W[i, j] = 1.0 - (abs(i - j) / (q - 1)) ** 2
            else:
                W[i, j] = 1.0 if i == j else 0.0

    p_a = float((W * P).sum())  # weighted observed agreement
    # mean marginal category prevalence
    pi = (P.sum(axis=0) + P.sum(axis=1)) / 2.0
    # Gwet chance agreement: pe = (Tw / (q(q-1))) * (1 - sum pi^2).
    # With identity weights Tw=q, so the coefficient is 1/(q-1) -> AC1.
    Tw = float(W.sum())
    p_e = (Tw / (q * (q - 1))) * (1.0 - float(np.sum(pi ** 2)))
    if 1.0 - p_e == 0:
        return float("nan")
    return (p_a - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# Krippendorff's alpha (interval/ordinal), secondary
# ---------------------------------------------------------------------------


def krippendorff_alpha(rater_matrix: dict[str, list[float]],
                       level: str = "interval") -> float:
    """Krippendorff's alpha via the coincidence-matrix method, supporting
    missing values (use None / np.nan). level: 'interval' (default) or
    'ordinal'. rater_matrix: {rater -> per-unit values}, all same length.
    """
    raters = list(rater_matrix)
    n_units = len(rater_matrix[raters[0]])
    # units as columns: list of per-unit value lists (drop missing)
    units: list[list[float]] = []
    for u in range(n_units):
        vals = []
        for r in raters:
            v = rater_matrix[r][u]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            vals.append(float(v))
        units.append(vals)

    # value domain
    all_vals = sorted({v for u in units for v in u})
    if len(all_vals) < 2:
        return float("nan")
    idx = {v: i for i, v in enumerate(all_vals)}
    V = len(all_vals)

    # coincidence matrix
    coinc = np.zeros((V, V), dtype=float)
    for u in units:
        m = len(u)
        if m < 2:
            continue
        for x in range(m):
            for y in range(m):
                if x == y:
                    continue
                coinc[idx[u[x]], idx[u[y]]] += 1.0 / (m - 1)

    n_total = coinc.sum()
    if n_total == 0:
        return float("nan")
    nc = coinc.sum(axis=1)  # marginal

    def metric2(i: int, j: int) -> float:
        vi, vj = all_vals[i], all_vals[j]
        if level == "ordinal":
            lo, hi = (i, j) if i <= j else (j, i)
            s = sum(nc[g] for g in range(lo, hi + 1)) - (nc[lo] + nc[hi]) / 2.0
            return s ** 2
        return (vi - vj) ** 2  # interval

    Do = sum(coinc[i, j] * metric2(i, j) for i in range(V) for j in range(V))
    De = 0.0
    for i in range(V):
        for j in range(V):
            De += nc[i] * nc[j] * metric2(i, j)
    De /= (n_total - 1)
    if De == 0:
        return float("nan")
    return 1.0 - Do / De


# ---------------------------------------------------------------------------
# Cross-judge ranking concordance (ROBUSTNESS HEADLINE)
# ---------------------------------------------------------------------------


def condition_rank_concordance(judge_means_by_condition: dict[str, dict[str, float]]
                               ) -> dict:
    """Kendall's W across judges' orderings of pipeline conditions.

    judge_means_by_condition: {judge -> {condition -> mean PDSQI-9 score}}.
    All judges must score the same set of conditions. Returns W (0..1; 1 =
    perfect agreement), the per-condition mean rank, and the implied ranking.
    """
    judges = list(judge_means_by_condition)
    conditions = list(judge_means_by_condition[judges[0]])
    m = len(judges)        # number of "judges" (rankers)
    n = len(conditions)    # number of conditions ranked
    if n < 2 or m < 2:
        return {"W": float("nan"), "n_conditions": n, "n_judges": m,
                "mean_rank": {}, "ranking": conditions}

    import pandas as pd
    from scipy.stats import rankdata

    # ranks per judge (rank conditions high->low so rank 1 = best)
    ranks = np.zeros((m, n), dtype=float)
    tie_correction = 0.0
    for r, j in enumerate(judges):
        scores = np.array([judge_means_by_condition[j][c] for c in conditions], dtype=float)
        rk = rankdata(-scores)  # 1 = highest score
        ranks[r] = rk
        # tie correction term Tj = sum(t^3 - t) over tie groups
        _, counts = np.unique(rk, return_counts=True)
        tie_correction += float(np.sum(counts ** 3 - counts))

    Rj = ranks.sum(axis=0)              # summed rank per condition
    Rbar = Rj.mean()
    S = float(np.sum((Rj - Rbar) ** 2))
    denom = (m ** 2) * (n ** 3 - n) - m * tie_correction
    W = (12.0 * S) / denom if denom != 0 else float("nan")

    mean_rank = {c: float(Rj[i] / m) for i, c in enumerate(conditions)}
    ranking = sorted(conditions, key=lambda c: mean_rank[c])  # best (lowest rank) first
    _ = pd  # keep import meaningful if extended later
    return {"W": W, "n_conditions": n, "n_judges": m,
            "mean_rank": mean_rank, "ranking": ranking}


# ---------------------------------------------------------------------------
# Disattenuation (supplementary sensitivity only)
# ---------------------------------------------------------------------------


def disattenuate(observed_icc: float, ceiling: float = 0.867) -> float:
    """Disattenuated ICC = observed / sqrt(ceiling). SUPPLEMENTARY only; the
    ceiling is the published PDSQI-9 human-human ICC, not our noisy n=4.
    Can exceed 1.0 when observed approaches the (noisy) ceiling — report as a
    bounded sensitivity, not a headline."""
    if ceiling <= 0:
        return float("nan")
    return observed_icc / np.sqrt(ceiling)
