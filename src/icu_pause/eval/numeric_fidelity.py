"""Type-segmented numeric-fidelity scorer — PRIMARY (pre-registered) endpoint
for the decomposition ablation.

Question this answers
---------------------
Given the *identical* source data bundle, how many clinically salient numeric
values does each arm's brief retain, within +-5% tolerance, broken down by data
type (vitals, vasopressor dose, ventilator/FiO2, antibiotic duration, dialysis
status)?

Denominator (methodological note for the manuscript)
----------------------------------------------------
Ground truth is extracted from ``patient_context_text`` — the typed bundle
produced by ``serialize_to_json`` that is fed *identically* to every arm
(early_fusion s0/n0; the monolith arms receive the same blob). This is the
correct denominator for a COMPARATIVE (architecture-vs-architecture) claim: it
is provably identical across arms, so a value that compaction dropped before any
arm could see it is never counted against any arm. We therefore report fidelity
as retention vs. "the source bundle provided to every arm," a deliberate
sharpening of the pre-registered "vs source EHR" wording (the alternative —
scoring against raw pre-compaction EHR — charges identical compaction loss to
every arm and only adds noise to the between-arm contrast).

Salient-value policy (the pre-registration knobs — easy to tweak here)
----------------------------------------------------------------------
A transfer brief summarizes; it does not restate every measurement. So per type
we score the values a transfer note is expected to carry:

  * vitals .............. per vital_category: most-recent value + window min + max
  * vasopressor dose .... most-recent dose of each ACTIVE vasopressor infusion
  * ventilator/FiO2 ..... most-recent fio2_set and peep_set (when on a vent device)
  * antibiotic duration . duration in days of each antibiotic course in-window
  * dialysis status ..... binary on/off (exact status match, not +-5%)

Matching rule
-------------
A numeric ground-truth value is "retained" if any number parsed from the brief's
section text is within ``tolerance`` relative error (default 0.05 = +-5%). FiO2 is
matched in both fraction (0.40) and percent (40) forms. Dialysis is a binary
status match. Retention% per type = retained / in_scope; overall is the
micro-average (sum retained / sum in_scope across types).

This module is intentionally pipeline-independent: it consumes the serialized
``patient_context_text`` dict and a brief's ``sections`` dict, so it can be
unit-tested on saved JSON without CLIF access or any LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# --- Data-type labels (stable keys for the results table) ---
VITALS = "vitals"
PRESSOR = "vasopressor_dose"
VENT = "vent_fio2"
ABX = "antibiotic_duration"
DIALYSIS = "dialysis_status"

DATA_TYPES = [VITALS, PRESSOR, VENT, ABX, DIALYSIS]

# Vasopressor med_category tokens (lowercased substring match). CLIF mCIDE
# continuous categories at Northwestern; extend if a site uses other names.
VASOPRESSOR_TOKENS = (
    "norepinephrine", "epinephrine", "phenylephrine", "vasopressin",
    "dopamine", "dobutamine", "angiotensin", "milrinone",
)

# Antibiotic med_category tokens (lowercased substring). Broad on purpose — the
# duration check only fires on categories that actually appear in-window.
ANTIBIOTIC_TOKENS = (
    "cillin", "cef", "penem", "micin", "mycin", "floxacin", "cycline",
    "vancomycin", "linezolid", "daptomycin", "metronidazole", "azithromycin",
    "clindamycin", "aztreonam", "bactrim", "sulfamethoxazole", "trimethoprim",
    "fluconazole", "micafungin", "caspofungin", "acyclovir", "meropenem",
    "piperacillin", "tazobactam", "cefepime", "ceftriaxone", "ampicillin",
)

# Numbers we never treat as "real" clinical values when scanned out of the
# bundle (years, obvious counts). Kept tiny — the matcher is value-driven, not
# extraction-driven, so false positives mostly self-limit.
_NUM_RE = re.compile(r"[-+]?\d{1,4}(?:\.\d+)?")

# Label/unit synonyms a value must appear NEAR (context-aware matching). Keys are
# serialized vital_category tokens; values are lowercased prose forms a brief uses.
VITAL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "heart_rate": ("hr", "heart rate", "pulse"),
    "map": ("map", "mean arterial", "bp", "blood pressure"),
    "sbp": ("sbp", "systolic", "bp", "blood pressure"),
    "dbp": ("dbp", "diastolic", "bp", "blood pressure"),
    "spo2": ("spo2", "spo₂", "o2 sat", "oxygen sat", "sao2", " sat"),
    "respiratory_rate": ("rr", "resp rate", "respiratory rate"),
    "temp_c": ("temp", "temperature", "tmax", "tmin", "tcurrent"),
    "weight_kg": ("weight", "wt", " kg"),
}

# Pressor drug-name synonyms (brand/abbrev). Falls back to the drug token itself.
PRESSOR_SYNONYMS: dict[str, tuple[str, ...]] = {
    "norepinephrine": ("norepinephrine", "levophed", "norepi", "ne "),
    "epinephrine": ("epinephrine", "epi "),
    "phenylephrine": ("phenylephrine", "neo", "neosynephrine"),
    "vasopressin": ("vasopressin", "adh", "avp"),
    "dopamine": ("dopamine",),
    "dobutamine": ("dobutamine",),
    "milrinone": ("milrinone",),
    "angiotensin": ("angiotensin", "giapreza"),
}


# ---------------------------------------------------------------------------
# Ground-truth value container
# ---------------------------------------------------------------------------
@dataclass
class GTValue:
    """One salient ground-truth value to look for in a brief."""

    data_type: str
    label: str               # human-readable, e.g. "HR max" / "norepinephrine dose"
    value: float | None      # numeric target (None for pure status checks)
    kind: str = "numeric"    # "numeric" | "status"
    status: bool | None = None  # for kind == "status"
    alt_values: tuple[float, ...] = ()  # accepted equivalents (e.g. FiO2 0.40 / 40)
    # Tolerance: "relative" values use the scorer's global tolerance (default
    # +-5%); "absolute" values carry their own (abx duration = +-1 day, because
    # admin-record sparsity makes a relative tolerance on a small day count both
    # too tight and clinically meaningless).
    tol_kind: str = "relative"   # "relative" | "absolute"
    tol: float = 1.0             # used only when tol_kind == "absolute"
    subtype: str | None = None   # vitals: "recent" | "min" | "max" (for sensitivity)
    # Label/unit synonyms the value must appear NEAR to count as retained. Empty
    # → fall back to whole-brief presence. This is what stops a wordy brief from
    # scoring on coincidental numeric proximity (verbosity confound).
    context_terms: tuple[str, ...] = ()


@dataclass
class GroundTruth:
    values: list[GTValue] = field(default_factory=list)

    def by_type(self, dt: str) -> list[GTValue]:
        return [v for v in self.values if v.data_type == dt]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _rows(obj: Any) -> list[dict]:
    """Coerce a serialized domain value into a list of row dicts."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    return []


def _latest(rows: list[dict], dttm_key: str) -> dict | None:
    """Most-recent row by an ISO dttm string key (lexicographic ISO sort)."""
    dated = [r for r in rows if r.get(dttm_key)]
    if not dated:
        return rows[-1] if rows else None
    return max(dated, key=lambda r: str(r.get(dttm_key)))


def _has_token(name: str, tokens: Iterable[str]) -> bool:
    n = (name or "").lower()
    return any(t in n for t in tokens)


def _vital_ctx(cat: str) -> tuple[str, ...]:
    """Context terms a vital value must appear near (synonyms + the prose category)."""
    syn = VITAL_SYNONYMS.get(cat, ())
    return tuple(dict.fromkeys(syn + (cat.replace("_", " "), cat)))


def _pressor_ctx(drug: str) -> tuple[str, ...]:
    d = (drug or "").lower()
    for key, syn in PRESSOR_SYNONYMS.items():
        if key in d:
            return syn
    return (d,)


# ---------------------------------------------------------------------------
# Ground-truth extraction (salient-value policy)
# ---------------------------------------------------------------------------
def extract_ground_truth(ctx: dict[str, Any]) -> GroundTruth:
    """Extract the salient typed values from a serialized ``patient_context_text``.

    Defensive to shape drift: tolerates missing keys/columns and both the
    tiered vitals dict ({recent_raw, bucketed_trends}) and a flat row list.
    """
    gt = GroundTruth()
    gt.values += _gt_vitals(ctx.get("vitals"))
    gt.values += _gt_pressors(ctx.get("meds"))
    gt.values += _gt_vent(ctx.get("respiratory"))
    gt.values += _gt_abx(ctx.get("meds"))
    gt.values += _gt_dialysis(ctx)
    return gt


def _gt_vitals(vitals: Any) -> list[GTValue]:
    """Salient vital values from the real serialized shape.

    serialize_to_json emits ``vitals = {"bucketed_trends": [...]}`` where each row
    is one (vital_category, 8h-bucket) with a ``mean`` (no raw min/max, and
    ``recent_raw`` is often absent). So:
      * recent = the latest bucket's mean per category (prefer recent_raw if present)
      * range  = min/max OF THE BUCKET MEANS per category (a smoothed range, the
        only range available in the bundle; raw extremes aren't serialized).
    Range values are only added when the category actually varies (min != max),
    so single-bucket categories contribute just their current value.
    """
    out: list[GTValue] = []
    if isinstance(vitals, dict):
        recent = _rows(vitals.get("recent_raw"))
        trends = _rows(vitals.get("bucketed_trends"))
    elif isinstance(vitals, list):
        recent = _rows(vitals)
        trends = []
    else:
        return out

    # Bucket means per category, with recency index (bucket_8h: higher = newer).
    by_cat: dict[str, list[tuple[float, float]]] = {}
    for r in trends:
        cat = str(r.get("vital_category") or r.get("vital_name") or "").strip()
        mean = _to_float(r.get("mean"))
        if not cat or mean is None:
            continue
        bidx = _to_float(r.get("bucket_8h"))
        by_cat.setdefault(cat, []).append((bidx if bidx is not None else 0.0, mean))

    # Current value per category: recent_raw latest if present, else latest bucket.
    recent_by_cat: dict[str, float] = {}
    for r in recent:
        cat = str(r.get("vital_category") or r.get("vital_name") or "").strip()
        val = _to_float(r.get("vital_value"))
        if cat and val is not None and cat not in recent_by_cat:
            recent_by_cat[cat] = val
    for cat, series in by_cat.items():
        if cat not in recent_by_cat:
            recent_by_cat[cat] = max(series, key=lambda t: t[0])[1]
    for cat, val in recent_by_cat.items():
        out.append(GTValue(VITALS, f"{cat} recent", val, subtype="recent",
                           context_terms=_vital_ctx(cat)))

    # Range from the pool of available values per category (bucket means + any raw).
    pool: dict[str, list[float]] = {c: [m for _, m in s] for c, s in by_cat.items()}
    for r in recent:
        cat = str(r.get("vital_category") or r.get("vital_name") or "").strip()
        val = _to_float(r.get("vital_value"))
        if cat and val is not None:
            pool.setdefault(cat, []).append(val)
    for cat, vals in pool.items():
        if not vals:
            continue
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin > 1e-9:  # only emit a range when the category actually varies
            ctx = _vital_ctx(cat)
            out.append(GTValue(VITALS, f"{cat} min", vmin, subtype="min", context_terms=ctx))
            out.append(GTValue(VITALS, f"{cat} max", vmax, subtype="max", context_terms=ctx))
    return out


def _gt_pressors(meds: Any) -> list[GTValue]:
    if not isinstance(meds, dict):
        return []
    out: list[GTValue] = []

    # Prefer the med_state classifier: it identifies ACTIVE infusions and
    # carries recent doses. Fall back to continuous rows.
    states = meds.get("states") or {}
    records = _rows(states.get("records")) if isinstance(states, dict) else []
    seen: set[str] = set()
    for rec in records:
        drug = str(rec.get("drug_name") or rec.get("med_category") or "")
        state = str(rec.get("state") or "").upper()
        if not _has_token(drug, VASOPRESSOR_TOKENS):
            continue
        if state and "ACTIVE" not in state:
            continue
        doses = rec.get("recent_doses") or []
        dose = _to_float(doses[-1]) if doses else _to_float(rec.get("last_dose"))
        if dose is not None:
            out.append(GTValue(PRESSOR, f"{drug.lower()} dose", dose,
                               context_terms=_pressor_ctx(drug)))
            seen.add(drug.lower())

    if not out:
        for r in _rows(meds.get("continuous")):
            name = str(r.get("med_category") or r.get("medication_name") or "")
            if not _has_token(name, VASOPRESSOR_TOKENS) or name.lower() in seen:
                continue
            dose = _to_float(r.get("med_dose") or r.get("dose"))
            if dose is not None:
                out.append(GTValue(PRESSOR, f"{name.lower()} dose", dose,
                                   context_terms=_pressor_ctx(name)))
                seen.add(name.lower())
    return out


def _gt_vent(respiratory: Any) -> list[GTValue]:
    rows = _rows(respiratory)
    if not rows:
        return []
    latest = _latest(rows, "recorded_dttm") or {}
    out: list[GTValue] = []
    fio2_ctx = ("fio2", "fio₂", "fdo2", "fract")
    fio2 = _to_float(latest.get("fio2_set"))
    if fio2 is not None:
        # Accept fraction (0.40) and percent (40) renderings.
        if fio2 <= 1.0:
            out.append(GTValue(VENT, "FiO2 set", fio2, alt_values=(round(fio2 * 100, 3),),
                               context_terms=fio2_ctx))
        else:
            out.append(GTValue(VENT, "FiO2 set", fio2, alt_values=(round(fio2 / 100, 4),),
                               context_terms=fio2_ctx))
    peep = _to_float(latest.get("peep_set"))
    if peep is not None:
        out.append(GTValue(VENT, "PEEP set", peep, context_terms=("peep",)))
    return out


def _gt_abx(meds: Any) -> list[GTValue]:
    """Antibiotic course duration (days) = span of in-window admin timestamps
    per antibiotic med_category."""
    if not isinstance(meds, dict):
        return []
    from datetime import datetime

    spans: dict[str, list[str]] = {}
    for bucket in ("continuous", "intermittent"):
        for r in _rows(meds.get(bucket)):
            name = str(r.get("med_category") or r.get("medication_name") or "")
            if not _has_token(name, ANTIBIOTIC_TOKENS):
                continue
            dttm = r.get("admin_dttm") or r.get("recorded_dttm")
            if dttm:
                spans.setdefault(name.lower(), []).append(str(dttm))

    out: list[GTValue] = []
    for name, stamps in spans.items():
        parsed = []
        for s in stamps:
            try:
                parsed.append(datetime.fromisoformat(s.replace("Z", "+00:00")))
            except (ValueError, TypeError):
                continue
        if len(parsed) >= 2:
            days = (max(parsed) - min(parsed)).total_seconds() / 86400.0
            # +-1 day absolute tolerance: admin-record spans under-estimate true
            # course length (no MAR/orders table at this site), and a relative
            # tolerance on a 2-3 day count is clinically meaningless. Report abx
            # as its own line; don't fix the noise by importing a biased
            # (pipeline-derived) reference.
            out.append(GTValue(ABX, f"{name} duration_d", max(days, 0.5),
                               tol_kind="absolute", tol=1.0, context_terms=(name,)))
    return out


def _gt_dialysis(ctx: dict[str, Any]) -> list[GTValue]:
    crrt = _rows(ctx.get("crrt"))
    on = False
    if crrt:
        # Any in-window CRRT row implies active RRT unless an explicit on/off
        # flag says otherwise on the latest row.
        latest = _latest(crrt, "recorded_dttm") or {}
        flag = latest.get("dialysis_on_off")
        on = bool(flag) if flag is not None else True
    return [GTValue(DIALYSIS, "RRT active", None, kind="status", status=on)]


# ---------------------------------------------------------------------------
# Brief scoring
# ---------------------------------------------------------------------------
_DIALYSIS_POS = ("crrt", "cvvh", "cvvhd", "cvvhdf", "hemodialysis", "dialysis",
                 "rrt", "ihd", "sled", "renal replacement")


@dataclass
class TypeScore:
    data_type: str
    in_scope: int
    retained: int
    missed_labels: list[str] = field(default_factory=list)

    @property
    def retention(self) -> float | None:
        return self.retained / self.in_scope if self.in_scope else None


@dataclass
class FidelityResult:
    per_type: dict[str, TypeScore]
    overall_in_scope: int
    overall_retained: int
    # (GTValue, retained?) for every scored value — powers the sensitivities.
    value_results: list[tuple[GTValue, bool]] = field(default_factory=list)

    @property
    def overall_retention(self) -> float | None:
        return (self.overall_retained / self.overall_in_scope
                if self.overall_in_scope else None)

    def overall_excluding(self, exclude: Iterable[str]) -> dict[str, Any]:
        """Composite retention over all types except ``exclude`` (e.g. {ABX}).

        Lets the writeup report a composite that isn't distorted by the noisier
        admin-span abx line, per the pre-registration.
        """
        ex = set(exclude)
        in_scope = sum(ts.in_scope for dt, ts in self.per_type.items() if dt not in ex)
        retained = sum(ts.retained for dt, ts in self.per_type.items() if dt not in ex)
        return {
            "retention": retained / in_scope if in_scope else None,
            "in_scope": in_scope,
            "retained": retained,
            "excluded": sorted(ex),
        }

    def vitals_current_only(self) -> dict[str, Any]:
        """Sensitivity: vitals retention counting only the current value per
        category (subtype == 'recent'), dropping window min/max. Reported as a
        one-line robustness check so a reviewer sees vitals fidelity both ways.
        """
        vals = [(v, ok) for v, ok in self.value_results
                if v.data_type == VITALS and v.subtype == "recent"]
        in_scope = len(vals)
        retained = sum(1 for _, ok in vals if ok)
        return {
            "retention": retained / in_scope if in_scope else None,
            "in_scope": in_scope,
            "retained": retained,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": {
                "retention": self.overall_retention,
                "in_scope": self.overall_in_scope,
                "retained": self.overall_retained,
            },
            "overall_excl_abx": self.overall_excluding({ABX}),
            "vitals_current_only_sensitivity": self.vitals_current_only(),
            "by_type": {
                dt: {
                    "retention": ts.retention,
                    "in_scope": ts.in_scope,
                    "retained": ts.retained,
                    "missed": ts.missed_labels,
                }
                for dt, ts in self.per_type.items()
            },
        }


def _brief_text(sections: dict[str, str]) -> str:
    if not isinstance(sections, dict):
        return str(sections or "")
    return "\n".join(str(v) for v in sections.values())


def _brief_numbers(text: str) -> list[float]:
    nums = []
    for m in _NUM_RE.finditer(text):
        f = _to_float(m.group())
        if f is not None:
            nums.append(f)
    return nums


def _numbers_near(text_lower: str, context_terms: tuple[str, ...],
                  window: int = 60) -> list[float]:
    """Numbers appearing within ``window`` chars of any context term.

    This is what makes the match label-anchored (HR 110 near "hr") instead of
    rewarding a verbose brief for a coincidental number somewhere in the text.
    """
    nums: list[float] = []
    for term in context_terms:
        if not term:
            continue
        start = 0
        while True:
            i = text_lower.find(term, start)
            if i < 0:
                break
            seg = text_lower[max(0, i - window): i + len(term) + window]
            for m in _NUM_RE.finditer(seg):
                f = _to_float(m.group())
                if f is not None:
                    nums.append(f)
            start = i + len(term)
    return nums


def _matches(target: float, alts: tuple[float, ...], numbers: list[float],
             tol_kind: str, tol: float) -> bool:
    cands = (target,) + alts
    for c in cands:
        for n in numbers:
            if tol_kind == "absolute":
                if abs(n - c) <= tol:
                    return True
            else:
                if abs(n - c) / max(abs(c), 1e-9) <= tol:
                    return True
    return False


def score_brief(sections: dict[str, str], gt: GroundTruth,
                tolerance: float = 0.05) -> FidelityResult:
    text = _brief_text(sections)
    text_lower = text.lower()
    all_numbers = _brief_numbers(text)  # fallback when a value carries no context

    per_type: dict[str, TypeScore] = {
        dt: TypeScore(dt, 0, 0) for dt in DATA_TYPES
    }
    value_results: list[tuple[GTValue, bool]] = []

    for v in gt.values:
        ts = per_type[v.data_type]
        ts.in_scope += 1
        if v.kind == "status":
            # Dialysis: does the brief's stated status match ground truth?
            mentioned = any(tok in text_lower for tok in _DIALYSIS_POS)
            ok = (mentioned == bool(v.status))
        else:
            val_tol = v.tol if v.tol_kind == "absolute" else tolerance
            nums = (_numbers_near(text_lower, v.context_terms)
                    if v.context_terms else all_numbers)
            ok = v.value is not None and _matches(
                v.value, v.alt_values, nums, v.tol_kind, val_tol)
        if ok:
            ts.retained += 1
        else:
            ts.missed_labels.append(v.label)
        value_results.append((v, ok))

    overall_in = sum(ts.in_scope for ts in per_type.values())
    overall_ret = sum(ts.retained for ts in per_type.values())
    return FidelityResult(per_type, overall_in, overall_ret, value_results)


def score_case(sections: dict[str, str], ctx: dict[str, Any],
               tolerance: float = 0.05) -> FidelityResult:
    """Convenience: extract ground truth from the bundle and score a brief."""
    return score_brief(sections, extract_ground_truth(ctx), tolerance)
