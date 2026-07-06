"""C3 objective-fidelity gold set — a FRESH I-PASS / Joint-Commission crosswalk.

This is the schema-EXOGENOUS, salience-RESTRICTED gold standard for the
decomposition fidelity comparison (plan C3). It is anchored to external handoff
standards, NOT to the pipeline's own section schema, so the same gold set scores
the multiagent (full) arm and the single-agent (monolith) arm identically:

  * I-PASS (Starmer et al., NEJM 2014): I llness severity, P atient summary,
    A ction list, S ituation-awareness/contingency, S ynthesis-by-receiver.
    (Synthesis-by-receiver is a read-back step and is NOT gradeable from a
    written brief — excluded by design.)
  * Joint Commission Sentinel Event Alert 58 (2017) + NPSG hand-off content:
    current condition, recent/anticipated changes, medications, allergies, code
    status, pending tests/results, care plan.

Each GoldElement maps an external handoff requirement to:
  - the brief section(s) where it should surface (audit only; not used for matching),
  - the SOURCE-of-truth keys (from eval.source_bundle / numeric_fidelity) that
    establish ground truth from the FULL CHART,
  - a deterministic match rule + tolerance where the data is structured, else an
    LLM extraction/verification route (the C3 "hybrid" decision),
  - a SALIENCE predicate: the element is only scored on cases where it is actually
    present/applicable in the source (e.g. pressor dose only when an active pressor
    exists) — this is the "salience-restricted" part; absent-and-correctly-absent
    is not charged or credited,
  - a HARM weight (3 = life-threatening if wrong/missing, 2 = high, 1 = moderate),
    used for the harm-weighted composite that is later VALIDATED as a proxy against
    the clinician PDSQI-9 `accurate`/`cited` scores.

Two-sided scoring (built on this registry elsewhere):
  - RECALL: for each gold element present in the chart, is it present + correct in
    the brief? (a miss is an omission)
  - PRECISION: for each value the brief asserts on a gold element, does it match the
    chart? (a mismatch / unsupported assertion is a fabrication)

This module is PURE DATA + validators (no LLM, no I/O) so it imports anywhere and
is unit-testable. Field content is informed by the signed Tier-1/2 list in
docs/METHODS_compression_strategy_comparison_for_experts.md §3.0, re-expressed
under the I-PASS/JC crosswalk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---- I-PASS buckets ----------------------------------------------------------
IPASS_I = "I:illness_severity"
IPASS_P = "P:patient_summary"
IPASS_A = "A:action_list"
IPASS_S = "S:situation_contingency"

# ---- match rules -------------------------------------------------------------
EXACT_CATEGORICAL = "exact_categorical"   # string/enum equality after normalization
NUMERIC_TOL = "numeric_tol"               # |a-b| <= max(abs,rel*|ref|) after unit norm
DAY_X_OF_Y = "day_x_of_y"                 # antibiotic day as exact integers
SET_EQUALITY = "set_equality"             # set match / recall threshold (e.g. weekday set)
NAME_MATCH = "name_match"                 # name equality after whitespace/case norm
PRESENCE = "presence"                     # documented-or-not (binary)

# ---- extractor routing (the C3 "hybrid" split) -------------------------------
DETERMINISTIC = "deterministic"           # structured source -> rule-based match
LLM = "llm"                               # prose-dependent -> LLM extract + verify
HYBRID = "hybrid"                         # deterministic core + LLM fallback for prose

# ---- brief section keys (audit only) -----------------------------------------
SECTIONS = ("I", "C", "U_unprescribing", "P", "A", "U_uncertainty", "S", "E")


@dataclass(frozen=True)
class GoldElement:
    id: str
    ipass: str
    jc_element: str
    label: str
    brief_sections: tuple            # where it SHOULD surface (audit)
    source_keys: tuple               # source_bundle / numeric_fidelity ground-truth keys
    match: str
    extractor: str
    harm: int                        # 3 critical, 2 high, 1 moderate
    salient_when: str                # plain-language predicate for "present in source"
    gt_type: str | None = None       # numeric_fidelity data-type, if deterministic-applicable
    tolerance: dict | None = None    # {"abs":x,"rel":y} or {"decimals":n} or {"recall":0.8}
    notes: str = ""


# Tolerances per METHODS §3.0 (exact-after-norm for Tier-1 meds/vent/labs;
# ±5% vitals trends; ±1 day device dwell; ≥80% recall for set fields).
_EXACT = {"abs": 0.0, "rel": 0.0}
_VITAL_TOL = {"abs": 0.0, "rel": 0.05}
_RECALL80 = {"recall": 0.80}


GOLD_REGISTRY: list[GoldElement] = [
    # ===================== HARM 3 — life-threatening if wrong/missing ==========
    GoldElement(
        "code_status", IPASS_S, "JC:code_status", "Code status / resuscitation",
        ("C",), ("code_status", "clinical_notes"), EXACT_CATEGORICAL, HYBRID, 3,
        "any code-status order or documented goals-of-care discussion exists",
        notes="exact categorical (Full/DNR/DNI/DNR-DNI/comfort); often prose -> hybrid"),
    GoldElement(
        "vent_dependence", IPASS_P, "JC:current_condition", "Ventilator dependence",
        ("E", "I"), ("respiratory_support",), EXACT_CATEGORICAL, DETERMINISTIC, 3,
        "an invasive/non-invasive support record exists in the window",
        notes="on-vent vs not; device/mode categorical"),
    GoldElement(
        "vent_fio2", IPASS_P, "JC:current_condition", "FiO2 set",
        ("E",), ("respiratory_support",), NUMERIC_TOL, DETERMINISTIC, 3,
        "patient is on respiratory support with a recorded FiO2",
        gt_type="vent_fio2", tolerance=_EXACT),
    GoldElement(
        "vent_peep", IPASS_P, "JC:current_condition", "PEEP set",
        ("E",), ("respiratory_support",), NUMERIC_TOL, DETERMINISTIC, 3,
        "patient is on invasive ventilation with a recorded PEEP",
        gt_type="vent_fio2", tolerance=_EXACT),
    GoldElement(
        "pressor_active", IPASS_P, "JC:current_condition", "Active vasopressor(s)",
        ("I", "E", "S"), ("meds_continuous",), SET_EQUALITY, DETERMINISTIC, 3,
        "an active vasopressor infusion exists at/near reference time",
        gt_type="vasopressor_dose", tolerance=_RECALL80,
        notes="drug-set equality (which pressors running)"),
    GoldElement(
        "pressor_dose", IPASS_P, "JC:current_condition", "Vasopressor dose",
        ("I", "E", "S"), ("meds_continuous",), NUMERIC_TOL, DETERMINISTIC, 3,
        "an active vasopressor infusion exists",
        gt_type="vasopressor_dose", tolerance=_EXACT,
        notes="exact after unit normalization"),
    GoldElement(
        "anticoagulation", IPASS_S, "JC:medications", "Anticoagulation (drug/dose/route/target)",
        ("S", "U_unprescribing"), ("meds_continuous", "meds_intermittent", "labs_recent"),
        EXACT_CATEGORICAL, HYBRID, 3,
        "an anticoagulant order (heparin/DOAC/warfarin/bivalirudin) exists",
        notes="drug+dose+route+INR target all exact; INR from labs"),
    GoldElement(
        "critical_labs", IPASS_P, "JC:pending_and_results", "Critical labs (K/Na/Cr/Hgb/Plt/INR/lactate/glucose)",
        ("E", "S"), ("labs_recent",), NUMERIC_TOL, DETERMINISTIC, 3,
        "the critical analyte has a recent result in the window",
        tolerance={"decimals": "display"},
        notes="exact at canonical display precision (half-ULP)"),
    GoldElement(
        "allergies", IPASS_P, "JC:allergies", "Allergies",
        ("C", "S"), ("demographics", "clinical_notes"), SET_EQUALITY, LLM, 3,
        "any allergy is documented (incl. NKDA assertion)",
        tolerance=_RECALL80,
        notes="allergy rendering currently disabled in brief — flag asymmetry"),
    GoldElement(
        "crrt_ecmo", IPASS_P, "JC:current_condition", "CRRT / ECMO status",
        ("I", "S"), ("meds_continuous", "procedures", "respiratory_support"),
        EXACT_CATEGORICAL, HYBRID, 3,
        "a CRRT or ECMO record/order exists",
        gt_type="dialysis_status",
        notes="CRRT via dialysis_status deterministic; ECMO often prose -> hybrid"),

    # ===================== HARM 2 — high =======================================
    GoldElement(
        "antibiotics_day", IPASS_S, "JC:medications", "Antibiotics (day x of y)",
        ("S",), ("meds_intermittent", "meds_continuous"), DAY_X_OF_Y, HYBRID, 2,
        "an active antimicrobial course exists in the window",
        gt_type="antibiotic_duration",
        notes="drug exact; day index exact integers; duration deterministic, x-of-y prose"),
    GoldElement(
        "hd_schedule", IPASS_S, "JC:care_plan", "Intermittent HD schedule",
        ("S",), ("procedures", "meds_intermittent", "clinical_notes"), SET_EQUALITY, LLM, 2,
        "patient is on intermittent (not continuous) hemodialysis",
        tolerance={"recall": 1.0}, notes="weekday-set equality, not string match"),
    GoldElement(
        "lines_drains", IPASS_P, "JC:current_condition", "Lines / drains / devices",
        ("E", "S"), ("procedures", "clinical_notes"), SET_EQUALITY, LLM, 2,
        "any indwelling line/drain/device is documented",
        tolerance=_RECALL80),
    GoldElement(
        "dpoa", IPASS_S, "JC:care_plan", "DPOA / surrogate decision-maker",
        ("C",), ("clinical_notes", "demographics"), NAME_MATCH, LLM, 2,
        "a surrogate / DPOA / contact is documented",
        notes="name match after whitespace/case norm; PHI-entangled (see redaction memo)"),
    GoldElement(
        "airway_status", IPASS_P, "JC:current_condition", "Airway (difficult / trach type-size)",
        ("E", "S"), ("respiratory_support", "procedures", "clinical_notes"),
        EXACT_CATEGORICAL, LLM, 2,
        "a difficult-airway flag or tracheostomy is documented"),
    GoldElement(
        "infusions_other", IPASS_P, "JC:medications", "Insulin / opioid / sedation infusions",
        ("S", "U_unprescribing"), ("meds_continuous",), NUMERIC_TOL, HYBRID, 2,
        "a continuous insulin/opioid/sedation infusion is active",
        tolerance=_EXACT, notes="dose exact after unit norm; not in pressor token set -> hybrid"),
    GoldElement(
        "pending_results", IPASS_A, "JC:pending_and_results", "Pending tests / results",
        ("P",), ("microbiology", "labs_recent", "clinical_notes"), SET_EQUALITY, HYBRID, 2,
        "any test is pending/resulted-but-unacknowledged at reference time",
        tolerance=_RECALL80, notes="≥80% recall of pending items"),

    # ===================== HARM 1 — moderate ===================================
    GoldElement(
        "vitals_trends", IPASS_P, "JC:current_condition", "Vital-sign trends",
        ("E",), ("vitals_summary",), NUMERIC_TOL, DETERMINISTIC, 1,
        "bucketed vital trends exist in the window",
        gt_type="vitals", tolerance=_VITAL_TOL,
        notes="±5% vs BUCKETED medians, never raw rows (METHODS §3.0)"),
    GoldElement(
        "neuro_scores", IPASS_P, "JC:current_condition", "GCS / RASS / CAM-ICU / SOFA",
        ("E",), ("assessments",), EXACT_CATEGORICAL, HYBRID, 1,
        "a neuro/sedation/delirium score is recorded"),
    GoldElement(
        "nutrition", IPASS_S, "JC:care_plan", "Nutrition / diet",
        ("S",), ("meds_continuous", "clinical_notes"), EXACT_CATEGORICAL, LLM, 1,
        "a diet order or nutrition (TPN/EN) is documented"),
    GoldElement(
        "mobility", IPASS_S, "JC:care_plan", "Mobility status",
        ("S",), ("clinical_notes",), EXACT_CATEGORICAL, LLM, 1,
        "mobility/activity is documented"),
    GoldElement(
        "consultants", IPASS_A, "JC:care_plan", "Active consultants",
        ("A",), ("clinical_notes", "diagnoses"), SET_EQUALITY, LLM, 1,
        "any consulting service is documented",
        tolerance=_RECALL80),
    GoldElement(
        "disposition", IPASS_A, "JC:care_plan", "Disposition / destination",
        ("S",), ("clinical_notes",), EXACT_CATEGORICAL, LLM, 1,
        "a disposition/transfer destination is documented"),

    # ===================== I-PASS narrative / contingency (LLM) ================
    GoldElement(
        "illness_severity", IPASS_I, "JC:current_condition", "Illness severity (stable/watcher/unstable)",
        ("I",), ("clinical_notes", "vitals_summary", "respiratory_support", "meds_continuous"),
        EXACT_CATEGORICAL, LLM, 2,
        "always salient (every transfer has an acuity)",
        notes="I-PASS 'I'; derived gestalt — LLM-judged against chart acuity markers"),
    GoldElement(
        "contingency_if_then", IPASS_S, "JC:anticipated_changes", "Situation awareness / if-then contingencies",
        ("U_uncertainty", "S"), ("clinical_notes",), PRESENCE, LLM, 2,
        "always salient for ICU->ward transfer",
        notes="I-PASS 'S'; presence + groundedness of anticipatory guidance"),
    GoldElement(
        "course_summary", IPASS_P, "JC:current_condition", "Hospital/ICU course summary",
        ("I",), ("clinical_notes", "diagnoses"), PRESENCE, LLM, 1,
        "always salient",
        notes="I-PASS 'P'; narrative grounding (no fabricated events)"),
]


# ---- helpers / accessors -----------------------------------------------------
def by_ipass(bucket: str) -> list[GoldElement]:
    return [g for g in GOLD_REGISTRY if g.ipass == bucket]


def by_harm(level: int) -> list[GoldElement]:
    return [g for g in GOLD_REGISTRY if g.harm == level]


def deterministic_elements() -> list[GoldElement]:
    return [g for g in GOLD_REGISTRY if g.extractor in (DETERMINISTIC, HYBRID)]


def llm_elements() -> list[GoldElement]:
    return [g for g in GOLD_REGISTRY if g.extractor in (LLM, HYBRID)]


def validate() -> list[str]:
    """Internal consistency checks; returns a list of problems (empty = OK)."""
    problems = []
    ids = [g.id for g in GOLD_REGISTRY]
    dups = {i for i in ids if ids.count(i) > 1}
    if dups:
        problems.append(f"duplicate ids: {sorted(dups)}")
    for g in GOLD_REGISTRY:
        if g.harm not in (1, 2, 3):
            problems.append(f"{g.id}: harm {g.harm} not in 1..3")
        if g.match not in (EXACT_CATEGORICAL, NUMERIC_TOL, DAY_X_OF_Y,
                           SET_EQUALITY, NAME_MATCH, PRESENCE):
            problems.append(f"{g.id}: bad match rule {g.match}")
        if g.extractor not in (DETERMINISTIC, LLM, HYBRID):
            problems.append(f"{g.id}: bad extractor {g.extractor}")
        if not set(g.brief_sections) <= set(SECTIONS):
            problems.append(f"{g.id}: brief_sections {g.brief_sections} not subset of {SECTIONS}")
        if g.match == NUMERIC_TOL and not (g.tolerance or g.gt_type):
            problems.append(f"{g.id}: NUMERIC_TOL needs a tolerance or gt_type")
        # gt_type is an OPTIONAL link to the existing numeric_fidelity engine; some
        # deterministic elements (categorical vent dependence, labs without a
        # numeric_fidelity type) use their own deterministic matcher and need none.
    return problems


if __name__ == "__main__":
    probs = validate()
    print(f"C3 gold registry: {len(GOLD_REGISTRY)} elements")
    from collections import Counter
    h = Counter(g.harm for g in GOLD_REGISTRY)
    e = Counter(g.extractor for g in GOLD_REGISTRY)
    ip = Counter(g.ipass for g in GOLD_REGISTRY)
    print(f"  harm:      3={h[3]}  2={h[2]}  1={h[1]}")
    print(f"  extractor: {dict(e)}")
    print(f"  I-PASS:    {dict(ip)}")
    print(f"  deterministic-capable: {len(deterministic_elements())}  llm-touch: {len(llm_elements())}")
    print("\n  id                     ipass                  harm  match              extractor")
    for g in GOLD_REGISTRY:
        print(f"  {g.id:22s} {g.ipass:22s} {g.harm}     {g.match:18s} {g.extractor}")
    print("\nVALIDATION:", "OK" if not probs else "PROBLEMS:")
    for p in probs:
        print("  -", p)
