"""Medication state classifier — ACTIVE / RECENTLY STOPPED / HISTORICAL.

Replaces the prior assumption that any med admin event in the 48h window is
"active." That assumption produced false-positive DDI alerts (e.g., flagging
fentanyl + propofol synergy when propofol was stopped 12h ago) and false
HISTORICAL classifications for q24h-dosed antibiotics where one dose 19h
ago is still mid-regimen (the cefepime + metronidazole aspiration-coverage
case). CLIF schema does not expose discontinuation flags, so state is
reconstructed from the timing of admin events alone — a deliberate scoping
decision documented in the manuscript methods.

Continuous infusions (propofol, fentanyl drip, norepinephrine, etc.)
    ACTIVE              most recent admin within 6h before reference_dttm
    RECENTLY_STOPPED    last admin 6-48h before reference_dttm
    HISTORICAL          last admin >48h before reference_dttm

Intermittent meds (scheduled IV/PO/PRN) — drug-aware (2026-05-08)
    ACTIVE_*            most recent dose within (expected_interval × 1.25)
    RECENTLY_STOPPED    expected_interval × 1.25 ≤ hours_since
                        < max(lookback_hours, 2 × expected_interval)
    HISTORICAL          beyond that

    expected_interval resolves in this order:
      1. (drug, route, renal_status) lookup table (DOSING_INTERVALS_HOURS)
      2. observed median inter-dose spacing if ≥3 admin events in window
      3. DEFAULT_INTERVAL_HOURS (24h) — biases toward ACTIVE/RECENTLY_STOPPED
         which is the safer error direction for a handoff (a missed
         antibiotic in HISTORICAL is invisible; an erroneously-active one
         is at worst noise the intensivist resolves).

    Boundary semantics: strict < on both upper bounds, so a value at exactly
    the threshold falls into the slower bucket (RECENTLY_STOPPED, not
    ACTIVE). The dedicated boundary test locks this in.

Trending-to-zero override (continuous only)
-------------------------------------------
A med whose most recent admin is within 6h is normally ACTIVE. But if the
last 2-3 admin doses are strictly decreasing AND the most recent dose is
< 25% of the peak in the window AND the most recent dose is at or near
zero, classify as RECENTLY_STOPPED instead. This catches the "vasopressor
weaned to off 5h ago" case where the patient is no longer pressor-dependent
even though a non-zero dose was charted within the 6h ACTIVE window.

Application
-----------
- DDI checking pairs only ACTIVE drugs (continuous + intermittent).
- U section narrative reports all three states with explicit labels.
- "(N active)" count in QA output reflects ACTIVE only.

Known v1 limitations
--------------------
- Renal status is point-in-time (latest creatinine in window). For AKI with
  rapidly-changing creatinine, doses given 30-50h ago get evaluated against
  current renal status rather than the renal status that was in effect when
  they were given. See test_renal_status_change_known_limitation (xfail).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Literal, Optional
from statistics import median

# State labels exposed to downstream consumers.
MedState = Literal[
    "ACTIVE",
    "ACTIVE_SCHEDULED",
    "ACTIVE_PRN",
    "RECENTLY_STOPPED",
    "HISTORICAL",
]

# Renal status banding for dosing-interval lookup.
RenalStatus = Literal[
    "normal",    # eGFR ≥ 90
    "mild",      # eGFR 60-89
    "moderate",  # eGFR 30-59
    "severe",    # eGFR 15-29
    "dialysis",  # eGFR < 15 OR currently on RRT (CRRT/IHD)
    "unknown",   # no creatinine in window OR missing demographics
]

# Time thresholds (hours)
CONTINUOUS_ACTIVE_HOURS = 6
CONTINUOUS_RECENTLY_STOPPED_HOURS = 48

# Drug-aware intermittent thresholds
ACTIVE_INTERVAL_MULTIPLIER = 1.25
DEFAULT_INTERVAL_HOURS = 24.0
DEFAULT_LOOKBACK_HOURS = 48
MIN_OBSERVED_DOSES_FOR_INTERVAL_ESTIMATE = 3

# Trending-to-zero override
TRENDING_DOSE_FRACTION = 0.25  # last dose < 25% of peak → trending to zero
TRENDING_NEAR_ZERO_FRACTION = 0.10  # last dose < 10% of peak → near zero

# Drug-name canonicalization. Extends the same canonical-name pattern used
# by drug_interactions.py — keep the two maps in sync if you add aliases.
_DRUG_ALIASES: dict[str, str] = {
    "cefepime": "cefepime",
    "maxipime": "cefepime",
    "ceftriaxone": "ceftriaxone",
    "rocephin": "ceftriaxone",
    "ertapenem": "ertapenem",
    "invanz": "ertapenem",
    "levofloxacin": "levofloxacin",
    "levaquin": "levofloxacin",
    "daptomycin": "daptomycin",
    "cubicin": "daptomycin",
    "aztreonam": "aztreonam",
    "azactam": "aztreonam",
    "vancomycin": "vancomycin",
    "vancocin": "vancomycin",
    "piperacillin/tazobactam": "piperacillin_tazobactam",
    "piperacillin-tazobactam": "piperacillin_tazobactam",
    "piperacillin tazobactam": "piperacillin_tazobactam",
    "piperacillin_tazobactam": "piperacillin_tazobactam",
    "pip-tazo": "piperacillin_tazobactam",
    "pip/tazo": "piperacillin_tazobactam",
    "zosyn": "piperacillin_tazobactam",
    "tazocin": "piperacillin_tazobactam",
    "meropenem": "meropenem",
    "merrem": "meropenem",
    "metronidazole": "metronidazole",
    "flagyl": "metronidazole",
}

# (canonical_drug, route) → {RenalStatus: hours}
# Source: IDSA + Sanford 2024 + Lexicomp ICU dosing references. Defaults
# reflect typical adult ICU regimens; institution-specific protocols (e.g.,
# extended-infusion pip-tazo, AUC-targeted vancomycin) may vary. The
# "unknown" column is deliberately conservative — picks the longest plausible
# interval so the classifier biases toward ACTIVE/RECENTLY_STOPPED when
# renal status can't be computed (no creatinine in window, missing demos).
DOSING_INTERVALS_HOURS: dict[tuple[str, str], dict[RenalStatus, float]] = {
    ("cefepime", "IV"): {
        "normal": 8.0, "mild": 8.0, "moderate": 12.0,
        "severe": 24.0, "dialysis": 24.0, "unknown": 24.0,
    },
    ("ceftriaxone", "IV"): {  # not renal-adjusted in routine practice
        "normal": 24.0, "mild": 24.0, "moderate": 24.0,
        "severe": 24.0, "dialysis": 24.0, "unknown": 24.0,
    },
    ("ertapenem", "IV"): {
        "normal": 24.0, "mild": 24.0, "moderate": 24.0,
        "severe": 24.0, "dialysis": 24.0, "unknown": 24.0,
    },
    ("levofloxacin", "IV"): {
        "normal": 24.0, "mild": 24.0, "moderate": 24.0,
        "severe": 48.0, "dialysis": 48.0, "unknown": 48.0,
    },
    ("daptomycin", "IV"): {
        "normal": 24.0, "mild": 24.0, "moderate": 24.0,
        "severe": 48.0, "dialysis": 48.0, "unknown": 48.0,
    },
    ("aztreonam", "IV"): {
        "normal": 8.0, "mild": 8.0, "moderate": 12.0,
        "severe": 24.0, "dialysis": 24.0, "unknown": 24.0,
    },
    # NOTE: vancomycin in modern practice is AUC-targeted, with q8h/q12h/q24h
    # variants depending on patient size, infection type, and target trough.
    # The intervals below are the modal frequency band for adult ICU patients
    # by renal status. Future readers: do not treat these as authoritative —
    # they're a classifier heuristic, not a dosing recommendation.
    ("vancomycin", "IV"): {
        "normal": 12.0, "mild": 12.0, "moderate": 24.0,
        "severe": 48.0, "dialysis": 72.0, "unknown": 24.0,
    },
    ("piperacillin_tazobactam", "IV"): {
        "normal": 8.0, "mild": 8.0, "moderate": 8.0,
        "severe": 12.0, "dialysis": 12.0, "unknown": 12.0,
    },
    # NOTE: standard-infusion meropenem at some institutions is q6h; extended-
    # infusion regimens push toward q8h. Defaults below reflect the common q8h
    # standard-infusion pattern. Same caveat as vancomycin above.
    ("meropenem", "IV"): {
        "normal": 8.0, "mild": 8.0, "moderate": 12.0,
        "severe": 24.0, "dialysis": 24.0, "unknown": 12.0,
    },
    ("metronidazole", "IV"): {  # primarily hepatic; no renal adjustment
        "normal": 8.0, "mild": 8.0, "moderate": 8.0,
        "severe": 8.0, "dialysis": 8.0, "unknown": 8.0,
    },
}


def _canonicalize_drug(name: str) -> Optional[str]:
    """Return the canonical drug key for a CLIF med_category/medication_name.

    Returns None if the name doesn't match any known antimicrobial in the
    lookup table. Matches case-insensitively against the alias map.
    """
    if not name:
        return None
    key = name.strip().lower()
    if key in _DRUG_ALIASES:
        return _DRUG_ALIASES[key]
    return None


def _lookup_dosing_interval(
    drug_name: str,
    route: Optional[str],
    renal_status: RenalStatus,
) -> Optional[float]:
    """Return expected dosing interval in hours from the lookup table.

    Returns None if (drug, route) is not in the table — caller falls back to
    observed-spacing estimation, then DEFAULT_INTERVAL_HOURS.
    """
    canonical = _canonicalize_drug(drug_name)
    if canonical is None:
        return None
    route_key = (route or "IV").strip().upper()
    # Normalize common route synonyms — CLIF route_category is mostly "IV"
    # but some sites emit "Intravenous", "PIV", etc.
    if route_key in ("INTRAVENOUS", "PIV", "CVC"):
        route_key = "IV"
    table_entry = DOSING_INTERVALS_HOURS.get((canonical, route_key))
    if table_entry is None:
        return None
    return table_entry.get(renal_status, table_entry.get("unknown"))


def _estimate_observed_interval(
    admin_dttms: list[datetime],
) -> Optional[float]:
    """Estimate dosing interval from ≥3 admin events using median spacing.

    Returns None if we don't have enough events to estimate. Used as a
    fallback for drugs not in the lookup table.
    """
    if len(admin_dttms) < MIN_OBSERVED_DOSES_FOR_INTERVAL_ESTIMATE:
        return None
    sorted_dttms = sorted(admin_dttms)
    gaps_hours = [
        (sorted_dttms[i] - sorted_dttms[i - 1]).total_seconds() / 3600
        for i in range(1, len(sorted_dttms))
    ]
    if not gaps_hours:
        return None
    return median(gaps_hours)


def _resolve_expected_interval(
    drug_name: str,
    route: Optional[str],
    renal_status: RenalStatus,
    admin_dttms: list[datetime],
) -> float:
    """Resolve expected dosing interval per the documented order:
    drug+route+renal lookup → observed median → DEFAULT_INTERVAL_HOURS.
    """
    interval = _lookup_dosing_interval(drug_name, route, renal_status)
    if interval is not None:
        return interval
    interval = _estimate_observed_interval(admin_dttms)
    if interval is not None:
        return interval
    return DEFAULT_INTERVAL_HOURS


# ---------------------------------------------------------------------------
# Renal status resolution (CKD-EPI 2021, race-free)
# ---------------------------------------------------------------------------

def _egfr_to_renal_status(egfr: float) -> RenalStatus:
    """Map eGFR (mL/min/1.73m²) to a renal-status band per KDIGO CKD stages.

    Note: the dosing lookup table uses the same band names but the
    classifier biases lookups toward the longer-interval column for
    severe/dialysis. Banding here is purely the eGFR → band mapping.
    """
    if egfr >= 90:
        return "normal"
    if egfr >= 60:
        return "mild"
    if egfr >= 30:
        return "moderate"
    if egfr >= 15:
        return "severe"
    return "dialysis"


def _ckd_epi_2021_egfr(
    creatinine_mg_dl: float,
    age_years: float,
    sex_category: str,
) -> Optional[float]:
    """CKD-EPI 2021 race-free eGFR (mL/min/1.73m²).

    Formula (Inker et al., NEJM 2021):
        eGFR = 142 × min(Scr/k, 1)^a × max(Scr/k, 1)^-1.200
                   × 0.9938^age × (1.012 if female else 1)
    where k = 0.7 (female) / 0.9 (male) and
          a = -0.241 (female) / -0.302 (male).

    Returns None if any input is invalid (non-positive Scr, missing sex).
    """
    if creatinine_mg_dl is None or creatinine_mg_dl <= 0:
        return None
    if age_years is None or age_years <= 0:
        return None
    sex_norm = (sex_category or "").strip().lower()
    if sex_norm in ("female", "f", "woman"):
        k, a, sex_factor = 0.7, -0.241, 1.012
    elif sex_norm in ("male", "m", "man"):
        k, a, sex_factor = 0.9, -0.302, 1.0
    else:
        return None
    ratio = creatinine_mg_dl / k
    egfr = (
        142.0
        * (min(ratio, 1.0) ** a)
        * (max(ratio, 1.0) ** -1.200)
        * (0.9938 ** age_years)
        * sex_factor
    )
    return egfr


def resolve_renal_status(
    *,
    creatinine_rows: list[dict],
    age_years: Optional[float],
    sex_category: Optional[str],
    on_rrt: bool,
) -> RenalStatus:
    """Resolve a single point-in-time renal-status band for dosing lookup.

    Resolution order:
        1. on_rrt=True (active CRRT/IHD in lookback window) → "dialysis"
           (creatinine values during RRT are not interpretable for
           clearance estimation, so we short-circuit).
        2. CKD-EPI 2021 eGFR from the **most recent** creatinine in
           ``creatinine_rows`` plus age + sex → banded.
        3. "unknown" otherwise (no creatinine, missing age, missing sex,
           or invalid values).

    Known v1 limitation: a single point-in-time creatinine doesn't capture
    rapidly-changing renal function (AKI). Doses given 30-50h ago get
    evaluated against current renal status rather than the renal status
    that was in effect when they were given. See
    test_renal_status_change_known_limitation for documented behavior.

    Parameters
    ----------
    creatinine_rows:
        List of lab dicts already filtered to ``lab_category == "creatinine"``,
        with numeric values in mg/dL. Caller is responsible for filtering;
        this function picks the latest by timestamp key.
    age_years:
        Patient age (typically ``ctx.age_at_admission``).
    sex_category:
        CLIF sex category string ("Female" / "Male" / etc.).
    on_rrt:
        True if patient has any CRRT row in lookback window OR an active
        IHD/HD procedure in window. Caller computes this.
    """
    if on_rrt:
        return "dialysis"
    if age_years is None or not sex_category:
        return "unknown"
    if not creatinine_rows:
        return "unknown"

    latest_dttm: Optional[datetime] = None
    latest_value: Optional[float] = None
    for row in creatinine_rows:
        value = _coerce_float(row.get("lab_value_numeric"))
        if value is None or value <= 0:
            continue
        # Prefer collect time over result time for ordering — collect is
        # when the kidney function was actually sampled.
        ts_raw = (
            row.get("lab_collect_dttm")
            or row.get("lab_result_dttm")
            or row.get("recorded_dttm")
        )
        ts = _parse_dttm(ts_raw)
        if ts is None:
            continue
        if latest_dttm is None or ts > latest_dttm:
            latest_dttm = ts
            latest_value = value
    if latest_value is None:
        return "unknown"

    egfr = _ckd_epi_2021_egfr(latest_value, float(age_years), sex_category)
    if egfr is None:
        return "unknown"
    return _egfr_to_renal_status(egfr)


@dataclass(frozen=True)
class MedStateRecord:
    """One medication's classified state at reference_dttm."""

    drug_name: str
    state: MedState
    last_admin_dttm: Optional[datetime]
    last_dose: Optional[float]
    admin_count_in_window: int
    is_continuous: bool
    trending_to_zero: bool = False
    # Drug-aware intermittent fields. None for continuous meds and for
    # intermittent meds with zero admin events in window.
    expected_interval_hours: Optional[float] = None
    hours_since_last_admin: Optional[float] = None

    @property
    def is_active(self) -> bool:
        return self.state in ("ACTIVE", "ACTIVE_SCHEDULED", "ACTIVE_PRN")


def _parse_dttm(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _drug_name(row: dict) -> str:
    return str(
        row.get("med_category") or row.get("medication_name") or ""
    ).strip()


def _detect_trending_to_zero(
    admin_events: list[tuple[datetime, Optional[float]]],
) -> bool:
    """Return True if continuous infusion is weaning toward zero.

    Heuristic: the last 2-3 doses are strictly non-increasing AND the most
    recent dose is < 25% of the in-window peak (or <= a near-zero floor).
    Requires at least 2 dose-bearing events; if doses are unavailable,
    cannot make this determination — return False (caller treats as ACTIVE).
    """
    dosed = [
        (dttm, dose) for dttm, dose in admin_events if dose is not None
    ]
    if len(dosed) < 2:
        return False

    peak = max(dose for _, dose in dosed)
    if peak <= 0:
        return False

    last_dose = dosed[-1][1]

    # Strict non-increasing across last 2-3 dosed events
    tail = dosed[-3:] if len(dosed) >= 3 else dosed[-2:]
    if not all(tail[i][1] <= tail[i - 1][1] for i in range(1, len(tail))):
        return False

    # Must actually be weaning, not just stable
    if last_dose >= peak * TRENDING_DOSE_FRACTION:
        return False

    return True


def _classify_continuous(
    drug: str,
    rows: list[dict],
    reference_dttm: datetime,
) -> MedStateRecord:
    """Classify a continuous-infusion drug at reference_dttm."""
    parsed: list[tuple[datetime, Optional[float]]] = []
    for row in rows:
        dttm = _parse_dttm(row.get("admin_dttm"))
        if dttm is None or dttm > reference_dttm:
            continue
        dose = _coerce_float(row.get("med_dose") or row.get("dose"))
        parsed.append((dttm, dose))

    if not parsed:
        return MedStateRecord(
            drug_name=drug,
            state="HISTORICAL",
            last_admin_dttm=None,
            last_dose=None,
            admin_count_in_window=0,
            is_continuous=True,
        )

    parsed.sort(key=lambda x: x[0])
    last_dttm, last_dose = parsed[-1]
    hours_since = (reference_dttm - last_dttm).total_seconds() / 3600

    if hours_since <= CONTINUOUS_ACTIVE_HOURS:
        # Trending-to-zero override: weaned-off vasopressor pattern.
        trending = _detect_trending_to_zero(parsed)
        if trending:
            return MedStateRecord(
                drug_name=drug,
                state="RECENTLY_STOPPED",
                last_admin_dttm=last_dttm,
                last_dose=last_dose,
                admin_count_in_window=len(parsed),
                is_continuous=True,
                trending_to_zero=True,
            )
        return MedStateRecord(
            drug_name=drug,
            state="ACTIVE",
            last_admin_dttm=last_dttm,
            last_dose=last_dose,
            admin_count_in_window=len(parsed),
            is_continuous=True,
        )
    if hours_since <= CONTINUOUS_RECENTLY_STOPPED_HOURS:
        return MedStateRecord(
            drug_name=drug,
            state="RECENTLY_STOPPED",
            last_admin_dttm=last_dttm,
            last_dose=last_dose,
            admin_count_in_window=len(parsed),
            is_continuous=True,
        )
    return MedStateRecord(
        drug_name=drug,
        state="HISTORICAL",
        last_admin_dttm=last_dttm,
        last_dose=last_dose,
        admin_count_in_window=len(parsed),
        is_continuous=True,
    )


def _classify_intermittent(
    drug: str,
    rows: list[dict],
    reference_dttm: datetime,
    renal_status: RenalStatus = "unknown",
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
) -> MedStateRecord:
    """Classify an intermittent (PRN/scheduled) drug at reference_dttm.

    Drug-aware: resolves expected dosing interval from the drug+route+renal
    lookup table, falling back to observed-spacing estimation, then to
    DEFAULT_INTERVAL_HOURS. ACTIVE iff hours_since < interval × 1.25;
    RECENTLY_STOPPED iff interval × 1.25 ≤ hours_since < max(lookback_hours,
    2 × interval); HISTORICAL beyond. Strict < on both upper bounds.
    """
    parsed: list[tuple[datetime, Optional[float]]] = []
    route_seen: Optional[str] = None
    for row in rows:
        dttm = _parse_dttm(row.get("admin_dttm"))
        if dttm is None or dttm > reference_dttm:
            continue
        dose = _coerce_float(row.get("med_dose") or row.get("dose"))
        parsed.append((dttm, dose))
        if route_seen is None:
            route_seen = row.get("med_route_category") or row.get("route") or row.get("med_route")

    if not parsed:
        return MedStateRecord(
            drug_name=drug,
            state="HISTORICAL",
            last_admin_dttm=None,
            last_dose=None,
            admin_count_in_window=0,
            is_continuous=False,
        )

    parsed.sort(key=lambda x: x[0])
    last_dttm, last_dose = parsed[-1]
    hours_since = (reference_dttm - last_dttm).total_seconds() / 3600
    n = len(parsed)

    expected_interval = _resolve_expected_interval(
        drug_name=drug,
        route=route_seen,
        renal_status=renal_status,
        admin_dttms=[dttm for dttm, _ in parsed],
    )
    active_threshold = expected_interval * ACTIVE_INTERVAL_MULTIPLIER
    historical_threshold = max(float(lookback_hours), 2.0 * expected_interval)

    if hours_since < active_threshold:
        # ACTIVE — distinguish SCHEDULED (≥2 doses observed) from PRN/single.
        # The count distinction is most informative for non-tabled drugs;
        # for known antimicrobials, a single observed dose at q24h is still
        # a scheduled regimen, so route through ACTIVE_SCHEDULED when the
        # drug is in the lookup table.
        is_known_scheduled = _canonicalize_drug(drug) is not None
        state: MedState = (
            "ACTIVE_SCHEDULED" if (n >= 2 or is_known_scheduled) else "ACTIVE_PRN"
        )
        return MedStateRecord(
            drug_name=drug,
            state=state,
            last_admin_dttm=last_dttm,
            last_dose=last_dose,
            admin_count_in_window=n,
            is_continuous=False,
            expected_interval_hours=expected_interval,
            hours_since_last_admin=hours_since,
        )
    if hours_since < historical_threshold:
        return MedStateRecord(
            drug_name=drug,
            state="RECENTLY_STOPPED",
            last_admin_dttm=last_dttm,
            last_dose=last_dose,
            admin_count_in_window=n,
            is_continuous=False,
            expected_interval_hours=expected_interval,
            hours_since_last_admin=hours_since,
        )
    return MedStateRecord(
        drug_name=drug,
        state="HISTORICAL",
        last_admin_dttm=last_dttm,
        last_dose=last_dose,
        admin_count_in_window=n,
        is_continuous=False,
        expected_interval_hours=expected_interval,
        hours_since_last_admin=hours_since,
    )


def classify_med_states(
    meds_data: dict,
    reference_dttm: Any,
    renal_status: RenalStatus = "unknown",
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
) -> list[MedStateRecord]:
    """Classify every med in a CLIF meds_data dict by state at reference_dttm.

    Parameters
    ----------
    meds_data:
        ``{"continuous": [...], "intermittent": [...]}`` — the same shape
        emitted by ``serialize_to_json`` and consumed by qa.py.
    reference_dttm:
        ISO-8601 string or ``datetime`` — transfer-note time point.
    renal_status:
        Patient renal-function band, used to look up expected dosing
        intervals for intermittent meds. Pass ``"unknown"`` (default) when
        no creatinine is available — the lookup table picks the longest
        plausible interval, biasing toward ACTIVE/RECENTLY_STOPPED.
    lookback_hours:
        Window over which admin events were retrieved. Sets the floor for
        the RECENTLY_STOPPED/HISTORICAL boundary so the boundary tracks
        the data window rather than hardcoded 48h. Defaults to 48.

    Returns one record per unique drug name; if a drug appears in both
    continuous and intermittent (rare), each list is classified separately
    and both records are returned. Sorted by drug name.
    """
    ref_dt = _parse_dttm(reference_dttm)
    if ref_dt is None or not meds_data:
        return []

    records: list[MedStateRecord] = []

    # Group continuous rows by drug name
    cont_rows: dict[str, list[dict]] = {}
    for row in meds_data.get("continuous") or []:
        name = _drug_name(row)
        if not name:
            continue
        cont_rows.setdefault(name, []).append(row)
    for drug, rows in cont_rows.items():
        records.append(_classify_continuous(drug, rows, ref_dt))

    # Group intermittent rows by drug name
    intermittent_rows: dict[str, list[dict]] = {}
    for row in meds_data.get("intermittent") or []:
        name = _drug_name(row)
        if not name:
            continue
        intermittent_rows.setdefault(name, []).append(row)
    for drug, rows in intermittent_rows.items():
        records.append(
            _classify_intermittent(
                drug, rows, ref_dt,
                renal_status=renal_status,
                lookback_hours=lookback_hours,
            )
        )

    records.sort(key=lambda r: (r.drug_name.lower(), r.is_continuous))
    return records


def active_drug_names(records: Iterable[MedStateRecord]) -> list[str]:
    """Return drug names whose state is any ACTIVE variant."""
    seen: set[str] = set()
    out: list[str] = []
    for r in records:
        if r.is_active and r.drug_name not in seen:
            seen.add(r.drug_name)
            out.append(r.drug_name)
    return out


def state_summary(records: Iterable[MedStateRecord]) -> dict[str, list[str]]:
    """Group drug names by coarse state for narrative rendering.

    Returns a dict with three buckets:
        "active"            — ACTIVE / ACTIVE_SCHEDULED / ACTIVE_PRN
        "recently_stopped"  — RECENTLY_STOPPED (incl. trending-to-zero)
        "historical"        — HISTORICAL
    """
    out: dict[str, list[str]] = {
        "active": [],
        "recently_stopped": [],
        "historical": [],
    }
    for r in records:
        if r.is_active:
            out["active"].append(r.drug_name)
        elif r.state == "RECENTLY_STOPPED":
            out["recently_stopped"].append(r.drug_name)
        else:
            out["historical"].append(r.drug_name)
    return out
