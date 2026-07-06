"""Deterministic lab trend / rate-of-change detector.

Analyses time-series lab values for clinically significant trends such as
acute kidney injury (rising creatinine), HIT screening (falling platelets),
and active bleeding (falling hemoglobin).  Designed to run as a deterministic
(no-LLM) step in the ICU-PAUSE pipeline, complementing the static
``lab_ranges`` checker.

Each rule encodes an evidence-based threshold, time window, and clinical
significance string so that flagged trends are immediately actionable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LabTrendRule(BaseModel):
    """A single trend detection rule."""

    lab_name: str
    direction: str  # "rise", "fall", "change" (absolute either direction)
    mode: str  # "absolute" (delta), "relative" (ratio from baseline/peak)
    threshold: float  # absolute delta or multiplier (e.g. 1.5 = 50% rise)
    window_hours: int
    severity: str  # "warning" or "critical"
    clinical_significance: str


class LabTrendFlag(BaseModel):
    """A flagged lab trend."""

    lab_name: str
    rule_name: str
    direction: str
    severity: str
    values: list[float]
    timestamps: list[str]
    delta: float  # absolute change or ratio
    time_hours: float  # elapsed hours between first and last value used
    clinical_significance: str


class LabTrendResult(BaseModel):
    """Aggregated result of a lab trend check for one patient."""

    flags: list[LabTrendFlag]
    labs_analyzed: int
    labs_with_insufficient_data: list[str]


# ---------------------------------------------------------------------------
# Clinical trend rules
# ---------------------------------------------------------------------------

LAB_TREND_RULES: list[LabTrendRule] = [
    # --- Renal ---
    LabTrendRule(
        lab_name="creatinine",
        direction="rise",
        mode="absolute",
        threshold=0.3,
        window_hours=48,
        severity="critical",
        clinical_significance="KDIGO AKI Stage 1: Cr rise >= 0.3 mg/dL in 48h",
    ),
    LabTrendRule(
        lab_name="creatinine",
        direction="rise",
        mode="relative",
        threshold=1.5,
        window_hours=168,  # 7 days
        severity="warning",
        clinical_significance="KDIGO AKI Stage 1 (alt): Cr >= 1.5x baseline in 7d",
    ),
    # --- Hematology ---
    LabTrendRule(
        lab_name="platelets",
        direction="fall",
        mode="relative",
        threshold=0.5,  # drop to 50% of peak = 50% decline
        window_hours=72,
        severity="critical",
        clinical_significance="Platelets dropped >= 50% from peak in 72h — consider HIT screening",
    ),
    LabTrendRule(
        lab_name="hemoglobin",
        direction="fall",
        mode="absolute",
        threshold=2.0,
        window_hours=24,
        severity="critical",
        clinical_significance="Hgb drop >= 2.0 g/dL in 24h — evaluate for active bleeding",
    ),
    # --- Perfusion ---
    LabTrendRule(
        lab_name="lactate",
        direction="rise",
        mode="absolute",
        threshold=1.0,
        window_hours=24,
        severity="critical",
        clinical_significance="Lactate rise >= 1.0 mmol/L in 24h — worsening perfusion / sepsis",
    ),
    # --- Electrolytes ---
    LabTrendRule(
        lab_name="potassium",
        direction="rise",
        mode="absolute",
        threshold=1.0,
        window_hours=12,
        severity="critical",
        clinical_significance="K+ rise >= 1.0 mEq/L in 12h — rapid hyperkalemia",
    ),
    LabTrendRule(
        lab_name="sodium",
        direction="change",
        mode="absolute",
        threshold=8.0,
        window_hours=24,
        severity="critical",
        clinical_significance="Na+ change >= 8 mEq/L in 24h — osmotic demyelination risk",
    ),
    # --- Coagulation ---
    LabTrendRule(
        lab_name="inr",
        direction="rise",
        mode="absolute",
        threshold=1.0,
        window_hours=24,
        severity="warning",
        clinical_significance="INR rise >= 1.0 in 24h — coagulopathy worsening",
    ),
    # --- Hepatic ---
    LabTrendRule(
        lab_name="bilirubin_total",
        direction="rise",
        mode="relative",
        threshold=2.0,  # doubling
        window_hours=48,
        severity="warning",
        clinical_significance="Bilirubin doubled in 48h — acute hepatic injury",
    ),
    # --- Infectious ---
    LabTrendRule(
        lab_name="wbc",
        direction="rise",
        mode="absolute",
        threshold=10.0,
        window_hours=24,
        severity="warning",
        clinical_significance="WBC rise >= 10 K/uL in 24h — acute infection / sepsis",
    ),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_dttm(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 datetime string, returning None on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _group_and_sort_labs(
    labs_data: list[dict],
) -> dict[str, list[tuple[datetime, float]]]:
    """Group labs by category and sort by timestamp ascending.

    Returns a dict mapping lowercase lab category to a list of
    (datetime, value) tuples sorted oldest-first.
    """
    groups: dict[str, list[tuple[datetime, float]]] = {}
    for row in labs_data:
        cat = str(row.get("lab_category", "")).strip().lower()
        if not cat:
            continue
        dttm = _parse_dttm(row.get("lab_result_dttm"))
        if dttm is None:
            continue
        val = row.get("lab_value_numeric")
        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        groups.setdefault(cat, []).append((dttm, val))

    # Sort each group by timestamp ascending
    for cat in groups:
        groups[cat].sort(key=lambda x: x[0])

    return groups


def _check_absolute_rise(
    values: list[tuple[datetime, float]],
    threshold: float,
    window: timedelta,
    ref_dt: datetime,
) -> Optional[tuple[float, float, list[float], list[str]]]:
    """Check for an absolute rise >= threshold within the window.

    Looks at the minimum value in the window and compares to the latest value.
    Returns (delta, time_hours, values_used, timestamps_used) or None.
    """
    cutoff = ref_dt - window
    in_window = [(dt, v) for dt, v in values if dt >= cutoff]
    if len(in_window) < 2:
        return None

    min_val = min(in_window, key=lambda x: x[1])
    latest = in_window[-1]

    # Only count rises that go from earlier→later
    if min_val[0] >= latest[0]:
        return None

    delta = latest[1] - min_val[1]
    if delta >= threshold:
        hours = (latest[0] - min_val[0]).total_seconds() / 3600
        return (
            round(delta, 2),
            round(hours, 1),
            [min_val[1], latest[1]],
            [min_val[0].isoformat(), latest[0].isoformat()],
        )
    return None


def _check_absolute_fall(
    values: list[tuple[datetime, float]],
    threshold: float,
    window: timedelta,
    ref_dt: datetime,
) -> Optional[tuple[float, float, list[float], list[str]]]:
    """Check for an absolute fall >= threshold within the window."""
    cutoff = ref_dt - window
    in_window = [(dt, v) for dt, v in values if dt >= cutoff]
    if len(in_window) < 2:
        return None

    max_val = max(in_window, key=lambda x: x[1])
    latest = in_window[-1]

    if max_val[0] >= latest[0]:
        return None

    delta = max_val[1] - latest[1]
    if delta >= threshold:
        hours = (latest[0] - max_val[0]).total_seconds() / 3600
        return (
            round(delta, 2),
            round(hours, 1),
            [max_val[1], latest[1]],
            [max_val[0].isoformat(), latest[0].isoformat()],
        )
    return None


def _check_absolute_change(
    values: list[tuple[datetime, float]],
    threshold: float,
    window: timedelta,
    ref_dt: datetime,
) -> Optional[tuple[float, float, list[float], list[str]]]:
    """Check for an absolute change (either direction) >= threshold."""
    result_rise = _check_absolute_rise(values, threshold, window, ref_dt)
    result_fall = _check_absolute_fall(values, threshold, window, ref_dt)

    if result_rise and result_fall:
        # Return the larger change
        return result_rise if result_rise[0] >= result_fall[0] else result_fall
    return result_rise or result_fall


def _check_relative_rise(
    values: list[tuple[datetime, float]],
    threshold: float,
    window: timedelta,
    ref_dt: datetime,
) -> Optional[tuple[float, float, list[float], list[str]]]:
    """Check for a relative rise (latest / baseline >= threshold).

    Uses the minimum value in the window as baseline.
    """
    cutoff = ref_dt - window
    in_window = [(dt, v) for dt, v in values if dt >= cutoff]
    if len(in_window) < 2:
        return None

    min_val = min(in_window, key=lambda x: x[1])
    latest = in_window[-1]

    if min_val[0] >= latest[0]:
        return None
    if min_val[1] <= 0:
        return None  # Avoid division by zero

    ratio = latest[1] / min_val[1]
    if ratio >= threshold:
        hours = (latest[0] - min_val[0]).total_seconds() / 3600
        return (
            round(ratio, 2),
            round(hours, 1),
            [min_val[1], latest[1]],
            [min_val[0].isoformat(), latest[0].isoformat()],
        )
    return None


def _check_relative_fall(
    values: list[tuple[datetime, float]],
    threshold: float,
    window: timedelta,
    ref_dt: datetime,
) -> Optional[tuple[float, float, list[float], list[str]]]:
    """Check for a relative fall (latest / peak <= threshold).

    Uses the maximum value in the window as peak.  threshold=0.5 means
    the latest value is <= 50% of the peak (i.e. a 50% drop).
    """
    cutoff = ref_dt - window
    in_window = [(dt, v) for dt, v in values if dt >= cutoff]
    if len(in_window) < 2:
        return None

    max_val = max(in_window, key=lambda x: x[1])
    latest = in_window[-1]

    if max_val[0] >= latest[0]:
        return None
    if max_val[1] <= 0:
        return None

    ratio = latest[1] / max_val[1]
    if ratio <= threshold:
        hours = (latest[0] - max_val[0]).total_seconds() / 3600
        pct_drop = round((1 - ratio) * 100, 1)
        return (
            pct_drop,
            round(hours, 1),
            [max_val[1], latest[1]],
            [max_val[0].isoformat(), latest[0].isoformat()],
        )
    return None


def _evaluate_rule(
    rule: LabTrendRule,
    values: list[tuple[datetime, float]],
    ref_dt: datetime,
) -> Optional[LabTrendFlag]:
    """Evaluate a single trend rule against a sorted list of (datetime, value)."""
    window = timedelta(hours=rule.window_hours)

    result = None
    if rule.mode == "absolute":
        if rule.direction == "rise":
            result = _check_absolute_rise(values, rule.threshold, window, ref_dt)
        elif rule.direction == "fall":
            result = _check_absolute_fall(values, rule.threshold, window, ref_dt)
        elif rule.direction == "change":
            result = _check_absolute_change(values, rule.threshold, window, ref_dt)
    elif rule.mode == "relative":
        if rule.direction == "rise":
            result = _check_relative_rise(values, rule.threshold, window, ref_dt)
        elif rule.direction == "fall":
            result = _check_relative_fall(values, rule.threshold, window, ref_dt)

    if result is None:
        return None

    delta, time_hours, vals, timestamps = result
    rule_name = f"{rule.lab_name}_{rule.direction}_{rule.mode}"

    return LabTrendFlag(
        lab_name=rule.lab_name,
        rule_name=rule_name,
        direction=rule.direction,
        severity=rule.severity,
        values=vals,
        timestamps=timestamps,
        delta=delta,
        time_hours=time_hours,
        clinical_significance=rule.clinical_significance,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_lab_trends(
    labs_data: list[dict],
    reference_dttm: Optional[str] = None,
) -> LabTrendResult:
    """Run deterministic lab trend detection on CLIF lab data.

    1. Group labs by ``lab_category``, sort by timestamp ascending.
    2. For each trend rule, check if the threshold is exceeded within
       the rule's time window.
    3. Return structured results (never raises).

    Parameters
    ----------
    labs_data:
        ``patient_data["labs"]`` — list of dicts with keys
        ``lab_category``, ``lab_result_dttm``, ``lab_value_numeric``.
    reference_dttm:
        ISO-8601 string — the time anchor for window calculations.
        If ``None``, uses the latest timestamp in the data.
    """
    if not labs_data:
        return LabTrendResult(
            flags=[],
            labs_analyzed=0,
            labs_with_insufficient_data=[],
        )

    try:
        grouped = _group_and_sort_labs(labs_data)
    except Exception as exc:
        logger.warning("Lab trend check: failed to group labs: %s", exc)
        return LabTrendResult(
            flags=[],
            labs_analyzed=0,
            labs_with_insufficient_data=[],
        )

    # Determine reference time
    ref_dt: Optional[datetime] = _parse_dttm(reference_dttm)
    if ref_dt is None:
        # Fall back to the latest timestamp across all labs
        all_times = [dt for vals in grouped.values() for dt, _ in vals]
        ref_dt = max(all_times) if all_times else None
    if ref_dt is None:
        return LabTrendResult(
            flags=[],
            labs_analyzed=0,
            labs_with_insufficient_data=[],
        )

    # Track which labs have too few data points for any rule
    insufficient: set[str] = set()
    relevant_labs = {r.lab_name for r in LAB_TREND_RULES}

    for lab_name in relevant_labs:
        vals = grouped.get(lab_name, [])
        if len(vals) < 2:
            insufficient.add(lab_name)

    # Evaluate each rule
    flags: list[LabTrendFlag] = []
    for rule in LAB_TREND_RULES:
        vals = grouped.get(rule.lab_name)
        if not vals or len(vals) < 2:
            continue
        flag = _evaluate_rule(rule, vals, ref_dt)
        if flag is not None:
            flags.append(flag)

    logger.info(
        "Lab trend check: %d labs analyzed, %d flags, %d insufficient data",
        len(grouped),
        len(flags),
        len(insufficient),
    )

    return LabTrendResult(
        flags=flags,
        labs_analyzed=len(grouped),
        labs_with_insufficient_data=sorted(insufficient),
    )
