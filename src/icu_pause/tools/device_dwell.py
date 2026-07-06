"""Deterministic device/line dwell-time calculator.

Scans CLIF ``patient_procedures`` data for device-insertion events, identifies
active devices (no recorded removal), and flags those exceeding clinical
threshold durations.  Designed to run as a deterministic (no-LLM) step in the
ICU-PAUSE pipeline.

Data window: full ICU stay (procedures are loaded without time-window filter).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DeviceThreshold(BaseModel):
    """Clinical dwell-time thresholds for a device type."""

    warn_days: int
    critical_days: int
    action: str


class DeviceDwellFlag(BaseModel):
    """A flagged device exceeding its dwell-time threshold."""

    device_type: str
    device_name: str
    insertion_dttm: str
    dwell_days: int
    threshold_days: int
    severity: str  # "warning" or "critical"
    recommended_action: str


class DeviceDwellResult(BaseModel):
    """Aggregated result of a device dwell-time check for one patient."""

    flags: list[DeviceDwellFlag]
    devices_checked: int
    devices_without_insertion_date: int


# ---------------------------------------------------------------------------
# Device recognition patterns (procedure_name / procedure_category substrings)
# ---------------------------------------------------------------------------

# Insertion keywords — presence of these (case-insensitive) in a procedure
# name/category indicates a device-insertion event for the given device type.
DEVICE_INSERTION_PATTERNS: dict[str, list[str]] = {
    "central_venous_catheter": [
        "central line", "central venous", "cvc insert", "picc insert",
        "picc line", "picc placement", "triple lumen", "central catheter",
        "subclavian line", "ij line", "femoral line",
    ],
    "foley_catheter": [
        "foley", "urinary catheter", "indwelling catheter",
        "bladder catheter",
    ],
    "arterial_line": [
        "arterial line", "a-line", "art line", "arterial catheter",
    ],
    "chest_tube": [
        "chest tube", "thoracostomy", "chest drain", "pigtail catheter",
    ],
    "endotracheal_tube": [
        "intubat", "endotracheal", "ett insert", "ett placement",
    ],
    "nasogastric_tube": [
        "ng tube", "nasogastric", "ogt", "dobhoff", "ogastric",
        "feeding tube",
    ],
}

# Keywords indicating a device was removed.
REMOVAL_KEYWORDS: list[str] = [
    "remov", "discontinu", "dc'd", "d/c'd", "pulled", "taken out",
    "extubat", "decannulat",
]

# Evidence-based dwell-time thresholds.
DEVICE_THRESHOLDS: dict[str, DeviceThreshold] = {
    "central_venous_catheter": DeviceThreshold(
        warn_days=5, critical_days=7,
        action="Assess for removal or replacement",
    ),
    "foley_catheter": DeviceThreshold(
        warn_days=2, critical_days=5,
        action="Assess for removal; trial void if appropriate",
    ),
    "arterial_line": DeviceThreshold(
        warn_days=5, critical_days=7,
        action="Assess ongoing need for invasive monitoring",
    ),
    "chest_tube": DeviceThreshold(
        warn_days=3, critical_days=5,
        action="Assess output trend and removal criteria",
    ),
    "endotracheal_tube": DeviceThreshold(
        warn_days=7, critical_days=10,
        action="Assess tracheostomy candidacy",
    ),
    "nasogastric_tube": DeviceThreshold(
        warn_days=3, critical_days=7,
        action="Assess for transition to oral or PEG",
    ),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_dttm(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 datetime string, returning None on failure."""
    if not value:
        return None
    try:
        # Handle both timezone-aware and naive strings
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _match_device_type(text: str) -> Optional[str]:
    """Return the device type if *text* matches any insertion pattern, else None."""
    text_lower = text.lower()
    for device_type, patterns in DEVICE_INSERTION_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                return device_type
    return None


def _is_removal(text: str) -> bool:
    """Return True if *text* contains a removal keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in REMOVAL_KEYWORDS)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_device_dwell(
    procedures_data: list[dict],
    reference_dttm: str,
    icu_admission_dttm: Optional[str] = None,
) -> DeviceDwellResult:
    """Run a deterministic device dwell-time check on CLIF procedure data.

    1. Scan procedures for device-insertion events via keyword matching.
    2. For each insertion, search for a later removal event of the same type.
    3. Compute dwell time for devices still active at ``reference_dttm``.
    4. Flag devices exceeding clinical thresholds.
    5. Return structured results (never raises).

    Parameters
    ----------
    procedures_data:
        ``patient_data["procedures"]`` — list of dicts with keys like
        ``procedure_name``, ``procedure_category``, ``procedure_dttm``.
    reference_dttm:
        ISO-8601 string — the time point from which dwell is measured.
    icu_admission_dttm:
        ISO-8601 string — ICU admission time (not currently used but
        reserved for future logic such as admission-implied devices).
    """
    ref_dt = _parse_dttm(reference_dttm)
    if not ref_dt or not procedures_data:
        return DeviceDwellResult(
            flags=[], devices_checked=0, devices_without_insertion_date=0,
        )

    # Step 1: identify insertion events
    # Each entry: (device_type, procedure_name, insertion_dttm)
    insertions: list[tuple[str, str, datetime]] = []
    no_date_count = 0

    for row in procedures_data:
        proc_name = str(row.get("procedure_name", row.get("procedure_category", "")))
        if not proc_name:
            continue

        device_type = _match_device_type(proc_name)
        if device_type is None:
            continue

        # Skip if this is actually a removal procedure
        if _is_removal(proc_name):
            continue

        dttm = _parse_dttm(
            row.get("procedure_dttm", row.get("performed_dttm"))
        )
        if dttm is None:
            no_date_count += 1
            continue

        insertions.append((device_type, proc_name, dttm))

    # Step 2: for each device type, find the latest insertion and check for removal
    # Group insertions by device type, keep the latest per type
    latest_insertion: dict[str, tuple[str, datetime]] = {}
    for device_type, proc_name, dttm in insertions:
        existing = latest_insertion.get(device_type)
        if existing is None or dttm > existing[1]:
            latest_insertion[device_type] = (proc_name, dttm)

    # Step 3: check for removal events after each insertion
    removal_times: dict[str, datetime] = {}
    for row in procedures_data:
        proc_name = str(row.get("procedure_name", row.get("procedure_category", "")))
        if not _is_removal(proc_name):
            continue
        device_type = _match_device_type(proc_name)
        if device_type is None:
            continue
        dttm = _parse_dttm(row.get("procedure_dttm", row.get("performed_dttm")))
        if dttm is not None:
            existing = removal_times.get(device_type)
            if existing is None or dttm > existing:
                removal_times[device_type] = dttm

    # Step 4: compute dwell time and flag
    flags: list[DeviceDwellFlag] = []
    for device_type, (proc_name, insertion_dt) in latest_insertion.items():
        # Check if device was removed after this insertion
        removal_dt = removal_times.get(device_type)
        if removal_dt is not None and removal_dt > insertion_dt:
            continue  # Device was removed

        threshold = DEVICE_THRESHOLDS.get(device_type)
        if threshold is None:
            continue

        dwell_days = (ref_dt - insertion_dt).days
        if dwell_days < 0:
            continue  # Insertion in the future relative to reference — data issue

        if dwell_days >= threshold.critical_days:
            severity = "critical"
            threshold_days = threshold.critical_days
        elif dwell_days >= threshold.warn_days:
            severity = "warning"
            threshold_days = threshold.warn_days
        else:
            continue  # Within safe range

        flags.append(
            DeviceDwellFlag(
                device_type=device_type,
                device_name=proc_name.strip(),
                insertion_dttm=insertion_dt.isoformat(),
                dwell_days=dwell_days,
                threshold_days=threshold_days,
                severity=severity,
                recommended_action=threshold.action,
            )
        )

    return DeviceDwellResult(
        flags=flags,
        devices_checked=len(latest_insertion),
        devices_without_insertion_date=no_date_count,
    )
