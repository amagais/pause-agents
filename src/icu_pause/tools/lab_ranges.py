"""Deterministic lab reference range validator.

Compares the most recent value for each tracked lab against standard
reference ranges and flags critical values or agent mischaracterizations.
Designed to run as a deterministic (no-LLM) step in the ICU-PAUSE pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LabRange(BaseModel):
    """Reference range for a single lab test."""

    low: float
    high: float
    unit: str
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None


class LabRangeFlag(BaseModel):
    """A flagged lab value outside its reference range."""

    lab_name: str
    value: float
    unit: str
    status: str  # "low", "high", "critical_low", "critical_high"
    reference_range: str  # e.g. "3.5-5.0 mEq/L"
    agent_characterization: Optional[str] = None
    mismatch: bool = False
    # Optional patient-context reframing (PR 3). When ``reframed_text``
    # is set, downstream stringification (qa.py) emits this in place of
    # the default "[LAB_RANGE/...]" message. ``status`` is NEVER changed
    # by reframing — the canonical severity stays intact, and reframing
    # is text-only. ``context_applied`` lists which clinical-context
    # flags produced the reframing (e.g. ["has_esrd_dialysis"]) for
    # audit. ``reframed_tier`` is "CHRONIC" or "REVIEW" — controls the
    # rendered tag and whether qa.py appends "confirm consistent with
    # prior" to the line. None when ``reframed_text`` is None.
    reframed_text: Optional[str] = None
    reframed_tier: Optional[str] = None
    context_applied: list[str] = Field(default_factory=list)


class LabRangeResult(BaseModel):
    """Aggregated result of a lab reference range check for one patient."""

    flags: list[LabRangeFlag]
    labs_checked: int
    labs_missing: list[str]


# ---------------------------------------------------------------------------
# Reference ranges (adult, general ICU)
# ---------------------------------------------------------------------------

LAB_REFERENCE_RANGES: dict[str, LabRange] = {
    "creatinine": LabRange(low=0.6, high=1.2, unit="mg/dL", critical_high=4.0),
    "potassium": LabRange(low=3.5, high=5.0, unit="mEq/L", critical_low=2.5, critical_high=6.5),
    "sodium": LabRange(low=136, high=145, unit="mEq/L", critical_low=120, critical_high=160),
    "lactate": LabRange(low=0.5, high=2.0, unit="mmol/L", critical_high=4.0),
    "hemoglobin": LabRange(low=12.0, high=17.5, unit="g/dL", critical_low=7.0, critical_high=20.0),
    "platelets": LabRange(low=150, high=400, unit="K/uL", critical_low=50, critical_high=1000),
    "inr": LabRange(low=0.8, high=1.1, unit="", critical_high=4.0),
    "bilirubin_total": LabRange(low=0.1, high=1.2, unit="mg/dL", critical_high=10.0),
    "albumin": LabRange(low=3.5, high=5.5, unit="g/dL", critical_low=1.5),
    "wbc": LabRange(low=4.5, high=11.0, unit="K/uL", critical_low=1.0, critical_high=30.0),
    "bun": LabRange(low=7, high=20, unit="mg/dL", critical_high=100),
    "glucose": LabRange(low=70, high=100, unit="mg/dL", critical_low=40, critical_high=500),
    "magnesium": LabRange(low=1.7, high=2.2, unit="mg/dL", critical_low=1.0, critical_high=4.0),
    "phosphorus": LabRange(low=2.5, high=4.5, unit="mg/dL", critical_low=1.0, critical_high=8.0),
    "calcium_ionized": LabRange(low=4.6, high=5.3, unit="mg/dL", critical_low=3.0, critical_high=6.5),
    "ph_arterial": LabRange(low=7.35, high=7.45, unit="", critical_low=7.1, critical_high=7.6),
    "troponin": LabRange(low=0, high=0.04, unit="ng/mL", critical_high=2.0),
}

# ---------------------------------------------------------------------------
# Mismatch detection keywords
# ---------------------------------------------------------------------------

NORMAL_KEYWORDS = [
    "normal", "within normal limits", "wnl", "unremarkable", "stable",
    "appropriate",
]
HIGH_KEYWORDS = ["elevated", "high", "rising", "increased", "above normal"]
LOW_KEYWORDS = ["low", "decreased", "declining", "below normal", "subtherapeutic"]


def _classify_value(value: float, ref: LabRange) -> Optional[str]:
    """Classify a lab value against its reference range.

    Returns None if normal, or one of "critical_low", "low", "high", "critical_high".
    """
    if ref.critical_low is not None and value < ref.critical_low:
        return "critical_low"
    if ref.critical_high is not None and value > ref.critical_high:
        return "critical_high"
    if value < ref.low:
        return "low"
    if value > ref.high:
        return "high"
    return None


def _find_characterization(
    lab_name: str, agent_text: str, window: int = 50,
) -> Optional[str]:
    """Search agent text for a characterization of the given lab near its name.

    Returns the matched keyword phrase, or None if no characterization found.
    Uses a window of ``window`` characters after the lab name mention.
    """
    if not agent_text:
        return None
    text_lower = agent_text.lower()
    lab_lower = lab_name.lower().replace("_", " ")

    for match in re.finditer(re.escape(lab_lower), text_lower):
        snippet = text_lower[match.start() : match.end() + window]
        for kw in NORMAL_KEYWORDS:
            if kw in snippet:
                return kw
        for kw in HIGH_KEYWORDS:
            if kw in snippet:
                return kw
        for kw in LOW_KEYWORDS:
            if kw in snippet:
                return kw
    return None


def _is_mismatch(status: str, characterization: str) -> bool:
    """Check if the agent's characterization contradicts the actual status."""
    if characterization in NORMAL_KEYWORDS and status in (
        "low", "high", "critical_low", "critical_high",
    ):
        return True
    if characterization in HIGH_KEYWORDS and status in ("low", "critical_low"):
        return True
    if characterization in LOW_KEYWORDS and status in ("high", "critical_high"):
        return True
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_lab_ranges(
    labs_data: list[dict],
    agent_text: str = "",
    clinical_context: Any = None,
) -> LabRangeResult:
    """Run a deterministic lab reference range check on CLIF lab data.

    1. Group labs by ``lab_category``, take the most recent value per category.
    2. Classify each against ``LAB_REFERENCE_RANGES``.
    3. Optionally detect mischaracterizations in ``agent_text``.
    4. When ``clinical_context`` is supplied, apply patient-context
       reframing (e.g. "Cr 4.2 mg/dL — chronic elevation from ESRD on
       HD" instead of a naked "creatinine high" warning). See
       ``safety/reframing.py``.
    5. Return structured results (never raises).

    Parameters
    ----------
    labs_data:
        ``patient_data["labs"]`` — list of dicts with keys
        ``lab_category``, ``lab_result_dttm``, ``lab_value_numeric``.
    agent_text:
        Concatenated text from all agent outputs for mismatch detection.
    clinical_context:
        Optional ``PatientClinicalContext`` (typed ``Any`` to avoid a
        circular import). When omitted, behavior is byte-identical to
        pre-PR-3 (no reframing applied).
    """
    if not labs_data:
        return LabRangeResult(
            flags=[],
            labs_checked=0,
            labs_missing=list(LAB_REFERENCE_RANGES.keys()),
        )

    # Group by lab_category, keep most recent per category
    most_recent: dict[str, dict] = {}
    for row in labs_data:
        cat = str(row.get("lab_category", "")).strip().lower()
        if not cat:
            continue
        dttm = str(row.get("lab_result_dttm", ""))
        existing = most_recent.get(cat)
        if existing is None or dttm > str(existing.get("lab_result_dttm", "")):
            most_recent[cat] = row

    flags: list[LabRangeFlag] = []
    labs_checked = 0
    labs_missing: list[str] = []

    for lab_name, ref in LAB_REFERENCE_RANGES.items():
        row = most_recent.get(lab_name)
        if row is None:
            labs_missing.append(lab_name)
            continue

        value = row.get("lab_value_numeric")
        if value is None:
            labs_missing.append(lab_name)
            continue

        try:
            value = float(value)
        except (ValueError, TypeError):
            labs_missing.append(lab_name)
            continue

        labs_checked += 1
        status = _classify_value(value, ref)
        if status is None:
            continue

        ref_str = f"{ref.low}-{ref.high} {ref.unit}".strip()
        characterization = _find_characterization(lab_name, agent_text)
        mismatch = (
            _is_mismatch(status, characterization) if characterization else False
        )

        flag = LabRangeFlag(
            lab_name=lab_name,
            value=value,
            unit=ref.unit,
            status=status,
            reference_range=ref_str,
            agent_characterization=characterization,
            mismatch=mismatch,
        )

        # PR 3: optional patient-context reframing. status is intentionally
        # left untouched (severity floor); only reframed_text and
        # context_applied get populated.
        if clinical_context is not None:
            from icu_pause.safety.reframing import reframe_lab_warning

            reframing = reframe_lab_warning(
                lab_name=lab_name,
                value=value,
                unit=ref.unit,
                status=status,
                ctx=clinical_context,
            )
            if reframing is not None:
                flag.reframed_text = reframing.text
                flag.reframed_tier = reframing.tier
                flag.context_applied = reframing.context_applied

        flags.append(flag)

    return LabRangeResult(
        flags=flags,
        labs_checked=labs_checked,
        labs_missing=labs_missing,
    )
