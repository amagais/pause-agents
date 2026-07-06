"""Pydantic models for the clinician review response."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class PDSQI9HumanScore(BaseModel):
    """Human-assigned PDSQI-9 scores. Mirrors PDSQI9Score in src/icu_pause/eval/pdsqi9.py."""

    cited: int = Field(ge=1, le=5)
    accurate: int = Field(ge=1, le=5)
    thorough: int = Field(ge=1, le=5)
    useful: int = Field(ge=1, le=5)
    organized: int = Field(ge=1, le=5)
    comprehensible: int = Field(ge=1, le=5)
    succinct: int = Field(ge=1, le=5)
    synthesized: int = Field(ge=1, le=5)
    stigmatizing: bool = False

    @property
    def total_score(self) -> float:
        vals = [self.cited, self.accurate, self.thorough, self.useful,
                self.organized, self.comprehensible, self.succinct, self.synthesized]
        return round(sum(vals) / len(vals), 3)


class HallucinationItem(BaseModel):
    """Reviewer verdict for a single extracted atomic claim.

    The legacy ``source_location`` field (vitals/labs/meds/notes/demographics
    /not_in_source dropdown) was removed 2026-05-08 — clinicians found it
    not informative enough to justify the click. Free-response ``brief_note``
    is now always available regardless of verdict (Streamlit's conditional
    re-render of widgets in a fixed-height container was unreliable for the
    incorrect-only gate). Most useful for "incorrect" verdicts; reviewers
    can leave it blank for verified/cannot_verify claims. Old responses on
    disk that still carry ``source_location`` load fine because Pydantic's
    default ``extra="ignore"`` silently drops the unknown field.
    """

    claim_id: str
    section: str
    claim_text: str
    verdict: Literal["verified", "cannot_verify", "incorrect"]
    brief_note: str = ""


class OmissionItem(BaseModel):
    """Whether something clinically important was missing from a given data domain."""

    domain: str   # e.g. "meds_continuous", "labs", "vitals", "respiratory", etc.
    domain_label: str  # human-readable label for display
    omitted: bool = False
    severity: Optional[Literal["pertinent", "potentially_pertinent"]] = None
    brief_note: str = ""

    @model_validator(mode="after")
    def _check_severity(self) -> "OmissionItem":
        if self.omitted and self.severity is None:
            raise ValueError("severity required when omitted=True")
        if not self.omitted:
            self.severity = None
            self.brief_note = ""
        return self


class ReviewResponse(BaseModel):
    """Complete review response for one reviewer-case pair."""

    review_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reviewer_id: str
    hosp_id: str
    phase: Literal["iterative", "final", "targeted", "pilot_batch"]
    batch: int = 0                 # pilot batch number; 0 = legacy/unbatched
    pipeline_version: str = ""     # snapshot of BatchInfo.pipeline_version at submit time
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    submitted_at: Optional[str] = None
    is_complete: bool = False
    # Set when this response was COPIED from another reviewer's submitted review
    # (e.g. r6's reviews prefilled into r1/r2 for read-only reference). Holds the
    # source reviewer_id ("r6"); None for a genuine first-party review. Drives the
    # "review by r6 (read-only)" labels in the dashboard + review page, and marks
    # these for exclusion from per-reviewer / inter-rater (IRR) analysis.
    prefilled_from: Optional[str] = None

    pdsqi9: Optional[PDSQI9HumanScore] = None
    hallucination_checks: list[HallucinationItem] = Field(default_factory=list)
    omission_checks: list[OmissionItem] = Field(default_factory=list)
    qa_issues_feedback: str = ""
    warnings_feedback: str = ""
    overall_comment: str = ""
    time_on_task_seconds: int = 0

    def mark_submitted(self) -> None:
        self.submitted_at = datetime.now(timezone.utc).isoformat()
        self.is_complete = True

    def to_blob_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_blob_dict(cls, data: dict[str, Any]) -> "ReviewResponse":
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Domain definitions for omission check
# ---------------------------------------------------------------------------

OMISSION_DOMAINS = [
    ("meds_continuous", "Continuous infusions (vasopressors, sedation, insulin drips)"),
    ("meds_intermittent", "Intermittent medications (antibiotics, anticoagulants, PRN)"),
    ("labs", "Recent lab results (last 24h)"),
    ("vitals", "Vital sign trends"),
    ("respiratory", "Respiratory / ventilator status"),
    ("microbiology", "Microbiology / culture results"),
    ("code_status", "Code status / goals of care"),
    ("consults", "Active consultants"),
    ("procedures", "Pending procedures or tests"),
]
