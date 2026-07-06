"""Specialty taxonomy used by the physician-note context floor.

The floor's job is to guarantee that progress-note-consuming domain agents
(Intensivist, Respiratory, Pharmacy, Dietitian) always see at least one
physician-authored note even when the 48-hour lookback window contains
only ancillary notes (Pharmacy, Nutrition Therapy, Respiratory Therapy,
etc.).

Design principle is **denylist, not allowlist**. The notes table at the
pilot site exposes 182+ unique specialty values plus a sizable null
bucket. An allowlist is unmaintainable and silently misses notes when
new specialties appear in the EHR's reference table. We instead enumerate
the small, stable set of non-physician specialties and treat everything
else as physician.

The Tier 1 / Tier 2 sets DO exist in this module — but they're used only
for *primary-team detection* (Step 2 of the floor algorithm: pick which
physician specialty leads the case when multiple are present). They are
not consulted by ``is_physician_note``; that predicate runs purely off
the deny set + a fallback for null specialty.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Denylist — specialties that never satisfy the physician-note floor.
# Curated from the full 182-specialty value-counts on the notes table.
# ---------------------------------------------------------------------------
NON_PHYSICIAN_SPECIALTIES: frozenset[str] = frozenset({
    # Allied health / pharmacy / nutrition
    "Pharmacy", "Nutrition Therapy", "Respiratory Therapy",
    "Physical Therapy", "Occupational Therapy",
    "Occupational/Physical Therapy", "Speech Pathology",
    "Diabetes Education", "Audiology", "Orthotics",
    "Athletic Training", "Athletic Training ",   # trailing space exists in source data
    # Social / behavioral (non-MD)
    "Social Work", "COUNSELOR", "Psychology", "Neuropsychology",
    "Sleep Psychologist",
    "Gastrointestinal Behavioral Health Psychology",
    "Cardiac Behavioral Medicine",
    # Nursing / technical
    "REGISTERED NURSE", "PERFUSIONIST", "Surgical Assistant",
    # Non-MD clinicians
    "Optometry", "Chiropractic Medicine", "Acupuncture",
    "Massage Therapy",
    # Coordination / home
    "Home Health", "Genetic Counseling",
    # Dentistry — DDS, not MD; excluded for ICU context purposes
    "Dentistry",
})

# ---------------------------------------------------------------------------
# Tier 1 — likely primary teams for an ICU stay.
# ---------------------------------------------------------------------------
TIER1_PHYSICIAN_SPECIALTIES: frozenset[str] = frozenset({
    "Hospital Medicine", "Internal Medicine", "Critical Care Medicine",
    "Neuro Critical Care", "Family Medicine",
    "Cardiac Surgery", "Neurological Surgery", "General Surgery",
    "Vascular Surgery", "Thoracic Surgery", "Urology",
    "Trauma Surgery", "Transplantation Surgery",
})

# ---------------------------------------------------------------------------
# Tier 2 — common consulting physician specialties.
# ---------------------------------------------------------------------------
TIER2_PHYSICIAN_SPECIALTIES: frozenset[str] = frozenset({
    "Pulmonology", "Cardiology", "Nephrology",
    "Hematology and Medical Oncology", "Hematology", "Medical Oncology",
    "Infectious Disease", "Neurology", "Endocrinology",
    "Gastroenterology", "Hepatology", "Transplant Hepatology",
    "Heart Failure and Heart Transplantation",
    "Interventional Cardiology", "Cardiac Electrophysiology",
    "Palliative Medicine", "Emergency Medicine", "Anesthesiology",
    "Physical Medicine and Rehab", "Interventional Pulmonology",
    "Vascular Neurology and Stroke", "Geriatric Medicine",
    "Rheumatology",
})


# Note types whose name alone implies physician authorship. Used as the
# fallback predicate when ``specialty`` is null/empty (the largest single
# bucket on the source table). See ``is_physician_note``.
_PHYSICIAN_NOTE_TYPES: frozenset[str] = frozenset({
    "hp_note", "consults_note", "progress_note",
})


def is_physician_note(specialty: Optional[str], note_type: Optional[str]) -> bool:
    """Predicate: does this note count as physician-authored for the floor?

    A note qualifies as physician-authored if it is **not** in the
    denylist AND either has a named specialty (any non-empty string outside
    the deny set is assumed physician) or — for the null/empty bucket —
    its ``note_type`` itself implies physician authorship.

    Edge cases worth flagging in the source data:

    - "Behavioral Health" (mixed psychiatrists + counselors): treated as
      physician — psychiatry is the more common author and the harm of
      false-positive inclusion is small.
    - "Unknown Physician Specialty": physician — handled by the default
      branch.
    - "Physician Assistant": mid-level, default-physician branch. Trivial
      volume.
    - Subspecialties not in Tier 1/2 (e.g., Hepatology, Rheumatology,
      Trauma Surgery) still satisfy this predicate via the default branch.
      They simply aren't candidates for primary-team *selection* — that
      logic is in ``ensure_physician_note_floor``.
    """
    if specialty in NON_PHYSICIAN_SPECIALTIES:
        return False
    if specialty is None or (isinstance(specialty, str) and specialty.strip() == ""):
        # Null bucket is the largest single bucket (~185k rows).
        # Accept only if the note type itself implies physician authorship.
        return note_type in _PHYSICIAN_NOTE_TYPES
    return True   # any other named specialty → assume physician


def primary_team_tier(specialty: Optional[str]) -> int:
    """Rank a specialty for primary-team selection.

    Returns 1 for Tier 1, 2 for Tier 2, 3 for any other named physician
    specialty, and 4 for null/non-physician (which should be filtered out
    upstream before this is called). Lower is better — used by the floor
    algorithm as a tie-break when two specialties share a count.
    """
    if specialty in TIER1_PHYSICIAN_SPECIALTIES:
        return 1
    if specialty in TIER2_PHYSICIAN_SPECIALTIES:
        return 2
    if specialty is None or specialty in NON_PHYSICIAN_SPECIALTIES:
        return 4
    return 3
