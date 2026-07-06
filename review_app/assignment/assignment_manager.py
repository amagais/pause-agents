"""Case-to-reviewer assignment logic with IRR overlap support."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Optional

from storage.blob_client import blob_exists, read_json, write_json
from assignment.assignment_schema import (
    AssignmentManifest,
    BatchInfo,
    CaseAssignment,
    Reviewer,
)

MANIFEST_PATH = "assignments/manifest.json"


# ---------------------------------------------------------------------------
# Load / save manifest
# ---------------------------------------------------------------------------

def load_manifest() -> AssignmentManifest | None:
    if not blob_exists(MANIFEST_PATH):
        return None
    data = read_json(MANIFEST_PATH)
    return AssignmentManifest.model_validate(data)


def save_manifest(manifest: AssignmentManifest) -> None:
    write_json(MANIFEST_PATH, manifest.model_dump())


# ---------------------------------------------------------------------------
# Generate assignment
# ---------------------------------------------------------------------------

def generate_manifest(
    reviewers: list[dict],
    case_ids: list[str],
    pilot_case_ids: list[str],
    targeted_assignments: list[dict] | None = None,
    irr_count: int = 5,
    seed: int = 42,
) -> AssignmentManifest:
    """
    Build an AssignmentManifest.

    Rules:
      - pilot_case_ids: assigned to ALL reviewers (iterative phase)
      - targeted_assignments: list of {"hosp_id": ..., "reviewer_ids": [...]} (targeted phase)
      - irr_count cases from non-pilot pool: assigned to reviewer pairs
      - remaining cases: round-robin, one reviewer each (final phase)

    Args:
        reviewers:            list of {"reviewer_id": ..., "display_name": ..., "email": ...}
        case_ids:             all non-pilot case IDs (final phase pool)
        pilot_case_ids:       case IDs assigned to every reviewer (iterative)
        targeted_assignments: explicit hosp_id → reviewer_id(s) mappings (targeted phase)
        irr_count:            number of IRR overlap cases (from case_ids pool)
        seed:                 random seed for reproducibility
    """
    rng = random.Random(seed)
    shuffled = list(case_ids)
    rng.shuffle(shuffled)

    reviewer_objs = [Reviewer(**r) for r in reviewers]
    n = len(reviewer_objs)
    assignments: list[CaseAssignment] = []

    # Pilot cases → all reviewers
    for cid in pilot_case_ids:
        assignments.append(CaseAssignment(
            hosp_id=cid,
            assigned_to=[r.reviewer_id for r in reviewer_objs],
            is_irr_case=True,
            phase="iterative",
        ))

    # Targeted cases → specific reviewer(s)
    for t in (targeted_assignments or []):
        assignments.append(CaseAssignment(
            hosp_id=t["hosp_id"],
            assigned_to=t["reviewer_ids"],
            is_irr_case=False,
            phase="targeted",
        ))

    # IRR cases → reviewer pairs
    irr_cases = shuffled[:irr_count]
    remaining = shuffled[irr_count:]
    pairs = _make_reviewer_pairs(reviewer_objs)
    for i, cid in enumerate(irr_cases):
        pair = pairs[i % len(pairs)]
        assignments.append(CaseAssignment(
            hosp_id=cid,
            assigned_to=pair,
            is_irr_case=True,
            phase="final",
        ))

    # Unique cases → round-robin
    for i, cid in enumerate(remaining):
        reviewer_id = reviewer_objs[i % n].reviewer_id
        assignments.append(CaseAssignment(
            hosp_id=cid,
            assigned_to=[reviewer_id],
            is_irr_case=False,
            phase="final",
        ))

    return AssignmentManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        random_seed=seed,
        reviewers=reviewer_objs,
        assignments=assignments,
    )


def _make_reviewer_pairs(reviewers: list[Reviewer]) -> list[list[str]]:
    """Return adjacent reviewer pairs: (r0,r1), (r2,r3), (r4,r5), ..."""
    pairs = []
    ids = [r.reviewer_id for r in reviewers]
    for i in range(0, len(ids) - 1, 2):
        pairs.append([ids[i], ids[i + 1]])
    if len(ids) % 2 == 1:
        pairs.append([ids[-1], ids[0]])
    return pairs


# ---------------------------------------------------------------------------
# Progress summary (for admin page)
# ---------------------------------------------------------------------------

def completion_summary(
    manifest: AssignmentManifest,
    response_exists_fn,
    batch: Optional[int] = None,
) -> list[dict]:
    """
    Return per-case completion info.

    Args:
        manifest: loaded manifest
        response_exists_fn: callable(reviewer_id, hosp_id) → bool
        batch: if provided, only include assignments in this batch

    Returns:
        list of {"hosp_id", "batch", "phase", "is_irr", "assigned_to", "completed_by"}
    """
    rows = []
    for a in manifest.assignments:
        if batch is not None and a.batch != batch:
            continue
        completed = [rid for rid in a.assigned_to if response_exists_fn(rid, a.hosp_id)]
        rows.append({
            "hosp_id": a.hosp_id,
            "batch": a.batch,
            "phase": a.phase,
            "is_irr": a.is_irr_case,
            "assigned_to": ", ".join(a.assigned_to),
            "n_assigned": len(a.assigned_to),
            "n_completed": len(completed),
            "completed_by": ", ".join(completed),
        })
    return rows


# ---------------------------------------------------------------------------
# Pilot batch workflow
# ---------------------------------------------------------------------------

def bootstrap_manifest(
    reviewers: list[dict],
    seed: int = 42,
) -> AssignmentManifest:
    """Create an empty pilot manifest with reviewer roster, no assignments, no active batch."""
    reviewer_objs = [Reviewer(**r) for r in reviewers]
    return AssignmentManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        random_seed=seed,
        reviewers=reviewer_objs,
        assignments=[],
        batches=[],
        active_batch=0,
    )


def add_batch(
    manifest: AssignmentManifest,
    batch_number: int,
    hosp_ids: list[str],
    pipeline_version: str,
    label: str = "",
    date_window_start: Optional[str] = None,
    date_window_end: Optional[str] = None,
    irr_overlap_count: int = 0,
    seed_offset: int = 0,
) -> AssignmentManifest:
    """
    Append a batch of assignments to an existing manifest.

    Each hosp_id in the batch is assigned to ALL reviewers (every clinician annotates
    every note in the batch — matches the "Pool of 6 annotates" pilot design).
    If irr_overlap_count > 0, that many cases are flagged is_irr_case=True for analysis.

    Does NOT change manifest.active_batch. Use set_active_batch() to open the batch
    to reviewers (two-step intent so a batch can be staged before being opened).

    Raises ValueError if batch_number already exists in manifest.batches or if any
    hosp_id is already assigned in the manifest.
    """
    if manifest.batch_info(batch_number) is not None:
        raise ValueError(f"Batch {batch_number} already exists in manifest")

    existing_ids = {a.hosp_id for a in manifest.assignments}
    duplicates = [h for h in hosp_ids if h in existing_ids]
    if duplicates:
        raise ValueError(
            f"hosp_id(s) already assigned in earlier batch: {', '.join(duplicates)}"
        )

    rng = random.Random(manifest.random_seed + batch_number + seed_offset)
    shuffled = list(hosp_ids)
    rng.shuffle(shuffled)

    all_reviewer_ids = [r.reviewer_id for r in manifest.reviewers]
    irr_set = set(shuffled[:irr_overlap_count]) if irr_overlap_count > 0 else set()

    for cid in shuffled:
        manifest.assignments.append(CaseAssignment(
            hosp_id=cid,
            assigned_to=list(all_reviewer_ids),
            is_irr_case=cid in irr_set,
            phase="pilot_batch",
            batch=batch_number,
        ))

    manifest.batches.append(BatchInfo(
        batch_number=batch_number,
        pipeline_version=pipeline_version,
        label=label,
        date_window_start=date_window_start,
        date_window_end=date_window_end,
        note_count=len(hosp_ids),
        created_at=datetime.now(timezone.utc).isoformat(),
    ))
    manifest.batches.sort(key=lambda b: b.batch_number)
    return manifest


def set_active_batch(
    manifest: AssignmentManifest,
    batch_number: int,
) -> AssignmentManifest:
    """Set the active batch. Must already exist via add_batch(). Pass 0 to close all batches."""
    if batch_number != 0 and manifest.batch_info(batch_number) is None:
        raise ValueError(f"Batch {batch_number} does not exist in manifest")
    manifest.active_batch = batch_number
    return manifest


def archive_batch(
    manifest: AssignmentManifest,
    batch_number: int,
) -> AssignmentManifest:
    """
    Move a batch and its assignments out of the active manifest into the
    archived lists. Frees up ``batch_number`` for reuse by ``add_batch``.

    Reviewer responses (``responses/{reviewer_id}/{hosp_id}.json`` on blob)
    are NOT touched — the audit trail of what reviewers submitted on the
    archived cases stays intact.

    If the archived batch was the active one, ``active_batch`` is reset
    to 0 (no batch open).

    Raises ValueError if the batch number isn't in the active list.
    """
    bi = manifest.batch_info(batch_number)
    if bi is None:
        raise ValueError(f"Batch {batch_number} not found in active batches")

    bi.archived_at = datetime.now(timezone.utc).isoformat()

    moved_assignments = [a for a in manifest.assignments if a.batch == batch_number]
    manifest.assignments = [a for a in manifest.assignments if a.batch != batch_number]
    manifest.archived_assignments.extend(moved_assignments)

    manifest.batches = [b for b in manifest.batches if b.batch_number != batch_number]
    manifest.archived_batches.append(bi)
    manifest.archived_batches.sort(key=lambda b: (b.batch_number, b.archived_at or ""))

    if manifest.active_batch == batch_number:
        manifest.active_batch = 0

    return manifest
