"""Pydantic models for case assignment manifest."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class Reviewer(BaseModel):
    reviewer_id: str          # e.g. "r01"
    display_name: str         # e.g. "Dr. Jane Smith"
    email: str = ""
    # bcrypt hash of the reviewer's password. Empty string means no
    # password set yet — login is denied until an admin runs
    # review_app/scripts/set_reviewer_password.py for that reviewer.
    # Never store plaintext passwords here.
    password_hash: str = ""
    # When True, the reviewer must change their password on next login
    # before reaching the dashboard. Set by the admin's --temp flow when
    # issuing a one-time temporary password; cleared automatically after
    # the reviewer sets a permanent password. This is the mechanism that
    # keeps admin-issued temps from being reused as permanent credentials.
    password_must_change: bool = False


class BatchInfo(BaseModel):
    batch_number: int                          # 1..5 in the pilot
    pipeline_version: str                      # e.g. "v2"
    label: str = ""                            # e.g. "Refined claim extractor"
    date_window_start: Optional[str] = None    # ISO date "2026-05-01"
    date_window_end: Optional[str] = None
    note_count: int = 0
    created_at: str
    archived_at: Optional[str] = None          # ISO timestamp set when archived


class CaseAssignment(BaseModel):
    hosp_id: str
    assigned_to: list[str]    # list of reviewer_ids
    is_irr_case: bool = False
    phase: Literal["iterative", "final", "targeted", "pilot_batch"] = "final"
    batch: int = 0            # 0 = legacy/unbatched (pre-pilot)


class AssignmentManifest(BaseModel):
    version: str = "2.0"
    created_at: str
    random_seed: int
    reviewers: list[Reviewer]
    assignments: list[CaseAssignment]
    batches: list[BatchInfo] = []
    active_batch: int = 0     # 0 = no batch open
    # Archived batches and their assignments. Moved here by archive_batch()
    # — out of the active lists so add_batch() can reuse the batch_number,
    # the dashboard skips them, but R2's responses + batch metadata are
    # preserved for audit. Defaults to [] for backward compatibility with
    # manifests written before this field existed.
    archived_batches: list[BatchInfo] = []
    archived_assignments: list[CaseAssignment] = []

    def cases_for_reviewer(
        self, reviewer_id: str, batch: Optional[int] = None
    ) -> list[CaseAssignment]:
        out = [a for a in self.assignments if reviewer_id in a.assigned_to]
        if batch is not None:
            out = [a for a in out if a.batch == batch]
        # Per-reviewer deterministic shuffle: blinds inter-rater anchor
        # cases by preventing them from clustering at the top of every
        # reviewer's queue (manifest insertion puts pilot-phase assignments
        # first; without this, all anchors would be positions 1..N for
        # every reviewer). Seeded by reviewer_id so the order is stable
        # across sessions for "Resume draft" UX, but distinct between
        # reviewers so anchor positions don't correlate cross-reviewer.
        import random
        random.Random(f"{reviewer_id}|case-order-v1").shuffle(out)
        return out

    def reviewer_ids(self) -> list[str]:
        return [r.reviewer_id for r in self.reviewers]

    def batch_info(self, batch_number: int) -> Optional[BatchInfo]:
        return next((b for b in self.batches if b.batch_number == batch_number), None)

    def assignment_for(self, hosp_id: str) -> Optional[CaseAssignment]:
        return next((a for a in self.assignments if a.hosp_id == hosp_id), None)
