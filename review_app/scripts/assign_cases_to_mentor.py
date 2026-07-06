"""Add (or overwrite) an assignment manifest entry for a single reviewer.

By default MERGES into any existing assignments/manifest.json in the
Azure container:
  - Adds the reviewer (dedup by reviewer_id; updates display_name/email if
    the id already exists).
  - For each hosp_id, if a CaseAssignment already exists, this reviewer_id
    is appended to its assigned_to list (dedup); is_irr_case is set to
    True when the resulting assigned_to list has 2+ reviewers. If no
    CaseAssignment exists yet, a new one is created.

Pass --overwrite to wipe the existing manifest and write only the
supplied reviewer + cases.

Usage:
    python review_app/scripts/assign_cases_to_mentor.py \\
        --reviewer-id r01 \\
        --display-name "Dr. Jane Smith" \\
        --email jane.smith@example.edu \\
        --hosp-ids <hospitalization_id> <hospitalization_id> <hospitalization_id> <hospitalization_id> <hospitalization_id>
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "review_app"))

from dotenv import load_dotenv
load_dotenv(_repo_root / "review_app" / ".env")

from assignment.assignment_manager import load_manifest, save_manifest
from assignment.assignment_schema import AssignmentManifest, CaseAssignment, Reviewer


def _upsert_reviewer(reviewers: list[Reviewer], new: Reviewer) -> list[Reviewer]:
    """Add reviewer or update existing record with same reviewer_id."""
    out = []
    replaced = False
    for r in reviewers:
        if r.reviewer_id == new.reviewer_id:
            out.append(new)
            replaced = True
        else:
            out.append(r)
    if not replaced:
        out.append(new)
    return out


def _upsert_assignments(
    existing: list[CaseAssignment],
    hosp_ids: list[str],
    reviewer_id: str,
    phase: str,
) -> list[CaseAssignment]:
    """For each hosp_id, ensure reviewer_id is in its assigned_to list."""
    by_id: dict[str, CaseAssignment] = {a.hosp_id: a for a in existing}
    for hid in hosp_ids:
        if hid in by_id:
            a = by_id[hid]
            if reviewer_id not in a.assigned_to:
                a.assigned_to.append(reviewer_id)
                a.is_irr_case = len(a.assigned_to) > 1
        else:
            by_id[hid] = CaseAssignment(
                hosp_id=hid,
                assigned_to=[reviewer_id],
                is_irr_case=False,
                phase=phase,
            )
    return list(by_id.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--email", default="")
    parser.add_argument("--hosp-ids", nargs="+", required=True)
    parser.add_argument(
        "--phase",
        default="iterative",
        choices=["iterative", "final", "targeted"],
        help="Phase label (default: iterative for early/pilot review)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Wipe the existing manifest and write only this reviewer's "
        "assignments. Default is to merge with existing.",
    )
    args = parser.parse_args()

    new_reviewer = Reviewer(
        reviewer_id=args.reviewer_id,
        display_name=args.display_name,
        email=args.email,
    )

    existing = None if args.overwrite else load_manifest()

    if existing is None:
        manifest = AssignmentManifest(
            created_at=datetime.now(timezone.utc).isoformat(),
            random_seed=0,
            reviewers=[new_reviewer],
            assignments=[
                CaseAssignment(
                    hosp_id=hid,
                    assigned_to=[args.reviewer_id],
                    is_irr_case=False,
                    phase=args.phase,
                )
                for hid in args.hosp_ids
            ],
        )
        mode = "wrote new manifest"
    else:
        existing.reviewers = _upsert_reviewer(existing.reviewers, new_reviewer)
        existing.assignments = _upsert_assignments(
            existing.assignments, args.hosp_ids, args.reviewer_id, args.phase,
        )
        manifest = existing
        mode = "merged into existing manifest"

    save_manifest(manifest)
    print(f"{mode}: reviewer={args.reviewer_id} ({args.display_name}); "
          f"{len(args.hosp_ids)} case(s) now reachable by this reviewer.")
    print(f"Manifest now has {len(manifest.reviewers)} reviewer(s) and "
          f"{len(manifest.assignments)} case assignment(s).")


if __name__ == "__main__":
    main()
