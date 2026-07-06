"""Surgically replace reviewer_id(s) inside targeted-phase assignments.

Use this when targeted cases were assigned to mistyped reviewer_ids
(e.g. zero-padded ``r02`` instead of the roster's ``r2``). It edits the
existing manifest in place — it does NOT regenerate via the admin form, so:

  * password_hash / password_must_change on every reviewer are preserved
  * reviewer responses (responses/{reviewer_id}/{hosp_id}.json) are untouched
  * pilot / final / IRR assignments are NOT reshuffled

Each OLD:NEW pair replaces OLD with NEW everywhere it appears in a
targeted assignment's assigned_to list. Every NEW must exist in the
reviewer roster. All pairs are applied in a single manifest write.

Dry-run by default. Pass --apply to write the change back to blob.

Usage:
    .venv/bin/python review_app/scripts/fix_targeted_reviewer_id.py r02:r2
    .venv/bin/python review_app/scripts/fix_targeted_reviewer_id.py r02:r2 r04:r4 r05:r5 r06:r6 r07:r7
    .venv/bin/python review_app/scripts/fix_targeted_reviewer_id.py r02:r2 r04:r4 --apply

Legacy two-argument form (single pair) is still accepted:
    .venv/bin/python review_app/scripts/fix_targeted_reviewer_id.py r02 r2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add review_app/ to path so submodules import cleanly.
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "review_app"))

try:
    from dotenv import load_dotenv
    load_dotenv(_repo_root / "review_app" / ".env")
except Exception:
    pass

from assignment.assignment_manager import load_manifest, save_manifest  # noqa: E402


def _parse_pairs(tokens: list[str]) -> list[tuple[str, str]]:
    """Parse OLD:NEW tokens, or the legacy two-positional [OLD, NEW] form."""
    if len(tokens) == 2 and ":" not in tokens[0] and ":" not in tokens[1]:
        return [(tokens[0], tokens[1])]
    pairs = []
    for tok in tokens:
        if ":" not in tok:
            print(f"ERROR: '{tok}' is not OLD:NEW. Use e.g. r02:r2")
            sys.exit(1)
        old, new = tok.split(":", 1)
        old, new = old.strip(), new.strip()
        if not old or not new:
            print(f"ERROR: '{tok}' has an empty side.")
            sys.exit(1)
        pairs.append((old, new))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pairs", nargs="+",
        help="One or more OLD:NEW reviewer_id pairs (e.g. r02:r2 r04:r4). "
             "The legacy 'OLD NEW' two-argument form is also accepted.",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write the change to blob. Without this flag, runs as a dry run.",
    )
    args = parser.parse_args()

    pairs = _parse_pairs(args.pairs)

    manifest = load_manifest()
    if manifest is None:
        print("ERROR: no manifest found on blob.")
        sys.exit(1)

    roster_ids = {r.reviewer_id for r in manifest.reviewers}
    bad = [new for _, new in pairs if new not in roster_ids]
    if bad:
        print(
            f"ERROR: target id(s) not in roster {sorted(roster_ids)}: "
            f"{', '.join(sorted(set(bad)))}. Aborting (no changes made)."
        )
        sys.exit(1)

    mapping = dict(pairs)
    changed = []
    for a in manifest.assignments:
        if a.phase != "targeted":
            continue
        if any(rid in mapping for rid in a.assigned_to):
            before = list(a.assigned_to)
            a.assigned_to = [mapping.get(rid, rid) for rid in a.assigned_to]
            changed.append((a.hosp_id, before, list(a.assigned_to)))

    if not changed:
        olds = ", ".join(o for o, _ in pairs)
        print(
            f"No targeted assignments contain any of: {olds}. Nothing to do.\n"
            "(Check spelling, or that these are 'targeted'-phase assignments.)"
        )
        return

    print(f"Pairs: {', '.join(f'{o}->{n}' for o, n in pairs)}")
    print(f"Targeted assignments to update ({len(changed)}):\n")
    for hosp_id, before, after in changed:
        print(f"  {hosp_id}:  {before}  ->  {after}")
    print()

    if not args.apply:
        print("DRY RUN — no changes written. Re-run with --apply to save to blob.")
        return

    save_manifest(manifest)
    print(f"Saved. {len(changed)} targeted assignment(s) updated on blob.")
    print("Passwords, responses, and all other assignments were left untouched.")


if __name__ == "__main__":
    main()
