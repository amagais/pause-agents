"""Wipe blobs from the review-app Azure container.

Safe by default — runs in dry-run mode unless --yes is passed. Scoped
to the container named in $BLOB_CONTAINER_NAME (default: icupause-review)
using $AZURE_BLOB_CONNECTION_STRING — never touches other containers or
other storage accounts.

Usage (dry-run — lists what would be deleted):
    python review_app/scripts/wipe_blobs.py

Actually delete (after reviewing dry-run):
    python review_app/scripts/wipe_blobs.py --yes

Limit to a prefix:
    python review_app/scripts/wipe_blobs.py --prefix cases/
    python review_app/scripts/wipe_blobs.py --prefix cases/12345/ --yes
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "review_app"))

from dotenv import load_dotenv
load_dotenv(_repo_root / "review_app" / ".env")

from storage.blob_client import _container, list_blobs_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default="",
        help="Only delete blobs under this prefix (default: empty = whole container)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete. Without this flag, runs in dry-run mode.",
    )
    args = parser.parse_args()

    container_name = os.environ.get("BLOB_CONTAINER_NAME", "icupause-review")
    names = list_blobs_prefix(args.prefix)

    if not names:
        print(f"No blobs found in container={container_name!r} "
              f"under prefix={args.prefix!r}. Nothing to do.")
        return

    mode = "DELETE" if args.yes else "DRY-RUN"
    print(f"[{mode}] container={container_name!r} prefix={args.prefix!r} "
          f"matched {len(names)} blob(s):")
    for n in names:
        print(f"  {n}")

    if not args.yes:
        print()
        print(f"This was a dry-run. Re-run with --yes to actually delete "
              f"these {len(names)} blob(s).")
        return

    container = _container()
    deleted = 0
    failed: list[tuple[str, str]] = []
    for n in names:
        try:
            container.delete_blob(n)
            deleted += 1
        except Exception as e:
            failed.append((n, str(e)))

    print()
    print(f"Deleted {deleted}/{len(names)} blob(s).")
    if failed:
        print(f"Failures ({len(failed)}):")
        for n, err in failed:
            print(f"  {n}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
