"""Archive buggy cases out of the live ``cases/`` namespace.

Copies every blob under ``cases/{hosp_id}/`` to
``archive/{archive_label}/{hosp_id}/`` and then deletes the original.
Reviewer responses under ``responses/{reviewer_id}/{hosp_id}.json`` are
intentionally NOT touched — the audit trail of what reviewers submitted
on the buggy cases stays intact.

Safe by default — dry-run unless ``--yes`` is passed. Container is
scoped to ``$BLOB_CONTAINER_NAME`` (default: ``icupause-review``) using
``$AZURE_BLOB_CONNECTION_STRING``.

Usage (dry-run):
    python review_app/scripts/archive_buggy_cases.py \\
        --hosp-ids CASE_001 CASE_002 CASE_003 CASE_004 CASE_005 \\
        --label batch1_buggy

Actually move:
    python review_app/scripts/archive_buggy_cases.py \\
        --hosp-ids CASE_001 ... --label batch1_buggy --yes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "review_app"))

from dotenv import load_dotenv
load_dotenv(_repo_root / "review_app" / ".env")

from storage.blob_client import _container, list_blobs_prefix


def _wait_for_copy(dst_client, timeout_s: int = 30) -> str:
    """Block until a server-side copy finishes (or fails). Returns final status."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        props = dst_client.get_blob_properties()
        status = props.copy.status
        if status != "pending":
            return status
        time.sleep(0.5)
    return "timeout"


def archive_hosp_id(hosp_id: str, label: str, yes: bool) -> tuple[int, int, list[str]]:
    """Returns (copied, deleted, failures)."""
    container = _container()
    src_prefix = f"cases/{hosp_id}/"
    dst_prefix = f"archive/{label}/{hosp_id}/"

    src_names = list_blobs_prefix(src_prefix)
    if not src_names:
        print(f"  [{hosp_id}] no blobs under {src_prefix!r} — skipping.")
        return 0, 0, []

    print(f"  [{hosp_id}] {len(src_names)} blob(s) under {src_prefix!r} → {dst_prefix!r}")
    for n in src_names:
        print(f"    src: {n}")

    if not yes:
        return 0, 0, []

    failures: list[str] = []
    copied = 0

    # Phase 1: copy each src blob to its archive location.
    for src_name in src_names:
        suffix = src_name[len(src_prefix):]  # e.g. "output.json"
        dst_name = f"{dst_prefix}{suffix}"
        src_client = container.get_blob_client(src_name)
        dst_client = container.get_blob_client(dst_name)
        try:
            dst_client.start_copy_from_url(src_client.url)
            status = _wait_for_copy(dst_client)
            if status != "success":
                failures.append(f"{src_name} → {dst_name}: copy status={status}")
                continue
            copied += 1
        except Exception as e:
            failures.append(f"{src_name} → {dst_name}: {e}")

    if failures:
        print(f"  [{hosp_id}] copy failed for {len(failures)} blob(s); skipping delete.")
        return copied, 0, failures

    # Phase 2: delete originals only after every copy succeeded.
    deleted = 0
    for src_name in src_names:
        try:
            container.get_blob_client(src_name).delete_blob()
            deleted += 1
        except Exception as e:
            failures.append(f"delete {src_name}: {e}")

    return copied, deleted, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hosp-ids",
        nargs="+",
        default=None,
        help="Hospitalization IDs to archive (one or more).",
    )
    parser.add_argument(
        "--ids-file",
        default=None,
        help="Text file with one hosp_id per line (alternative to --hosp-ids).",
    )
    parser.add_argument(
        "--label",
        default="batch1_buggy",
        help="Archive sub-folder name (default: batch1_buggy). Goes under archive/.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually copy + delete. Without this flag, runs in dry-run mode.",
    )
    args = parser.parse_args()

    hosp_ids = list(args.hosp_ids or [])
    if args.ids_file:
        with open(args.ids_file) as f:
            hosp_ids += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not hosp_ids:
        parser.error("provide --hosp-ids or --ids-file")

    mode = "ARCHIVE" if args.yes else "DRY-RUN"
    print(f"[{mode}] archiving {len(hosp_ids)} case(s) to archive/{args.label}/")

    total_copied = 0
    total_deleted = 0
    all_failures: list[str] = []
    for hosp_id in hosp_ids:
        c, d, f = archive_hosp_id(hosp_id, args.label, args.yes)
        total_copied += c
        total_deleted += d
        all_failures.extend(f)

    print()
    if not args.yes:
        print(f"Dry-run complete. Re-run with --yes to actually archive.")
        return

    print(f"Copied {total_copied} blob(s); deleted {total_deleted} original(s).")
    if all_failures:
        print(f"Failures ({len(all_failures)}):")
        for msg in all_failures:
            print(f"  {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
