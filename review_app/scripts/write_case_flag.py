"""Publish a per-case UI flag blob (e.g. the crfix creatinine badge).

Writes ``flags/{flag_id}.json = {"label": ..., "hosp_ids": [...]}`` to Blob.
The dashboard reads it to badge those cases in the reviewer queue (see
``storage/case_flags.py``). Runs wherever ``review_app/.env`` + azure SDK exist.

Usage:
  python review_app/scripts/write_case_flag.py \\
      --flag-id crfix_creatinine --label Cr --ids-file crfix_ids.txt

  python review_app/scripts/write_case_flag.py \\
      --flag-id crfix_creatinine --label Cr --hosp-ids <hosp_id_1> <hosp_id_2> ...

  python review_app/scripts/write_case_flag.py --flag-id crfix_creatinine --clear
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REVIEW_APP = os.path.join(_REPO_ROOT, "review_app")
sys.path.insert(0, REVIEW_APP)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REVIEW_APP, ".env"))

from storage.case_flags import FLAGS_PREFIX, write_case_flag  # noqa: E402
from storage.blob_client import delete_blob  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flag-id", required=True, help="flag blob id, e.g. crfix_creatinine")
    ap.add_argument("--label", default="", help="short badge label shown in the queue, e.g. Cr")
    ap.add_argument("--ids-file", default=None, help="newline-delimited hosp_ids")
    ap.add_argument("--hosp-ids", nargs="+", default=None, help="explicit hosp_ids")
    ap.add_argument("--clear", action="store_true", help="delete the flag blob instead of writing")
    args = ap.parse_args()

    if args.clear:
        path = f"{FLAGS_PREFIX}{args.flag_id}.json"
        ok = delete_blob(path)
        print(f"{'deleted' if ok else 'not found'}: {path}")
        return

    ids: list[str] = list(args.hosp_ids or [])
    if args.ids_file:
        with open(args.ids_file) as f:
            ids += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    ids = sorted(set(ids))
    if not ids:
        ap.error("provide --ids-file or --hosp-ids (or --clear)")

    path = write_case_flag(args.flag_id, args.label, ids)
    print(f"wrote {path}  label={args.label!r}  {len(ids)} hosp_id(s)")


if __name__ == "__main__":
    main()
