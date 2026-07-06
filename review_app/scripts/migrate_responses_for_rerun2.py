"""Migrate completed reviewer responses for the rerun2 (scribe-fix + PMH-fallback) re-review.

Runs LOCALLY on the Mac (needs review_app/.env + azure SDK; HPC can't import review_app).

Policy (decided 2026-06-08, pragmatic burden-min — see memory
project_icu_pause_rereview_carryover):
  * KEEP   : pdsqi9 (rubric) + qa_issues_feedback + warnings_feedback + overall_comment
  * CLEAR  : hallucination_checks (Step 1 — claim-anchored, old claim_ids no longer exist
             in the regenerated briefs) + omission_checks (Step 2)
  * REOPEN : is_complete -> False, submitted_at -> None  (case re-enters reviewer's queue
             as a pre-filled "Resume draft")
  * No anchor carve-out — applied uniformly to all responses.

Backup-first: every response is copied to BOTH
  * blob prefix  responses_v1_backup/{reviewer_id}/{hosp_id}.json
  * local dir    review_app/_response_backups_v1/{reviewer_id}/{hosp_id}.json
before anything is rewritten. --apply refuses to run unless both backups succeed.

Usage (from repo root, Mac venv):
  .venv/bin/python review_app/scripts/migrate_responses_for_rerun2.py          # dry-run (default)
  .venv/bin/python review_app/scripts/migrate_responses_for_rerun2.py --apply  # write changes
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # review_app/scripts
REVIEW_APP = os.path.dirname(HERE)                          # review_app
sys.path.insert(0, REVIEW_APP)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REVIEW_APP, ".env"))

from storage.blob_client import list_blobs_prefix, read_json, write_json  # noqa: E402

# Backup destinations — derived from --backup-label (default "v1" for the
# original rerun2 migration). A second migration (e.g. the crfix creatinine
# re-review) MUST pass a fresh label (e.g. "v2_crfix") so it snapshots the
# current responses into a NEW namespace instead of overwriting the v1 backups.
BACKUP_BLOB_PREFIX = "responses_v1_backup/"
LOCAL_BACKUP_DIR = os.path.join(REVIEW_APP, "_response_backups_v1")

# Fields preserved (carried over) — everything else on the dict is left as-is too;
# we only explicitly MUTATE the four below.
CLEARED = ("hallucination_checks", "omission_checks")


def _backup(resp: dict) -> bool:
    """Copy a response to the blob backup prefix AND a local file. Return True on success."""
    rid, hid = resp["reviewer_id"], resp["hosp_id"]
    ok = True
    # blob backup
    try:
        write_json(f"{BACKUP_BLOB_PREFIX}{rid}/{hid}.json", resp)
    except Exception as e:  # noqa: BLE001
        print(f"  !! blob backup FAILED for {rid}/{hid}: {e}")
        ok = False
    # local backup
    try:
        d = os.path.join(LOCAL_BACKUP_DIR, rid)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{hid}.json"), "w") as f:
            json.dump(resp, f, indent=2)
    except Exception as e:  # noqa: BLE001
        print(f"  !! local backup FAILED for {rid}/{hid}: {e}")
        ok = False
    return ok


def _transform(resp: dict) -> dict:
    """Return the migrated copy: clear Step 1/2, reopen, keep rubric + free-text."""
    out = dict(resp)
    out["hallucination_checks"] = []
    out["omission_checks"] = []
    out["is_complete"] = False
    out["submitted_at"] = None
    # time_on_task_seconds is intentionally PRESERVED: the resumed review
    # continues the timer from where round 1 left off (review_page accumulates
    # onto it). Round-1's standalone time is also kept in responses_v1_backup/,
    # so round-2-only time = live_total - backup_time if ever needed.
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Write changes. Without this, dry-run only.")
    ap.add_argument("--ids-file", default=None,
                    help="newline-delimited hosp_ids to scope the migration to "
                         "(e.g. the 28 crfix-affected). Omit = all responses.")
    ap.add_argument("--hosp-ids", nargs="+", default=None,
                    help="explicit hosp_ids to scope to (alternative to --ids-file).")
    ap.add_argument("--backup-label", default="v1",
                    help="backup namespace label (default v1). Use a fresh label "
                         "for a second migration, e.g. v2_crfix, so it does NOT "
                         "overwrite the v1 backups.")
    args = ap.parse_args()

    # Resolve the backup destinations from the label (default v1 = unchanged).
    global BACKUP_BLOB_PREFIX, LOCAL_BACKUP_DIR
    BACKUP_BLOB_PREFIX = f"responses_{args.backup_label}_backup/"
    LOCAL_BACKUP_DIR = os.path.join(REVIEW_APP, f"_response_backups_{args.backup_label}")

    only_ids: set[str] | None = None
    if args.ids_file:
        with open(args.ids_file) as f:
            only_ids = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
    if args.hosp_ids:
        only_ids = (only_ids or set()) | set(args.hosp_ids)
    if only_ids is not None:
        print(f"Scoped to {len(only_ids)} hosp_id(s).")

    blobs = [b for b in list_blobs_prefix("responses/") if b.endswith(".json")]
    print(f"Found {len(blobs)} response blobs under responses/")
    print(f"Backup namespace: {BACKUP_BLOB_PREFIX} + {LOCAL_BACKUP_DIR}\n")

    n_complete = n_drafts = n_changed = 0
    for path in sorted(blobs):
        try:
            resp = read_json(path)
        except Exception as e:  # noqa: BLE001
            print(f"SKIP {path}: unreadable ({e})")
            continue

        rid, hid = resp.get("reviewer_id", "?"), resp.get("hosp_id", "?")
        if only_ids is not None and hid not in only_ids:
            continue
        complete = bool(resp.get("is_complete"))
        n_complete += complete
        n_drafts += (not complete)
        n_halluc = len(resp.get("hallucination_checks", []) or [])
        n_omit = len(resp.get("omission_checks", []) or [])
        pdsqi = "yes" if resp.get("pdsqi9") else "no"
        ft = sum(len(resp.get(k, "") or "") for k in
                 ("qa_issues_feedback", "warnings_feedback", "overall_comment"))

        status = "COMPLETE" if complete else "draft"
        print(f"{rid:8} {hid:14} [{status:8}] "
              f"clear Step1={n_halluc} Step2={n_omit} | keep pdsqi9={pdsqi} freetext_chars={ft}")

        if args.apply:
            if not _backup(resp):
                print(f"  -> backup failed; NOT migrating {rid}/{hid}")
                continue
            write_json(path, _transform(resp))
            n_changed += 1

    print(f"\n=== {'APPLIED' if args.apply else 'DRY-RUN (no writes)'} ===")
    print(f"  responses: {len(blobs)}  (complete={n_complete}, drafts={n_drafts})")
    if args.apply:
        print(f"  migrated:  {n_changed}")
        print(f"  blob backup:  {BACKUP_BLOB_PREFIX}")
        print(f"  local backup: {LOCAL_BACKUP_DIR}")
    else:
        print("  re-run with --apply to back up + write changes.")


if __name__ == "__main__":
    main()
