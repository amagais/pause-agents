"""Export reviewer responses with a `round` column distinguishing v1 (original)
from v2 (rerun2 re-review).

Round-1 responses are preserved under the `responses_v1_backup/` blob prefix by
migrate_responses_for_rerun2.py; current/live responses live under `responses/`.
This reads BOTH and emits one CSV with a leading `round` column so you can:
  * take TIME-ON-TASK from round == "v1_original" (the genuine full-review latency),
  * take RATINGS from round == "v2_rereview" (judgments on the regenerated briefs),
  * join the two on (reviewer_id, hosp_id).

Runs LOCALLY on the Mac (needs review_app/.env + azure SDK).

Usage (from repo root):
  .venv/bin/python review_app/scripts/export_responses_by_round.py
  .venv/bin/python review_app/scripts/export_responses_by_round.py --out /tmp/responses_by_round.csv
"""

from __future__ import annotations

import argparse
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW_APP = os.path.dirname(HERE)
sys.path.insert(0, REVIEW_APP)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REVIEW_APP, ".env"))

import pandas as pd  # noqa: E402

from storage.blob_client import list_blobs_prefix, read_json  # noqa: E402
from storage.response_writer import export_summary_csv  # noqa: E402

PREFIXES = [
    ("v1_original", "responses_v1_backup/"),
    ("v2_rereview", "responses/"),
]


def _load_prefix(prefix: str) -> list[dict]:
    out = []
    for b in list_blobs_prefix(prefix):
        if b.endswith(".json"):
            try:
                out.append(read_json(b))
            except Exception:  # noqa: BLE001
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="responses_by_round.csv")
    args = ap.parse_args()

    frames = []
    for round_label, prefix in PREFIXES:
        resps = _load_prefix(prefix)
        if not resps:
            print(f"  (no responses under {prefix})")
            continue
        df = pd.read_csv(io.BytesIO(export_summary_csv(resps)))
        df.insert(0, "round", round_label)
        frames.append(df)
        print(f"  {round_label:12} {len(df):4} responses from {prefix}")

    if not frames:
        print("No responses found under either prefix.")
        return

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} rows -> {args.out}")
    print("  latency  -> filter round=='v1_original', use time_on_task_s")
    print("  ratings  -> filter round=='v2_rereview'")
    print("  join key -> (reviewer_id, hosp_id)")


if __name__ == "__main__":
    main()
