"""Save, load, and export ReviewResponse objects to/from Azure Blob Storage."""

from __future__ import annotations

import io
import os
from typing import Any

import pandas as pd

from storage.blob_client import blob_exists, list_blobs_prefix, read_json, write_json


# ---------------------------------------------------------------------------
# Read-only (frozen) mode
# ---------------------------------------------------------------------------

def is_read_only() -> bool:
    """True when the app is frozen to read-only (no draft saves / submissions).

    Toggled by the REVIEW_APP_READ_ONLY env var (Azure App Service Application
    Setting). Set to 1/true/yes/on to freeze; unset or 0/false to re-enable
    editing. No code redeploy is needed to flip it — just change the setting
    and restart the app.
    """
    return os.environ.get("REVIEW_APP_READ_ONLY", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


# ---------------------------------------------------------------------------
# Save / load individual responses
# ---------------------------------------------------------------------------

def _response_path(reviewer_id: str, hosp_id: str) -> str:
    return f"responses/{reviewer_id}/{hosp_id}.json"


def save_response(response_dict: dict[str, Any]) -> None:
    """Write a ReviewResponse dict to blob (overwrite).

    Authoritative write choke point — blocked when the app is frozen so no
    write can slip through any UI path.
    """
    if is_read_only():
        raise RuntimeError(
            "Review app is in read-only mode; saving is disabled. "
            "Unset REVIEW_APP_READ_ONLY to re-enable editing."
        )
    reviewer_id = response_dict["reviewer_id"]
    hosp_id = response_dict["hosp_id"]
    write_json(_response_path(reviewer_id, hosp_id), response_dict)


def load_response(reviewer_id: str, hosp_id: str) -> dict[str, Any] | None:
    """Load a ReviewResponse dict, or None if it doesn't exist."""
    path = _response_path(reviewer_id, hosp_id)
    if not blob_exists(path):
        return None
    return read_json(path)


def response_exists(reviewer_id: str, hosp_id: str) -> bool:
    return blob_exists(_response_path(reviewer_id, hosp_id))


def load_prior_round_response(reviewer_id: str, hosp_id: str) -> dict[str, Any] | None:
    """Load the round-1 response from the rerun2 backup prefix, or None.

    Written by scripts/migrate_responses_for_rerun2.py before the re-review.
    Used to surface a reviewer's prior non-verified claims / omitted domains
    as read-only reference (the live response's Step 1/2 were cleared because
    the regenerated brief's sentences changed).
    """
    path = f"responses_v1_backup/{reviewer_id}/{hosp_id}.json"
    if not blob_exists(path):
        return None
    return read_json(path)


def response_complete(reviewer_id: str, hosp_id: str) -> bool:
    """Return True if a completed response exists for this reviewer-case pair."""
    resp = load_response(reviewer_id, hosp_id)
    return bool(resp and resp.get("is_complete"))


# ---------------------------------------------------------------------------
# Admin: aggregate all responses
# ---------------------------------------------------------------------------

def load_all_responses() -> list[dict[str, Any]]:
    """Return all response dicts across all reviewers."""
    blobs = list_blobs_prefix("responses/")
    results = []
    for blob_path in blobs:
        if blob_path.endswith(".json"):
            try:
                results.append(read_json(blob_path))
            except Exception:
                pass
    return results


def export_summary_csv(responses: list[dict[str, Any]]) -> bytes:
    """Build the flat summary CSV (one row per reviewer-case pair)."""
    rows = []
    for r in responses:
        pdsqi = r.get("pdsqi9") or {}
        likert_fields = ["cited", "accurate", "thorough", "useful", "organized",
                         "comprehensible", "succinct", "synthesized"]
        likert_values = {f"pdsqi9_{k}": pdsqi.get(k) for k in likert_fields}
        valid_vals = [v for v in likert_values.values() if v is not None]
        pdsqi_total = round(sum(valid_vals) / len(valid_vals), 3) if valid_vals else None

        halluc = r.get("hallucination_checks", [])
        n_verified = sum(1 for c in halluc if c.get("verdict") == "verified")
        n_cannot = sum(1 for c in halluc if c.get("verdict") == "cannot_verify")
        n_incorrect = sum(1 for c in halluc if c.get("verdict") == "incorrect")
        n_claims = len(halluc)
        halluc_rate = round(n_incorrect / n_claims, 3) if n_claims else None

        omissions = r.get("omission_checks", [])
        n_pertinent = sum(1 for o in omissions if o.get("omitted") and o.get("severity") == "pertinent")
        n_potentially = sum(1 for o in omissions if o.get("omitted") and o.get("severity") == "potentially_pertinent")

        row = {
            "reviewer_id": r.get("reviewer_id"),
            "hosp_id": r.get("hosp_id"),
            "batch": r.get("batch", 0),
            "pipeline_version": r.get("pipeline_version", ""),
            "phase": r.get("phase"),
            "submitted_at": r.get("submitted_at"),
            "is_complete": r.get("is_complete"),
            "time_on_task_s": r.get("time_on_task_seconds"),
            **likert_values,
            "pdsqi9_stigmatizing": pdsqi.get("stigmatizing"),
            "pdsqi9_total": pdsqi_total,
            "n_claims_total": n_claims,
            "n_claims_verified": n_verified,
            "n_claims_cannot_verify": n_cannot,
            "n_claims_incorrect": n_incorrect,
            "hallucination_rate": halluc_rate,
            "n_omissions_pertinent": n_pertinent,
            "n_omissions_potentially_pertinent": n_potentially,
            "qa_issues_feedback": r.get("qa_issues_feedback", ""),
            "warnings_feedback": r.get("warnings_feedback", ""),
            "overall_comment": r.get("overall_comment", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def export_claims_csv(responses: list[dict[str, Any]]) -> bytes:
    """Claim-level detail CSV."""
    rows = []
    for r in responses:
        for claim in r.get("hallucination_checks", []):
            rows.append({
                "reviewer_id": r.get("reviewer_id"),
                "hosp_id": r.get("hosp_id"),
                "batch": r.get("batch", 0),
                "pipeline_version": r.get("pipeline_version", ""),
                "claim_id": claim.get("claim_id"),
                "section": claim.get("section"),
                "claim_text": claim.get("claim_text"),
                "verdict": claim.get("verdict"),
                "source_location": claim.get("source_location"),
            })
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def export_omissions_csv(responses: list[dict[str, Any]]) -> bytes:
    """Omission-level detail CSV."""
    rows = []
    for r in responses:
        for o in r.get("omission_checks", []):
            rows.append({
                "reviewer_id": r.get("reviewer_id"),
                "hosp_id": r.get("hosp_id"),
                "batch": r.get("batch", 0),
                "pipeline_version": r.get("pipeline_version", ""),
                "domain": o.get("domain"),
                "omitted": o.get("omitted"),
                "severity": o.get("severity"),
                "brief_note": o.get("brief_note"),
            })
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()
