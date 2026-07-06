"""Lightweight per-case UI flags (e.g. the crfix creatinine badge).

A flag blob at ``flags/{flag_id}.json`` holds
``{"label": "Cr", "hosp_ids": [...]}``. The dashboard reads these to badge
affected cases in the queue WITHOUT loading each full brief (output.json is
~270 KB; the flag blob is a few hundred bytes). Decoupled from the brief so the
badge works for reviewed and not-yet-reviewed cases alike.
"""

from __future__ import annotations

from .blob_client import list_blobs_prefix, read_json, write_json

FLAGS_PREFIX = "flags/"


def load_case_badges() -> dict[str, str]:
    """Return ``{hosp_id: badge_label}`` merged across all flag blobs.

    Best-effort: a missing/unreadable flag blob is skipped, never raised, so a
    flags problem can't break the dashboard. Empty dict when no flags exist.
    """
    out: dict[str, str] = {}
    try:
        paths = list_blobs_prefix(FLAGS_PREFIX)
    except Exception:  # noqa: BLE001
        return out
    for path in paths:
        if not path.endswith(".json"):
            continue
        try:
            data = read_json(path)
        except Exception:  # noqa: BLE001
            continue
        label = str(data.get("label") or "")
        for hid in data.get("hosp_ids", []) or []:
            out[str(hid)] = label
    return out


def write_case_flag(flag_id: str, label: str, hosp_ids: list[str]) -> str:
    """Write/overwrite a flag blob. Returns the blob path."""
    path = f"{FLAGS_PREFIX}{flag_id}.json"
    write_json(path, {"label": label, "hosp_ids": list(hosp_ids)})
    return path
