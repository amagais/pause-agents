"""Load case data (output, source bundle, claims) from Azure Blob Storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from storage.blob_client import read_json


@dataclass
class CaseBundle:
    hosp_id: str
    output: dict[str, Any]           # ICUPauseOutput JSON
    source: dict[str, Any]           # source_bundle JSON
    claims: list[dict[str, Any]]     # list of {claim_id, section, text}


def load_case(hosp_id: str) -> CaseBundle:
    """Load all three blobs for a case and return a CaseBundle."""
    output = read_json(f"cases/{hosp_id}/output.json")
    source = read_json(f"cases/{hosp_id}/source_bundle.json")
    claims = read_json(f"cases/{hosp_id}/claims.json")
    return CaseBundle(hosp_id=hosp_id, output=output, source=source, claims=claims)
