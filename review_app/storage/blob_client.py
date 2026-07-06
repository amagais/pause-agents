"""Azure Blob Storage client wrapper for the review app."""

from __future__ import annotations

import json
import os
from typing import Any

from azure.storage.blob import BlobServiceClient, ContainerClient


def _container() -> ContainerClient:
    conn_str = os.environ["AZURE_BLOB_CONNECTION_STRING"]
    container = os.environ.get("BLOB_CONTAINER_NAME", "icupause-review")
    return BlobServiceClient.from_connection_string(conn_str).get_container_client(container)


def blob_exists(blob_path: str) -> bool:
    client = _container().get_blob_client(blob_path)
    return client.exists()


def read_json(blob_path: str) -> Any:
    """Download and parse a JSON blob. Raises if not found."""
    client = _container().get_blob_client(blob_path)
    data = client.download_blob().readall()
    return json.loads(data)


def write_json(blob_path: str, obj: Any) -> None:
    """Serialize obj to JSON and upload (overwrite)."""
    client = _container().get_blob_client(blob_path)
    payload = json.dumps(obj, indent=2, default=str).encode("utf-8")
    client.upload_blob(payload, overwrite=True)


def list_blobs_prefix(prefix: str) -> list[str]:
    """Return all blob names with the given prefix."""
    return [b.name for b in _container().list_blobs(name_starts_with=prefix)]


def delete_blob(blob_path: str) -> bool:
    """Delete a single blob. Returns True if deleted, False if not found."""
    client = _container().get_blob_client(blob_path)
    if not client.exists():
        return False
    client.delete_blob()
    return True


def delete_blobs_prefix(prefix: str) -> list[str]:
    """Delete every blob under *prefix*. Returns the list of paths deleted."""
    container = _container()
    deleted: list[str] = []
    for b in container.list_blobs(name_starts_with=prefix):
        container.get_blob_client(b.name).delete_blob()
        deleted.append(b.name)
    return deleted


def list_case_files_with_timestamps() -> list[dict]:
    """Return one row per hosp_id under ``cases/`` with last_modified per file.

    Each row: ``{"hosp_id", "output.json", "source_bundle.json", "claims.json"}``.
    Timestamp values are the blob's ``last_modified`` (UTC datetime); missing
    files are ``None``. Used by the admin page to surface upload recency.
    """
    by_hosp: dict[str, dict[str, Any]] = {}
    for b in _container().list_blobs(name_starts_with="cases/"):
        # Path shape: cases/{hosp_id}/{filename}
        parts = b.name.split("/", 2)
        if len(parts) != 3:
            continue
        _, hosp_id, fname = parts
        row = by_hosp.setdefault(
            hosp_id,
            {"hosp_id": hosp_id, "output.json": None, "source_bundle.json": None, "claims.json": None},
        )
        if fname in row:
            row[fname] = b.last_modified
    return list(by_hosp.values())
