"""
PHI redaction wrapper around philter-ucsf.

Calls a Python 3.11 sub-venv where philter-ucsf is installed, since v1.0.3
does not run on Python 3.12 (regex flag syntax + removed distutils). Set
the env var ICUPAUSE_REDACT_PYTHON to point at the 3.11 venv's python:

    export ICUPAUSE_REDACT_PYTHON=/path/to/.venv-redact/bin/python
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

REDACT_PYTHON_ENV = "ICUPAUSE_REDACT_PYTHON"
RUN_PHILTER_SCRIPT = Path(__file__).parent / "run_philter.py"


class RedactionConfigError(RuntimeError):
    """Raised when the sub-venv pointer is missing or unusable."""


def _resolve_python() -> str:
    p = os.environ.get(REDACT_PYTHON_ENV)
    if not p:
        raise RedactionConfigError(
            f"{REDACT_PYTHON_ENV} is not set. Point it at a Python 3.11 venv "
            f"with philter-ucsf installed. See review_app/redaction/README.md."
        )
    if not os.path.exists(p):
        raise RedactionConfigError(f"{REDACT_PYTHON_ENV}={p} does not exist on disk.")
    return p


def redact_strings(strs: list[str]) -> list[str]:
    """Redact PHI from each string. Returns a list of redacted strings same
    length as input. Empty strings pass through untouched."""
    if not strs:
        return []

    python = _resolve_python()

    with tempfile.TemporaryDirectory(prefix="philter_") as tmp:
        in_dir = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        non_empty_indices: list[int] = []
        for i, s in enumerate(strs):
            if isinstance(s, str) and s.strip():
                (in_dir / f"{i:06d}.txt").write_text(s, encoding="utf-8")
                non_empty_indices.append(i)

        if not non_empty_indices:
            return list(strs)

        proc = subprocess.run(
            [
                python, str(RUN_PHILTER_SCRIPT),
                "--input-dir", str(in_dir),
                "--output-dir", str(out_dir),
            ],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"philter subprocess failed (rc={proc.returncode}).\n"
                f"stderr (last 2KB):\n{proc.stderr[-2000:]}"
            )

        result: list[str] = list(strs)
        for i in non_empty_indices:
            redacted_path = out_dir / f"{i:06d}.txt"
            if redacted_path.exists():
                result[i] = redacted_path.read_text(encoding="utf-8")
            else:
                raise RuntimeError(
                    f"philter output missing for index {i}; subprocess may have "
                    f"silently skipped a note."
                )
        return result


def _collect_strings(payload_pair: tuple[dict, dict]) -> tuple[list[str], list[Callable[[str], None]]]:
    """Walk the source_bundle + output dicts, collect every free-text string
    we want redacted. Returns parallel (values, setters) lists. Each setter
    writes a redacted string back into the dict it came from."""
    source_bundle, output = payload_pair

    values: list[str] = []
    setters: list[Callable[[str], None]] = []

    def add(value: Any, setter: Callable[[str], None]) -> None:
        if isinstance(value, str) and value.strip():
            values.append(value)
            setters.append(setter)

    def walk_clinical_notes(notes_by_type: Any) -> None:
        if not isinstance(notes_by_type, dict):
            return
        for notes in notes_by_type.values():
            if not isinstance(notes, list):
                continue
            for note in notes:
                if not isinstance(note, dict):
                    continue
                add(note.get("note_text"), lambda v, n=note: n.__setitem__("note_text", v))

    # Free-text keys to redact when an item in a list-of-dicts has them.
    # Excludes structural fields like "id", "severity", "category", "code".
    TEXT_KEYS = {"text", "message", "description", "content", "note_text"}

    def walk_string_list(lst: Any) -> None:
        if not isinstance(lst, list):
            return
        for i, item in enumerate(lst):
            if isinstance(item, str):
                add(item, lambda v, lst=lst, i=i: lst.__setitem__(i, v))
            elif isinstance(item, dict):
                for k, v in list(item.items()):
                    if k in TEXT_KEYS and isinstance(v, str):
                        add(v, lambda new, d=item, k=k: d.__setitem__(k, new))

    # source_bundle.clinical_notes
    walk_clinical_notes(source_bundle.get("clinical_notes"))

    # output.sections — dict[str, str]
    sections = output.get("sections")
    if isinstance(sections, dict):
        for sid, val in sections.items():
            if isinstance(val, str):
                add(val, lambda v, s=sections, k=sid: s.__setitem__(k, v))

    # output.todo_checklist, warnings, qa_issues
    walk_string_list(output.get("todo_checklist"))
    walk_string_list(output.get("warnings"))
    walk_string_list(output.get("qa_issues"))

    # output.metadata.source_data.clinical_notes (mirror of source_bundle)
    md = output.get("metadata") or {}
    md_source = md.get("source_data") or {}
    walk_clinical_notes(md_source.get("clinical_notes"))

    # output.metadata.agent_source_data.<role>.clinical_notes
    md_agent = md.get("agent_source_data") or {}
    if isinstance(md_agent, dict):
        for ctx in md_agent.values():
            if isinstance(ctx, dict):
                walk_clinical_notes(ctx.get("clinical_notes"))

    return values, setters


def redact_case_payload(
    source_bundle: dict[str, Any],
    output: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Redact PHI from a case's source_bundle.json and output.json payloads.

    Returns deep-copied (redacted_source_bundle, redacted_output). Inputs are
    not mutated. Dates are preserved (DATE patterns dropped from philter
    config at runtime in run_philter.py).
    """
    sb = deepcopy(source_bundle)
    out = deepcopy(output)

    values, setters = _collect_strings((sb, out))
    if not values:
        return sb, out

    redacted = redact_strings(values)
    for setter, new_value in zip(setters, redacted):
        setter(new_value)

    return sb, out
