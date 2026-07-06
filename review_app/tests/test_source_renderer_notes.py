"""Reviewer-panel note caption formatter.

Locks down _format_note_caption — the per-note header in the left source
panel. Lines up with:

- Single timestamp display: revision_dttm preferred, creation_dttm
  fallback. Matches the cite-tag the model emits inline in S, so the
  reviewer can cross-reference one timestamp. creation_dttm is never
  surfaced even when it differs from revision_dttm — reviewers were
  reading the panel's "created M-DD HH:MM" alongside the tooltip's
  anchor time as a contradiction (tooltip only carries revision_dttm)
  and flagging notes as inaccurate.
- note_author_service / specialty / type priority order matches
  citation_index._trim_note so panel header and cite tooltip read
  consistently.
- Graceful degradation when fields are missing (no revision_dttm yet,
  no service attribution, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

# review_app is not installed as a package; add its root to sys.path so the
# display.source_renderer import resolves.
_REVIEW_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_REVIEW_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_REVIEW_APP_ROOT))

from display.source_renderer import _format_note_caption  # noqa: E402


# ---------------------------------------------------------------------------
# Common case: note never revised (revision_dttm == creation_dttm).
# Single timestamp display, no (revised)/(created) labels.
# ---------------------------------------------------------------------------


def test_caption_collapses_when_revision_equals_creation():
    note = {
        "revision_dttm": "2026-05-27T19:23:00+00:00",
        "creation_dttm": "2026-05-27T19:23:00+00:00",
        "note_author_service": "Critical Care Medicine",
    }
    cap = _format_note_caption(note)
    assert cap == "Date: 5-27 14:23 · Critical Care Medicine"
    # No labels in the simple case — visual cleanliness for ~90% of notes.
    assert "(revised)" not in cap
    assert "created" not in cap


def test_caption_collapses_no_service_attribution():
    note = {
        "revision_dttm": "2026-05-27T19:23:00+00:00",
        "creation_dttm": "2026-05-27T19:23:00+00:00",
    }
    cap = _format_note_caption(note)
    assert cap == "Date: 5-27 14:23"


# ---------------------------------------------------------------------------
# Revised note: revision_dttm != creation_dttm. creation_dttm is NOT
# surfaced — reviewers were reading the panel's "created M-DD HH:MM"
# alongside the tooltip's anchor as a contradiction (tooltip only
# carries revision_dttm) and flagging notes as inaccurate.
# ---------------------------------------------------------------------------


def test_caption_shows_only_revision_when_revision_differs_from_creation():
    note = {
        "revision_dttm": "2026-05-27T19:23:00+00:00",  # 14:23 CDT
        "creation_dttm": "2026-05-27T16:21:00+00:00",  # 11:21 CDT
        "note_author_service": "Critical Care Medicine",
    }
    cap = _format_note_caption(note)
    assert cap == "Date: 5-27 14:23 · Critical Care Medicine"
    assert "created" not in cap
    assert "(revised)" not in cap
    assert "11:21" not in cap


def test_caption_shows_only_revision_no_service():
    note = {
        "revision_dttm": "2026-05-27T19:23:00+00:00",
        "creation_dttm": "2026-05-27T16:21:00+00:00",
    }
    cap = _format_note_caption(note)
    assert cap == "Date: 5-27 14:23"


# ---------------------------------------------------------------------------
# Missing revision_dttm (note not yet revised). Caption falls back to
# creation_dttm so reviewer still sees a timestamp; cite tag for the
# note anchors on creation_dttm too (see _add_cite_fields call sites
# in data/context.py), keeping panel and tooltip consistent.
# ---------------------------------------------------------------------------


def test_caption_shows_creation_when_revision_missing():
    note = {
        "creation_dttm": "2026-05-27T16:21:00+00:00",
        "note_author_service": "Critical Care Medicine",
    }
    cap = _format_note_caption(note)
    assert cap == "Date: 5-27 11:21 · Critical Care Medicine"
    assert "(revised)" not in cap


def test_caption_shows_creation_when_revision_is_empty_string():
    note = {
        "revision_dttm": "",
        "creation_dttm": "2026-05-27T16:21:00+00:00",
        "note_author_service": "Critical Care Medicine",
    }
    cap = _format_note_caption(note)
    assert cap == "Date: 5-27 11:21 · Critical Care Medicine"


# ---------------------------------------------------------------------------
# Service attribution priority — matches citation_index._trim_note.
# Same priority order keeps the panel header and the cite-tooltip
# consistent for the same note.
# ---------------------------------------------------------------------------


def test_caption_prefers_note_author_service():
    note = {
        "creation_dttm": "2026-05-27T16:21:00+00:00",
        "note_author_service": "Critical Care Medicine",
        "note_author_specialty": "Pulmonology",
        "note_author_type": "Physician",
    }
    cap = _format_note_caption(note)
    assert cap.endswith("· Critical Care Medicine")


def test_caption_falls_back_to_specialty_when_service_missing():
    note = {
        "creation_dttm": "2026-05-27T16:21:00+00:00",
        "note_author_specialty": "Nephrology",
        "note_author_type": "Physician",
    }
    cap = _format_note_caption(note)
    assert cap.endswith("· Nephrology")


def test_caption_falls_back_to_author_type_when_service_specialty_missing():
    note = {
        "creation_dttm": "2026-05-27T16:21:00+00:00",
        "note_author_type": "Nurse Practitioner",
    }
    cap = _format_note_caption(note)
    assert cap.endswith("· Nurse Practitioner")


def test_caption_omits_service_suffix_when_all_attribution_missing():
    note = {"creation_dttm": "2026-05-27T16:21:00+00:00"}
    cap = _format_note_caption(note)
    assert cap == "Date: 5-27 11:21"
    assert "·" not in cap  # no trailing separator when nothing to append
