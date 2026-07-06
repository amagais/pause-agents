"""Canonical citation-rendering spec.

The **single source of truth** for how citation tags in a rendered section
are transformed into superscript footnote markers.  The JavaScript port in
``frontend/src/utils/citations.js`` re-implements the same algorithm against
a shared fixture — if the two drift, CI fails.

Algorithm:
    1. Scan the section text for citation tags (matching ``CITE_PATTERN``).
    2. Assign each DISTINCT tag a per-section number in first-appearance
       order.  Numbering resets at every section boundary (callers invoke
       this once per section).
    3. Dedupe within each sentence — if a tag appears N times in one
       sentence, emit one marker (not N).  Sentence boundaries are
       ``[.!?]`` + whitespace/EOL, or a newline.
    4. Produce a list of text/citation segments the renderer walks.

Returns both segments (for inline rendering) and footnotes (for
end-of-section lists in DOCX/PDF exports).

See tests/test_citation_rendering.py for the shared fixture both
Python and JS implementations validate against.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict
from zoneinfo import ZoneInfo

from icu_pause.data.context import CITE_PATTERN

# Display TZ for row-time tooltip strings. Kept in sync with
# ``review_app/display/_datetimes.DISPLAY_TZ`` (and with
# ``review_app/display/source_renderer._DISPLAY_TZ``) so that any
# surface rendering an ISO row-time agrees on what hour the clinician
# sees. The pipeline writes row.time as raw UTC ISO; without the
# conversion below, a tooltip would say "Jul 01 04:00" while the
# source-data table renders the same instant as "6-30 23:00" CDT.
# If you change this constant, change the others too — there is no
# review_app dependency available here (canonical module must stay
# package-standalone) so the constant is duplicated by design.
_DISPLAY_TZ = ZoneInfo("America/Chicago")

# Unicode superscript digits — cover 0-9; two-digit numbers concatenate
# (¹⁰, ¹¹) rather than falling back to ``[10]`` brackets.  Renders
# correctly in browsers, python-docx, and fpdf2 per manual verification.
_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def superscript(n: int) -> str:
    """Return the Unicode superscript representation of a positive integer."""
    return "".join(_SUPERSCRIPT_DIGITS[int(c)] for c in str(n))


# Short month names for tooltip time formatting.  Index 0 unused.
_MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_TAG_TIME_RE = re.compile(
    r"\((?:lab|vital|med|resp|assess|code|proc"
    r"|exam-vitals|exam-neuro|exam-resp"
    r"|progress_note|hp_note|consults_note|plan_of_care_note"
    r"|nursing_note|case_management_note|social_work_note|therapy_note) "
    r"(\d{1,2})-(\d{2}) (\d{2}:\d{2})\)"
)


def _short_time_from_tag(tag: str) -> str:
    """Extract ``"Mon DD HH:MM"`` from a tag string — the tag is authoritative.

    Even for unverified tags (no matching row in the index), we can still
    display the timestamp the agent claimed, since the tag regex guarantees
    month-day-time structure.
    """
    m = _TAG_TIME_RE.match(tag)
    if not m:
        return tag
    month, day, hhmm = m.group(1), m.group(2), m.group(3)
    try:
        mon_name = _MONTHS[int(month)]
    except (ValueError, IndexError):
        mon_name = month
    return f"{mon_name} {int(day):02d} {hhmm}"


def _format_one_row(label: Any, value: Any, unit: Any) -> Optional[str]:
    if label and value and unit:
        return f"{label} {value} {unit}"
    if label and value:
        return f"{label} {value}"
    if label:
        return str(label)
    if value:
        return str(value)
    return None


def _short_time_from_iso(iso: Optional[str]) -> Optional[str]:
    """Convert an ISO ``row.time`` to ``"Mon DD HH:MM"`` in the display TZ.

    Naive timestamps are assumed UTC (matches the pipeline's cite-tag
    builder + ``review_app/display/source_renderer._compact_dttm``).
    Returns ``None`` for non-parseable inputs so the caller can fall
    back cleanly. Day component is zero-padded to match
    ``_short_time_from_tag``'s output.

    The display TZ conversion (``astimezone(_DISPLAY_TZ)``) is
    intentional: ``row.time`` is stored raw-UTC in the citation index,
    but every clinician-visible surface (source table, brief, doc
    export) renders in America/Chicago. Without the conversion, the
    tooltip's per-row ``(time)`` parenthetical disagrees with the
    source table by the UTC offset — surfacing a real instant as an
    apparent contradiction. See the same-named helper in
    ``review_app/display/citations.py`` for the parallel fix.
    """
    if not iso:
        return None
    s = str(iso)
    try:
        normalized = s.replace(" ", "T", 1) if "T" not in s[:11] else s
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_local = dt.astimezone(_DISPLAY_TZ)
    try:
        mon = _MONTHS[dt_local.month]
    except IndexError:
        return None
    return f"{mon} {dt_local.day:02d} {dt_local.hour:02d}:{dt_local.minute:02d}"


_NOTE_SOURCE_TYPES = frozenset({
    "progress_note", "hp_note", "consults_note", "plan_of_care_note",
    "nursing_note", "case_management_note", "social_work_note", "therapy_note",
})


def _is_note_tag(tag: str) -> bool:
    """True iff *tag* names one of the clinical-note source types.

    Note row.time is always the same logical timestamp as the tag's
    anchor (revision_dttm, or creation_dttm fallback — see
    _add_cite_fields call sites for notes in data/context.py). The two
    only diverge as strings because the tag anchor is formatted in the
    display tz (America/Chicago) while row.time is the raw ISO UTC.
    Suppressing the per-row time parenthetical for notes avoids
    surfacing that TZ artifact as an apparent contradiction.
    """
    inner = tag[1:].split(" ", 1)[0] if tag.startswith("(") else ""
    return inner in _NOTE_SOURCE_TYPES


def format_tooltip(tag: str, entry: Optional[dict[str, Any]]) -> str:
    """Deterministic tooltip text for a single citation.

    Both the Python and JS renderers call this so the hover string is
    identical regardless of where the brief is displayed.

    When a tag resolves to multiple sibling rows (e.g. an exam summary
    citing HR/BP/MAP/SpO2 at one bucket → 4 vital rows under one tag),
    every sibling is rendered, separated by ``;``. Single-row tags
    fall back to the legacy ``label value unit`` form.

    Per-row timestamps (``row.time``, populated for Phase-3 exam-*
    source types) are appended to the row's display when they differ
    from the tag's anchor. The anchor time is always appended at the
    end as ``" · <time>"``. When all rows share the anchor (the
    pre-Phase-3 common case for vitals / labs / etc.), no per-row times
    appear — the tooltip looks identical to the prior behavior. Note
    source types skip the per-row time entirely; see _is_note_tag.

    Backwards compatibility: on-disk citation_index from before the
    ``rows`` field was added has only singular label/value/unit;
    ``entry.get("rows", [])`` returns ``[]`` and we fall back to the
    singular fields automatically. Pre-Phase-3 ``rows`` entries without
    a ``time`` field skip the divergence check.
    """
    time_short = _short_time_from_tag(tag)
    if entry is None or entry.get("tier") == "unverified":
        return f"⚠ unverified source · {time_short}"

    rows = entry.get("rows") or []
    row_strs: list[str] = []
    suppress_row_time = _is_note_tag(tag)
    if rows:
        for r in rows:
            formatted = _format_one_row(r.get("label"), r.get("value"), r.get("unit"))
            if not formatted:
                continue
            if not suppress_row_time:
                row_time = _short_time_from_iso(r.get("time"))
                if row_time and row_time != time_short:
                    formatted = f"{formatted} ({row_time})"
            row_strs.append(formatted)
    else:
        # Pre-``rows`` index shape — fall back to the singular fields so
        # legacy output.json files keep rendering.
        legacy = _format_one_row(
            entry.get("label"), entry.get("value"), entry.get("unit")
        )
        if legacy:
            row_strs.append(legacy)

    parts: list[str] = []
    if row_strs:
        parts.append("; ".join(row_strs))
    parts.append(time_short)
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Segment types — renderer-agnostic intermediate representation
# ---------------------------------------------------------------------------


class TextSegment(TypedDict):
    kind: str  # "text"
    text: str


class CitationSegment(TypedDict):
    kind: str  # "cite"
    number: int
    tag: str
    tooltip: str
    tier: str  # "decision_critical" | "unverified"
    marker: str  # superscript string


class Footnote(TypedDict):
    number: int
    tag: str
    tooltip: str
    tier: str
    marker: str


class RenderedSection(TypedDict):
    segments: list[dict[str, Any]]
    footnotes: list[Footnote]


# ---------------------------------------------------------------------------
# Sentence boundary detection
# ---------------------------------------------------------------------------
#
# Sentence break = [.!?] followed by whitespace or end-of-input, OR a newline.
# We need spans (not split results) so we can group citations by sentence
# while preserving the original text exactly.

_SENT_BOUNDARY = re.compile(r"(?:(?<=[.!?])(?=\s)|(?=\n))")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans covering *text* split at sentence boundaries.

    Boundaries are zero-width — no characters are consumed — so the spans
    concatenate back to exactly *text*.  A single sentence with no terminator
    returns one span ``(0, len(text))``.
    """
    if not text:
        return []
    boundaries: list[int] = [0]
    for m in _SENT_BOUNDARY.finditer(text):
        pos = m.start()
        # Advance past any whitespace / newline so the next sentence starts
        # on non-whitespace content, not the break character.
        while pos < len(text) and text[pos] in " \t":
            pos += 1
        if pos > boundaries[-1]:
            boundaries.append(pos)
    if boundaries[-1] < len(text):
        boundaries.append(len(text))
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


# ---------------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------------


def render_section(
    text: str,
    citation_index: dict[str, dict[str, Any]],
) -> RenderedSection:
    """Transform raw section text into (segments, footnotes).

    Args:
        text: The merged section content as it will appear in the brief,
            including newlines and any markdown.  Should NOT be split
            by the caller — this function handles per-section numbering
            internally.
        citation_index: Map of tag → CitationEntry dict (from
            ``ICUPauseOutput.metadata['citation_index']``).  Missing tags
            produce ``unverified`` segments.

    Returns:
        A ``RenderedSection`` with two lists:

        * ``segments``: ``{"kind": "text", ...}`` and ``{"kind": "cite", ...}``
          entries in order.  Concatenating every ``text`` segment plus each
          ``marker`` reproduces the final displayed string.
        * ``footnotes``: one entry per distinct cited tag, in numbered
          order, for DOCX/PDF end-of-section rendering.
    """
    if not text:
        return {"segments": [], "footnotes": []}

    # Pass 1: assign a per-section number to each distinct tag in
    # first-appearance order.  Built up as we walk sentences so a tag
    # that's removed by intra-sentence dedupe still keeps its number.
    tag_number: dict[str, int] = {}

    def _assign(tag: str) -> int:
        if tag not in tag_number:
            tag_number[tag] = len(tag_number) + 1
        return tag_number[tag]

    segments: list[dict[str, Any]] = []

    # Pass 2: walk sentences.  For each sentence, compute the set of
    # distinct tags, remove ALL inline occurrences, then emit markers
    # for each distinct tag (in first-occurrence order within the
    # sentence) at the sentence's content tail — immediately before
    # any trailing terminator punctuation, else at the very end.
    for s_start, s_end in _sentence_spans(text):
        sentence = text[s_start:s_end]
        matches = list(CITE_PATTERN.finditer(sentence))

        if not matches:
            segments.append({"kind": "text", "text": sentence})
            continue

        # Distinct tags in first-occurrence order within this sentence.
        seen: dict[str, None] = {}
        for m in matches:
            seen.setdefault(m.group(), None)
        distinct_tags = list(seen.keys())

        # Register numbers now (preserves global first-appearance order).
        for tag in distinct_tags:
            _assign(tag)

        # Build the cleaned sentence: remove every inline citation.
        # Strip a leading space before the citation too so we don't leave
        # ``"HR 84 ."`` behind after dropping ``"(vital 1-12 07:00)"``.
        cleaned_parts: list[str] = []
        cursor = 0
        for m in matches:
            start = m.start()
            # Consume trailing whitespace IN the retained left-segment so
            # the inline cite and its leading space are both removed.
            left_end = start
            while left_end > cursor and sentence[left_end - 1] == " ":
                left_end -= 1
            cleaned_parts.append(sentence[cursor:left_end])
            cursor = m.end()
        cleaned_parts.append(sentence[cursor:])
        cleaned = "".join(cleaned_parts)

        # Find where to insert markers: just before any trailing
        # terminator punctuation (+ trailing whitespace).
        tail_start = len(cleaned)
        # Walk back over trailing whitespace + a single terminator char.
        while tail_start > 0 and cleaned[tail_start - 1] in " \t\n":
            tail_start -= 1
        if tail_start > 0 and cleaned[tail_start - 1] in ".!?":
            # Insert marker BEFORE this terminator.
            insert_at = tail_start - 1
        else:
            # No terminator — insert at tail_start (before trailing whitespace).
            insert_at = tail_start

        prefix = cleaned[:insert_at]
        suffix = cleaned[insert_at:]

        if prefix:
            segments.append({"kind": "text", "text": prefix})

        for tag in distinct_tags:
            num = tag_number[tag]
            entry = citation_index.get(tag)
            tier = (entry or {}).get("tier", "unverified")
            segments.append({
                "kind": "cite",
                "number": num,
                "tag": tag,
                "tooltip": format_tooltip(tag, entry),
                "tier": tier,
                "marker": superscript(num),
            })

        if suffix:
            segments.append({"kind": "text", "text": suffix})

    # Footnotes: one entry per distinct tag, in numeric order.
    footnotes: list[Footnote] = []
    for tag, num in sorted(tag_number.items(), key=lambda kv: kv[1]):
        entry = citation_index.get(tag)
        tier = (entry or {}).get("tier", "unverified")
        footnotes.append({
            "number": num,
            "tag": tag,
            "tooltip": format_tooltip(tag, entry),
            "tier": tier,
            "marker": superscript(num),
        })

    return {"segments": segments, "footnotes": footnotes}


def render_plain_text(rendered: RenderedSection) -> str:
    """Reproduce the section's displayed string from segments.

    Useful for DOCX/PDF renderers that want to lay out text and superscripts
    without re-parsing, and for tests that assert the superscript-injected
    output.
    """
    parts: list[str] = []
    for seg in rendered["segments"]:
        if seg["kind"] == "text":
            parts.append(seg["text"])
        elif seg["kind"] == "cite":
            parts.append(seg["marker"])
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTML helper (used by Streamlit; the React port emits JSX from segments)
# ---------------------------------------------------------------------------

from html import escape as _html_escape


def segments_to_html(segments: list[dict[str, Any]]) -> str:
    """Render segments as HTML, emitting ``<sup>`` for citation markers.

    The CSS class encodes the tier so the Streamlit stylesheet and the
    React stylesheet can pick the same visual treatment:

      * ``icp-cite icp-cite--decision_critical`` — bold deep-blue
      * ``icp-cite icp-cite--unverified`` — bold amber, ⚠ prefix in tooltip
    """
    out: list[str] = []
    for seg in segments:
        if seg["kind"] == "text":
            out.append(_html_escape(seg["text"]))
        elif seg["kind"] == "cite":
            tier = seg.get("tier", "unverified")
            tooltip = _html_escape(seg["tooltip"], quote=True)
            tag = _html_escape(seg["tag"], quote=True)
            # tabindex makes the marker focusable so clicking/tabbing holds
            # the CSS tooltip open (via the :focus pseudo-class in theme CSS).
            # Keeping the native ``title=`` too gives us a fallback if a
            # host app's CSS overrides the ``::after`` tooltip.
            out.append(
                f'<sup class="icp-cite icp-cite--{tier}" tabindex="0" '
                f'title="{tooltip}" data-tag="{tag}">{seg["marker"]}</sup>'
            )
    return "".join(out)


def segments_by_line(
    segments: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split segments into per-line lists.

    The canonical renderer preserves ``\\n`` characters verbatim inside text
    segments and guarantees citation segments contain no newlines, so this
    split is unambiguous.  DOCX/PDF renderers use it to iterate lines while
    still honoring the section-level numbering.
    """
    lines: list[list[dict[str, Any]]] = [[]]
    for seg in segments:
        if seg["kind"] == "text":
            parts = seg["text"].split("\n")
            for i, part in enumerate(parts):
                if i > 0:
                    lines.append([])
                if part:
                    lines[-1].append({"kind": "text", "text": part})
        else:
            lines[-1].append(seg)
    return lines


def line_plain_prefix(line_segments: list[dict[str, Any]]) -> str:
    """Concatenate just the text-segment content of one line for structural detection.

    Used by renderers to check ``startswith("#")`` / ``[ ]`` / ``## `` without
    being thrown off by inline citation markers.
    """
    return "".join(s["text"] for s in line_segments if s["kind"] == "text").strip()


def render_section_html(
    text: str,
    citation_index: dict[str, dict[str, Any]],
) -> tuple[str, list[Footnote]]:
    """Convenience: ``render_section`` + ``segments_to_html`` in one call.

    Returns ``(html, footnotes)``.  Newlines in text segments are preserved
    verbatim so callers that split-on-newline (e.g. the Streamlit S-section
    parser) still see the original line structure.
    """
    rendered = render_section(text, citation_index)
    return segments_to_html(rendered["segments"]), rendered["footnotes"]
