"""Render ICUPauseOutput as styled Streamlit expanders."""

from __future__ import annotations

import logging
import re
from typing import Any

import streamlit as st

_logger = logging.getLogger(__name__)

from display.citations import render_section_html
from display.warnings import (
    CATEGORY_LABEL,
    SEVERITY_LABEL,
    audit_only_warnings,
    render_warnings_for_clinician,
)

# Matches SECTION_ORDER from src/icu_pause/rendering/formatter.py
SECTION_ORDER = [
    ("I", "I", "ICU Admission Reason & Brief ICU Course"),
    ("C", "C", "Code Status / DPOA Info / Goals of Care / ACP Note"),
    ("U_unprescribing", "U", "Unprescribing & Pertinent High-Risk Medications"),
    ("P", "P", "Pending Tests at the Time of Transfer"),
    ("A", "A", "Active Consultants, including Rehab"),
    ("U_uncertainty", "U", "Uncertainty Measure / Diagnostic Pause"),
    ("S", "S", "Summary of Major Problems and To-Do's"),
    ("E", "E", "Exam at the Time of Transfer, incl. Lines/Drains/Airways & Data Review"),
]


_TODO_BUCKET_LABELS = {
    "BEFORE TRANSFER": "Before Transfer",
    "ON THE WARD": "On the Ward",
    "AT DISCHARGE": "At Discharge",
}

# S-section spacing. Tune by eye, not pixel: the rule is "the gap between
# problems must be visibly larger than the gap between a problem's header
# and its body" so reviewers instantly pair each body with its header.
# Roughly 4–5x ratio between the two; adjust together if rhythm regresses.
_S_PROBLEM_BLOCK_GAP_REM = 1.25   # gap below each problem block (inter-problem)
_S_HEADER_BODY_GAP_REM = 0.25     # gap below header (tight pair with body)


# Order and display labels for structured todo_checklist items
# ({"bucket": ..., "text": ...}). Mirrors src/icu_pause/rendering/formatter.py
# in the agents repo — keep in sync if upstream order/labels change.
_BUCKET_ORDER = ["pre_transfer", "ward_ongoing", "discharge"]
_BUCKET_LABELS = {
    "pre_transfer": "Before Transfer (ICU team)",
    "ward_ongoing": "On the Ward (receiving team)",
    "discharge": "At Discharge (case manager / team)",
}


def _cite_html(text: str, citation_index: dict[str, Any]) -> str:
    """Return *text* with (cite) tags replaced by hoverable ``<sup>`` markers.

    Numbering resets at every call — the canonical renderer in
    display/citations.py assigns per-section numbers.
    """
    html, _footnotes = render_section_html(text, citation_index or {})
    return html


# Parent header from the pharmacy payload that wraps the 5 anticoagulation
# sub-headers. Dropped at render time only; the payload still carries it
# so QA can cross-reference rendered output against raw section text.
_U_SUPPRESS_PARENT_HEADERS: frozenset[str] = frozenset({
    "anticoagulation",
})

# Verbatim payload labels (normalized) that render as top-level groups.
# Order in the rendered output follows payload order, not this set —
# this is membership-only. Keep labels in sync with config/prompts/
# pharmacy.yaml lines 25-54; that file is the contract.
_U_KNOWN_HEADERS: frozenset[str] = frozenset({
    "changes to home meds",
    "active anticoagulation at transfer",
    "home anticoagulation (status at transfer)",
    "vte prophylaxis at transfer",
    "antiplatelet therapy at transfer",
    "transition/bridging plan (when applicable)",
    "transition/bridging plan",
    "antibiotics",
})

# Groups that must render even when their body is empty — clinical
# absence is itself meaningful (a brief with no VTE prophylaxis info
# is materially different from one where the group was omitted).
_U_FORCE_RENDER_EMPTY: frozenset[str] = frozenset({
    "vte prophylaxis at transfer",
})

# Groups whose items carry formatting that breaks the inline single-
# value collapse (checkbox prefix, two named sub-fields). Always
# multi-line, even with one body line.
_U_NEVER_COLLAPSE: frozenset[str] = frozenset({
    "antibiotics",
    "vte prophylaxis at transfer",
})


def _u_normalize_header(text: str) -> str:
    """Lowercased, colon-stripped header text for matching against known set."""
    return text.rstrip(":").strip().lower()


def _build_meds_markdown(content: str, citation_index: dict[str, Any]) -> str:
    """Pure markdown builder for U_unprescribing — no Streamlit dependency.

    See _render_meds_section for the rendering spec. Split out as a
    pure function so the structural rules (flatten, inline collapse,
    force-render VTE, never-collapse Antibiotics, unknown-header fall-
    through) are testable without a Streamlit runtime.
    """
    full_html = _cite_html(content, citation_index)
    raw_lines = content.splitlines()
    html_lines = full_html.splitlines()
    if len(html_lines) == len(raw_lines):
        html_by_raw = dict(zip(raw_lines, html_lines))
    else:
        html_by_raw = {line: _cite_html(line, citation_index) for line in raw_lines}

    def _line_html(raw_line: str) -> str:
        return html_by_raw.get(raw_line, _cite_html(raw_line, citation_index))

    groups: list[tuple[str, list[str]]] = []  # (header_label, body_html_lines)
    current_header: str | None = None
    current_body: list[str] = []

    def _flush() -> None:
        nonlocal current_header, current_body
        if current_header is not None:
            groups.append((current_header, current_body))
        current_header = None
        current_body = []

    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        head, sep, rest = stripped.partition(":")
        is_headerish = (
            sep == ":"
            and 0 < len(head.strip()) <= 80
            and len(head.split()) <= 8
        )
        normalized = _u_normalize_header(head) if is_headerish else ""

        if is_headerish and normalized in _U_SUPPRESS_PARENT_HEADERS:
            _flush()
            continue

        if is_headerish and normalized in _U_KNOWN_HEADERS:
            _flush()
            current_header = head.strip()
            inline = rest.strip()
            if inline:
                # Inline body case (e.g. "Changes to home meds: None").
                # Pull the post-colon HTML out of the pre-rendered full
                # line so per-section cite numbering is preserved.
                full_line_html = _line_html(raw_line)
                _h, _s, inline_html = full_line_html.partition(":")
                current_body.append(inline_html.lstrip())
            continue

        # Regular body line.
        body_html = _line_html(raw_line).lstrip()
        if current_header is None:
            # Content emitted before any recognized header — open a
            # synthetic group so the line still renders rather than
            # silently disappearing.
            current_header = stripped
            continue
        current_body.append(body_html)

    _flush()

    out_lines: list[str] = []
    for header, body in groups:
        normalized = _u_normalize_header(header)
        body_nonempty = [b for b in body if b.strip()]
        header_clean = header.rstrip(":").strip()

        if not body_nonempty and normalized not in _U_FORCE_RENDER_EMPTY:
            continue

        force_multiline = normalized in _U_NEVER_COLLAPSE
        if len(body_nonempty) == 1 and not force_multiline:
            out_lines.append(f"**{header_clean}:** {body_nonempty[0]}")
        else:
            out_lines.append(f"**{header_clean}**")
            for b in body_nonempty:
                out_lines.append(f"&emsp;{b}")

        out_lines.append("")

    while out_lines and out_lines[-1] == "":
        out_lines.pop()

    return "  \n".join(out_lines)


# U_uncertainty group headers — case-insensitive, line-start, two-word
# phrase, tolerant of leading whitespace, optional ** bold wrappers, and
# any descriptors before the colon. Match is against the whitespace-
# stripped line; partition on the first colon separates label from inline
# value. Update both the regex and the display label here when prompt
# wording shifts.
_U_UNCERTAINTY_HEADER_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "working_diagnosis",
        "Working diagnosis at the time of transfer",
        re.compile(r"^\*{0,2}\s*working\s+diagnosis\b", re.IGNORECASE),
    ),
    (
        "differential",
        "Differential includes",
        re.compile(r"^\*{0,2}\s*differential\s+(?:includes|diagnos)", re.IGNORECASE),
    ),
    (
        "less_likely",
        "Less likely",
        re.compile(r"^\*{0,2}\s*less\s+likely\b", re.IGNORECASE),
    ),
    (
        "pending_data",
        "Pending data (to confirm/exclude)",
        re.compile(r"^\*{0,2}\s*pending\s+data\b", re.IGNORECASE),
    ),
)

_U_UNCERTAINTY_BULLET_RE = re.compile(r"^[-*•]\s+")


def _build_uncertainty_markdown(
    content: str, citation_index: dict[str, Any]
) -> str:
    """Pure markdown builder for U_uncertainty — no Streamlit dependency.

    Parse-and-rebuild band-aid: the LLM's emitted U_uncertainty markdown
    is structurally chaotic (two-space-indented "Pending data:" gets
    rendered as a continuation of a preceding less-likely bullet; the
    cultures that follow become visual siblings of differential items, so
    a clinician scanning fast can misread "Urine culture" as a
    differential). Operating on already-emitted briefs, this renderer
    parses the four recognized group headers out of the raw text and
    re-emits them as flat one-indent-level groups. The durable fix is
    prompt-side, deferred until brief regeneration is available.

    Rules:
      * Header recognition: line-start match against the four phrases in
        ``_U_UNCERTAINTY_HEADER_PATTERNS``. Case-insensitive, tolerant
        of leading whitespace, optional ``**`` bold wrappers, and any
        descriptors before the colon.
      * Reassignment: every line is assigned to the most recently
        recognized header — this is how "Pending data" lifts out from
        under a sibling less-likely bullet regardless of how the LLM
        indented it.
      * Inline collapse: header line with post-colon content renders as
        ``**Label:** <value>``.
      * Multi-line list/prose: header followed by body lines renders as
        ``**Label:**`` then one indent-step body lines. Bullet markers
        are normalized; prose body lines render without a bullet.
      * Empty matched group: kept with an empty body so the reviewer can
        see the slot was emitted but blank.
      * Fallback: zero headers matched → fall through to the generic
        markdown path so a broken brief doesn't blank-screen.
      * Partial match (1-2 of 4 headers): render matched groups; any
        pre-first-header content renders as a leading prose block.
    """
    raw_lines = content.splitlines()
    full_html = _cite_html(content, citation_index)
    html_lines = full_html.splitlines()
    if len(html_lines) == len(raw_lines):
        html_by_raw = dict(zip(raw_lines, html_lines))
    else:
        html_by_raw = {
            line: _cite_html(line, citation_index) for line in raw_lines
        }

    def _line_html(raw_line: str) -> str:
        return html_by_raw.get(raw_line, _cite_html(raw_line, citation_index))

    def _match_header(stripped: str) -> tuple[str, str] | None:
        for key, label, pattern in _U_UNCERTAINTY_HEADER_PATTERNS:
            if pattern.match(stripped):
                return key, label
        return None

    groups: list[dict[str, Any]] = []
    orphan_body: list[tuple[bool, str]] = []
    current: dict[str, Any] | None = None
    seen_keys: set[str] = set()

    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        header_match = _match_header(stripped)
        if header_match is not None:
            key, label = header_match
            if key in seen_keys:
                # Duplicate header (LLM repeated itself) — append to the
                # existing group rather than opening a second one so the
                # rendered output stays one slot per kind.
                current = next(g for g in groups if g["key"] == key)
            else:
                current = {
                    "key": key,
                    "label": label,
                    "inline_html": "",
                    "body": [],  # list of (is_bullet, content_html)
                }
                groups.append(current)
                seen_keys.add(key)
            full_line_html = _line_html(raw_line)
            _h, sep, inline_html = full_line_html.partition(":")
            inline_text = inline_html.strip() if sep else ""
            if inline_text and not current["inline_html"]:
                current["inline_html"] = inline_text
            elif inline_text:
                # Already had inline; treat the new value as a body line.
                current["body"].append((False, inline_text))
            continue

        body_html = _line_html(raw_line).strip()
        if not body_html:
            continue
        bullet = _U_UNCERTAINTY_BULLET_RE.match(body_html)
        if bullet:
            body_html = body_html[bullet.end():].strip()
            is_bullet = True
        else:
            is_bullet = False
        target = orphan_body if current is None else current["body"]
        target.append((is_bullet, body_html))

    if not groups:
        # No recognized structure — fall through to generic markdown so
        # legacy or malformed briefs still render rather than blank-
        # screening. The dispatch logs the fall-through.
        return full_html.replace("\n", "  \n")

    out_lines: list[str] = []

    if orphan_body:
        for is_bullet, item in orphan_body:
            if is_bullet:
                out_lines.append(f"- {item}")
            else:
                out_lines.append(item)
        out_lines.append("")

    for group in groups:
        label = group["label"]
        inline = group["inline_html"]
        body = group["body"]

        if inline and not body:
            out_lines.append(f"**{label}:** {inline}")
        elif inline and body:
            out_lines.append(f"**{label}:** {inline}")
            for is_bullet, item in body:
                if is_bullet:
                    out_lines.append(f"&emsp;- {item}")
                else:
                    out_lines.append(f"&emsp;{item}")
        elif body:
            out_lines.append(f"**{label}:**")
            for is_bullet, item in body:
                if is_bullet:
                    out_lines.append(f"&emsp;- {item}")
                else:
                    out_lines.append(f"&emsp;{item}")
        else:
            # Header recognized but no content — keep the labeled slot
            # visible so the reviewer can see the agent emitted it blank.
            out_lines.append(f"**{label}:**")
        out_lines.append("")

    while out_lines and out_lines[-1] == "":
        out_lines.pop()

    return "  \n".join(out_lines)


def _render_uncertainty_section(
    content: str, citation_index: dict[str, Any]
) -> None:
    """Render U_uncertainty as flat 2-level groups (display-only).

    Parse-and-rebuild band-aid against chaotic LLM markdown for this
    section; durable fix is prompt-side, deferred until regeneration is
    available. See ``_build_uncertainty_markdown`` for the parse rules.
    """
    st.markdown(
        _build_uncertainty_markdown(content, citation_index),
        unsafe_allow_html=True,
    )


# E-section line classifier patterns. Walked in order: top header (most
# specific) → section header → bullet → label:value (split via
# _split_label_value_at_punct). Anything else falls through as plain text.
# Section header: capitalized words + colon, nothing after. Currently
# observed shapes: "Lines/drains/airways:", "Skin/wounds:".
# Label:value: capitalized words + ":" OR "?" + whitespace + content. The
# punctuation stays with the bold label ("**Difficult airway?** N").
# Dash separators ("Label - value") are NOT treated as label:value — they
# fall through as plain text, by spec.
_E_TOP_HEADER_RE = re.compile(r"^TRANSFER\s+EXAM\b", re.IGNORECASE)
_E_SECTION_HEADER_RE = re.compile(r"^[A-Z][A-Za-z/ -]+:$")
_E_LABEL_VALUE_RE = re.compile(r"^[A-Z][A-Za-z/ -]+[:?]\s+\S")
_E_BULLET_RE = re.compile(r"^[-*]\s+")

# E-section bucket router (rev 2 + addendum).
#
# Section E renders into FIVE fixed-order buckets, regardless of payload
# order. Each bucket's subheader text is template-mandated and is
# injected by the renderer (the upstream subheaders, if any, are dropped
# so the rendered text matches the official ICU pause template verbatim):
#
#   1. lead           — TRANSFER EXAM (Neuro, Vitals, Respiratory rows)
#   2. lda            — Lines / Drains / Airways
#                       (Devices: glance line → device bullets → ☐/☑ Y/N)
#   3. skin           — Skin/wounds
#   4. isolation      — Isolation precautions
#   5. positioning    — Positioning requirements and precautions
#                       (bullets + C-collar / spine precautions: label:value)
#
# Items that don't match any bucket route to an "unrouted" catch-all
# that renders at the very end with a per-item WARNING log. The
# field-name → bucket maps below are matched against
# ``_e_normalize_label_key(raw_label_text)`` so "Difficult airway?",
# "  difficult airway  ", and "DIFFICULT AIRWAY?" all normalize to
# "difficult airway".

_E_LEAD_LABEL_KEYS = frozenset({"neuro", "vitals", "respiratory"})

# Upstream "Lines/drains/airways:" section_header opens the LDA bucket.
# The header text itself is dropped (the LDA subheader is injected).
# "lines/drains" is the abbreviated variant the model sometimes emits for
# a SECOND device-inventory block; without it that block escapes to the
# unrouted catch-all and renders as its own stray section instead of
# merging into the one LDA block. (Slash-spacing variants like
# "Lines / Drains / Airways" are folded in by _e_normalize_label_key.)
_E_LDA_SECTION_KEYS = frozenset({"lines/drains/airways", "lines/drains"})
# Fixed Y/N rows that render INSIDE the LDA block with a ☐ / ☑ glyph.
_E_LDA_YN_LABEL_KEYS = frozenset({
    "difficult airway",
    "lines/drains assessed for removal",
})
# "Active lines/drains/airways" + "Lines/drains" route to the glance
# slot — rendered as a flush-left "Devices:" line at the TOP of the
# LDA block (per rev2 addendum: the field provides framing the
# bullets don't, so it is NOT dropped). The divergence check still
# runs and logs a WARNING when an Active item is absent from the
# device-bullet text.
_E_LDA_GLANCE_LABEL_KEYS = frozenset({
    "active lines/drains/airways",
    "lines/drains",
    # Inline "Lines/drains/airways: <description>" — the model sometimes puts
    # the LDA narrative after the colon instead of as a bare header + bullets;
    # route it to the glance slot rather than letting it escape to unrouted.
    "lines/drains/airways",
    # The pre-render consolidator emits its glance line as "Devices:"; recognize
    # it so consolidated brief.json re-parses to the same slot at display time.
    "devices",
})

# Skin/wounds bucket. Upstream may emit as a section_header + bullets
# OR as a label:value row — both shapes route here.
_E_SKIN_SECTION_KEYS = frozenset({"skin/wounds"})
_E_SKIN_LABEL_KEYS = frozenset({"skin/wounds", "skin", "wounds"})

# Isolation precautions bucket. Upstream commonly uses a flat
# "Isolation: ..." label:value, which the renderer converts to a
# single bullet under the injected "Isolation precautions" subheader.
_E_ISOLATION_SECTION_KEYS = frozenset({"isolation precautions"})
_E_ISOLATION_LABEL_KEYS = frozenset({
    "isolation",
    "isolation precautions",
})

# Positioning bucket. The named related fields (mobility level,
# activity restrictions, device positioning notes) all surface here
# as bullets under the injected subheader. "C-collar / spine
# precautions" stays as a label:value row (no bullet, no glyph) per
# the spec's example output.
_E_POSITIONING_SECTION_KEYS = frozenset({
    "positioning requirements and precautions",
})
_E_POSITIONING_LABEL_KEYS = frozenset({
    "positioning",
    "positioning requirements and precautions",
    "positioning/mobility precautions",
    "current mobility level",
    "activity restrictions",
    "device positioning notes",
})
_E_POSITIONING_LABEL_VALUE_KEYS = frozenset({
    "c-collar / spine precautions",
    "c-collar/spine precautions",
})

# Subheader text per bucket — template-mandated. The lead bucket's
# header is "TRANSFER EXAM" (synthesized if upstream omitted it).
_E_LEAD_SUBHEADER = "TRANSFER EXAM"
_E_LDA_SUBHEADER = "Lines / Drains / Airways"
_E_SKIN_SUBHEADER = "Skin/wounds"
_E_ISOLATION_SUBHEADER = "Isolation precautions"
_E_POSITIONING_SUBHEADER = "Positioning requirements and precautions"

# Backwards-compat alias for the old single-subheader constant. Tests
# and downstream callers may still import this name; kept so renaming
# the public symbol doesn't cascade into a separate breaking change.
_E_TEMPLATE_SUBHEADER = _E_LDA_SUBHEADER

# Checkbox glyph map for the two fixed Y/N rows. Matched case-insensitively
# on the first whitespace-separated token of the value (with trailing
# punctuation stripped) — so "No.", "no", "N", "negative" all map to ☐.
# Anything outside both sets renders without a glyph and logs a warning,
# per the spec's "unhandled case is visible" requirement.
_E_GLYPH_NEGATIVE = frozenset({"n", "no", "false", "neg", "negative"})
_E_GLYPH_POSITIVE = frozenset({"y", "yes", "true", "pos", "positive"})
_E_GLYPH_BOX_EMPTY = "☐"
_E_GLYPH_BOX_CHECK = "☑"


def _split_label_value_at_punct(text: str) -> tuple[int, str, str]:
    """Partition *text* at the first ':' or '?' (whichever comes first).

    Returns ``(punct_index, label_with_punct, value)``. Caller must have
    validated *text* against ``_E_LABEL_VALUE_RE`` so at least one of
    ":" or "?" is present after a valid label.
    """
    colon_idx = text.find(":")
    qmark_idx = text.find("?")
    candidates = [i for i in (colon_idx, qmark_idx) if i != -1]
    idx = min(candidates)
    return idx, text[: idx + 1], text[idx + 1:].lstrip()


def _strip_html_tags(html: str) -> str:
    """Remove HTML tags for normalization/comparison — not for display."""
    return re.sub(r"<[^>]+>", "", html)


def _e_normalize_label_key(text: str) -> str:
    """Lowercased, ':'/'?' stripped, whitespace-collapsed label key.

    Used to compare a raw label string against the template routing sets.
    Strips whitespace BEFORE removing trailing punctuation so labels with
    incidental trailing whitespace ("Foo : ") still normalize correctly.
    """
    cleaned = text.strip().rstrip(":?").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Fold spacing around slashes so "Lines / Drains / Airways" matches the
    # canonical "lines/drains/airways" routing key.
    cleaned = re.sub(r"\s*/\s*", "/", cleaned)
    return cleaned.lower()


def _e_active_item_set(value_html: str) -> set[str]:
    """Tokenize an Active line value into a normalized set for subset compare."""
    raw = _strip_html_tags(value_html).lower()
    return {
        re.sub(r"\s+", " ", item).strip()
        for item in re.split(r"[;,]", raw)
        if item.strip()
    }


def _e_glance_value_html(items: list[tuple[str, str]]) -> str | None:
    """Merge multiple Active/Lines-drains candidates into one Devices: value.

    *items* is a list of ``(label_key, value_html)``. The spec's addendum
    expects a single Devices: glance line; when the upstream emits more
    than one candidate they get joined with ``"; "`` so no field is
    silently dropped. Returns ``None`` if no candidates.
    """
    if not items:
        return None
    if len(items) == 1:
        return items[0][1]
    return "; ".join(value_html for _, value_html in items)


def _e_active_items_not_in_bullets(
    glance_value_html: str, bullet_body_htmls: list[str]
) -> list[str]:
    """Return Active items whose text is NOT covered by any device bullet.

    "Covered" = the item's whitespace-collapsed lowercase plain-text form
    appears as a substring of the joined-lowercase device-bullet text.
    Returned items are plain text (HTML stripped) so the divergence-
    warning log can name them. The check is intentionally loose: a clean
    summary like "peripheral IV, urinary catheter" with bullets that
    say "Peripheral IV x2 documented: ..." matches because "peripheral
    iv" is a substring of "peripheral iv x2 documented: ...".
    """
    if not bullet_body_htmls:
        return [
            re.sub(r"\s+", " ", item.strip())
            for item in re.split(r"[;,]", _strip_html_tags(glance_value_html))
            if item.strip()
        ]
    bullets_text = " ".join(
        _strip_html_tags(b) for b in bullet_body_htmls
    ).lower()
    bullets_text = re.sub(r"\s+", " ", bullets_text)
    items = [
        re.sub(r"\s+", " ", item.strip())
        for item in re.split(r"[;,]", _strip_html_tags(glance_value_html))
        if item.strip()
    ]
    return [item for item in items if item.lower() not in bullets_text]


def _e_checkbox_glyph(value_html: str) -> str | None:
    """Map first token of a Y/N row's value to ☐/☑, or None for unhandled.

    Trailing punctuation on the first token is stripped before lookup
    ("No." → "no"). Empty values and tokens outside both glyph sets log
    a WARNING and return ``None`` so the caller can fall back to plain
    label:value rendering and the spec's "no glyph, render as plain
    label + value with a log line" rule is satisfied.
    """
    raw = _strip_html_tags(value_html).strip()
    if not raw:
        _logger.warning(
            "E-section Y/N row unhandled value: empty (no glyph rendered)"
        )
        return None
    first = raw.split(None, 1)[0]
    first_clean = re.sub(r"[.,;:?!—–-]+$", "", first).lower()
    if first_clean in _E_GLYPH_NEGATIVE:
        return _E_GLYPH_BOX_EMPTY
    if first_clean in _E_GLYPH_POSITIVE:
        return _E_GLYPH_BOX_CHECK
    _logger.warning(
        "E-section Y/N row unhandled value (no glyph rendered): %r", raw
    )
    return None


def _build_exam_markdown(content: str, citation_index: dict[str, Any]) -> str:
    """Pure HTML builder for E section — no Streamlit dependency.

    Three-pass classify → route → render against a FIXED 5-bucket layout
    (lead / lda / skin / isolation / positioning) that overrides upstream
    payload order. The bucket subheaders are injected by the renderer;
    upstream subheaders, if any, are dropped.

      Pass 1 — classify each non-blank line into a typed item
        (top_header / section_header / bullet / label_value / plain).

      Pass 2 — route each item into one of the 5 fixed buckets via
        ``_E_*_LABEL_KEYS`` / ``_E_*_SECTION_KEYS`` (case-insensitive,
        whitespace-and-punct-normalized via ``_e_normalize_label_key``).
        Bullets attach to the most recent section_header so a bullet
        emitted under "Lines/drains/airways:" routes to the LDA bucket
        even if an orphan label:value intervenes. Items that don't
        match any bucket route to an "unrouted" catch-all that renders
        at the end with a per-item WARNING log.

      Pass 3 — render buckets in their fixed order. The LDA block has
        a specific internal order per the rev2 addendum:
          1. Devices: glance line (from "Active lines/drains/airways"
             and/or "Lines/drains" values — joined with "; " if both
             are present). Flush-left under the subheader, not bullet-
             indented, so it reads as an orientation row.
          2. Device bullets (from the bullets that followed the
             "Lines/drains/airways:" section_header upstream).
          3. ☐ / ☑ Y/N rows for "Difficult airway?" and "Lines/drains
             assessed for removal?". Unhandled Y/N values fall back to
             plain label:value rendering and log a WARNING.

        Divergence between the glance value and the device bullets
        is still flagged as a WARNING (per spec): "Active items not in
        bullets: X". The glance line still renders (rev2-addendum
        reversal of rev2's drop rule).

        Skin / Isolation / Positioning buckets render as subheader +
        bullets. If upstream emitted a label:value (e.g., "Isolation:
        ..." as a single row), the renderer synthesizes a single bullet
        from the VALUE — the label is implied by the subheader. The
        special "C-collar / spine precautions:" key stays as a label:
        value row inside the Positioning bucket (no bullet, no glyph).

    Spacing reuses the S-section primitives so the cross-section visual
    rhythm is one decision in one place: tight header→first-child pair,
    looser gap before each new bucket header.

    Renderer-only. The scribe-field-as-source-of-truth contract is
    preserved — no field content is altered, only its presentation and
    placement. Durable fixes for content ambiguity are prompt-side.
    """
    raw_lines = content.splitlines()
    full_html = _cite_html(content, citation_index)
    html_lines = full_html.splitlines()
    if len(html_lines) == len(raw_lines):
        html_by_raw = dict(zip(raw_lines, html_lines))
    else:
        html_by_raw = {
            line: _cite_html(line, citation_index) for line in raw_lines
        }

    def _line_html(raw_line: str) -> str:
        return html_by_raw.get(raw_line, _cite_html(raw_line, citation_index))

    # ----- Pass 1: classify -------------------------------------------------
    items: list[dict[str, Any]] = []
    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        line_html = _line_html(raw_line).strip()

        if _E_TOP_HEADER_RE.match(stripped):
            items.append({"kind": "top_header", "html": line_html})
            continue

        if _E_SECTION_HEADER_RE.match(stripped):
            text = stripped.rstrip(":").strip()
            items.append({
                "kind": "section_header",
                "text": text,
                "key": _e_normalize_label_key(text),
            })
            continue

        bullet_match = _E_BULLET_RE.match(stripped)
        if bullet_match:
            body_raw = stripped[bullet_match.end():].strip()
            body_html = line_html
            bullet_html = _E_BULLET_RE.match(body_html)
            if bullet_html:
                body_html = body_html[bullet_html.end():].strip()
            # If the bullet body is itself "Inner label: value", bold the
            # inner label so the bullet stays scannable.
            if _E_LABEL_VALUE_RE.match(body_raw):
                colon_html = body_html.find(":")
                qmark_html = body_html.find("?")
                html_candidates = [c for c in (colon_html, qmark_html) if c != -1]
                if html_candidates:
                    html_idx = min(html_candidates)
                    bullet_label_html = body_html[: html_idx + 1]
                    bullet_value_html = body_html[html_idx + 1:].lstrip()
                    body_html = (
                        f"<strong>{bullet_label_html}</strong> "
                        f"{bullet_value_html}"
                    )
            items.append({"kind": "bullet", "body_html": body_html})
            continue

        if _E_LABEL_VALUE_RE.match(stripped):
            idx, _label_raw, _value_raw = _split_label_value_at_punct(stripped)
            label_text_raw = stripped[: idx].strip()

            colon_html = line_html.find(":")
            qmark_html = line_html.find("?")
            html_candidates = [c for c in (colon_html, qmark_html) if c != -1]
            html_idx = min(html_candidates)
            label_html = line_html[: html_idx + 1]
            value_html = line_html[html_idx + 1:].lstrip()

            items.append({
                "kind": "label_value",
                "label_html": label_html,
                "value_html": value_html,
                "key": _e_normalize_label_key(label_text_raw),
            })
            continue

        items.append({"kind": "plain", "html": line_html})

    if not items:
        # Empty or unparseable — fall through to generic markdown so a
        # broken brief still renders rather than blank-screens.
        return full_html.replace("\n", "  \n")

    # ----- Pass 2: route into the 5 fixed buckets ---------------------------
    top_header_html: str | None = None
    lead_rows: list[dict[str, Any]] = []
    lda_devices: list[str] = []                       # bullet body HTMLs
    lda_yn: list[tuple[str, str]] = []                # (label_html, value_html)
    lda_glance: list[tuple[str, str]] = []            # (label_key, value_html)
    skin_items: list[dict[str, Any]] = []
    isolation_items: list[dict[str, Any]] = []
    positioning_items: list[dict[str, Any]] = []
    positioning_labels: list[tuple[str, str]] = []    # (label_html, value_html)
    unrouted: list[dict[str, Any]] = []
    current_section_bucket: str | None = None         # "lda" / "skin" / "isolation" / "positioning"

    def _route_bullet_or_plain(item: dict[str, Any]) -> None:
        if current_section_bucket == "lda":
            if item["kind"] == "bullet":
                lda_devices.append(item["body_html"])
            else:
                # Plain text under the LDA section_header — surface as
                # an unrouted item so it's visible without polluting the
                # device-bullet list.
                unrouted.append(item)
        elif current_section_bucket == "skin":
            skin_items.append(item)
        elif current_section_bucket == "isolation":
            isolation_items.append(item)
        elif current_section_bucket == "positioning":
            positioning_items.append(item)
        else:
            unrouted.append(item)

    for it in items:
        kind = it["kind"]
        if kind == "top_header":
            top_header_html = it["html"]
            continue
        if kind == "section_header":
            key = it["key"]
            if key in _E_LDA_SECTION_KEYS:
                current_section_bucket = "lda"
                continue  # subheader text dropped; LDA subheader injected
            if key in _E_SKIN_SECTION_KEYS:
                current_section_bucket = "skin"
                continue
            if key in _E_ISOLATION_SECTION_KEYS:
                current_section_bucket = "isolation"
                continue
            if key in _E_POSITIONING_SECTION_KEYS:
                current_section_bucket = "positioning"
                continue
            # Unrecognized section_header — render as-is at end via
            # unrouted catch-all (with a WARNING log later).
            current_section_bucket = None
            unrouted.append(it)
            continue
        if kind == "label_value":
            key = it["key"]
            if key in _E_LEAD_LABEL_KEYS:
                lead_rows.append(it)
                continue
            if key in _E_LDA_YN_LABEL_KEYS:
                lda_yn.append((it["label_html"], it["value_html"]))
                continue
            if key in _E_LDA_GLANCE_LABEL_KEYS:
                lda_glance.append((key, it["value_html"]))
                continue
            if key in _E_SKIN_LABEL_KEYS:
                skin_items.append(it)
                continue
            if key in _E_ISOLATION_LABEL_KEYS:
                isolation_items.append(it)
                continue
            if key in _E_POSITIONING_LABEL_VALUE_KEYS:
                positioning_labels.append((it["label_html"], it["value_html"]))
                continue
            if key in _E_POSITIONING_LABEL_KEYS:
                positioning_items.append(it)
                continue
            unrouted.append(it)
            continue
        # bullet or plain — route by most-recent section_header.
        _route_bullet_or_plain(it)

    # Divergence detection on the LDA glance line vs the device bullets.
    # Per spec addendum: still log a WARNING if Active content names a
    # device the bullets don't, but DO NOT drop the glance line.
    for label_key, value_html in lda_glance:
        unique = _e_active_items_not_in_bullets(value_html, lda_devices)
        if unique:
            _logger.warning(
                "E-section LDA glance divergence: %s lists items absent "
                "from device bullets: %r — rendering Devices: line "
                "anyway (no silent drop)",
                label_key, unique,
            )

    # Surface unrouted content as a WARNING per item so a stray label
    # or a renamed prompt field doesn't disappear silently.
    for it in unrouted:
        label = (
            it.get("key")
            or it.get("text")
            or _strip_html_tags(it.get("html", ""))[:80]
            or it.get("body_html", "")[:80]
        )
        _logger.warning(
            "E-section unrouted item (kind=%s key/text=%r) — "
            "rendered as fallback at end of section",
            it["kind"], label,
        )

    # ----- Pass 3: render in fixed bucket order -----------------------------
    out_parts: list[str] = []
    is_first = True

    def _emit_header(text: str) -> None:
        nonlocal is_first
        top_margin = 0 if is_first else _S_PROBLEM_BLOCK_GAP_REM
        out_parts.append(
            f"<div style='margin-top:{top_margin}rem;"
            f"margin-bottom:{_S_HEADER_BODY_GAP_REM}rem;'>"
            f"<strong>{text}</strong></div>"
        )
        is_first = False

    def _emit_html(html: str) -> None:
        nonlocal is_first
        out_parts.append(html)
        is_first = False

    # 1. Lead block — TRANSFER EXAM + Neuro/Vitals/Respiratory rows.
    if top_header_html or lead_rows:
        # Use upstream's top header text verbatim when present (so any
        # parenthetical the LLM added survives); otherwise synthesize.
        _emit_header(top_header_html if top_header_html else _E_LEAD_SUBHEADER)
        for it in lead_rows:
            _emit_html(
                f"<div><strong>{it['label_html']}</strong> "
                f"{it['value_html']}</div>"
            )

    # 2. Lines / Drains / Airways block — Devices: glance → bullets → ☐/☑.
    glance_value_html = _e_glance_value_html(lda_glance)
    has_lda_content = bool(glance_value_html or lda_devices or lda_yn)
    if has_lda_content:
        _emit_header(_E_LDA_SUBHEADER)
        if glance_value_html:
            # Flush-left glance row (no &emsp; indent), per the spec
            # addendum: "Devices: line sits flush-left under the
            # subheader (not bullet-indented), so it visually reads as
            # a topic line rather than another bullet."
            _emit_html(
                f"<div><strong>Devices:</strong> {glance_value_html}</div>"
            )
        for body_html in lda_devices:
            _emit_html(f"<div>&emsp;• {body_html}</div>")
        for label_html, value_html in lda_yn:
            glyph = _e_checkbox_glyph(value_html)
            if glyph is None:
                _emit_html(
                    f"<div>&emsp;<strong>{label_html}</strong> "
                    f"{value_html}</div>"
                )
            else:
                _emit_html(
                    f"<div>&emsp;{glyph} <strong>{label_html}</strong> "
                    f"{value_html}</div>"
                )

    # 3. Skin/wounds block. Upstream label:value rows synthesize as a
    #    single bullet (the label is implied by the subheader); upstream
    #    bullets pass through.
    def _emit_subheader_bucket(
        subheader: str,
        items_list: list[dict[str, Any]],
        extra_label_rows: list[tuple[str, str]] | None = None,
    ) -> None:
        if not items_list and not extra_label_rows:
            return
        _emit_header(subheader)
        for it in items_list:
            if it["kind"] == "bullet":
                _emit_html(f"<div>&emsp;• {it['body_html']}</div>")
            elif it["kind"] == "label_value":
                # Drop the label, use the VALUE as the bullet body —
                # the subheader carries the label's meaning.
                _emit_html(f"<div>&emsp;• {it['value_html']}</div>")
            else:  # plain
                _emit_html(f"<div>&emsp;{it['html']}</div>")
        if extra_label_rows:
            for label_html, value_html in extra_label_rows:
                _emit_html(
                    f"<div>&emsp;<strong>{label_html}</strong> "
                    f"{value_html}</div>"
                )

    _emit_subheader_bucket(_E_SKIN_SUBHEADER, skin_items)
    _emit_subheader_bucket(_E_ISOLATION_SUBHEADER, isolation_items)
    _emit_subheader_bucket(
        _E_POSITIONING_SUBHEADER,
        positioning_items,
        extra_label_rows=positioning_labels,
    )

    # 6. Unrouted catch-all — render at the very end, no subheader, so
    #    nothing disappears even when a prompt change introduces a new
    #    field key. The per-item WARNING log was emitted above.
    for it in unrouted:
        kind = it["kind"]
        if kind == "section_header":
            # Rendered without margin trick — just bold, no group gap,
            # to make obvious this is a fallback path. Don't crash on
            # missing keys.
            text = it.get("text", "")
            if text:
                _emit_html(f"<div><strong>{text}</strong></div>")
        elif kind == "label_value":
            _emit_html(
                f"<div><strong>{it['label_html']}</strong> "
                f"{it['value_html']}</div>"
            )
        elif kind == "bullet":
            _emit_html(f"<div>&emsp;• {it['body_html']}</div>")
        else:
            _emit_html(f"<div>{it['html']}</div>")

    return "".join(out_parts)


def _render_exam_section(content: str, citation_index: dict[str, Any]) -> None:
    """Render the E section with two-weight typography (display-only).

    Parse-and-rebuild band-aid: the generic markdown path renders the
    E section as a wall of identical-weight lines, leaving the bottom
    assessment block (Current mobility level / Positioning / Activity
    restrictions / Difficult airway? / Lines assessed / Active lines)
    unscannable. This renderer classifies each line and emits headers
    as bold-strong blocks with a wider top margin, label:value rows
    with the label portion bolded, and bullets at one consistent indent.
    See ``_build_exam_markdown`` for the classifier rules.

    Durable fix is prompt-side, deferred until brief regeneration is
    available.
    """
    st.markdown(
        _build_exam_markdown(content, citation_index),
        unsafe_allow_html=True,
    )


def _render_meds_section(content: str, citation_index: dict[str, Any]) -> None:
    """Render U_unprescribing as flat 2-level groups (display-only).

    Payload structure (see config/prompts/pharmacy.yaml:25-54) wraps
    five anticoagulation sub-headers under a parent `Anticoagulation:`
    header, producing three visual indent levels. This renderer drops
    the parent at display time and promotes the five children to top-
    level groups, preserving the payload labels verbatim so reviewers
    can cross-reference rendered output against raw section text
    without wondering whether a label was rewritten in transit.

    Layout rules:
      * Group header: bold, no trailing colon when introducing a
        multi-item body.
      * Single body line: collapses to "**Header:** body" inline.
      * Body lines: regular weight, one consistent indent step.
      * Empty groups: omitted, except VTE prophylaxis at transfer.
      * Antibiotics and VTE prophylaxis never collapse.

    Header recognition is explicit against `_U_KNOWN_HEADERS`. Unknown
    headerish lines (e.g. an off-spec `History:` line under Antibiotics)
    are treated as body of the current group rather than starting a
    new one — the contract with the prompt stays visible, and drift
    surfaces as visibly-misplaced content rather than silent indent
    chaos.
    """
    st.markdown(
        _build_meds_markdown(content, citation_index),
        unsafe_allow_html=True,
    )


def _render_s_section(content: str, citation_index: dict[str, Any]) -> None:
    """Render the S section with #Problem: headers and grouped to-do items."""
    lines = content.splitlines()
    current_problem: str | None = None
    current_body: list[str] = []
    problems: list[tuple[str, list[str]]] = []  # (header, body_lines)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            # Save previous problem
            if current_problem is not None:
                problems.append((current_problem, current_body))
            # New problem header. Handles two forms:
            #   "#Problem title:"            (header alone, body on next lines)
            #   "#Problem title: body text"  (inline body — split at first ":")
            # The inline form is what _normalize_s_format produces when the
            # LLM writes prose; without splitting, the whole line ends up
            # bolded as the header and citations skip rendering.
            header_text = stripped.lstrip("#").strip()
            title, _sep, rest = header_text.partition(":")
            current_problem = title.strip()
            current_body = []
            rest = rest.strip()
            if rest:
                current_body.append(rest)
        else:
            current_body.append(stripped)

    if current_problem is not None:
        problems.append((current_problem, current_body))

    if not problems:
        # No # structure -- render as-is (supports markdown bullets and checkboxes).
        # Citation rendering runs on the whole block so numbering is per-section.
        st.markdown(_cite_html(content, citation_index), unsafe_allow_html=True)
        return

    # Run the canonical renderer over the full S-section text ONCE so numbering
    # resets at the section boundary, not per problem block. Then split the
    # resulting HTML back on line boundaries.  The `<sup>` tags never contain
    # newlines, so splitlines is safe.
    full_html = _cite_html(content, citation_index)
    html_by_raw_line: dict[str, str] = {}
    for raw_line, html_line in zip(lines, full_html.splitlines()):
        html_by_raw_line[raw_line.strip()] = html_line.strip()

    for header, body_lines in problems:
        # Build each problem (header + body) as a single HTML container
        # so the header→body gap is controlled separately from the gap
        # between problems. Emitting via st.markdown breaks each call
        # into a paragraph with identical rhythm — that's the visual
        # ambiguity this renderer is fixing.
        body_chunks: list[str] = []
        for bline in body_lines:
            # Detect to-do timing buckets (e.g. "[] BEFORE TRANSFER")
            clean = bline.lstrip("☐[]- ").strip()
            bucket_label = None
            bucket_key = None
            for bkey, bdisp in _TODO_BUCKET_LABELS.items():
                if clean.upper().startswith(bkey):
                    bucket_label = bdisp
                    bucket_key = bkey
                    remainder = clean[len(bkey):].strip(": ").strip()
                    break
            if bucket_label:
                items = [i.strip() for i in remainder.split(" / ") if i.strip()]
                body_chunks.append(
                    f"<div>&emsp;<em>{bucket_label}:</em></div>"
                )
                for item in items:
                    # Re-run per-item HTML so item-level tags pick up
                    # section-level numbering when available.
                    item_html = html_by_raw_line.get(
                        f"[] {bucket_key}: {' / '.join(items)}",
                        item,
                    )
                    if not item_html or "<sup" not in item_html:
                        item_html = _cite_html(item, citation_index)
                    body_chunks.append(
                        f"<div>&emsp;&emsp;- {item_html}</div>"
                    )
            else:
                bline_html = html_by_raw_line.get(bline, _cite_html(bline, citation_index))
                # " / " separator → soft break inside the body div.
                formatted_line = bline_html.replace(" / ", "<br>")
                body_chunks.append(f"<div>{formatted_line}</div>")

        header_html = (
            f"<div style='margin-bottom:{_S_HEADER_BODY_GAP_REM}rem;'>"
            f"<strong>{header}</strong>"
            f"</div>"
        )
        if body_chunks:
            body_html = (
                f"<div style='margin-top:0;'>{''.join(body_chunks)}</div>"
            )
        else:
            # Header with no body (edge case: two consecutive #headers).
            # Skip the body div entirely rather than emit empty whitespace,
            # but keep the inter-problem gap on the container below.
            body_html = ""

        st.markdown(
            f"<div style='margin-bottom:{_S_PROBLEM_BLOCK_GAP_REM}rem;'>"
            f"{header_html}{body_html}"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_note(output: dict[str, Any]) -> None:
    """Render an ICUPauseOutput dict as expandable sections in Streamlit."""

    hosp_id = output.get("hospitalization_id", "Unknown")
    generated_at = output.get("generated_at", "")
    st.markdown(f"**Hospitalization:** `{hosp_id}`")
    if generated_at:
        st.caption(f"Generated: {generated_at}")

    # Citation index may be absent on legacy outputs generated before the
    # hoverable-footnotes rollout; fall back to empty so tags render as
    # unverified (the canonical module handles the missing-entry path).
    citation_index = output.get("metadata", {}).get("citation_index", {}) or {}

    # QA issues and Warnings render BEFORE the I section so reviewers see
    # pipeline-flagged concerns before reading the note itself.
    qa_issues = output.get("qa_issues", [])
    if qa_issues:
        with st.expander("QA Issues", expanded=True):
            for issue in qa_issues:
                st.warning(issue)

    raw_warnings = output.get("warnings", [])
    visible_warnings = render_warnings_for_clinician(raw_warnings)
    if visible_warnings:
        with st.expander("Warnings", expanded=True):
            for w in visible_warnings:
                sev = w.get("severity", "info")
                cat = w.get("category", "safety_flag")
                badge = (
                    f"**[{SEVERITY_LABEL.get(sev, sev.upper())} \u00b7 "
                    f"{CATEGORY_LABEL.get(cat, cat)}]** "
                )
                msg = w.get("message", "")
                if sev == "safety_critical":
                    st.error(badge + msg)
                elif sev == "clinical":
                    st.warning(badge + msg)
                else:
                    st.info(badge + msg)
    audit = audit_only_warnings(raw_warnings)
    if audit:
        with st.expander(f"Audit-only warnings ({len(audit)})", expanded=False):
            st.caption(
                "Editorial revisions and pipeline-internal notes. Not shown to "
                "the clinician at the bedside; surfaced here for prompt tuning."
            )
            for w in audit:
                cat = CATEGORY_LABEL.get(w.get("category", "qa_process"), "QA Process")
                st.markdown(f"- **{cat}** — {w.get('message', '')}")

    sections = output.get("sections", {})
    for key, letter, label in SECTION_ORDER:
        content = sections.get(key, "")
        title = f"**{letter}** — {label}"
        with st.expander(title, expanded=True):
            if content and content.strip():
                if key == "S":
                    _render_s_section(content, citation_index)
                elif key == "U_unprescribing":
                    _render_meds_section(content, citation_index)
                elif key == "U_uncertainty":
                    _render_uncertainty_section(content, citation_index)
                elif key == "E":
                    _render_exam_section(content, citation_index)
                else:
                    html = _cite_html(content, citation_index)
                    formatted = html.replace(" / ", "  \n").replace("\n", "  \n")
                    st.markdown(formatted, unsafe_allow_html=True)
            else:
                st.caption("No content generated for this section.")

    # To-Do checklist (grouped by temporal bucket).
    # Items are dicts {"bucket": ..., "text": ...}; legacy outputs may be plain
    # strings — normalize to ward_ongoing so they still render.
    todos = output.get("todo_checklist", [])
    if todos:
        normalized = [
            t if isinstance(t, dict) else {"bucket": "ward_ongoing", "text": t}
            for t in todos
        ]
        with st.expander("**To-Do List**", expanded=True):
            buckets: dict[str, list[str]] = {b: [] for b in _BUCKET_ORDER}
            for item in normalized:
                b = item.get("bucket", "ward_ongoing")
                if b not in buckets:
                    b = "ward_ongoing"
                buckets[b].append(item.get("text", ""))
            for bkey in _BUCKET_ORDER:
                items = buckets[bkey]
                if not items:
                    continue
                st.markdown(f"**{_BUCKET_LABELS[bkey]}**")
                for i, text in enumerate(items, 1):
                    item_html = _cite_html(text, citation_index)
                    # Checkbox + number prefix: "☐" reads as actionable, the
                    # number gives a stable reference for reviewer feedback.
                    # Indented with &emsp; so the row hangs neatly under the
                    # bucket header. unsafe_allow_html=True is required for
                    # both the ☐ glyph and any <sup> citation markers.
                    st.markdown(
                        f"&emsp;☐ {i}. {item_html}",
                        unsafe_allow_html=True,
                    )
