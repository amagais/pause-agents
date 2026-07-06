"""Section E (Exam at Transfer — Lines/Drains/Airways & Data Review) consolidator.

Pre-render consolidator for Section E. The reviewer-app router re-buckets E
into a FIXED 5-bucket layout at DISPLAY time (review_app/display/
note_renderer.py:_build_exam_markdown), so the brief.json itself could carry a
fragmented/duplicated E (e.g. a stray "Lines/drains/airways:" block separate
from the fixed checkbox bucket, or the model emitting the LDA description
inline after the colon). This module performs the SAME classify→route→reorder
but emits canonical TEXT, so the consolidation lands in brief.json BEFORE it's
stored — and the reviewer-app router then re-parses the already-consolidated
text idempotently.

Routing rules (bucket key-maps + normalization) are kept in sync with the
reviewer app so the two can't diverge in *behavior*; the only difference is the
emit (text here, HTML there). The routing is content-preserving: no field text
is altered, only its placement/order (the scribe-source-of-truth contract).

Self-idempotency: the emitted glance line uses a "Devices:" label that is
itself a recognized glance key, so consolidate(consolidate(x)) == consolidate(x).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# --- classify regexes (mirror note_renderer.py) ---------------------------
_E_TOP_HEADER_RE = re.compile(r"^TRANSFER\s+EXAM\b", re.IGNORECASE)
_E_SECTION_HEADER_RE = re.compile(r"^[A-Z][A-Za-z/ -]+:$")
_E_LABEL_VALUE_RE = re.compile(r"^[A-Z][A-Za-z/ -]+[:?]\s+\S")
_E_BULLET_RE = re.compile(r"^[-*]\s+")

# --- bucket key-maps (kept in sync with note_renderer.py) -----------------
_E_LEAD_LABEL_KEYS = frozenset({"neuro", "vitals", "respiratory"})
# Bare "Lines/drains/airways:" / "Lines/drains:" section_headers open the LDA
# bucket. "lines/drains" is the abbreviated second-block header the model
# sometimes emits; without it that block escapes to the unrouted catch-all.
_E_LDA_SECTION_KEYS = frozenset({"lines/drains/airways", "lines/drains"})
_E_LDA_YN_LABEL_KEYS = frozenset({
    "difficult airway",
    "lines/drains assessed for removal",
})
# Inline label:value forms that route to the glance ("Devices:") slot.
# "lines/drains/airways" catches the case where the model writes the LDA
# description inline after the colon instead of as a bare header + bullets.
# "devices" makes the consolidator's OWN emitted glance line re-parse to the
# same slot (self-idempotency).
_E_LDA_GLANCE_LABEL_KEYS = frozenset({
    "active lines/drains/airways",
    "lines/drains",
    "lines/drains/airways",
    "devices",
})
_E_SKIN_SECTION_KEYS = frozenset({"skin/wounds"})
_E_SKIN_LABEL_KEYS = frozenset({"skin/wounds", "skin", "wounds"})
_E_ISOLATION_SECTION_KEYS = frozenset({"isolation precautions"})
_E_ISOLATION_LABEL_KEYS = frozenset({"isolation", "isolation precautions"})
_E_POSITIONING_SECTION_KEYS = frozenset({"positioning requirements and precautions"})
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

# Canonical TEXT headers the consolidator emits. These are the SECTION-HEADER
# forms the reviewer-app router recognizes (it drops them and injects its
# display subheaders), so re-parsing consolidated text is idempotent.
_LDA_HEADER = "Lines/drains/airways:"
_SKIN_HEADER = "Skin/wounds:"
_ISOLATION_HEADER = "Isolation precautions:"
_POSITIONING_HEADER = "Positioning requirements and precautions:"
_LEAD_HEADER = "TRANSFER EXAM"


def _normalize_label_key(text: str) -> str:
    cleaned = text.strip().rstrip(":?").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Fold spacing around slashes so "Lines / Drains / Airways" matches the
    # canonical "lines/drains/airways" routing key.
    cleaned = re.sub(r"\s*/\s*", "/", cleaned)
    return cleaned.lower()


def _split_label_value_at_punct(text: str) -> tuple[str, str]:
    colon_idx = text.find(":")
    qmark_idx = text.find("?")
    candidates = [i for i in (colon_idx, qmark_idx) if i != -1]
    idx = min(candidates)
    return text[: idx + 1], text[idx + 1:].lstrip()


def consolidate_e_section(content: str) -> str:
    """Re-bucket a Section E payload into the fixed 5-bucket order, as text.

    Returns the consolidated text. Idempotent: consolidating already-
    consolidated text yields the same result. Unrouted lines are preserved at
    the end (never silently dropped) with a per-item WARNING, matching the
    reviewer app's no-silent-drop contract.
    """
    if not content or not content.strip():
        return content

    raw_lines = content.splitlines()

    # ----- Pass 1: classify (text, not HTML) -----
    items: list[dict] = []
    for raw in raw_lines:
        s = raw.strip()
        if not s:
            continue
        if _E_TOP_HEADER_RE.match(s):
            items.append({"kind": "top_header", "text": s})
            continue
        if _E_SECTION_HEADER_RE.match(s):
            text = s.rstrip(":").strip()
            items.append({"kind": "section_header", "key": _normalize_label_key(text)})
            continue
        bm = _E_BULLET_RE.match(s)
        if bm:
            items.append({"kind": "bullet", "body": s[bm.end():].strip()})
            continue
        if _E_LABEL_VALUE_RE.match(s):
            label, value = _split_label_value_at_punct(s)
            items.append({
                "kind": "label_value",
                "label": label, "value": value,
                "key": _normalize_label_key(label),
            })
            continue
        items.append({"kind": "plain", "text": s})

    if not items:
        return content

    # ----- Pass 2: route into the 5 fixed buckets -----
    top_header: str | None = None
    lead: list[tuple[str, str]] = []
    lda_devices: list[str] = []
    lda_yn: list[tuple[str, str]] = []
    lda_glance: list[str] = []
    skin: list[dict] = []
    isolation: list[dict] = []
    positioning: list[dict] = []
    positioning_labels: list[tuple[str, str]] = []
    unrouted: list[dict] = []
    cur: str | None = None

    def route_bullet_or_plain(it: dict) -> None:
        if cur == "lda":
            if it["kind"] == "bullet":
                lda_devices.append(it["body"])
            else:
                unrouted.append(it)
        elif cur == "skin":
            skin.append(it)
        elif cur == "isolation":
            isolation.append(it)
        elif cur == "positioning":
            positioning.append(it)
        else:
            unrouted.append(it)

    for it in items:
        kind = it["kind"]
        if kind == "top_header":
            top_header = it["text"]
            continue
        if kind == "section_header":
            key = it["key"]
            if key in _E_LDA_SECTION_KEYS:
                cur = "lda"; continue
            if key in _E_SKIN_SECTION_KEYS:
                cur = "skin"; continue
            if key in _E_ISOLATION_SECTION_KEYS:
                cur = "isolation"; continue
            if key in _E_POSITIONING_SECTION_KEYS:
                cur = "positioning"; continue
            cur = None
            unrouted.append(it)
            continue
        if kind == "label_value":
            key = it["key"]
            if key in _E_LEAD_LABEL_KEYS:
                lead.append((it["label"], it["value"])); continue
            if key in _E_LDA_YN_LABEL_KEYS:
                lda_yn.append((it["label"], it["value"])); continue
            if key in _E_LDA_GLANCE_LABEL_KEYS:
                lda_glance.append(it["value"]); continue
            if key in _E_SKIN_LABEL_KEYS:
                skin.append(it); continue
            if key in _E_ISOLATION_LABEL_KEYS:
                isolation.append(it); continue
            if key in _E_POSITIONING_LABEL_VALUE_KEYS:
                positioning_labels.append((it["label"], it["value"])); continue
            if key in _E_POSITIONING_LABEL_KEYS:
                positioning.append(it); continue
            unrouted.append(it)
            continue
        route_bullet_or_plain(it)

    for it in unrouted:
        label = it.get("key") or it.get("text") or it.get("body", "")[:80]
        logger.warning("E-section unrouted item (kind=%s key/text=%r) — "
                       "kept at end of section", it["kind"], label)

    # ----- Pass 3: emit in fixed bucket order, as text -----
    out: list[str] = []

    # 1. lead
    if top_header or lead:
        out.append(top_header if top_header else _LEAD_HEADER)
        for label, value in lead:
            out.append(f"{label} {value}")

    # 2. lda — Devices glance -> device bullets -> Y/N rows (merged into ONE block)
    glance = "; ".join(lda_glance) if lda_glance else ""
    if glance or lda_devices or lda_yn:
        out.append(_LDA_HEADER)
        if glance:
            out.append(f"Devices: {glance}")
        for body in lda_devices:
            out.append(f"- {body}")
        for label, value in lda_yn:
            out.append(f"{label} {value}")

    # 3-5. skin / isolation / positioning — header + bullets (+ positioning labels)
    def emit_bucket(header: str, bucket: list[dict],
                    extra_labels: list[tuple[str, str]] | None = None) -> None:
        if not bucket and not extra_labels:
            return
        out.append(header)
        for it in bucket:
            if it["kind"] == "bullet":
                out.append(f"- {it['body']}")
            elif it["kind"] == "label_value":
                # label implied by subheader -> single bullet from the value
                out.append(f"- {it['value']}")
            elif it["kind"] == "plain":
                out.append(f"- {it['text']}")
        for label, value in (extra_labels or []):
            out.append(f"{label} {value}")

    emit_bucket(_SKIN_HEADER, skin)
    emit_bucket(_ISOLATION_HEADER, isolation)
    emit_bucket(_POSITIONING_HEADER, positioning, positioning_labels)

    # unrouted catch-all at the very end (preserved, never dropped)
    for it in unrouted:
        if it["kind"] == "bullet":
            out.append(f"- {it['body']}")
        elif it["kind"] == "label_value":
            out.append(f"{it['label']} {it['value']}")
        else:
            out.append(it.get("text", ""))

    return "\n".join(out)
