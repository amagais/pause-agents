"""E-section render: 5-bucket fixed-order layout with LDA Devices: glance.

Locks down _build_exam_markdown — the pure HTML builder behind
_render_exam_section. The renderer is display-only: it takes the section
text the intensivist agent emits (see config/prompts/intensivist.yaml
lines 557-617) and re-emits it against a FIVE-bucket fixed-order layout
that overrides upstream payload order:

  1. TRANSFER EXAM        — Neuro / Vitals / Respiratory rows.
  2. Lines / Drains /     — Devices: glance line (flush-left),
     Airways                device bullets, ☐/☑ Y/N rows
                            (Difficult airway?, Lines/drains assessed
                            for removal?).
  3. Skin/wounds          — bullets (synthesized from label:value if
                            upstream emitted a flat row).
  4. Isolation precautions — bullets (same shape rule as Skin).
  5. Positioning          — bullets + C-collar / spine precautions:
     requirements and       label:value row (no glyph).
     precautions

The Active lines/drains/airways: value is NOT dropped (per rev2
addendum); it renders as the Devices: glance line at the top of the LDA
block. A divergence WARNING is logged when an item in the glance line is
absent from the device-bullet text — the line still renders so no
content is silently dropped.

Tests cover:
  - Bucket routing: each label/section key → correct bucket.
  - Fixed render order: LDA always before Skin, regardless of payload
    order.
  - Bucket subheader text (template-mandated wording).
  - LDA internal order: Devices: → bullets → ☐/☑ Y/N rows.
  - Devices: glance is flush-left (no &emsp; indent).
  - Active divergence warning + the glance line still renders.
  - Multiple Active candidates concatenate with "; " into one Devices:.
  - Y/N glyph: No → ☐, Yes → ☑, unknown → no glyph + WARNING,
    explanatory tail preserved verbatim.
  - Skin/wounds: bullets and label:value both supported.
  - Isolation: label:value upstream → bullet under injected subheader.
  - Positioning: bullets + C-collar special label:value row.
  - Empty-bucket subheader omission.
  - Unrouted items render at end + per-item WARNING.
  - Spec example round-trip (modulo whitespace).
"""

from __future__ import annotations

import html as _html
import logging
import re
import sys
from pathlib import Path

_REVIEW_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_REVIEW_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_REVIEW_APP_ROOT))

from display.note_renderer import (  # noqa: E402
    _build_exam_markdown,
    _e_active_item_set,
    _e_active_items_not_in_bullets,
    _e_checkbox_glyph,
    _e_glance_value_html,
    _e_normalize_label_key,
    _E_LDA_SUBHEADER,
    _E_LEAD_SUBHEADER,
    _E_SKIN_SUBHEADER,
    _E_ISOLATION_SUBHEADER,
    _E_POSITIONING_SUBHEADER,
    _E_TEMPLATE_SUBHEADER,  # backwards-compat alias for LDA subheader
)


def _strip_html(html: str) -> str:
    """Plain-text view of the rendered HTML — for ordering assertions.

    Strips tags AND decodes HTML entities (``&gt;`` → ``>``, ``&amp;`` → ``&``)
    so substring checks against payload-equivalent text work even when
    the renderer escapes special characters like ``>30 deg``.
    """
    return _html.unescape(re.sub(r"<[^>]+>", "", html))


# ---------------------------------------------------------------------------
# Lead block (TRANSFER EXAM + Neuro / Vitals / Respiratory).
# ---------------------------------------------------------------------------


def test_lead_block_renders_with_top_header_and_bold_labels():
    payload = (
        "TRANSFER EXAM\n"
        "Neuro: GCS 7 (E2 V1 **), RASS -2\n"
        "Vitals: BP 114/61, MAP 79, HR 85, RR 20, SpO2 98%, Temp 37.2°C\n"
        "Respiratory: Nasal Cannula 2 L/min\n"
    )
    out = _build_exam_markdown(payload, {})
    # TRANSFER EXAM is the lead-block header, bold-strong, no group gap.
    assert "<strong>TRANSFER EXAM</strong>" in out
    assert "margin-top:0rem" in out
    # Each lead-row label is bold, value follows.
    assert "<strong>Neuro:</strong> GCS 7 (E2 V1 **), RASS -2" in out
    assert "<strong>Vitals:</strong> BP 114/61" in out
    assert "<strong>Respiratory:</strong> Nasal Cannula 2 L/min" in out


def test_lead_synthesizes_transfer_exam_header_when_upstream_omits_it():
    # The bucket router still injects the lead subheader if upstream
    # emitted Neuro/Vitals without a TRANSFER EXAM top header — so the
    # rendered Section E always opens with a labeled lead block.
    payload = "Neuro: GCS 14\n"
    out = _build_exam_markdown(payload, {})
    assert f"<strong>{_E_LEAD_SUBHEADER}</strong>" in out
    assert _E_LEAD_SUBHEADER == "TRANSFER EXAM"


def test_lead_top_header_passes_through_verbatim_with_parenthetical():
    # Out-of-scope content (asterisk artifacts, parentheticals) survives
    # verbatim — the renderer never strips upstream noise.
    payload = (
        "TRANSFER EXAM (********** — ** *** paraphrase or duplicate)\n"
        "Neuro: GCS 7\n"
    )
    out = _build_exam_markdown(payload, {})
    assert (
        "<strong>TRANSFER EXAM (********** — ** *** "
        "paraphrase or duplicate)</strong>"
    ) in out


# ---------------------------------------------------------------------------
# Bucket subheader text — template-mandated wording.
# ---------------------------------------------------------------------------


def test_lda_subheader_text_matches_official_template():
    # "Lines / Drains / Airways" — spaces around slashes, title-case.
    assert _E_LDA_SUBHEADER == "Lines / Drains / Airways"
    # Backwards-compat alias still points at the LDA subheader.
    assert _E_TEMPLATE_SUBHEADER == _E_LDA_SUBHEADER
    payload = "Difficult airway? No\n"
    out = _build_exam_markdown(payload, {})
    assert "<strong>Lines / Drains / Airways</strong>" in out


def test_all_bucket_subheader_constants_match_spec():
    assert _E_LEAD_SUBHEADER == "TRANSFER EXAM"
    assert _E_LDA_SUBHEADER == "Lines / Drains / Airways"
    assert _E_SKIN_SUBHEADER == "Skin/wounds"
    assert _E_ISOLATION_SUBHEADER == "Isolation precautions"
    assert (
        _E_POSITIONING_SUBHEADER
        == "Positioning requirements and precautions"
    )


# ---------------------------------------------------------------------------
# LDA block — Devices: glance / device bullets / ☐/☑ Y/N rows.
# ---------------------------------------------------------------------------


def test_lda_devices_glance_renders_above_bullets_and_yn():
    payload = (
        "Lines/drains/airways:\n"
        "- Peripheral IV x2 documented: left upper arm and left wrist\n"
        "- Indwelling urethral catheter documented\n"
        "Active lines/drains/airways: peripheral IV x2, urinary catheter; "
        "no airway devices documented\n"
        "Difficult airway? No\n"
        "Lines/drains assessed for removal? No — no explicit removal "
        "assessment documented\n"
    )
    out = _build_exam_markdown(payload, {})
    # LDA subheader present.
    assert "<strong>Lines / Drains / Airways</strong>" in out
    # Devices: glance line is FLUSH-LEFT (no &emsp; before it) and
    # carries the Active value as its content.
    assert (
        "<div><strong>Devices:</strong> peripheral IV x2, urinary catheter; "
        "no airway devices documented</div>"
    ) in out
    # Devices: line appears BEFORE the first device bullet.
    pos_devices = out.find("<strong>Devices:</strong>")
    pos_bullet = out.find("&emsp;• Peripheral IV x2 documented")
    assert pos_devices != -1
    assert pos_bullet != -1
    assert pos_devices < pos_bullet
    # Device bullets render with &emsp;• indent under the subheader.
    assert (
        "&emsp;• Peripheral IV x2 documented: "
        "left upper arm and left wrist"
    ) in out
    assert "&emsp;• Indwelling urethral catheter documented" in out
    # ☐ Y/N rows render AFTER device bullets, with &emsp; indent.
    pos_yn = out.find("&emsp;☐ <strong>Difficult airway?</strong>")
    assert pos_yn != -1
    assert pos_bullet < pos_yn


def test_lda_glance_is_flush_left_not_bullet_indented():
    # Spec addendum: Devices: line "sits flush-left under the subheader
    # (not bullet-indented), so it visually reads as a topic line
    # rather than another bullet."
    payload = (
        "Active lines/drains/airways: peripheral IV, urinary catheter\n"
    )
    out = _build_exam_markdown(payload, {})
    # No &emsp; immediately before the Devices: glance.
    assert (
        "<div><strong>Devices:</strong> peripheral IV, urinary catheter</div>"
    ) in out
    assert "&emsp;<strong>Devices:</strong>" not in out
    assert "&emsp;• <strong>Devices:" not in out


def test_lda_glance_renders_even_with_no_bullets_or_yn():
    # If the upstream only emits the Active row (no Lines/drains/airways:
    # section header, no Y/N rows), the LDA block still renders with the
    # subheader + Devices: glance line.
    payload = "Active lines/drains/airways: peripheral IV\n"
    out = _build_exam_markdown(payload, {})
    assert "<strong>Lines / Drains / Airways</strong>" in out
    assert "<strong>Devices:</strong> peripheral IV" in out


def test_lda_two_glance_sources_concat_with_semicolon():
    # Both Lines/drains: and Active lines/drains/airways: route to the
    # glance slot. When both present, they concat with "; " so no
    # content is silently dropped.
    payload = (
        "Lines/drains: peripheral IV\n"
        "Active lines/drains/airways: peripheral IV, urinary catheter\n"
    )
    out = _build_exam_markdown(payload, {})
    # Both values appear in the single Devices: line.
    plain = _strip_html(out)
    assert "Devices: peripheral IV; peripheral IV, urinary catheter" in plain
    # Still exactly ONE Devices: row.
    assert out.count("<strong>Devices:</strong>") == 1


def test_lda_subheader_omitted_when_no_lda_content():
    # No LDA section header, no Active glance, no Y/N rows → LDA
    # subheader is NOT injected.
    payload = "Neuro: GCS 14\nSkin/wounds:\n- Sacral wound\n"
    out = _build_exam_markdown(payload, {})
    assert "<strong>Lines / Drains / Airways</strong>" not in out
    assert "<strong>Devices:</strong>" not in out


def test_lda_subheader_renders_when_only_yn_rows_present():
    # Y/N rows alone are sufficient to open the LDA block — the
    # template Y/N questions must always have a labeled home.
    payload = "Difficult airway? No\n"
    out = _build_exam_markdown(payload, {})
    assert "<strong>Lines / Drains / Airways</strong>" in out
    assert "&emsp;☐ <strong>Difficult airway?</strong> No" in out


def test_lda_devices_glance_divergence_logs_warning_but_renders(caplog):
    # Glance value lists "arterial line" which is absent from the
    # device bullets → WARNING logged, but Devices: still renders
    # (rev2 addendum reversed rev2's drop rule).
    payload = (
        "Lines/drains/airways:\n"
        "- Peripheral IV documented: left arm\n"
        "Active lines/drains/airways: peripheral IV, arterial line\n"
    )
    with caplog.at_level(logging.WARNING, logger="display.note_renderer"):
        out = _build_exam_markdown(payload, {})
    # Devices: line still renders.
    assert (
        "<strong>Devices:</strong> peripheral IV, arterial line"
    ) in out
    # WARNING captured naming the divergent item.
    divergence_logs = [
        r for r in caplog.records
        if r.levelname == "WARNING" and "divergence" in r.getMessage()
    ]
    assert len(divergence_logs) == 1
    msg = divergence_logs[0].getMessage()
    assert "active lines/drains/airways" in msg
    assert "arterial line" in msg


def test_lda_devices_glance_no_divergence_no_warning(caplog):
    # When every Active item is a substring of the joined bullet text,
    # no divergence warning is emitted.
    payload = (
        "Lines/drains/airways:\n"
        "- Peripheral IV x2 documented: left arm and right wrist\n"
        "- Urinary catheter documented\n"
        "Active lines/drains/airways: peripheral IV, urinary catheter\n"
    )
    with caplog.at_level(logging.WARNING, logger="display.note_renderer"):
        _build_exam_markdown(payload, {})
    assert not any(
        "divergence" in r.getMessage()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Y/N row glyph mapping (inside the LDA block, ☐/☑ + &emsp; indent).
# ---------------------------------------------------------------------------


def test_yn_no_renders_empty_checkbox_indented():
    payload = "Difficult airway? No\n"
    out = _build_exam_markdown(payload, {})
    # ☐ + &emsp; indent + bold label + value, all under LDA block.
    assert "&emsp;☐ <strong>Difficult airway?</strong> No" in out
    assert "&emsp;☑ <strong>Difficult airway?</strong>" not in out


def test_yn_yes_renders_checked_checkbox_indented():
    payload = "Difficult airway? Yes\n"
    out = _build_exam_markdown(payload, {})
    assert "&emsp;☑ <strong>Difficult airway?</strong> Yes" in out
    assert "&emsp;☐ <strong>Difficult airway?</strong>" not in out


def test_yn_preserves_explanatory_tail_verbatim():
    # Spec: "do not strip the explanatory tail on Lines/drains
    # assessed for removal? — the '...' rationale must be preserved
    # verbatim."
    payload = (
        "Lines/drains assessed for removal? "
        "No — no explicit removal assessment documented for ****, *****, "
        "or enteral tube in available notes\n"
    )
    out = _build_exam_markdown(payload, {})
    assert (
        "&emsp;☐ <strong>Lines/drains assessed for removal?</strong> "
        "No — no explicit removal assessment documented for ****, *****, "
        "or enteral tube in available notes"
    ) in out


def test_yn_unhandled_value_falls_back_plain_with_warning(caplog):
    # Multi-word answer falls outside both glyph sets → no glyph,
    # plain label:value, WARNING logged.
    payload = "Difficult airway? Unknown — anesthesia consult pending\n"
    with caplog.at_level(logging.WARNING, logger="display.note_renderer"):
        out = _build_exam_markdown(payload, {})
    assert "&emsp;☐ <strong>Difficult airway?</strong>" not in out
    assert "&emsp;☑ <strong>Difficult airway?</strong>" not in out
    assert (
        "&emsp;<strong>Difficult airway?</strong> "
        "Unknown — anesthesia consult pending"
    ) in out
    warnings = [
        r for r in caplog.records
        if r.levelname == "WARNING" and "unhandled value" in r.getMessage()
    ]
    assert len(warnings) == 1


def test_yn_case_insensitive_with_trailing_punct():
    # First-token + trailing-punct strip handles "No.", "YES,",
    # "neg", "positive", etc.
    cases = [
        ("Difficult airway? no\n", "☐"),
        ("Difficult airway? No.\n", "☐"),
        ("Difficult airway? N\n", "☐"),
        ("Difficult airway? negative\n", "☐"),
        ("Difficult airway? YES\n", "☑"),
        ("Difficult airway? yes,\n", "☑"),
        ("Difficult airway? Y\n", "☑"),
        ("Difficult airway? positive\n", "☑"),
    ]
    for payload, glyph in cases:
        out = _build_exam_markdown(payload, {})
        assert (
            f"&emsp;{glyph} <strong>Difficult airway?</strong>" in out
        ), payload


# ---------------------------------------------------------------------------
# Skin/wounds bucket.
# ---------------------------------------------------------------------------


def test_skin_bucket_subheader_with_bullets_passthrough():
    payload = (
        "Skin/wounds:\n"
        "- Sacral wound: location documented as sacral\n"
        "- Stoma skin: laryngectomy stoma clean, dry, intact\n"
    )
    out = _build_exam_markdown(payload, {})
    assert f"<strong>{_E_SKIN_SUBHEADER}</strong>" in out
    # Bullets pass through with their inner label:value bold (existing
    # bullet-body label-bold rule from the prior renderer).
    assert (
        "&emsp;• <strong>Sacral wound:</strong> "
        "location documented as sacral"
    ) in out
    assert (
        "&emsp;• <strong>Stoma skin:</strong> "
        "laryngectomy stoma clean, dry, intact"
    ) in out


def test_skin_label_value_upstream_renders_as_synthesized_bullet():
    # Upstream emitted a flat "Skin/wounds: ..." label:value, not a
    # section_header + bullets. Renderer drops the label (implied by
    # the injected subheader) and synthesizes a single bullet from the
    # value.
    payload = (
        "Skin/wounds: bilateral arm gauze dressings with underlying "
        "erythema noted but extent not assessable in note\n"
    )
    out = _build_exam_markdown(payload, {})
    assert f"<strong>{_E_SKIN_SUBHEADER}</strong>" in out
    assert (
        "&emsp;• bilateral arm gauze dressings with underlying "
        "erythema noted but extent not assessable in note"
    ) in out
    # The "Skin/wounds:" label is NOT rendered as a separate row — it's
    # been folded into the subheader.
    assert "<strong>Skin/wounds:</strong> bilateral" not in out


# ---------------------------------------------------------------------------
# Isolation precautions bucket.
# ---------------------------------------------------------------------------


def test_isolation_label_value_upstream_renders_as_synthesized_bullet():
    payload = (
        "Isolation: no isolation precautions documented in available "
        "structured data/notes\n"
    )
    out = _build_exam_markdown(payload, {})
    assert f"<strong>{_E_ISOLATION_SUBHEADER}</strong>" in out
    assert (
        "&emsp;• no isolation precautions documented in available "
        "structured data/notes"
    ) in out


def test_isolation_section_header_with_bullets_passthrough():
    payload = (
        "Isolation precautions:\n"
        "- No isolation precautions documented; confirm at bedside\n"
    )
    out = _build_exam_markdown(payload, {})
    assert f"<strong>{_E_ISOLATION_SUBHEADER}</strong>" in out
    assert (
        "&emsp;• No isolation precautions documented; confirm at bedside"
    ) in out


# ---------------------------------------------------------------------------
# Positioning bucket + C-collar special label:value row.
# ---------------------------------------------------------------------------


def test_positioning_label_value_renders_as_synthesized_bullet():
    payload = (
        "Positioning requirements and precautions: "
        "frequent side-to-side repositioning; head of bed >30 deg\n"
    )
    out = _build_exam_markdown(payload, {})
    assert f"<strong>{_E_POSITIONING_SUBHEADER}</strong>" in out
    # ">" is HTML-escaped to "&gt;" by the citation renderer's escaping
    # pass — assert against the escaped form so the test mirrors the
    # actual rendered HTML, not the source payload.
    assert (
        "&emsp;• frequent side-to-side repositioning; "
        "head of bed &gt;30 deg"
    ) in out


def test_positioning_c_collar_renders_as_label_value_no_glyph():
    # C-collar / spine precautions: keeps its label as a label:value
    # row (no bullet, no glyph) per the spec's example output.
    payload = (
        "Positioning requirements and precautions: alternate side-lying\n"
        "C-collar / spine precautions: None documented\n"
    )
    out = _build_exam_markdown(payload, {})
    assert f"<strong>{_E_POSITIONING_SUBHEADER}</strong>" in out
    # Positioning row: synthesized bullet from value.
    assert "&emsp;• alternate side-lying" in out
    # C-collar row: label:value with bold label, no bullet glyph.
    assert (
        "&emsp;<strong>C-collar / spine precautions:</strong> "
        "None documented"
    ) in out
    # NOT rendered as a bullet.
    assert "&emsp;• <strong>C-collar / spine precautions:" not in out
    assert "&emsp;• None documented" not in out


def test_positioning_related_field_keys_route_to_same_bucket():
    # "Current mobility level", "Activity restrictions", and "Device
    # positioning notes" all surface under the Positioning subheader.
    payload = (
        "Current mobility level: bed-bound\n"
        "Activity restrictions: no specific therapy-based restrictions\n"
        "Device positioning notes: ETT 22 cm at lip\n"
    )
    out = _build_exam_markdown(payload, {})
    assert f"<strong>{_E_POSITIONING_SUBHEADER}</strong>" in out
    assert "&emsp;• bed-bound" in out
    assert "&emsp;• no specific therapy-based restrictions" in out
    assert "&emsp;• ETT 22 cm at lip" in out


# ---------------------------------------------------------------------------
# Fixed bucket order — LDA always before Skin, regardless of payload order.
# ---------------------------------------------------------------------------


def test_fixed_bucket_order_overrides_payload_order():
    # Upstream emits Skin/wounds FIRST, then Positioning, then LDA — but
    # the renderer puts them in template order: LDA → Skin → Positioning.
    payload = (
        "Skin/wounds:\n- Sacral wound\n"
        "Positioning requirements and precautions: bed-bound\n"
        "Lines/drains/airways:\n- Foley\n"
        "Difficult airway? No\n"
    )
    out = _build_exam_markdown(payload, {})
    pos_lda = out.find(f"<strong>{_E_LDA_SUBHEADER}</strong>")
    pos_skin = out.find(f"<strong>{_E_SKIN_SUBHEADER}</strong>")
    pos_pos = out.find(f"<strong>{_E_POSITIONING_SUBHEADER}</strong>")
    assert pos_lda != -1
    assert pos_skin != -1
    assert pos_pos != -1
    # LDA before Skin before Positioning, even though payload had
    # Skin first.
    assert pos_lda < pos_skin < pos_pos


def test_empty_buckets_omit_their_subheader():
    # Lead block only — none of LDA / Skin / Isolation / Positioning
    # subheaders appear.
    payload = "Neuro: GCS 14\nVitals: BP 120/80\n"
    out = _build_exam_markdown(payload, {})
    assert f"<strong>{_E_LEAD_SUBHEADER}</strong>" in out
    for subheader in (
        _E_LDA_SUBHEADER,
        _E_SKIN_SUBHEADER,
        _E_ISOLATION_SUBHEADER,
        _E_POSITIONING_SUBHEADER,
    ):
        assert f"<strong>{subheader}</strong>" not in out


# ---------------------------------------------------------------------------
# Unrouted catch-all — preserves visibility for unknown content.
# ---------------------------------------------------------------------------


def test_unrouted_label_value_logs_warning_and_renders_at_end(caplog):
    # "Mystery field: foo" doesn't match any bucket — it logs a WARNING
    # and renders at the very end so nothing disappears.
    payload = (
        "Neuro: GCS 14\n"
        "Mystery field: surprise content\n"
    )
    with caplog.at_level(logging.WARNING, logger="display.note_renderer"):
        out = _build_exam_markdown(payload, {})
    # Rendered, not dropped.
    assert "<strong>Mystery field:</strong> surprise content" in out
    # WARNING captured.
    warnings = [
        r for r in caplog.records
        if r.levelname == "WARNING" and "unrouted item" in r.getMessage()
    ]
    assert len(warnings) >= 1
    assert any("mystery field" in r.getMessage() for r in warnings)
    # And the mystery row renders AFTER all known buckets.
    lead_pos = out.find(f"<strong>{_E_LEAD_SUBHEADER}</strong>")
    mystery_pos = out.find("<strong>Mystery field:</strong>")
    assert lead_pos < mystery_pos


# ---------------------------------------------------------------------------
# Citation preservation through bucket routing.
# ---------------------------------------------------------------------------


def test_citation_sup_tokens_preserved_in_lead_block():
    # Tag format: (source_type M-DD HH:MM). Lead rows route to lead
    # bucket; citation HTML survives intact.
    payload = "Neuro: GCS 7 (exam-neuro 1-09 06:00)\n"
    out = _build_exam_markdown(payload, {})
    assert "<strong>Neuro:</strong>" in out
    assert "<sup" in out
    assert ">¹</sup>" in out


def test_citation_sup_tokens_preserved_in_lda_bullet():
    payload = (
        "Lines/drains/airways:\n"
        "- Right chest port (nursing_note 1-09 06:00)\n"
    )
    out = _build_exam_markdown(payload, {})
    assert "&emsp;• Right chest port" in out
    assert "<sup" in out


def test_citation_sup_tokens_preserved_in_devices_glance():
    payload = "Active lines/drains/airways: peripheral IV (nursing_note 1-09 06:00)\n"
    out = _build_exam_markdown(payload, {})
    assert "<strong>Devices:</strong>" in out
    assert "<sup" in out


# ---------------------------------------------------------------------------
# Helper-level unit tests — exercise routing/normalization in isolation.
# ---------------------------------------------------------------------------


def test_normalize_label_key_strips_punct_and_lowers():
    assert _e_normalize_label_key("Difficult airway?") == "difficult airway"
    assert _e_normalize_label_key("Lines/drains:") == "lines/drains"
    assert (
        _e_normalize_label_key("  Active lines/drains/airways  :  ")
        == "active lines/drains/airways"
    )
    assert (
        _e_normalize_label_key("C-collar / spine precautions:")
        == "c-collar / spine precautions"
    )


def test_active_item_set_splits_on_comma_and_semicolon():
    assert _e_active_item_set(
        "peripheral IV, urinary catheter; nasal cannula"
    ) == {"peripheral iv", "urinary catheter", "nasal cannula"}


def test_glance_value_html_handles_empty_single_and_multiple():
    assert _e_glance_value_html([]) is None
    assert (
        _e_glance_value_html([("lines/drains", "peripheral IV")])
        == "peripheral IV"
    )
    # Multiple sources concat with "; ".
    assert (
        _e_glance_value_html([
            ("lines/drains", "peripheral IV"),
            ("active lines/drains/airways", "peripheral IV, urinary catheter"),
        ])
        == "peripheral IV; peripheral IV, urinary catheter"
    )


def test_active_items_not_in_bullets_clean_summary_returns_empty():
    # Each Active item is a case-insensitive substring of the joined
    # bullet text → no divergent items.
    bullets = [
        "Peripheral IV x2 documented: left arm and right wrist",
        "Urinary catheter documented",
    ]
    assert (
        _e_active_items_not_in_bullets(
            "peripheral IV, urinary catheter", bullets
        )
        == []
    )


def test_active_items_not_in_bullets_unique_item_flagged():
    bullets = ["Peripheral IV documented"]
    divergent = _e_active_items_not_in_bullets(
        "peripheral IV, arterial line", bullets
    )
    assert divergent == ["arterial line"]


def test_active_items_not_in_bullets_no_bullets_treats_all_as_unique():
    # When upstream emits the Active row but no Lines/drains/airways:
    # section_header (so no device bullets), every Active item is
    # "unique" — the warning surfaces the situation so we know the
    # glance line stands alone.
    divergent = _e_active_items_not_in_bullets(
        "peripheral IV, urinary catheter", []
    )
    assert set(divergent) == {"peripheral IV", "urinary catheter"}


def test_checkbox_glyph_helper_directly():
    assert _e_checkbox_glyph("No") == "☐"
    assert _e_checkbox_glyph("Yes") == "☑"
    assert _e_checkbox_glyph("Unknown") is None
    assert _e_checkbox_glyph("") is None
    # First-token tokenization works through HTML markers too.
    assert _e_checkbox_glyph("No <sup>1</sup>") == "☐"


# ---------------------------------------------------------------------------
# Dash separators — fall through as plain text (handled via unrouted /
# bucket-specific behavior). Defensive guard preserved from prior spec.
# ---------------------------------------------------------------------------


def test_dash_separator_does_not_trigger_bold_label():
    payload = "Status - stable on room air\n"
    out = _build_exam_markdown(payload, {})
    assert "<strong>Status" not in out
    assert "Status - stable on room air" in out


# ---------------------------------------------------------------------------
# Spec example round-trip (rev2 + addendum) — the desired output block
# reproduced in the spec must appear in the rendered output (modulo
# whitespace + bold markup), in the bucket-mandated order.
# ---------------------------------------------------------------------------


def test_spec_example_renders_to_desired_output_modulo_whitespace():
    # Plausible upstream payload that, after rendering, must produce the
    # spec's rev2-addendum "Desired output" block.
    payload = (
        "TRANSFER EXAM\n"
        "Neuro: GCS 11 (** V1 **), RASS 0, ***-*** negative\n"
        "Vitals: BP 97/62, MAP 74, HR 82, RR 20, SpO2 92%, Temp 35.9°C\n"
        "Respiratory: Room air\n"
        "Lines/drains/airways:\n"
        "- Peripheral IV x2 documented: left upper arm and left wrist; "
        "current site condition not documented in available notes, "
        "status unclear — confirm at bedside\n"
        "- Indwelling urethral catheter documented; chronic catheter noted "
        "in progress/H&P, current insertion site condition not documented "
        "in available notes, status unclear — confirm at bedside\n"
        "- Jejunostomy tube / enteral tube documented in ***; current site "
        "condition not documented in available notes, status unclear — "
        "confirm at bedside\n"
        "- Laryngectomy tube/***** **** 8 documented at stoma; stoma clean, "
        "dry, intact; tube midline and secured with trach tie\n"
        "Active lines/drains/airways: Peripheral IV x2, indwelling urethral "
        "catheter, jejunostomy/enteral tube, laryngectomy Tube 8 / surgical "
        "airway\n"
        "Difficult airway? No\n"
        "Lines/drains assessed for removal? No — no explicit removal "
        "assessment documented for ****, *****, or enteral tube in "
        "available notes\n"
        "Skin/wounds:\n"
        "- Sacral wound: location documented as sacral; "
        "type/stage/size/drainage/odor/periwound not documented\n"
        "- Stoma skin: laryngectomy stoma clean, dry, intact\n"
        "Isolation: No isolation precautions documented in available "
        "structured data/notes; confirm at bedside\n"
        "Positioning requirements and precautions: ***** self; alternate "
        "side-lying/supine positioning; head of bed >30 deg\n"
        "C-collar / spine precautions: None documented\n"
    )
    out = _build_exam_markdown(payload, {})
    plain = re.sub(r"\s+", " ", _strip_html(out)).strip()
    # Every snippet of the spec's desired output, in bucket-mandated
    # order. Substring + in-order check (not strict equality — modulo
    # whitespace per spec).
    expected_in_order = [
        "TRANSFER EXAM",
        "Neuro: GCS 11 (** V1 **), RASS 0, ***-*** negative",
        "Vitals: BP 97/62, MAP 74, HR 82, RR 20, SpO2 92%, Temp 35.9°C",
        "Respiratory: Room air",
        "Lines / Drains / Airways",
        "Devices: Peripheral IV x2, indwelling urethral catheter, "
        "jejunostomy/enteral tube, laryngectomy Tube 8 / surgical airway",
        "• Peripheral IV x2 documented: left upper arm",
        "• Indwelling urethral catheter documented",
        "• Jejunostomy tube / enteral tube documented",
        "• Laryngectomy tube/***** **** 8 documented at stoma",
        "☐ Difficult airway? No",
        "☐ Lines/drains assessed for removal? No — no explicit removal "
        "assessment documented for ****, *****, or enteral tube in "
        "available notes",
        "Skin/wounds",
        "• Sacral wound: location documented as sacral",
        "• Stoma skin: laryngectomy stoma clean, dry, intact",
        "Isolation precautions",
        "• No isolation precautions documented in available structured "
        "data/notes; confirm at bedside",
        "Positioning requirements and precautions",
        "• ***** self; alternate side-lying/supine positioning; "
        "head of bed >30 deg",
        "C-collar / spine precautions: None documented",
    ]
    last_idx = -1
    for snippet in expected_in_order:
        idx = plain.find(snippet)
        assert idx != -1, f"missing from rendered output: {snippet!r}"
        assert idx > last_idx, f"out of order: {snippet!r}"
        last_idx = idx
