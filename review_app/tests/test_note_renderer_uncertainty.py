"""U_uncertainty flat-group renderer.

Locks down _build_uncertainty_markdown — the pure markdown builder
behind _render_uncertainty_section. The renderer is display-only:
parse-and-rebuild band-aid against chaotic LLM markdown for this
section (the durable fix is prompt-side, deferred until brief
regeneration is available).

Tests cover:
  * The "Pending data masquerades as continuation of preceding bullet"
    bug from the reviewer-app screenshot — the parser must reassign
    "Pending data" to its own top-level group regardless of how the
    LLM indented it, with the cultures that follow becoming children
    of Pending data rather than siblings of Less-likely items.
  * Inline-collapse for single-value headers (Working diagnosis,
    Differential includes).
  * Multi-line list group for Less likely / Pending data.
  * Empty Pending data (single bullet item) renders without inventing
    extra slots.
  * Less likely with only one item renders cleanly (parser shouldn't
    over-fit to multi-item lists).
  * Pending data missing entirely — render the three matched groups,
    do not synthesize a Pending data placeholder.
  * Header wording variance ("Working diagnosis at transfer" instead
    of "Working diagnosis at the time of transfer") still matches.
  * Bold-wrapped headers (LLM emits ``**Less likely:**``) match.
  * Fallback: zero headers matched → return generic markdown rather
    than blank-screening.
  * Partial match: 1-2 of 4 headers found → render matched groups
    cleanly; orphan content before the first header rendered as
    leading prose.
  * Citation tokens (``(cite source=... id=...)``) pass through into
    rendered ``<sup>`` markers intact and are not stripped.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REVIEW_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_REVIEW_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_REVIEW_APP_ROOT))

from display.note_renderer import _build_uncertainty_markdown  # noqa: E402


# ---------------------------------------------------------------------------
# Headline regression: the screenshot bug. Two-space-indented "Pending data:"
# was being rendered as a markdown continuation of the preceding bullet, and
# the cultures that followed displayed as siblings of less-likely items.
# After the fix, Pending data lifts out to its own group and the cultures
# become its children.
# ---------------------------------------------------------------------------


def test_pending_data_lifts_out_from_under_less_likely_bullet():
    payload = (
        "Working diagnosis at the time of transfer: Sepsis/UTI vs possible "
        "intra-abdominal source\n"
        "Differential includes: urinary tract infection, intra-abdominal "
        "infection, pulmonary infarction, aspiration pneumonia\n"
        "Less likely:\n"
        "  - Aspiration pneumonia: no aspiration event documented\n"
        "  - Hemorrhagic shock: hemoglobin stable\n"
        "  Pending data (to confirm/exclude):\n"
        "  - Urine culture\n"
        "  - Blood cultures\n"
        "  - Stool culture\n"
        "  - VRE screen culture\n"
    )
    out = _build_uncertainty_markdown(payload, {})

    # All four group headers present, bolded, in order.
    pos_wd = out.find("**Working diagnosis at the time of transfer:**")
    pos_dx = out.find("**Differential includes:**")
    pos_ll = out.find("**Less likely:**")
    pos_pd = out.find("**Pending data (to confirm/exclude):**")
    assert pos_wd >= 0
    assert pos_dx > pos_wd
    assert pos_ll > pos_dx
    assert pos_pd > pos_ll

    # Cultures must appear AFTER the Pending data header, not embedded
    # in the Less likely block.
    pos_urine = out.find("Urine culture")
    pos_blood = out.find("Blood cultures")
    assert pos_urine > pos_pd
    assert pos_blood > pos_pd

    # And specifically AFTER the hemorrhagic-shock bullet (which is the
    # last less-likely item).
    pos_hem = out.find("Hemorrhagic shock")
    assert pos_urine > pos_hem


def test_pending_data_cultures_are_indented_as_bullets():
    payload = (
        "Pending data (to confirm/exclude):\n"
        "- Urine culture\n"
        "- Blood cultures\n"
    )
    out = _build_uncertainty_markdown(payload, {})
    # Cultures render with the &emsp;- prefix used elsewhere in the app
    # for indented bullets within a streamlit markdown block.
    assert "&emsp;- Urine culture" in out
    assert "&emsp;- Blood cultures" in out


# ---------------------------------------------------------------------------
# Inline collapse for single-value headers.
# ---------------------------------------------------------------------------


def test_working_diagnosis_inline_collapses_to_one_line():
    payload = "Working diagnosis at the time of transfer: Septic shock\n"
    out = _build_uncertainty_markdown(payload, {})
    assert "**Working diagnosis at the time of transfer:** Septic shock" in out
    # Not the multi-line shape.
    assert "**Working diagnosis at the time of transfer:**\n" not in out


def test_differential_inline_collapses_with_comma_list():
    payload = (
        "Differential includes: UTI, intra-abdominal infection, "
        "pulmonary infarction\n"
    )
    out = _build_uncertainty_markdown(payload, {})
    assert (
        "**Differential includes:** UTI, intra-abdominal infection, "
        "pulmonary infarction"
    ) in out


# ---------------------------------------------------------------------------
# Less likely with a single item renders cleanly — parser shouldn't
# over-fit to multi-item lists.
# ---------------------------------------------------------------------------


def test_less_likely_single_item_renders_as_one_bullet():
    payload = (
        "Less likely:\n"
        "- Intracranial hemorrhage: noncontrast head CT without acute findings\n"
    )
    out = _build_uncertainty_markdown(payload, {})
    assert "**Less likely:**" in out
    assert (
        "&emsp;- Intracranial hemorrhage: noncontrast head CT without acute findings"
        in out
    )


# ---------------------------------------------------------------------------
# Pending data missing entirely: render the three present groups; do NOT
# invent an empty Pending data slot.
# ---------------------------------------------------------------------------


def test_pending_data_missing_does_not_synthesize_empty_slot():
    payload = (
        "Working diagnosis at the time of transfer: Sepsis\n"
        "Differential includes: UTI, pneumonia\n"
        "Less likely:\n"
        "- Aspiration: no aspiration event documented\n"
    )
    out = _build_uncertainty_markdown(payload, {})
    assert "**Working diagnosis at the time of transfer:** Sepsis" in out
    assert "**Differential includes:** UTI, pneumonia" in out
    assert "**Less likely:**" in out
    # Pending data was never in the payload — must not appear in output.
    assert "Pending data" not in out


# ---------------------------------------------------------------------------
# Header wording variance.
# ---------------------------------------------------------------------------


def test_working_diagnosis_at_transfer_variant_matches():
    payload = "Working diagnosis at transfer: Septic shock\n"
    out = _build_uncertainty_markdown(payload, {})
    # Matched as a working_diagnosis header; canonical display label
    # used in output regardless of input wording.
    assert "**Working diagnosis at the time of transfer:** Septic shock" in out


def test_differential_diagnosis_variant_matches():
    # Some LLMs emit "Differential diagnosis:" instead of "Differential
    # includes:" — both should be recognized as the differential header.
    payload = "Differential diagnosis: UTI, pneumonia\n"
    out = _build_uncertainty_markdown(payload, {})
    assert "**Differential includes:** UTI, pneumonia" in out


def test_bold_wrapped_header_matches():
    payload = (
        "**Less likely:**\n"
        "- Aspiration: no aspiration event documented\n"
    )
    out = _build_uncertainty_markdown(payload, {})
    assert "**Less likely:**" in out
    # The bold markers in the input shouldn't pollute the body either.
    assert "&emsp;- Aspiration: no aspiration event documented" in out


# ---------------------------------------------------------------------------
# Fallback: zero headers matched → generic markdown rather than blank.
# ---------------------------------------------------------------------------


def test_fallback_when_no_recognized_headers():
    payload = "Patient stable, no diagnostic uncertainty documented.\n"
    out = _build_uncertainty_markdown(payload, {})
    # Original content survives — we don't blank-screen the section.
    assert "Patient stable, no diagnostic uncertainty documented." in out
    # No group-header structure is invented.
    assert "**Working diagnosis" not in out
    assert "**Pending data" not in out


# ---------------------------------------------------------------------------
# Partial match: 1-2 of 4 headers + orphan prose before the first header.
# ---------------------------------------------------------------------------


def test_partial_match_renders_matched_groups_and_keeps_orphan_prose():
    payload = (
        "Some leading prose before any header.\n"
        "Working diagnosis at the time of transfer: Sepsis\n"
        "Less likely:\n"
        "- Aspiration: no aspiration event\n"
    )
    out = _build_uncertainty_markdown(payload, {})
    # Orphan prose preserved at the top.
    assert "Some leading prose before any header." in out
    # Both matched headers rendered.
    assert "**Working diagnosis at the time of transfer:** Sepsis" in out
    assert "**Less likely:**" in out
    # Unmatched groups not invented.
    assert "**Differential includes" not in out
    assert "**Pending data" not in out


# ---------------------------------------------------------------------------
# Citation passthrough — (cite ...) tokens become <sup> markers and are
# not stripped or mangled by the parse-and-rebuild.
# ---------------------------------------------------------------------------


def test_citation_tokens_survive_into_sup_markers():
    # Cite tags must match CITE_PATTERN — (source_type M-DD HH:MM) —
    # otherwise render_section_html passes the token through as text.
    tag = "(progress_note 8-18 14:00)"
    citation_index = {
        tag: {
            "tier": "decision_critical",
            "row": {"label": "Progress note", "value": "transfer summary"},
        },
    }
    payload = (
        f"Working diagnosis at the time of transfer: Septic shock {tag}\n"
    )
    out = _build_uncertainty_markdown(payload, citation_index)
    # The cite token has been transformed into a <sup> marker. The tag
    # text is preserved verbatim inside the marker's data-tag attribute
    # (intentional — tooling reads it back); what matters is that the
    # raw token no longer appears as inline body text.
    assert "<sup" in out
    assert 'data-tag="(progress_note 8-18 14:00)"' in out
    assert f"Septic shock {tag}" not in out  # not standalone trailing text
    # The label and value text are still there alongside the marker.
    assert "**Working diagnosis at the time of transfer:**" in out
    assert "Septic shock" in out


def test_citation_tokens_inside_less_likely_bullet_preserved():
    tag = "(progress_note 8-18 14:00)"
    citation_index = {
        tag: {
            "tier": "decision_critical",
            "row": {"label": "Imaging", "value": "head CT"},
        },
    }
    payload = (
        "Less likely:\n"
        f"- Intracranial hemorrhage: head CT without acute findings {tag}\n"
    )
    out = _build_uncertainty_markdown(payload, citation_index)
    assert "**Less likely:**" in out
    assert "&emsp;- Intracranial hemorrhage:" in out
    assert "<sup" in out
    assert 'data-tag="(progress_note 8-18 14:00)"' in out
    # Token isn't left as inline text following the rationale.
    assert f"acute findings {tag}" not in out


# ---------------------------------------------------------------------------
# Smoke test: a realistic full payload renders end-to-end without errors
# and contains every group in the expected order.
# ---------------------------------------------------------------------------


def test_full_payload_smoke():
    payload = (
        "Working diagnosis at the time of transfer: Septic shock secondary "
        "to urinary source\n"
        "Differential includes: UTI, intra-abdominal infection, "
        "pulmonary infarction, aspiration pneumonia\n"
        "Less likely:\n"
        "  - Aspiration pneumonia: no aspiration event documented in nursing notes\n"
        "  - Hemorrhagic shock: hemoglobin stable at 9.8\n"
        "Pending data (to confirm/exclude):\n"
        "  - Urine culture\n"
        "  - Blood cultures\n"
        "  - CT abdomen to exclude intra-abdominal source\n"
    )
    out = _build_uncertainty_markdown(payload, {})

    # All four group labels present.
    assert "**Working diagnosis at the time of transfer:**" in out
    assert "**Differential includes:**" in out
    assert "**Less likely:**" in out
    assert "**Pending data (to confirm/exclude):**" in out

    # Bullets indented with &emsp;- prefix.
    assert "&emsp;- Aspiration pneumonia:" in out
    assert "&emsp;- Hemorrhagic shock:" in out
    assert "&emsp;- Urine culture" in out
    assert "&emsp;- Blood cultures" in out
    assert "&emsp;- CT abdomen to exclude intra-abdominal source" in out

    # Order: WD < DX < LL < PD
    positions = [
        out.find("**Working diagnosis at the time of transfer:**"),
        out.find("**Differential includes:**"),
        out.find("**Less likely:**"),
        out.find("**Pending data (to confirm/exclude):**"),
    ]
    assert positions == sorted(positions)
