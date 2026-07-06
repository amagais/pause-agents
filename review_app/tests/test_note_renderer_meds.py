"""U_unprescribing flat 2-level renderer.

Locks down _build_meds_markdown — the pure markdown builder behind
_render_meds_section. The renderer is display-only: it takes the
section text the pharmacy agent emits (see config/prompts/pharmacy.yaml
lines 25-54) and restructures it into the two-level "group header +
indented items" shape the clinician spec called for.

Tests cover:
  - Parent "Anticoagulation:" header dropped, children promoted to
    top-level groups (no three-level nesting).
  - Verbatim payload labels (no wording rewrites in the renderer).
  - Single-body-line groups collapse to "**Header:** body" inline.
  - VTE prophylaxis always renders even when both Pharmacologic /
    Mechanical sub-fields are "Not documented" — clinical absence
    is itself meaningful.
  - Antibiotics never inline-collapses (checkbox formatting needs
    its own line).
  - Unknown header lines (e.g. an off-spec "History:" line under
    Antibiotics) fall through as body of the current group rather
    than starting a new group — keeps the prompt/renderer contract
    visible.
  - Empty optional groups (e.g. Transition/bridging plan when no
    plan documented) are omitted entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REVIEW_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_REVIEW_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_REVIEW_APP_ROOT))

from display.note_renderer import _build_meds_markdown  # noqa: E402


# ---------------------------------------------------------------------------
# Three-level payload → two-level rendered output. Parent "Anticoagulation:"
# is dropped; its children become top-level bold groups.
# ---------------------------------------------------------------------------


def test_anticoagulation_parent_is_dropped_children_promoted():
    payload = (
        "Changes to home meds: None\n"
        "Anticoagulation:\n"
        "  Active anticoagulation at transfer:\n"
        "    Heparin gtt 12 units/kg/hr for PE\n"
        "  Home anticoagulation (status at transfer):\n"
        "    Apixaban 5 mg BID — held for procedure\n"
    )
    out = _build_meds_markdown(payload, {})
    # Parent header never appears as its own line.
    assert "**Anticoagulation**" not in out
    assert "**Anticoagulation:**" not in out
    # Children appear as top-level bold groups, verbatim labels.
    assert "**Active anticoagulation at transfer:**" in out
    assert "**Home anticoagulation (status at transfer):**" in out


def test_labels_are_verbatim_no_rewrites():
    payload = (
        "Anticoagulation:\n"
        "  Active anticoagulation at transfer:\n"
        "    Heparin gtt\n"
    )
    out = _build_meds_markdown(payload, {})
    # No em-dash rewrites of the spec like "Anticoagulation — active at transfer".
    assert "Anticoagulation — active" not in out
    assert "Anticoagulation — active" not in out
    # Original payload label preserved.
    assert "Active anticoagulation at transfer" in out


# ---------------------------------------------------------------------------
# Single body line → inline collapse "**Header:** body".
# ---------------------------------------------------------------------------


def test_single_value_group_collapses_to_inline_one_line():
    payload = (
        "Anticoagulation:\n"
        "  Antiplatelet therapy at transfer:\n"
        "    Not documented in available data\n"
    )
    out = _build_meds_markdown(payload, {})
    assert "**Antiplatelet therapy at transfer:** Not documented in available data" in out
    # No header-then-indented-line pattern for single-value case.
    assert "**Antiplatelet therapy at transfer**\n" not in out
    assert "**Antiplatelet therapy at transfer**  \n" not in out


def test_changes_to_home_meds_inline_payload_collapses():
    payload = "Changes to home meds: None\n"
    out = _build_meds_markdown(payload, {})
    assert "**Changes to home meds:** None" in out


# ---------------------------------------------------------------------------
# VTE prophylaxis: always renders + never collapses (two named sub-fields).
# ---------------------------------------------------------------------------


def test_vte_prophylaxis_renders_when_both_subfields_not_documented():
    payload = (
        "Anticoagulation:\n"
        "  VTE prophylaxis at transfer:\n"
        "    Pharmacologic: Not documented in available data\n"
        "    Mechanical: not documented\n"
    )
    out = _build_meds_markdown(payload, {})
    assert "**VTE prophylaxis at transfer**" in out
    # Sub-fields render as body lines with regular weight (no bold).
    assert "Pharmacologic: Not documented in available data" in out
    assert "Mechanical: not documented" in out
    # Sub-labels are NOT bolded as competing headers.
    assert "**Pharmacologic" not in out
    assert "**Mechanical" not in out


def test_vte_prophylaxis_never_inline_collapses_even_with_one_subfield():
    # Defensive: even if the agent emits only one of the two sub-fields,
    # VTE prophylaxis stays multi-line (its visual identity is the two
    # named slots, not a one-liner).
    payload = (
        "Anticoagulation:\n"
        "  VTE prophylaxis at transfer:\n"
        "    Pharmacologic: enoxaparin 40 mg SQ daily\n"
    )
    out = _build_meds_markdown(payload, {})
    assert "**VTE prophylaxis at transfer**" in out
    assert "**VTE prophylaxis at transfer:** Pharmacologic" not in out


# ---------------------------------------------------------------------------
# Antibiotics: never collapses (checkbox formatting), each item own line.
# ---------------------------------------------------------------------------


def test_antibiotics_never_collapses_single_item_kept_multiline():
    payload = (
        "Antibiotics:\n"
        "[ ] N/A - no planned antimicrobials\n"
    )
    out = _build_meds_markdown(payload, {})
    assert "**Antibiotics**" in out
    # NOT inline-collapsed — checkbox needs its own indented line.
    assert "**Antibiotics:** [ ] N/A" not in out
    assert "[ ] N/A - no planned antimicrobials" in out


def test_antibiotics_multiple_active_items_each_own_line():
    payload = (
        "Antibiotics:\n"
        "[x] vancomycin - indication: MRSA bacteremia, day 3\n"
        "[x] cefepime - indication: HCAP, day 2\n"
    )
    out = _build_meds_markdown(payload, {})
    assert "**Antibiotics**" in out
    assert "vancomycin - indication: MRSA bacteremia" in out
    assert "cefepime - indication: HCAP" in out


# ---------------------------------------------------------------------------
# Empty optional groups omitted entirely (no empty headers in output).
# ---------------------------------------------------------------------------


def test_empty_transition_plan_omitted():
    # Transition/bridging plan is allowed to be absent — agent omits the
    # whole block when no plan documented. If it ever does emit the
    # header with no body, the renderer omits it.
    payload = (
        "Anticoagulation:\n"
        "  Active anticoagulation at transfer:\n"
        "    Heparin gtt\n"
        "  Transition/bridging plan (when applicable):\n"
    )
    out = _build_meds_markdown(payload, {})
    assert "Transition/bridging plan" not in out
    assert "**Active anticoagulation at transfer:** Heparin gtt" in out


# ---------------------------------------------------------------------------
# Unknown headerish lines fall through as body of current group, not as
# new top-level groups. Keeps the explicit prompt/renderer contract
# visible — drift surfaces as visibly-misplaced content, not silent
# indent chaos.
# ---------------------------------------------------------------------------


def test_unknown_header_falls_through_as_body_of_current_group():
    # The agent has been observed emitting a "History:" line under
    # Antibiotics despite the prompt only specifying current items
    # (flagged in clinician review). "History:" is NOT in the known
    # header set, so it renders as a body line under Antibiotics — not
    # as a new top-level group, and not silently dropped.
    payload = (
        "Antibiotics:\n"
        "[x] cefepime - indication: HCAP, day 2\n"
        "History: vancomycin (7/12 - 7/13)\n"
    )
    out = _build_meds_markdown(payload, {})
    assert "**Antibiotics**" in out
    # "History: ..." appears under Antibiotics body, NOT as a new group.
    assert "**History" not in out
    assert "History: vancomycin (7/12 - 7/13)" in out
    # Cefepime is still there too.
    assert "cefepime - indication: HCAP" in out


# ---------------------------------------------------------------------------
# Group ordering: rendered groups follow payload order, not the order in
# the known-header set. Ensures the renderer doesn't reorder content.
# ---------------------------------------------------------------------------


def test_group_order_follows_payload_not_known_header_set():
    # Payload emits Antibiotics first, then Changes to home meds — the
    # rendered order must match (no reordering).
    payload = (
        "Antibiotics:\n"
        "[x] vancomycin - indication: MRSA, day 3\n"
        "Changes to home meds: None\n"
    )
    out = _build_meds_markdown(payload, {})
    antibiotics_pos = out.find("**Antibiotics**")
    changes_pos = out.find("**Changes to home meds")
    assert antibiotics_pos != -1
    assert changes_pos != -1
    assert antibiotics_pos < changes_pos


# ---------------------------------------------------------------------------
# Smoke test: a realistic full payload renders end-to-end without errors
# and contains every group we expect.
# ---------------------------------------------------------------------------


def test_full_payload_smoke():
    payload = (
        "Changes to home meds: Lisinopril held — AKI on admission\n"
        "Anticoagulation:\n"
        "  Active anticoagulation at transfer:\n"
        "    Heparin gtt 12 units/kg/hr for PE; titrate per aPTT\n"
        "  Home anticoagulation (status at transfer):\n"
        "    Apixaban 5 mg BID — held; transitioning to heparin gtt\n"
        "  VTE prophylaxis at transfer:\n"
        "    Pharmacologic: covered by therapeutic heparin gtt\n"
        "    Mechanical: SCDs in place\n"
        "  Antiplatelet therapy at transfer:\n"
        "    Aspirin 81 mg PO daily — continued for CAD\n"
        "  Transition/bridging plan (when applicable):\n"
        "    Bridge with heparin gtt until INR therapeutic on warfarin\n"
        "Antibiotics:\n"
        "[x] cefepime - indication: HCAP, start: 7/12, duration: 7 days\n"
        "[x] vancomycin - indication: MRSA coverage, start: 7/12, duration: 7 days\n"
    )
    out = _build_meds_markdown(payload, {})
    # Every expected group header present in the right shape.
    assert "**Changes to home meds:** Lisinopril held" in out  # single-value collapse
    assert "**Active anticoagulation at transfer:**" in out
    assert "**Home anticoagulation (status at transfer):**" in out
    assert "**VTE prophylaxis at transfer**" in out  # multi-line, no colon
    assert "**Antiplatelet therapy at transfer:**" in out  # single-value collapse
    assert "**Transition/bridging plan (when applicable):**" in out  # single-value collapse
    assert "**Antibiotics**" in out  # never collapses
    # Anticoagulation parent dropped.
    assert "**Anticoagulation**" not in out
    assert "**Anticoagulation:**" not in out
