"""Streamlit widget for structured hallucination (claim verification) check."""

from __future__ import annotations

from typing import Any

import streamlit as st

from display.citations import render_section_html

from review.form_schema import HallucinationItem

_VERDICT_OPTIONS = {
    "verified": "Verified",
    "cannot_verify": "Cannot verify",
    "incorrect": "Incorrect",
}

_VERDICT_DESCRIPTIONS = {
    "verified": "Supported by source data",
    "cannot_verify": "Not visible in provided source data",
    "incorrect": "Contradicts or fabricated from source",
}

_SECTION_LABELS = {
    "I": "I — ICU Admission & Course",
    "C": "C — Code Status / Goals of Care",
    "U_unprescribing": "U — High-Risk Medications",
    "P": "P — Pending Tests",
    "A": "A — Active Consultants",
    "U_uncertainty": "U — Diagnostic Uncertainty",
    "S": "S — Summary / To-Dos",
    "E": "E — Exam at Transfer",
}


def render_hallucination_widget(
    claims: list[dict],
    existing: list[HallucinationItem] | None = None,
    citation_index: dict[str, dict[str, Any]] | None = None,
) -> list[HallucinationItem]:
    """
    Render hallucination check for each extracted claim.

    Args:
        claims:   list of {claim_id, section, text} from claims.json
        existing: previously saved items (for draft resume)
        citation_index: optional metadata.citation_index — when supplied,
            raw ``(source M-DD HH:MM)`` tags in claim text are replaced with
            hoverable superscript markers (per-claim numbering).

    Returns:
        List of HallucinationItem (one per claim, incomplete ones excluded).
    """
    citation_index = citation_index or {}
    if not claims:
        st.info("No claims were extracted for this case.")
        return []

    # Index existing items by claim_id for easy lookup
    existing_map: dict[str, HallucinationItem] = {}
    if existing:
        for item in existing:
            existing_map[item.claim_id] = item

    # Group claims by section for readability
    by_section: dict[str, list[dict]] = {}
    for c in claims:
        by_section.setdefault(c["section"], []).append(c)

    st.markdown(
        "For each statement below, indicate whether it is supported by the source data shown on the left. "
        "Focus on whether the claim is plausible and traceable."
    )

    results: list[HallucinationItem] = []

    for section_key, section_claims in by_section.items():
        section_label = _SECTION_LABELS.get(section_key, section_key)
        st.markdown(f"**{section_label}**")

        for claim in section_claims:
            cid = claim["claim_id"]
            prev = existing_map.get(cid)

            claim_html, _ = render_section_html(claim["text"], citation_index)
            # Block-mode claims (Anticoagulation: + ☑/☐ option lines) are
            # multi-line — preserve newlines as <br> so they don't collapse
            # to a single line inside the blockquote.
            claim_html = claim_html.replace("\n", "<br>")
            st.markdown(
                f"<blockquote>{claim_html}</blockquote>",
                unsafe_allow_html=True,
            )

            verdict_key = f"halluc_verdict_{cid}"
            verdict_val = st.radio(
                "Verdict",
                options=list(_VERDICT_OPTIONS.keys()),
                format_func=lambda v: _VERDICT_OPTIONS[v],
                index=list(_VERDICT_OPTIONS.keys()).index(prev.verdict) if prev else 0,
                horizontal=True,
                key=verdict_key,
                label_visibility="collapsed",
            )

            note_key = f"halluc_note_{cid}"
            brief_note = st.text_area(
                "Notes (optional) — where is the discrepancy / what does the source actually show?",
                value=prev.brief_note if prev else "",
                key=note_key,
                height=80,
            )

            results.append(HallucinationItem(
                claim_id=cid,
                section=section_key,
                claim_text=claim["text"],
                verdict=verdict_val,
                brief_note=brief_note,
            ))
            st.markdown("")  # spacer

    return results
