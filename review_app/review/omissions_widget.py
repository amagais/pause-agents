"""Streamlit widget for structured critical omissions check."""

from __future__ import annotations

import streamlit as st

from review.form_schema import OMISSION_DOMAINS, OmissionItem


def render_omissions_widget(
    source: dict,
    existing: list[OmissionItem] | None = None,
) -> list[OmissionItem]:
    """
    Render omission check for each data domain present in source_bundle.

    Only shows domains that have non-empty data in the source bundle.

    Args:
        source:   source_bundle dict
        existing: previously saved items (for draft resume)

    Returns:
        List of OmissionItem (one per visible domain).
    """
    existing_map: dict[str, OmissionItem] = {}
    if existing:
        for item in existing:
            existing_map[item.domain] = item

    # Determine which domains have data in the source bundle. ``vitals_summary``
    # can be either the new dict shape ({"bucketed_trends": [...],
    # "recent_raw": [...]}) or a legacy flat list — a non-empty dict with
    # empty sub-lists must NOT count as "present", so check content rather
    # than the container.
    vitals_summary = source.get("vitals_summary")
    if isinstance(vitals_summary, dict):
        has_vitals = bool(vitals_summary.get("bucketed_trends")) or bool(
            vitals_summary.get("recent_raw")
        )
    else:
        has_vitals = bool(vitals_summary)

    domain_present = {
        "meds_continuous": bool(source.get("meds_continuous")),
        "meds_intermittent": bool(source.get("meds_intermittent")),
        "labs": bool(source.get("labs_recent")),
        "vitals": has_vitals,
        "respiratory": bool(source.get("respiratory_support")),
        "microbiology": bool(source.get("microbiology")),
        "code_status": True,   # always shown
        "consults": bool(source.get("diagnoses")),   # approximate; show by default
        "procedures": bool(source.get("procedures")),
    }

    st.markdown(
        "For each data type below, indicate whether any **clinically important** information "
        "was missing from the generated note. Only flag things you actually noticed — "
        "leave unchecked if nothing is missing."
    )

    results: list[OmissionItem] = []

    for domain_key, domain_label in OMISSION_DOMAINS:
        if not domain_present.get(domain_key, False):
            continue

        prev = existing_map.get(domain_key)
        st.markdown(f"**{domain_label}**")

        omit_key = f"omit_{domain_key}"
        omitted = st.radio(
            "Omitted?",
            options=[False, True],
            format_func=lambda v: ("Something important was missing" if v else "Nothing important missing"),
            index=1 if (prev and prev.omitted) else 0,
            horizontal=True,
            key=omit_key,
            label_visibility="collapsed",
        )

        severity = None
        brief_note = ""

        if omitted:
            sev_key = f"omit_sev_{domain_key}"
            severity = st.selectbox(
                "How important is the omission?",
                options=["pertinent", "potentially_pertinent"],
                format_func=lambda v: (
                    "Pertinent — could directly impact care decisions"
                    if v == "pertinent"
                    else "Potentially pertinent — useful context but not immediately actionable"
                ),
                index=0 if not prev or prev.severity == "pertinent" else 1,
                key=sev_key,
            )
            note_key = f"omit_note_{domain_key}"
            brief_note = st.text_input(
                "Briefly describe what was missing (optional):",
                value=prev.brief_note if prev else "",
                key=note_key,
            )

        results.append(OmissionItem(
            domain=domain_key,
            domain_label=domain_label,
            omitted=omitted,
            severity=severity if omitted else None,
            brief_note=brief_note if omitted else "",
        ))
        st.write("")  # spacer

    return results
