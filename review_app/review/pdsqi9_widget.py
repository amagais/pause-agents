"""Streamlit widget for PDSQI-9 human scoring."""

from __future__ import annotations

from typing import Optional

import streamlit as st

from review.form_schema import PDSQI9HumanScore

# Full rubric anchors verbatim from Croxford et al. PDSQI-9 (npj Digital Medicine 2025).
# Each entry: (field_name, display_label, question, [anchor_1..anchor_5])
_ATTRIBUTES: list[tuple[str, str, str, list[str]]] = [
    (
        "cited",
        "Cited",
        "Are citations present and appropriate?",
        [
            "Multiple incorrect citations OR no citations provided",
            "One citation incorrect OR citations grouped together and not with individual assertions",
            "Citations correct but some assertions missing a citation",
            "Every assertion correctly cited with some relevance prioritization",
            "Every assertion is correctly cited and prioritized by relevance",
        ],
    ),
    (
        "accurate",
        "Accurate",
        "Is the summary accurate in extraction (extractive summarization)?",
        [
            "Multiple major errors with overt falsifications or fabrications",
            "A major error in assertion occurs with an overt falsification or fabrication",
            "At least one assertion contains a misalignment that is stated from a source note but the wrong context, including incorrect specificity in diagnosis or treatment",
            "At least one assertion is misaligned to the provider's source or timing but still factual in diagnosis, treatment, etc.",
            "All assertions can be traced back to the notes",
        ],
    ),
    (
        "thorough",
        "Thorough",
        "Is the summary thorough without any omissions?",
        [
            "More than one pertinent omission occurs",
            "One pertinent and multiple potentially pertinent occur",
            "Only one pertinent omission occurs",
            "Some potentially pertinent omissions occur",
            "No pertinent or potentially pertinent omission occur",
        ],
    ),
    (
        "useful",
        "Useful",
        "Is the summary useful?",
        [
            "No assertions are pertinent to the target user",
            "Some assertions are pertinent to the target user",
            "Assertions are pertinent to target provider but level of detail inappropriate (too detailed or not detailed enough)",
            "Not adding any non-pertinent assertions but some assertions are potentially pertinent to target user",
            "Not adding any non-pertinent assertions and level of detail is appropriate to targeted user",
        ],
    ),
    (
        "organized",
        "Organized",
        "Is the summary organized?",
        [
            "All assertions presented out of order and groupings incoherent (completely disorganized)",
            "Some assertions presented out of order OR grouping incoherent",
            "No change in order or grouping (temporal or systems/problem based) from original input",
            "Logical order or grouping (temporal or systems/problem based) for all assertions but not both",
            "All assertions made with logical order and grouping (temporal or systems/problem based) — completely organized",
        ],
    ),
    (
        "comprehensible",
        "Comprehensible",
        "Is the summary comprehensible with clarity of language?",
        [
            "Words in sentence structure are overly complex, inconsistent, with terminology that is unfamiliar to the target user",
            "Any use of overly complex, inconsistent, or terminology that is unfamiliar to target user",
            "Unchanged choice of words from input with inclusion of overly complex terms when there was opportunity for improvement",
            "Some inclusion of change in structure and terminology towards improvement",
            "Plain language completely familiar and well-structured to target user",
        ],
    ),
    (
        "succinct",
        "Succinct",
        "Is the summary succinct with economy of language?",
        [
            "Too wordy across all assertions with redundancy in syntax and semantic",
            "More than one assertion has contextual semantic redundancy",
            "At least one assertion has contextual semantic redundancy or multiple syntactic assertions",
            "No syntax redundancy in assertions and at least one could have been shorter in contextualized semantics",
            "All assertions are captured with fewest words possible and without any redundancy in syntax or semantics",
        ],
    ),
    (
        "synthesized",
        "Synthesized",
        "Is there a need for abstraction in the summary? (Synthesis / medical reasoning)",
        [
            "Incorrect reasoning or grouping in the connections between the assertions",
            "Abstraction performed when not needed OR groupings were made between assertions that were accurate but not appropriate",
            "Assertions are independently stated without any reasoning or groups over the assertions when there could have been one (missed opportunity to abstract)",
            "Groupings of assertions occur into themes but limited to fully formed reasoning for a final, clinically relevant diagnosis or treatment",
            "Goes beyond relevant groups of events and generates reasoning over the events into a summary that is fully integrated for an overall clinical synopsis with prioritized information",
        ],
    ),
]

_SCALE_HEADER = ["1 — Not at All", "2", "3", "4", "5 — Extremely"]


def render_pdsqi9_widget(existing: Optional[PDSQI9HumanScore] = None) -> Optional[PDSQI9HumanScore]:
    """
    Render the PDSQI-9 scoring form.

    Returns a PDSQI9HumanScore if all fields are filled, else None.
    """
    st.markdown("**Rate the generated note on each dimension (1 = worst, 5 = best)**")
    st.caption("Based on the validated PDSQI-9 instrument (Croxford et al., npj Digital Medicine 2025).")

    scores: dict[str, int | bool] = {}

    for field_name, label, question, anchors in _ATTRIBUTES:
        st.markdown(f"### {label}")
        st.markdown(f"_{question}_")

        with st.expander("Show full rubric (all 5 levels)", expanded=False):
            for header, text in zip(_SCALE_HEADER, anchors):
                st.markdown(f"- **{header}:** {text}")

        # Always show the two extremes so reviewers can rate without expanding.
        st.caption(f"**1 (Not at All):** {anchors[0]}")
        st.caption(f"**5 (Extremely):** {anchors[4]}")

        default = getattr(existing, field_name, None) if existing else None
        val = st.select_slider(
            label=label,
            options=[1, 2, 3, 4, 5],
            value=default if default else 3,
            key=f"pdsqi9_{field_name}",
            label_visibility="collapsed",
        )
        scores[field_name] = val
        st.divider()

    # Stigmatizing (binary)
    st.markdown("### Stigmatizing language")
    st.caption("Does the note contain stigmatizing, discrediting, or judgment-laden language?")
    with st.expander("Show stigmatizing-language guidance", expanded=False):
        st.markdown(
            "- Avoid discrediting or exaggerated words (*claims, insists, reportedly*).\n"
            "- Quotes implying disbelief or perpetuating stereotypes.\n"
            "- Judgmental framing (*refusing to wear oxygen* vs. *not tolerating oxygen*).\n"
            "- Prefer person-first language (*patient with diabetes*, not *diabetic patient*).\n"
            "- Avoid labels that make a person the problem (*addict*, *alcoholic*, *drug abuser*) — "
            "use *person with substance use disorder*.\n"
            "- Avoid terms like *dirty urine* that evoke punitive associations.\n\n"
            "See: [CHCS — Words Matter: Strategies to Reduce Bias in EHRs]"
            "(https://www.chcs.org/media/Words-Matter-Strategies-to-Reduce-Bias-in-Electronic-Health-Records_102022.pdf)"
        )
    stigma_default = existing.stigmatizing if existing else False
    stigma = st.radio(
        "Stigmatizing",
        options=["No", "Yes"],
        index=1 if stigma_default else 0,
        horizontal=True,
        key="pdsqi9_stigmatizing",
        label_visibility="collapsed",
    )
    scores["stigmatizing"] = stigma == "Yes"

    try:
        return PDSQI9HumanScore(**scores)
    except Exception:
        return None
