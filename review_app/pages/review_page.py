"""Main review page: side-by-side source data + generated note + review form."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import streamlit as st

from assignment.assignment_manager import load_manifest
from auth.msal_auth import get_reviewer_name
from display.note_renderer import render_note
from display.source_renderer import render_source
from review.form_schema import OmissionItem, ReviewResponse
from review.hallucination_widget import render_hallucination_widget
from review.omissions_widget import render_omissions_widget
from review.pdsqi9_widget import render_pdsqi9_widget
from storage.case_loader import load_case
from storage.response_writer import (
    is_read_only,
    load_prior_round_response,
    load_response,
    save_response,
)


def render_prior_hallucination_panel(prior_dict: dict) -> None:
    """Read-only panel above Step 1: claims the reviewer did NOT verify last
    round (carried-forward re-reviews only). No flags / empty dict -> renders
    nothing. The regenerated brief's claims changed, so verdicts can't carry
    over — this is reference context only."""
    flags = [
        c for c in (prior_dict.get("hallucination_checks") or [])
        if c.get("verdict") in ("cannot_verify", "incorrect")
    ]
    if not flags:
        return
    vlabel = {"incorrect": "❌ incorrect", "cannot_verify": "❓ cannot verify"}
    st.info(
        "**From your last review — claims you did NOT mark verified.** "
        "The brief was regenerated and its sentences changed, so please "
        "re-check Step 1 fresh; this is for reference only."
    )
    for c in flags:
        sec = c.get("section") or ""
        txt = (c.get("claim_text") or "").strip()
        note = (c.get("brief_note") or "").strip()
        head = vlabel.get(c.get("verdict"), c.get("verdict"))
        line = f"- **{head}**" + (f" · _{sec}_" if sec else "") + f" — “{txt}”"
        if note:
            line += f"\n    - your note: _{note}_"
        st.markdown(line)


def render_prior_omission_panel(prior_dict: dict) -> None:
    """Read-only panel above Step 2: domains the reviewer flagged as omitted
    last round (carried-forward re-reviews only). No flags -> renders nothing."""
    flags = [o for o in (prior_dict.get("omission_checks") or []) if o.get("omitted")]
    if not flags:
        return
    st.info(
        "**From your last review — domains you flagged as omitted.** "
        "Re-check against the updated brief; reference only."
    )
    for o in flags:
        label = o.get("domain_label") or o.get("domain") or "?"
        sev = o.get("severity") or ""
        note = (o.get("brief_note") or "").strip()
        line = f"- **{label}**" + (f" · _{sev}_" if sev else "")
        if note:
            line += f"\n    - your note: _{note}_"
        st.markdown(line)


def render_review_notice(output: dict) -> None:
    """Per-case correction banner. Renders when the brief carries
    ``metadata.review_notice`` (stamped on the re-uploaded crfix cases), so a
    reviewer is told to double-check the flagged content (e.g. creatinine /
    renal). Absent on every other case -> renders nothing."""
    notice = (output.get("metadata") or {}).get("review_notice")
    if not notice:
        return
    title = notice.get("title", "Notice")
    body = (notice.get("body") or "").strip()
    st.warning(f"**⚠️ {title}**" + (f"\n\n{body}" if body else ""))


def render_readonly_review(case, existing: ReviewResponse) -> None:
    """Read-only display of an already-submitted review's answers, alongside the
    note. Used so r1/r2 can READ r6's prefilled reviews — no editable widgets are
    instantiated, so nothing can be changed, saved, or have its timer disturbed."""
    src = existing.prefilled_from or existing.reviewer_id
    vlabel = {"verified": "✅", "cannot_verify": "❓", "incorrect": "❌"}

    def _claim_line(c) -> str:
        sec = (getattr(c, "section", "") or "").strip()
        txt = (getattr(c, "claim_text", "") or "").strip()
        note = (getattr(c, "brief_note", "") or "").strip()
        line = f"- {vlabel.get(getattr(c, 'verdict', ''), '•')}"
        line += f" _{sec}_ —" if sec else " —"
        line += f" “{txt}”"
        if note:
            line += f"\n    - note: _{note}_"
        return line

    left, right = st.columns([55, 45])
    with left:
        st.markdown("#### Generated ICU-PAUSE Note")
        render_note(case.output)
        with st.expander("Source data", expanded=False):
            render_source(case.source)

    with right:
        st.markdown(f"#### {src}'s Review &ensp;·&ensp; _read-only_")

        checks = existing.hallucination_checks or []
        vc = {"verified": 0, "cannot_verify": 0, "incorrect": 0}
        for c in checks:
            v = getattr(c, "verdict", None)
            if v in vc:
                vc[v] += 1
        st.markdown(
            f"**Step 1 — Accuracy ({len(checks)} claims):** &ensp;"
            f"✅ {vc['verified']} verified &ensp;·&ensp; ❓ {vc['cannot_verify']} cannot verify "
            f"&ensp;·&ensp; ❌ {vc['incorrect']} incorrect"
        )
        flagged = [c for c in checks if getattr(c, "verdict", None) in ("cannot_verify", "incorrect")]
        if flagged:
            st.caption("Claims not marked verified:")
            for c in flagged:
                st.markdown(_claim_line(c))
        with st.expander(f"Show all {len(checks)} claims", expanded=False):
            for c in checks:
                st.markdown(_claim_line(c))

        st.divider()
        oms = existing.omission_checks or []
        omitted = [o for o in oms if getattr(o, "omitted", False)]
        st.markdown(f"**Step 2 — Omissions:** {len(omitted)} flagged of {len(oms)} domains")
        for o in omitted:
            lbl = getattr(o, "domain_label", "") or getattr(o, "domain", "") or "?"
            sev = getattr(o, "severity", "") or ""
            note = (getattr(o, "brief_note", "") or "").strip()
            line = f"- **{lbl}**" + (f" · _{sev}_" if sev else "")
            if note:
                line += f" — _{note}_"
            st.markdown(line)

        st.divider()
        st.markdown("**Step 3 — PDSQI-9 scores**")
        p = existing.pdsqi9
        if p:
            labels = [
                ("cited", "Cited"), ("accurate", "Accurate"), ("thorough", "Thorough"),
                ("useful", "Useful"), ("organized", "Organized"),
                ("comprehensible", "Comprehensible"), ("succinct", "Succinct"),
                ("synthesized", "Synthesized"),
            ]
            rows = "\n".join(f"| {lab} | {getattr(p, key, '—')} |" for key, lab in labels)
            st.markdown("| Dimension | Score (1–5) |\n|---|---|\n" + rows)
            st.markdown(f"**Stigmatizing language:** {'Yes' if p.stigmatizing else 'No'}")
        else:
            st.caption("No PDSQI-9 scores recorded.")

        free = [
            ("QA issues", (existing.qa_issues_feedback or "").strip()),
            ("Warnings", (existing.warnings_feedback or "").strip()),
            ("Overall", (existing.overall_comment or "").strip()),
        ]
        free = [(k, v) for k, v in free if v]
        if free:
            st.divider()
            st.markdown("**Step 4–5 — Free-text feedback**")
            for k, v in free:
                st.markdown(f"- _{k}:_ {v}")


def render_review_page() -> None:
    reviewer_id = get_reviewer_name()
    hosp_id = st.session_state.get("current_case")
    phase = st.session_state.get("current_phase", "final")

    if not reviewer_id or not hosp_id:
        st.error("No case selected. Return to dashboard.")
        if st.button("Back to dashboard"):
            st.session_state["page"] = "dashboard"
            st.rerun()
        return

    # --- Defense-in-depth: enforce active-batch gate ---
    manifest = load_manifest()
    assignment = manifest.assignment_for(hosp_id) if manifest else None
    if not manifest or assignment is None:
        st.error("This case is not in the assignment manifest.")
        if st.button("Back to dashboard"):
            st.session_state["page"] = "dashboard"
            st.rerun()
        return
    if assignment.batch != manifest.active_batch:
        st.warning(
            f"Batch {assignment.batch} is closed. The active batch is "
            f"Batch {manifest.active_batch}."
        )
        if st.button("Back to dashboard"):
            st.session_state["current_case"] = None
            st.session_state["page"] = "dashboard"
            st.rerun()
        return
    if reviewer_id not in assignment.assigned_to:
        st.error("This case is not assigned to you.")
        if st.button("Back to dashboard"):
            st.session_state["page"] = "dashboard"
            st.rerun()
        return
    binfo = manifest.batch_info(assignment.batch)
    pipeline_version = binfo.pipeline_version if binfo else ""

    # Load case data
    with st.spinner("Loading case data..."):
        try:
            case = load_case(hosp_id)
        except Exception as e:
            st.error(f"Failed to load case {hosp_id}: {e}")
            return

    # Load or initialize draft. Always heal batch/pipeline_version from the live
    # assignment so older drafts pick up the right values on save.
    existing_dict = load_response(reviewer_id, hosp_id)
    if existing_dict:
        existing = ReviewResponse.from_blob_dict(existing_dict)
        existing.batch = assignment.batch
        existing.pipeline_version = pipeline_version
    else:
        existing = ReviewResponse(
            reviewer_id=reviewer_id,
            hosp_id=hosp_id,
            phase=assignment.phase,
            batch=assignment.batch,
            pipeline_version=pipeline_version,
        )

    if existing.is_complete:
        if existing.prefilled_from:
            st.info(
                f"📋 This is **{existing.prefilled_from}**'s review, shown read-only for "
                f"reference. It is **not your own review** — you don't need to do anything here."
            )
            st.caption("Contact the study coordinator with any questions.")
            if st.button("Back to dashboard", key="back_readonly"):
                st.session_state["page"] = "dashboard"
                st.rerun()
            st.divider()
            render_readonly_review(case, existing)
            return
        st.success("This review has already been submitted.")
        st.caption("Your submitted review is locked. Contact the study coordinator to make changes.")
        if st.button("Back to dashboard"):
            st.session_state["page"] = "dashboard"
            st.rerun()
        return

    # Initialize timer state, scoped to this case so resuming a different draft
    # doesn't inherit a stale baseline. timer_baseline carries previously
    # accumulated seconds across draft-resume cycles.
    if st.session_state.get("timer_case") != hosp_id:
        st.session_state["timer_case"] = hosp_id
        st.session_state["timer_baseline"] = existing.time_on_task_seconds
        st.session_state["timer_start"] = time.time()

    # --- Page header ---
    demo = case.source.get("demographics", {})
    age = demo.get("age_at_icu_admission") or demo.get("age_at_admission", "?")
    sex = demo.get("sex_category", "?")
    admit = demo.get("admission_type_category", "?")
    icu_adm = demo.get("icu_admission_dttm", "?")
    icu_los = demo.get("icu_los_hours")
    los_str = f" | ICU LOS: {icu_los:.1f}h" if icu_los else ""

    chip = (
        f"Batch {assignment.batch} · {pipeline_version}"
        if pipeline_version else f"Batch {assignment.batch}"
    )
    col_header, col_back = st.columns([5, 1])
    with col_header:
        st.markdown(f"### Case `{hosp_id}`  ·  _{chip}_")
        st.caption(f"{age}{sex[0] if sex else '?'} | {admit} | ICU admission: {icu_adm}{los_str}")
    with col_back:
        if st.button("Back to dashboard"):
            st.session_state["page"] = "dashboard"
            st.rerun()

    # Per-case correction banner (full width, top of page). Fires only on a
    # carried-over re-review of a flagged case — i.e. a prior rubric exists to
    # reconcile against the corrected brief. A fresh first review has nothing to
    # re-check, so it shows no banner (matches the dashboard Cr badge).
    carried_over = bool(existing_dict and existing_dict.get("pdsqi9"))
    if carried_over:
        render_review_notice(case.output)

    st.divider()

    # --- Two-column layout ---
    # The right column is made sticky via CSS so the review form follows the
    # reviewer as they scroll the page down to read clinical notes on the
    # left. Without this, the claim being verified scrolls out of view as
    # soon as the reviewer scrolls down to the notes section (which sits at
    # the bottom of the source bundle), forcing a scroll-up-to-remember
    # round trip per claim.
    #
    # Mechanism: a marker element rendered inside the right column lets a
    # single CSS rule pick out exactly that column via :has(), so other
    # st.columns calls on the page (header back-button, save/submit row)
    # are unaffected.
    st.markdown(
        """
        <style>
        [data-testid="stColumn"]:has(.icu-pause-sticky-anchor) {
            position: sticky;
            top: 1rem;
            max-height: calc(100vh - 2rem);
            overflow-y: auto;
            align-self: flex-start;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([55, 45])

    with left:
        st.markdown("#### Source Data")
        st.caption("Review the source data and generated note below.")
        render_source(case.source)

        st.divider()
        st.markdown("#### Generated ICU-PAUSE Note")
        render_note(case.output)

    with right:
        st.markdown(
            '<div class="icu-pause-sticky-anchor"></div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### Review Form")
        read_only = is_read_only()
        if read_only:
            st.warning(
                "🔒 The review app is temporarily in **read-only mode**. "
                "You can view cases and your prior responses, but saving "
                "drafts and submitting reviews is disabled for now."
            )
        else:
            st.caption(
                "Complete all five steps below, then submit. "
                "You can save a draft at any point and return later."
            )
        # ---- Carried-over reference (rerun2 re-review) ----
        # The brief was regenerated and its sentences/claims changed, so prior
        # verdicts can't be reused. Surface ONLY the prior non-verified claims
        # and omitted domains (+ the reviewer's own note) as read-only context.
        # Sourced from the v1 backup written by migrate_responses_for_rerun2.py;
        # cached per case to avoid a blob read on every Streamlit rerun.
        _prior_key = f"prior_round_{reviewer_id}_{hosp_id}"
        if _prior_key not in st.session_state:
            st.session_state[_prior_key] = load_prior_round_response(reviewer_id, hosp_id)
        prior_dict = st.session_state[_prior_key] or {}

        render_prior_hallucination_panel(prior_dict)

        # ---- Step 1: Hallucination check ----
        with st.expander("**Step 1 — Accuracy Check (Claim Verification)**", expanded=True):
            with st.container(height=500):
                citation_index = case.output.get("metadata", {}).get("citation_index", {}) or {}
                halluc_items = render_hallucination_widget(
                    case.claims,
                    existing.hallucination_checks,
                    citation_index=citation_index,
                )

        render_prior_omission_panel(prior_dict)

        # ---- Step 2: Omissions check ----
        with st.expander("**Step 2 — Completeness Check (Critical Omissions)**", expanded=True):
            with st.container(height=500):
                omission_items = render_omissions_widget(case.source, existing.omission_checks)

        # ---- Step 3: PDSQI-9 ----
        with st.expander("**Step 3 — PDSQI-9 Quality Scores**", expanded=True):
            with st.container(height=500):
                pdsqi9 = render_pdsqi9_widget(existing.pdsqi9)

        # ---- Step 4: QA Issues & Warnings feedback ----
        with st.expander(
            "**Step 4 — QA Issues & Warnings Feedback (optional)**",
            expanded=False,
        ):
            st.caption(
                "If the note displayed QA Issues or Warnings, share feedback "
                "on whether they were accurate, useful, or missed anything."
            )
            qa_issues_feedback = st.text_area(
                "Feedback on QA Issues",
                value=existing.qa_issues_feedback,
                key="qa_issues_feedback",
                height=100,
            )
            warnings_feedback = st.text_area(
                "Feedback on Warnings",
                value=existing.warnings_feedback,
                key="warnings_feedback",
                height=100,
            )

        # ---- Step 5: Comments ----
        with st.expander("**Step 5 — Overall Comments (optional)**", expanded=False):
            comment = st.text_area(
                "Any additional observations about this note?",
                value=existing.overall_comment,
                key="overall_comment",
                height=120,
            )

        # ---- Timer display ----
        session_elapsed = int(time.time() - st.session_state["timer_start"])
        elapsed = st.session_state["timer_baseline"] + session_elapsed
        st.caption(f"Time on this case: {elapsed // 60}m {elapsed % 60}s")

        # ---- Save / Submit buttons (outside scrollable container) ----
        col_save, col_submit = st.columns(2)

        with col_save:
            if st.button("Save Draft", use_container_width=True, disabled=read_only):
                draft = _build_response(
                    existing=existing,
                    pdsqi9=pdsqi9,
                    halluc=halluc_items,
                    omissions=omission_items,
                    qa_issues_feedback=qa_issues_feedback,
                    warnings_feedback=warnings_feedback,
                    comment=comment,
                    elapsed=elapsed,
                    is_complete=False,
                )
                save_response(draft.to_blob_dict())
                st.success("Draft saved.")

        with col_submit:
            if st.button("Submit Final Review", type="primary", use_container_width=True, disabled=read_only):
                if pdsqi9 is None:
                    st.error("Please complete all PDSQI-9 scores before submitting.")
                elif not halluc_items:
                    st.error("Accuracy check (Step 2) is incomplete.")
                else:
                    final = _build_response(
                        existing=existing,
                        pdsqi9=pdsqi9,
                        halluc=halluc_items,
                        omissions=omission_items,
                        qa_issues_feedback=qa_issues_feedback,
                        warnings_feedback=warnings_feedback,
                        comment=comment,
                        elapsed=elapsed,
                        is_complete=True,
                    )
                    final.mark_submitted()
                    save_response(final.to_blob_dict())
                    st.success("Review submitted! Returning to dashboard...")
                    st.balloons()
                    st.session_state.pop("current_case", None)
                    st.session_state.pop("timer_start", None)
                    st.session_state.pop("timer_baseline", None)
                    st.session_state.pop("timer_case", None)
                    st.session_state["page"] = "dashboard"
                    st.rerun()


def _build_response(
    existing: ReviewResponse,
    pdsqi9,
    halluc: list,
    omissions: list[OmissionItem],
    qa_issues_feedback: str,
    warnings_feedback: str,
    comment: str,
    elapsed: int,
    is_complete: bool,
) -> ReviewResponse:
    return ReviewResponse(
        review_id=existing.review_id,
        reviewer_id=existing.reviewer_id,
        hosp_id=existing.hosp_id,
        phase=existing.phase,
        batch=existing.batch,
        pipeline_version=existing.pipeline_version,
        started_at=existing.started_at,
        submitted_at=existing.submitted_at,
        is_complete=is_complete,
        pdsqi9=pdsqi9,
        hallucination_checks=halluc,
        omission_checks=omissions,
        qa_issues_feedback=qa_issues_feedback,
        warnings_feedback=warnings_feedback,
        overall_comment=comment,
        time_on_task_seconds=elapsed,
    )
