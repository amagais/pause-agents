"""Dashboard: reviewer case queue and progress tracking."""

from __future__ import annotations

import streamlit as st

from assignment.assignment_manager import load_manifest
from auth.msal_auth import get_reviewer_name, logout
from storage.case_flags import load_case_badges
from storage.response_writer import (
    is_read_only,
    load_response,
    response_complete,
    save_response,
)


def _badge(badges: dict[str, str], hosp_id: str) -> str:
    """Short inline chip (e.g. ' · 🩸 Cr') for a flagged case, else ''."""
    label = badges.get(hosp_id)
    return f" · 🩸 {label}" if label else ""

# A completed review with less time-on-task than this is likely an accidental
# submit — surface an "are you sure?" nudge plus a Reopen action.
LOW_TIME_SECONDS = 60


def render_dashboard_page() -> None:
    reviewer_id = get_reviewer_name()
    if not reviewer_id:
        st.error("Not logged in.")
        return

    manifest = load_manifest()
    if not manifest:
        st.error("No assignment manifest found.")
        return

    # Find this reviewer's display name
    reviewer = next((r for r in manifest.reviewers if r.reviewer_id == reviewer_id), None)
    display_name = reviewer.display_name if reviewer else reviewer_id

    st.markdown(f"## Welcome back, {display_name}")

    # Two modes:
    #  - Pilot batch mode (manifest.batches populated): active_batch gates
    #    which cases are visible; batch header rendered.
    #  - Final-round mode (manifest.batches empty): single-shot manifest,
    #    all assigned cases are visible to their assigned reviewer, no
    #    batch header.
    if manifest.batches:
        active = manifest.active_batch
        binfo = manifest.batch_info(active) if active else None
        if not active or binfo is None:
            st.info("The study has not opened yet — please check back soon.")
            st.divider()
            if st.button("Sign out"):
                logout()
                st.rerun()
            return

        total_batches = len(manifest.batches)
        window = ""
        if binfo.date_window_start and binfo.date_window_end:
            window = f"  ·  {binfo.date_window_start} – {binfo.date_window_end}"
        elif binfo.date_window_start:
            window = f"  ·  starting {binfo.date_window_start}"
        label = f"  ·  {binfo.label}" if binfo.label else ""
        st.markdown(
            f"### Batch {binfo.batch_number} of {total_batches} — pipeline `{binfo.pipeline_version}`"
            f"{label}{window}"
        )
        st.divider()

        assignments = manifest.cases_for_reviewer(reviewer_id, batch=active)
    else:
        assignments = manifest.cases_for_reviewer(reviewer_id)
    if not assignments:
        if manifest.batches:
            st.info(f"No cases in Batch {binfo.batch_number} are assigned to you.")
        else:
            st.info("No cases are currently assigned to you.")
        st.divider()
        if st.button("Sign out"):
            logout()
            st.rerun()
        return

    # Compute progress
    completed = [a for a in assignments if response_complete(reviewer_id, a.hosp_id)]
    n_done = len(completed)
    n_total = len(assignments)

    # Progress section
    col_prog, col_metric = st.columns([3, 1])
    with col_prog:
        st.progress(n_done / n_total if n_total else 0, text=f"{n_done} of {n_total} cases completed")
    with col_metric:
        pct = int(100 * n_done / n_total) if n_total else 0
        st.metric("Completion", f"{pct}%")

    st.markdown("")

    # Split into pending and completed
    pending = [a for a in assignments if not response_complete(reviewer_id, a.hosp_id)]
    done = [a for a in assignments if response_complete(reviewer_id, a.hosp_id)]

    badges = load_case_badges()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### To Review")
        if not pending:
            st.success("All done! No pending cases.")
        for a in pending:
            draft = load_response(reviewer_id, a.hosp_id)
            # A draft that already carries a rubric is a re-review whose answers
            # were carried over from last round (Step 1/2 were cleared by the
            # rerun2 migration); flag so the reviewer knows what to re-check.
            carried_over = bool(draft and not draft.get("is_complete") and draft.get("pdsqi9"))
            if draft and not draft.get("is_complete"):
                marker, label = "🟡", "Resume draft"
            else:
                marker, label = "⬜", "Start review"
            # Badge only on carried-over re-reviews: a case this reviewer scored
            # before and must now re-check. On a fresh (never-reviewed) case there
            # is no prior rating to reconcile, so the chip would be noise.
            badge_str = _badge(badges, a.hosp_id) if carried_over else ""
            if st.button(f"{marker}  **{a.hosp_id}**{badge_str} — {label}", key=f"open_{a.hosp_id}_{a.phase}"):
                st.session_state["current_case"] = a.hosp_id
                st.session_state["current_phase"] = a.phase
                st.session_state["current_batch"] = a.batch
                st.session_state["page"] = "review"
                st.rerun()
            if carried_over:
                st.caption(
                    "↻ Carried over from last round — your rubric & comments are kept; "
                    "Step 1 & 2 were cleared, please re-check them against the updated brief."
                )

    with col2:
        st.markdown("#### Completed")
        if not done:
            st.caption("No completed reviews yet.")
        for a in done:
            resp = load_response(reviewer_id, a.hosp_id) or {}
            # Copied from another reviewer (e.g. r6's reviews prefilled here for
            # reference): label it as theirs, read-only, and skip the time-warning
            # and Reopen action — it isn't this reviewer's to edit.
            src = resp.get("prefilled_from")
            if src:
                if st.button(
                    f"👁️  **{a.hosp_id}** — review by {src} (read-only)",
                    key=f"view_{a.hosp_id}_{a.phase}",
                ):
                    st.session_state["current_case"] = a.hosp_id
                    st.session_state["current_phase"] = a.phase
                    st.session_state["current_batch"] = a.batch
                    st.session_state["page"] = "review"
                    st.rerun()
                st.caption(f"{src}'s submitted review — open to read it (read-only, not your own).")
                continue
            t = int(resp.get("time_on_task_seconds", 0) or 0)
            st.markdown(f"✅ &ensp;**{a.hosp_id}**")
            if t < LOW_TIME_SECONDS:
                st.caption(
                    f"⚠️ Only {t}s recorded on this review — are you sure you finished? "
                    "If you hit submit by accident, reopen it below."
                )
            if is_read_only():
                st.caption("_Editing is frozen (read-only mode)._")
            elif st.button("Reopen", key=f"reopen_{a.hosp_id}_{a.phase}"):
                resp["is_complete"] = False
                resp["submitted_at"] = None
                save_response(resp)
                st.session_state["current_case"] = a.hosp_id
                st.session_state["current_phase"] = a.phase
                st.session_state["current_batch"] = a.batch
                st.session_state["page"] = "review"
                st.rerun()

    st.divider()
    if st.button("Sign out"):
        logout()
        st.rerun()
