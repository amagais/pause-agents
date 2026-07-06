"""Admin page: assignment management, completion tracking, CSV export, IRR kappa."""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from assignment.assignment_manager import (
    add_batch,
    archive_batch,
    bootstrap_manifest,
    completion_summary,
    generate_manifest,
    load_manifest,
    save_manifest,
    set_active_batch,
)
from storage.blob_client import (
    delete_blobs_prefix,
    list_blobs_prefix,
    list_case_files_with_timestamps,
)
from storage.response_writer import (
    export_claims_csv,
    export_omissions_csv,
    export_summary_csv,
    load_all_responses,
    response_complete,
)


def render_admin_page() -> None:
    st.markdown("## Admin Panel")

    # Admin password gate
    if not st.session_state.get("admin_auth"):
        _, center, _ = st.columns([1, 2, 1])
        with center:
            pw = st.text_input("Admin password", type="password", key="admin_pw")
            if st.button("Sign in", type="primary"):
                if pw == os.environ.get("ADMIN_PASSWORD", ""):
                    st.session_state["admin_auth"] = True
                    st.rerun()
                else:
                    st.error("Incorrect admin password.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Assignment Setup", "Progress", "Export Data", "IRR Analysis",
        "Manage Cases", "Cases on Blob",
    ])

    with tab1:
        _render_assignment_setup()

    with tab2:
        _render_progress()

    with tab3:
        _render_export()

    with tab4:
        _render_irr()

    with tab5:
        _render_manage_cases()

    with tab6:
        _render_cases_on_blob()


# ---------------------------------------------------------------------------
# Tab 1: Assignment setup (Pilot batch workflow + legacy final-phase generator)
# ---------------------------------------------------------------------------

def _render_assignment_setup() -> None:
    sub_a, sub_b, sub_c, sub_d = st.tabs([
        "1a. Bootstrap & Roster",
        "1b. Add batch",
        "1c. Active batch",
        "1d. Final-phase (post-pilot)",
    ])
    with sub_a:
        _render_bootstrap()
    with sub_b:
        _render_add_batch()
    with sub_c:
        _render_active_batch()
    with sub_d:
        _render_legacy_generate()


def _render_bootstrap() -> None:
    st.markdown("#### Bootstrap pilot manifest")
    st.caption(
        "Create an empty manifest with the reviewer roster. Use **1b. Add batch** "
        "to add Batch 1..5 as you generate notes; use **1c. Active batch** to open "
        "a batch to reviewers."
    )

    existing = load_manifest()
    if existing:
        st.info(
            f"Manifest exists: version {existing.version}, "
            f"{len(existing.reviewers)} reviewers, {len(existing.batches)} batch(es), "
            f"active_batch={existing.active_batch}."
        )

    default_reviewers = (
        "\n".join(f"{r.reviewer_id},{r.display_name},{r.email}" for r in existing.reviewers)
        if existing else _PLACEHOLDER_REVIEWERS
    )
    default_seed = existing.random_seed if existing else 42

    with st.form("bootstrap_form"):
        st.markdown("**Reviewers** (one per line: `reviewer_id,Display Name,email@nu.edu`)")
        reviewers_text = st.text_area(
            "Reviewers",
            value=default_reviewers,
            height=150,
            label_visibility="collapsed",
        )
        seed = st.number_input("Random seed", value=default_seed)
        submitted = st.form_submit_button("Bootstrap (or update roster)")

    if submitted:
        reviewers = _parse_reviewers(reviewers_text)
        if not reviewers:
            st.error("Please provide at least one reviewer.")
            return
        if existing is None:
            new_manifest = bootstrap_manifest(reviewers=reviewers, seed=int(seed))
            save_manifest(new_manifest)
            st.success(f"Empty pilot manifest created with {len(reviewers)} reviewers.")
        else:
            existing.reviewers = [r for r in bootstrap_manifest(reviewers, int(seed)).reviewers]
            existing.random_seed = int(seed)
            save_manifest(existing)
            st.success("Reviewer roster and seed updated. Existing batches/assignments preserved.")
        st.rerun()

    # Reset & bootstrap (destructive)
    if existing:
        st.divider()
        with st.expander("Reset & bootstrap fresh manifest (destructive)"):
            st.warning(
                "This wipes all batches, assignments, and active_batch from the manifest. "
                "Reviewer responses on blob are NOT deleted — but they will become orphaned."
            )
            confirm = st.text_input("Type `RESET` to enable the button:")
            if st.button(
                "Reset & bootstrap with current roster",
                type="primary",
                disabled=(confirm != "RESET"),
            ):
                fresh = bootstrap_manifest(
                    reviewers=[
                        {"reviewer_id": r.reviewer_id, "display_name": r.display_name, "email": r.email}
                        for r in existing.reviewers
                    ],
                    seed=existing.random_seed,
                )
                save_manifest(fresh)
                st.success("Manifest reset. All batches/assignments cleared.")
                st.rerun()


def _render_add_batch() -> None:
    st.markdown("#### Add a batch")
    st.caption(
        "Append a new batch of cases to the manifest. Each hosp_id is assigned to ALL "
        "reviewers (every clinician annotates every note in the batch). Adding a batch "
        "does NOT open it — use **1c. Active batch** to do that."
    )

    manifest = load_manifest()
    if manifest is None:
        st.info("No manifest yet. Bootstrap one in tab 1a first.")
        return

    existing_numbers = sorted(b.batch_number for b in manifest.batches)
    next_default = (max(existing_numbers) + 1) if existing_numbers else 1

    with st.form("add_batch_form"):
        col1, col2 = st.columns(2)
        with col1:
            batch_number = st.number_input(
                "Batch number", min_value=1, max_value=99, value=next_default, step=1
            )
        with col2:
            pipeline_version = st.text_input(
                "Pipeline version label", value=f"v{int(next_default)}",
                help="Free-text label, e.g. v1, v2, v2.1",
            )

        label = st.text_input("Optional batch label", value="")

        col3, col4 = st.columns(2)
        with col3:
            date_start = st.date_input("Date window start (optional)", value=None)
        with col4:
            date_end = st.date_input("Date window end (optional)", value=None)

        st.markdown("**hosp_ids in this batch** (one per line, ~5 expected)")
        hosp_text = st.text_area(
            "hosp_ids", value="", height=130, label_visibility="collapsed"
        )
        irr_overlap = st.number_input(
            "IRR overlap cases within this batch (flag is_irr_case)",
            min_value=0, max_value=10, value=0,
        )
        submitted = st.form_submit_button("Add batch")

    if submitted:
        hosp_ids = [x.strip() for x in hosp_text.strip().splitlines() if x.strip()]
        if not hosp_ids:
            st.error("Provide at least one hosp_id.")
            return
        if len(hosp_ids) != 5:
            st.warning(f"Got {len(hosp_ids)} hosp_ids — pilot design expects 5 per batch.")
        try:
            add_batch(
                manifest=manifest,
                batch_number=int(batch_number),
                hosp_ids=hosp_ids,
                pipeline_version=pipeline_version.strip(),
                label=label.strip(),
                date_window_start=date_start.isoformat() if date_start else None,
                date_window_end=date_end.isoformat() if date_end else None,
                irr_overlap_count=int(irr_overlap),
            )
            save_manifest(manifest)
            st.success(
                f"Batch {int(batch_number)} added with {len(hosp_ids)} cases. "
                f"Activate it in tab 1c when ready."
            )
            st.rerun()
        except ValueError as e:
            st.error(str(e))

    # Existing batches table
    st.markdown("##### Batches in manifest")
    if not manifest.batches:
        st.caption("No batches yet.")
    else:
        rows = []
        for b in manifest.batches:
            n_assigned = sum(1 for a in manifest.assignments if a.batch == b.batch_number)
            rows.append({
                "batch": b.batch_number,
                "pipeline_version": b.pipeline_version,
                "label": b.label,
                "dates": _format_window(b.date_window_start, b.date_window_end),
                "n_cases": n_assigned,
                "active": "yes" if manifest.active_batch == b.batch_number else "",
                "created_at": b.created_at,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Archive a batch — moves it (and its assignments) to archived_batches
        # / archived_assignments. Frees the batch_number for reuse.
        with st.expander("Archive a batch (preserves audit trail; frees batch_number)"):
            st.caption(
                "Moves the batch and its assignments out of the active lists "
                "into the archived ones. Reviewer responses on blob are NOT "
                "deleted. The batch_number becomes available for reuse."
            )
            choices = [b.batch_number for b in manifest.batches]
            chosen = st.selectbox(
                "Batch to archive", choices, key="archive_batch_pick",
                format_func=lambda n: f"Batch {n}",
            )
            confirm = st.text_input(
                "Type `ARCHIVE` to confirm:", key="archive_batch_confirm",
            )
            if st.button(
                "Archive batch",
                type="primary",
                disabled=(confirm != "ARCHIVE"),
                key="archive_batch_btn",
            ):
                try:
                    archive_batch(manifest, int(chosen))
                    save_manifest(manifest)
                    st.success(
                        f"Batch {int(chosen)} archived. "
                        f"batch_number={int(chosen)} is now available for reuse."
                    )
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

    # Archived batches table — read-only audit view.
    if manifest.archived_batches:
        st.markdown("##### Archived batches")
        archived_rows = []
        for b in manifest.archived_batches:
            n_archived = sum(
                1 for a in manifest.archived_assignments if a.batch == b.batch_number
            )
            archived_rows.append({
                "batch": b.batch_number,
                "pipeline_version": b.pipeline_version,
                "label": b.label,
                "dates": _format_window(b.date_window_start, b.date_window_end),
                "n_cases": n_archived,
                "created_at": b.created_at,
                "archived_at": b.archived_at or "",
            })
        st.dataframe(pd.DataFrame(archived_rows), use_container_width=True, hide_index=True)


def _render_active_batch() -> None:
    st.markdown("#### Active batch (gate for reviewers)")
    st.caption(
        "Reviewers see only cases in the active batch on their dashboard, and the "
        "review page refuses to load cases from a closed batch. Advancing closes "
        "the previous batch — drafts are preserved on blob but no longer visible."
    )

    manifest = load_manifest()
    if manifest is None:
        st.info("No manifest yet. Bootstrap one in tab 1a first.")
        return

    if not manifest.batches:
        st.info("No batches yet. Add Batch 1 in tab 1b first.")
        return

    current = manifest.active_batch
    if current == 0:
        st.info("No batch is currently active.")
    else:
        binfo = manifest.batch_info(current)
        if binfo:
            st.success(
                f"Active: **Batch {current}** — pipeline `{binfo.pipeline_version}` · "
                f"{_format_window(binfo.date_window_start, binfo.date_window_end)}"
            )

    options = [0] + sorted(b.batch_number for b in manifest.batches)
    labels = ["0 — close all (no batch open)"] + [
        f"{b.batch_number} — {b.pipeline_version}" for b in sorted(manifest.batches, key=lambda x: x.batch_number)
    ]
    idx = options.index(current) if current in options else 0
    chosen_label = st.selectbox("Set active batch to:", labels, index=idx, key="active_batch_pick")
    chosen = options[labels.index(chosen_label)]

    if chosen != current:
        st.warning(
            f"Advancing will close Batch {current} for reviewers. "
            f"Drafts in Batch {current} are preserved on blob but no longer visible."
            if current else "Opening this batch will make it visible to all reviewers."
        )
        confirm = st.text_input("Type `OPEN` to confirm:", key="active_batch_confirm")
        if st.button(
            f"Advance to Batch {chosen}" if chosen else "Close all batches",
            type="primary",
            disabled=(confirm != "OPEN"),
        ):
            set_active_batch(manifest, chosen)
            save_manifest(manifest)
            st.success(
                f"Active batch set to {chosen}." if chosen else "All batches closed."
            )
            st.rerun()


def _render_legacy_generate() -> None:
    st.markdown("#### Final-phase manifest (post-pilot)")
    st.caption(
        "Use this for the post-pilot final study: one-shot pilot/IRR/round-robin "
        "assignment across all reviewers. NOT the pilot workflow — see tabs 1a–1c "
        "for the pilot batch flow. **Generating here will overwrite the manifest.**"
    )

    existing = load_manifest()
    defaults = _form_defaults_from_manifest(existing)

    with st.form("legacy_assignment_form"):
        st.markdown("**Reviewers** (one per line: `reviewer_id,Display Name,email@nu.edu`)")
        reviewers_text = st.text_area(
            "Reviewers", value=defaults["reviewers"], height=150, label_visibility="collapsed"
        )
        st.markdown("**Pilot (iterative) case IDs** (one per line, assigned to ALL reviewers)")
        pilot_text = st.text_area(
            "Pilot cases", value=defaults["pilot"], height=60, label_visibility="collapsed"
        )
        st.markdown("**Targeted case IDs** (one per line: `hosp_id,reviewer_id1,...`)")
        targeted_text = st.text_area(
            "Targeted cases", value=defaults["targeted"], height=80, label_visibility="collapsed"
        )
        st.markdown("**Final case IDs** (one per line, round-robin assigned)")
        final_text = st.text_area(
            "Final cases", value=defaults["final"], height=200, label_visibility="collapsed"
        )
        col1, col2 = st.columns(2)
        with col1:
            irr_count = st.number_input(
                "IRR overlap cases (from final pool)",
                min_value=0, max_value=20, value=defaults["irr_count"],
            )
        with col2:
            seed = st.number_input("Random seed", value=defaults["seed"])
        submitted = st.form_submit_button("Generate & Preview")

    if submitted:
        reviewers = _parse_reviewers(reviewers_text)
        pilot_ids = [x.strip() for x in pilot_text.strip().splitlines() if x.strip()]
        final_ids = [x.strip() for x in final_text.strip().splitlines() if x.strip()]
        targeted = _parse_targeted(targeted_text)
        if not reviewers or (not final_ids and not pilot_ids and not targeted):
            st.error("Please provide at least one reviewer and at least one case ID.")
            return
        manifest = generate_manifest(
            reviewers=reviewers,
            case_ids=final_ids,
            pilot_case_ids=pilot_ids,
            targeted_assignments=targeted,
            irr_count=int(irr_count),
            seed=int(seed),
        )
        st.session_state["preview_manifest"] = manifest
        st.success("Manifest generated. Preview below:")
        rows = [
            {"hosp_id": a.hosp_id, "phase": a.phase, "is_irr": a.is_irr_case,
             "assigned_to": ", ".join(a.assigned_to)}
            for a in manifest.assignments
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if "preview_manifest" in st.session_state:
        if st.button("Push Manifest to Blob Storage", type="primary"):
            save_manifest(st.session_state["preview_manifest"])
            st.success("Manifest saved to Azure Blob Storage.")
            st.session_state.pop("preview_manifest", None)


def _format_window(start, end) -> str:
    if start and end:
        return f"{start} – {end}"
    if start:
        return f"from {start}"
    if end:
        return f"until {end}"
    return "—"


_PLACEHOLDER_REVIEWERS = (
    "r01,Dr. Reviewer One,r1@northwestern.edu\n"
    "r02,Dr. Reviewer Two,r2@northwestern.edu\n"
    "r03,Dr. Reviewer Three,r3@northwestern.edu\n"
    "r04,Dr. Reviewer Four,r4@northwestern.edu\n"
    "r05,Dr. Reviewer Five,r5@northwestern.edu\n"
    "r06,Dr. Reviewer Six,r6@northwestern.edu"
)


def _form_defaults_from_manifest(manifest) -> dict:
    """Build form-field defaults from an existing manifest, falling back to placeholders."""
    if manifest is None:
        return {
            "reviewers": _PLACEHOLDER_REVIEWERS,
            "pilot": "",
            "targeted": "",
            "final": "",
            "irr_count": 5,
            "seed": 42,
        }

    reviewers = "\n".join(
        f"{r.reviewer_id},{r.display_name},{r.email}" for r in manifest.reviewers
    )
    pilot, targeted, final, irr_in_final = [], [], [], 0
    for a in manifest.assignments:
        if a.phase == "iterative":
            pilot.append(a.hosp_id)
        elif a.phase == "targeted":
            targeted.append(",".join([a.hosp_id, *a.assigned_to]))
        elif a.phase == "final":
            final.append(a.hosp_id)
            if a.is_irr_case:
                irr_in_final += 1

    return {
        "reviewers": reviewers,
        "pilot": "\n".join(pilot),
        "targeted": "\n".join(targeted),
        "final": "\n".join(final),
        "irr_count": irr_in_final,
        "seed": manifest.random_seed,
    }


def _parse_reviewers(text: str) -> list[dict]:
    reviewers = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            reviewers.append({
                "reviewer_id": parts[0],
                "display_name": parts[1],
                "email": parts[2] if len(parts) > 2 else "",
            })
    return reviewers


def _parse_targeted(text: str) -> list[dict]:
    """Parse 'hosp_id,reviewer_id1,reviewer_id2,...' lines into targeted assignment dicts."""
    result = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            result.append({"hosp_id": parts[0], "reviewer_ids": parts[1:]})
    return result


# ---------------------------------------------------------------------------
# Tab 2: Progress
# ---------------------------------------------------------------------------

def _render_progress() -> None:
    st.markdown("#### Completion Status")
    manifest = load_manifest()
    if not manifest:
        st.info("No manifest found.")
        return

    # Per-batch summary first (one row per batch)
    if manifest.batches:
        st.markdown("##### Per-batch summary")
        summary_rows = []
        for b in sorted(manifest.batches, key=lambda x: x.batch_number):
            batch_rows = completion_summary(manifest, response_complete, batch=b.batch_number)
            n_assigned_total = sum(r["n_assigned"] for r in batch_rows)
            n_completed_total = sum(r["n_completed"] for r in batch_rows)
            cases_done = sum(1 for r in batch_rows if r["n_completed"] >= r["n_assigned"])
            pct = round(100 * n_completed_total / n_assigned_total, 1) if n_assigned_total else 0.0
            summary_rows.append({
                "batch": b.batch_number,
                "pipeline_version": b.pipeline_version,
                "active": "yes" if manifest.active_batch == b.batch_number else "",
                "n_cases": len(batch_rows),
                "cases_fully_done": f"{cases_done} / {len(batch_rows)}",
                "reviews_complete": f"{n_completed_total} / {n_assigned_total}",
                "completion_pct": pct,
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        st.divider()

    # Per-case detail with batch filter
    st.markdown("##### Per-case detail")
    batch_options = ["All"] + [str(b.batch_number) for b in sorted(manifest.batches, key=lambda x: x.batch_number)]
    pick = st.selectbox("Filter by batch", batch_options, index=0, key="progress_batch_filter")
    batch_arg = None if pick == "All" else int(pick)

    rows = completion_summary(manifest, response_complete, batch=batch_arg)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    total = len(rows)
    done = sum(1 for r in rows if r["n_completed"] >= r["n_assigned"])
    st.metric(
        "Cases fully reviewed" + (f" (Batch {batch_arg})" if batch_arg else ""),
        f"{done} / {total}",
    )


# ---------------------------------------------------------------------------
# Tab 3: Export
# ---------------------------------------------------------------------------

def _render_export() -> None:
    st.markdown("#### Export Review Data")
    if st.button("Load all responses from blob"):
        responses = load_all_responses()
        st.session_state["admin_responses"] = responses
        st.success(f"Loaded {len(responses)} responses.")

    responses = st.session_state.get("admin_responses", [])
    if not responses:
        st.info("Click 'Load all responses' first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Summary CSV",
            data=export_summary_csv(responses),
            file_name="icupause_review_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Claims Detail CSV",
            data=export_claims_csv(responses),
            file_name="icupause_review_claims.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            "Omissions Detail CSV",
            data=export_omissions_csv(responses),
            file_name="icupause_review_omissions.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 4: IRR
# ---------------------------------------------------------------------------

def _render_irr() -> None:
    st.markdown("#### Inter-Rater Reliability (Cohen's Kappa)")
    st.caption("Computed on cases assigned to 2+ reviewers (IRR cases).")

    responses = st.session_state.get("admin_responses")
    if not responses:
        st.info("Load responses in the Export tab first.")
        return

    try:
        from scipy.stats import cohen_kappa_score
    except ImportError:
        st.error("scipy not installed.")
        return

    # Find IRR cases (hosp_id appearing in 2+ completed responses)
    from collections import defaultdict
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in responses:
        if r.get("is_complete") and r.get("pdsqi9"):
            by_case[r["hosp_id"]].append(r)

    irr_cases = {cid: rs for cid, rs in by_case.items() if len(rs) >= 2}

    if not irr_cases:
        st.info("No IRR cases with 2+ completed responses found.")
        return

    attrs = ["cited", "accurate", "thorough", "useful", "organized", "comprehensible", "succinct", "synthesized"]
    kappa_rows = []

    for attr in attrs:
        rater1_scores, rater2_scores = [], []
        for cid, rs in irr_cases.items():
            # Take first two reviewers alphabetically for consistency
            sorted_rs = sorted(rs, key=lambda r: r["reviewer_id"])
            s1 = sorted_rs[0]["pdsqi9"].get(attr)
            s2 = sorted_rs[1]["pdsqi9"].get(attr)
            if s1 is not None and s2 is not None:
                rater1_scores.append(s1)
                rater2_scores.append(s2)

        if len(rater1_scores) >= 2:
            kappa = cohen_kappa_score(rater1_scores, rater2_scores, weights="linear")
            kappa_rows.append({
                "Attribute": attr,
                "Kappa (linear weighted)": round(kappa, 3),
                "Interpretation": _kappa_label(kappa),
                "N cases": len(rater1_scores),
            })

    if kappa_rows:
        st.dataframe(pd.DataFrame(kappa_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Not enough paired data to compute kappa.")


def _kappa_label(k: float) -> str:
    if k < 0:
        return "Poor (< 0)"
    elif k < 0.20:
        return "Slight (0-0.20)"
    elif k < 0.40:
        return "Fair (0.21-0.40)"
    elif k < 0.60:
        return "Moderate (0.41-0.60)"
    elif k < 0.80:
        return "Substantial (0.61-0.80)"
    else:
        return "Near-perfect (0.81-1.0)"


# ---------------------------------------------------------------------------
# Tab 5: Manage Cases — list / inspect / delete cases from blob storage
# ---------------------------------------------------------------------------

def _render_manage_cases() -> None:
    st.markdown("#### Manage Cases in Blob Storage")
    st.caption(
        "Hard-deletes case blobs (`cases/<hosp_id>/`).  Optionally also "
        "wipes all reviewer responses for the deleted cases.  This action "
        "cannot be undone."
    )

    # --- Discover cases ---
    if st.button("Refresh case list"):
        st.session_state.pop("manage_cases_index", None)

    if "manage_cases_index" not in st.session_state:
        with st.spinner("Listing cases…"):
            st.session_state["manage_cases_index"] = _build_case_index()

    case_index: dict[str, dict] = st.session_state["manage_cases_index"]
    if not case_index:
        st.info("No cases found in blob storage under `cases/`.")
        return

    # --- Display cases as a selectable table ---
    rows = []
    for hosp_id, info in sorted(case_index.items()):
        rows.append({
            "Select": False,
            "hosp_id": hosp_id,
            "batch": info.get("batch", 0),
            "blob_files": info["blob_count"],
            "drafts": info["n_drafts"],
            "submitted": info["n_submitted"],
            "in_manifest": "yes" if info["in_manifest"] else "no",
            "assigned_to": ", ".join(info["assigned_to"]) or "—",
        })

    edited = st.data_editor(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        disabled=["hosp_id", "batch", "blob_files", "drafts", "submitted", "in_manifest", "assigned_to"],
        key="manage_cases_editor",
    )

    selected_ids = [r["hosp_id"] for r in edited.to_dict("records") if r["Select"]]
    if not selected_ids:
        st.caption("Tick the **Select** column on rows you want to delete.")
        return

    # --- Show what will be deleted ---
    st.markdown(f"**{len(selected_ids)} case(s) selected for deletion:**")
    affected_drafts = sum(case_index[h]["n_drafts"] for h in selected_ids)
    affected_subs = sum(case_index[h]["n_submitted"] for h in selected_ids)
    affected_manifest = [h for h in selected_ids if case_index[h]["in_manifest"]]
    if affected_manifest:
        st.warning(
            f"{len(affected_manifest)} of these cases are still in the assignment manifest "
            f"({', '.join(affected_manifest)}).  The manifest will NOT be updated automatically; "
            "regenerate it from the Assignment Setup tab if you want the dangling assignments "
            "cleaned up."
        )

    also_delete_responses = st.checkbox(
        f"Also delete reviewer responses ({affected_drafts} draft, {affected_subs} submitted)",
        value=False,
        key="manage_cases_also_responses",
    )

    confirm_phrase = "DELETE"
    typed = st.text_input(
        f"Type `{confirm_phrase}` to confirm:",
        key="manage_cases_confirm",
        value="",
    )
    if st.button("Hard-delete selected cases", type="primary", disabled=(typed != confirm_phrase)):
        deleted_summary = _hard_delete_cases(selected_ids, also_delete_responses)
        st.success(
            f"Deleted {deleted_summary['cases']} case file(s)"
            + (f" and {deleted_summary['responses']} response file(s)." if also_delete_responses else ".")
        )
        # Force a fresh list on next render. Use pop (delete) — assigning
        # a new value to a widget-bound key after the widget rendered raises
        # StreamlitAPIException, but deletion is allowed and recreates the
        # widget at default on rerun.
        st.session_state.pop("manage_cases_index", None)
        st.session_state.pop("manage_cases_editor", None)
        st.session_state.pop("manage_cases_confirm", None)
        st.rerun()


def _build_case_index() -> dict[str, dict]:
    """Inspect blob storage and build per-case metadata.

    Returns dict keyed by hosp_id with: blob_count, n_drafts, n_submitted,
    in_manifest, assigned_to.
    """
    case_blobs = list_blobs_prefix("cases/")
    case_ids: dict[str, int] = {}
    for path in case_blobs:
        # path looks like "cases/<hosp_id>/output.json"
        parts = path.split("/", 2)
        if len(parts) < 3 or not parts[1]:
            continue
        case_ids[parts[1]] = case_ids.get(parts[1], 0) + 1

    # Build manifest assignment lookup (hosp_id -> [reviewer_ids]) and batch lookup
    manifest_lookup: dict[str, list[str]] = {}
    batch_lookup: dict[str, int] = {}
    manifest = load_manifest()
    if manifest:
        for a in manifest.assignments:
            manifest_lookup.setdefault(a.hosp_id, []).extend(a.assigned_to)
            batch_lookup[a.hosp_id] = a.batch

    # Count responses per case
    response_blobs = list_blobs_prefix("responses/")
    drafts: dict[str, int] = {}
    submitted: dict[str, int] = {}
    for path in response_blobs:
        # path: "responses/<reviewer_id>/<hosp_id>.json"
        parts = path.split("/")
        if len(parts) != 3 or not parts[2].endswith(".json"):
            continue
        hosp_id = parts[2][: -len(".json")]
        # Cheap check: differentiate draft vs submitted via load
        try:
            from storage.blob_client import read_json
            data = read_json(path)
            if data.get("is_complete"):
                submitted[hosp_id] = submitted.get(hosp_id, 0) + 1
            else:
                drafts[hosp_id] = drafts.get(hosp_id, 0) + 1
        except Exception:
            drafts[hosp_id] = drafts.get(hosp_id, 0) + 1

    index: dict[str, dict] = {}
    for hosp_id, blob_count in case_ids.items():
        index[hosp_id] = {
            "blob_count": blob_count,
            "n_drafts": drafts.get(hosp_id, 0),
            "n_submitted": submitted.get(hosp_id, 0),
            "in_manifest": hosp_id in manifest_lookup,
            "assigned_to": manifest_lookup.get(hosp_id, []),
            "batch": batch_lookup.get(hosp_id, 0),
        }
    return index


def _hard_delete_cases(hosp_ids: list[str], also_responses: bool) -> dict[str, int]:
    """Hard-delete the given case blobs (and optionally their responses).

    Returns counts: {"cases": N_case_files_deleted, "responses": N_response_files_deleted}.
    """
    n_cases = 0
    for hosp_id in hosp_ids:
        deleted = delete_blobs_prefix(f"cases/{hosp_id}/")
        n_cases += len(deleted)

    n_responses = 0
    if also_responses:
        # Responses live at responses/<reviewer_id>/<hosp_id>.json — no per-case prefix,
        # so iterate the full responses/ tree and delete by suffix match.
        targets = set(hosp_ids)
        for path in list_blobs_prefix("responses/"):
            parts = path.split("/")
            if len(parts) == 3 and parts[2].endswith(".json"):
                hosp_id = parts[2][: -len(".json")]
                if hosp_id in targets:
                    deleted = delete_blobs_prefix(path)  # delete_blobs_prefix on a single path also works
                    n_responses += len(deleted)

    return {"cases": n_cases, "responses": n_responses}


# ---------------------------------------------------------------------------
# Tab 6: Cases on Blob — last-modified timestamps for upload version control
# ---------------------------------------------------------------------------

def _render_cases_on_blob() -> None:
    st.markdown("#### Cases on Azure Blob")
    st.caption(
        "Last-modified timestamps for each case's three files. Use this to "
        "verify a fresh upload landed (sort by output.json descending)."
    )

    if st.button("Refresh from Blob", key="refresh_cases_on_blob"):
        st.session_state.pop("admin_blob_cases", None)

    rows = st.session_state.get("admin_blob_cases")
    if rows is None:
        with st.spinner("Listing blobs…"):
            rows = list_case_files_with_timestamps()
            st.session_state["admin_blob_cases"] = rows

    if not rows:
        st.info("No cases found under cases/ prefix.")
        return

    df = pd.DataFrame(rows)
    for col in ("output.json", "source_bundle.json", "claims.json"):
        df[col] = df[col].apply(
            lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S UTC") if dt is not None else "—"
        )
    df = df.sort_values("hosp_id").reset_index(drop=True)

    st.metric("Total cases on blob", len(df))
    st.dataframe(df, use_container_width=True, hide_index=True)
