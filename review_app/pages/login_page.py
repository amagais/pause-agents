"""Login page: Entra SSO + per-reviewer password (combined gate)."""

from __future__ import annotations

import streamlit as st

from auth.msal_auth import check_auth, get_reviewer_name


def render_login_page() -> bool:
    """
    Render the authentication flow.

    The per-reviewer password gate (Entra SSO -> name + personal password)
    now lives inside check_auth() and sets ``reviewer_id`` in session on
    success — so this page is just a thin wrapper that hands off to the
    auth module and reports done when reviewer_id is set.
    """
    # Centered layout for login
    _, center, _ = st.columns([1, 2, 1])

    with center:
        st.markdown("")
        st.markdown("")
        st.markdown("## ICU-PAUSE Clinician Review")
        st.caption("Northwestern University — Study Access Portal")
        st.divider()

        if not check_auth():
            return False

        # Auth passed AND reviewer_id is set in session — done.
        return get_reviewer_name() is not None
