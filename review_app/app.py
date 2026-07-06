"""ICU-PAUSE Clinician Review App — Streamlit entry point."""

from __future__ import annotations

import os
import sys

# Add review_app/ to path so all submodules are importable
sys.path.insert(0, os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env: {e}", file=sys.stderr)

import streamlit as st

st.set_page_config(
    page_title="ICU-PAUSE Review",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit's auto-discovered multipage nav. The app.py router below
# is the only routing surface; the per-file links Streamlit injects under
# pages/ lead to dead pages because each module is invoked through a
# render_*_page() function, not as a standalone Streamlit script.
st.markdown(
    """
    <style>
        [data-testid='stSidebarNav'],
        [data-testid='stSidebarNavItems'],
        [data-testid='stSidebarNavLink'],
        section[data-testid='stSidebar'] > div:first-child > div:first-child > ul {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Apply Apple HIG theme
from display.theme import inject_apple_hig_css
inject_apple_hig_css()

# ---------------------------------------------------------------------------
# Sidebar: quick nav
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ICU-PAUSE Review")
    reviewer_id = st.session_state.get("reviewer_id", "")
    if reviewer_id:
        st.caption(f"Logged in as: {reviewer_id}")
        if st.button("Dashboard"):
            st.session_state["page"] = "dashboard"
            st.rerun()
        if st.button("Sign out"):
            from auth.msal_auth import logout
            logout()
            st.session_state["page"] = "login"
            st.rerun()

    st.divider()
    if st.button("Demo"):
        st.session_state["page"] = "demo"
        st.rerun()
    if st.button("Admin"):
        st.session_state["page"] = "admin"
        st.rerun()

# ---------------------------------------------------------------------------
# Page router
# ---------------------------------------------------------------------------

page = st.session_state.get("page", "login")

if page == "login":
    from pages.login_page import render_login_page
    authenticated = render_login_page()
    if authenticated:
        st.session_state["page"] = "dashboard"
        st.rerun()

elif page == "dashboard":
    from auth.msal_auth import get_reviewer_name
    if not get_reviewer_name():
        st.session_state["page"] = "login"
        st.rerun()
    from pages.dashboard_page import render_dashboard_page
    render_dashboard_page()

elif page == "review":
    from auth.msal_auth import get_reviewer_name
    if not get_reviewer_name():
        st.session_state["page"] = "login"
        st.rerun()
    from pages.review_page import render_review_page
    render_review_page()

elif page == "demo":
    from pages.demo_page import render_demo_page
    render_demo_page()

elif page == "admin":
    from pages.admin_page import render_admin_page
    render_admin_page()

else:
    st.error(f"Unknown page: {page!r}")
    st.session_state["page"] = "login"
    st.rerun()
