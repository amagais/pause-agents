"""MSAL-based Azure Entra authentication + per-reviewer password gate for Streamlit."""

from __future__ import annotations

import os
import urllib.parse

import bcrypt
import msal
import streamlit as st


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _auth_mode() -> str:
    return os.environ.get("REVIEW_APP_AUTH_MODE", "entra_plus_password")


def _entra_required() -> bool:
    return _auth_mode() == "entra_plus_password"


def _build_msal_app() -> msal.PublicClientApplication:
    tenant_id = os.environ["AZURE_TENANT_ID"]
    client_id = os.environ["AZURE_CLIENT_ID"]
    authority = f"https://login.microsoftonline.com/{tenant_id}"
    return msal.PublicClientApplication(client_id=client_id, authority=authority)


SCOPES = ["User.Read"]


# ---------------------------------------------------------------------------
# Entra (MSAL) flow
# ---------------------------------------------------------------------------

def _get_entra_token() -> dict | None:
    """Return cached Entra token or None if not authenticated."""
    return st.session_state.get("entra_token")


def _try_acquire_token_by_code(msal_app: msal.PublicClientApplication) -> dict | None:
    """Attempt to exchange the auth code from the URL query params for a token."""
    params = st.query_params
    code = params.get("code")
    state = params.get("state")
    if not code:
        return None
    if state != st.session_state.get("msal_state"):
        st.error("State mismatch — possible CSRF. Please try again.")
        return None
    redirect_uri = _redirect_uri()
    result = msal_app.acquire_token_by_authorization_code(
        code=code,
        scopes=SCOPES,
        redirect_uri=redirect_uri,
    )
    if "error" in result:
        st.error(f"Entra authentication failed: {result.get('error_description', result['error'])}")
        return None
    # Clear code from URL
    st.query_params.clear()
    return result


def _redirect_uri() -> str:
    """Build the redirect URI from Streamlit's current URL."""
    # Prefer explicit env var; fall back to localhost for local dev
    return os.environ.get("STREAMLIT_REDIRECT_URI", "http://localhost:8501/")


def _entra_login_url(msal_app: msal.PublicClientApplication) -> str:
    import secrets
    state = secrets.token_urlsafe(16)
    st.session_state["msal_state"] = state
    auth_url = msal_app.get_authorization_request_url(
        scopes=SCOPES,
        redirect_uri=_redirect_uri(),
        state=state,
    )
    return auth_url


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_auth() -> bool:
    """
    Run the full auth gate. Returns True if the user is fully authenticated
    AND has selected their reviewer identity via per-reviewer password.

    Handles:
      1. Entra SSO (MSAL) — only if REVIEW_APP_AUTH_MODE=entra_plus_password
      2. Per-reviewer password gate (replaces previous shared password +
         separate reviewer-selector flow). User picks their name and enters
         their personal password; bcrypt-verified against the manifest.
         On success, ``reviewer_id`` is set in session.

    Loads the assignment manifest internally so callers don't have to.
    """
    # --- Step 1: Entra (MSAL) ---
    if _entra_required():
        if not _get_entra_token():
            msal_app = _build_msal_app()
            # Check if we're returning from Entra with a code
            token = _try_acquire_token_by_code(msal_app)
            if token:
                st.session_state["entra_token"] = token
                st.session_state["entra_user"] = token.get("id_token_claims", {}).get("name", "NU User")
            else:
                # Show login prompt
                _render_entra_login(msal_app)
                return False

    # --- Step 2: Per-reviewer password gate ---
    if not st.session_state.get("reviewer_id"):
        _render_reviewer_password_gate()
        return False

    # --- Step 3: Forced password change (first login on a temp password) ---
    # Set by the reviewer-password gate when bcrypt-match succeeds against a
    # reviewer flagged password_must_change=True. The admin issued a temp
    # password via set_reviewer_password.py --temp; reviewer sets their own
    # before reaching the rest of the app.
    if st.session_state.get("password_change_pending"):
        _render_change_password_form()
        return False

    return True


def _render_entra_login(msal_app: msal.PublicClientApplication) -> None:
    login_url = _entra_login_url(msal_app)
    st.markdown("### Sign in with your Northwestern account")
    st.markdown(
        f'<a href="{login_url}" target="_self">'
        '<button style="background:#4E2A84;color:white;padding:10px 24px;'
        'border:none;border-radius:4px;font-size:16px;cursor:pointer;">'
        "Sign in with NU Account"
        "</button></a>",
        unsafe_allow_html=True,
    )


def _render_reviewer_password_gate() -> None:
    """Combined gate: pick reviewer + enter their personal password.

    Replaces the previous flow of shared password gate + separate reviewer
    selector. On bcrypt-verified password match, sets ``reviewer_id`` in
    session and reruns. Reviewer-id-bearing session is then the gate for
    subsequent pages (see app.py routing).
    """
    # Lazy import to avoid a circular dep with assignment_manager.
    from assignment.assignment_manager import load_manifest

    st.markdown("### Study Access")
    if _entra_required():
        user = st.session_state.get("entra_user", "")
        if user:
            st.info(f"Signed in as: **{user}**")

    manifest = load_manifest()
    if manifest is None:
        st.error(
            "No assignment manifest found. "
            "Please ask the study administrator to set up the reviewer assignments."
        )
        return

    reviewers = manifest.reviewers
    if not reviewers:
        st.error(
            "No reviewers configured. Ask the study admin to set up the manifest."
        )
        return

    # Display-name dropdown so the reviewer recognizes themselves.
    options = {r.display_name: r for r in reviewers}
    chosen_name = st.selectbox(
        "Select your name:",
        options=list(options.keys()),
        index=None,
        placeholder="— choose your name —",
        key="reviewer_select_login",
    )
    pw = st.text_input(
        "Enter your study password:", type="password", key="reviewer_pw_input"
    )

    if st.button("Sign in", type="primary"):
        if not chosen_name:
            st.error("Please select your name from the list.")
            return
        reviewer = options[chosen_name]
        if not reviewer.password_hash:
            st.error(
                f"No password is set for {chosen_name}. "
                "Please contact the study administrator."
            )
            return
        try:
            ok = bcrypt.checkpw(pw.encode("utf-8"), reviewer.password_hash.encode("utf-8"))
        except (ValueError, TypeError):
            # Malformed hash in the manifest — treat as non-matching but
            # surface a coordinator-actionable message.
            st.error(
                "Stored password hash is malformed. Please contact the "
                "study administrator to reset your password."
            )
            return
        if ok:
            st.session_state["reviewer_id"] = reviewer.reviewer_id
            # If admin issued a temp password (must_change=True), gate the
            # rest of the app behind a forced password change so the temp
            # cannot be reused as a permanent credential.
            if reviewer.password_must_change:
                st.session_state["password_change_pending"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please check with the study coordinator.")


def _render_change_password_form() -> None:
    """Forced password change after first login on a temp password.

    Reads the current reviewer's session_state.reviewer_id, prompts for a
    new password (twice for confirmation), bcrypt-hashes it, writes back
    to the manifest with password_must_change cleared, and removes the
    pending flag from session. Does NOT allow the reviewer to skip — the
    pending flag stays set until a successful change.
    """
    from assignment.assignment_manager import load_manifest, save_manifest

    MIN_LENGTH = 12

    rid = st.session_state.get("reviewer_id")
    if not rid:
        # Defensive: shouldn't happen because check_auth gates this on
        # reviewer_id being set, but if session state is corrupted bail out
        # rather than crash.
        st.error("Session error: reviewer not identified. Please sign out and back in.")
        return

    st.markdown("### Set a new password")
    st.caption(
        "You signed in with a temporary password. Choose a permanent "
        "password to continue. The study admin will not see your new "
        "password — only you and the system will know it."
    )
    st.markdown(f"**Account:** `{rid}`")

    new_pw = st.text_input(
        f"New password (minimum {MIN_LENGTH} characters):",
        type="password", key="new_password_input",
    )
    confirm_pw = st.text_input(
        "Confirm new password:",
        type="password", key="confirm_password_input",
    )

    if st.button("Save and continue", type="primary"):
        if len(new_pw) < MIN_LENGTH:
            st.error(f"Password too short. Need at least {MIN_LENGTH} characters.")
            return
        if new_pw != confirm_pw:
            st.error("Passwords do not match. Please re-enter both.")
            return

        # Load fresh manifest to avoid clobbering concurrent admin edits
        manifest = load_manifest()
        if manifest is None:
            st.error("Manifest unavailable. Please contact the study coordinator.")
            return
        target = next((r for r in manifest.reviewers if r.reviewer_id == rid), None)
        if target is None:
            st.error("Reviewer record not found. Please contact the study coordinator.")
            return

        new_hash = bcrypt.hashpw(new_pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        target.password_hash = new_hash
        target.password_must_change = False
        try:
            save_manifest(manifest)
        except Exception as e:
            st.error(f"Could not save new password: {e}. Please try again.")
            return

        st.session_state.pop("password_change_pending", None)
        # Clear the input keys so a subsequent rerun doesn't leave stale
        # plaintext in widget state.
        st.session_state.pop("new_password_input", None)
        st.session_state.pop("confirm_password_input", None)
        st.success("Password updated. Continuing to your case queue...")
        st.rerun()


def get_reviewer_name() -> str | None:
    """Return the currently selected reviewer_id, or None if not selected."""
    return st.session_state.get("reviewer_id")


def logout() -> None:
    """Clear all auth state."""
    # app_password_ok is from the legacy shared-password gate; kept in the
    # clear-list for backward-compat with sessions established before the
    # per-reviewer migration. Safe to remove after one deployment cycle.
    for key in [
        "entra_token", "app_password_ok", "reviewer_id",
        "msal_state", "entra_user",
        "reviewer_select_login", "reviewer_pw_input",
        "password_change_pending",
        "new_password_input", "confirm_password_input",
    ]:
        st.session_state.pop(key, None)
