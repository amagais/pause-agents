"""Set or rotate a reviewer's password in the assignment manifest.

Hashes the supplied password with bcrypt and writes it to the
``password_hash`` field of the matching Reviewer entry in the manifest
stored in Azure Blob Storage. Never persists plaintext.

Usage
-----
Issue a one-time temporary password (recommended — admin never sees the
final password the reviewer will use):

    .venv/bin/python review_app/scripts/set_reviewer_password.py r01 --temp

The script generates a cryptographically random password, prints it ONCE
to the terminal, sets the reviewer's password_must_change flag, and
hashes it into the manifest. Send the temp via secure channel; the
reviewer is forced to set their own permanent password on first login.

Set an admin-chosen password (admin sees the password during entry):

    .venv/bin/python review_app/scripts/set_reviewer_password.py r01

Bulk reset (interactive prompts for each reviewer in the manifest):

    .venv/bin/python review_app/scripts/set_reviewer_password.py --all
    .venv/bin/python review_app/scripts/set_reviewer_password.py --all --temp

Show which reviewers have / don't have passwords set, and which still
need to change a temp:

    .venv/bin/python review_app/scripts/set_reviewer_password.py --status

The reviewer_id is the short identifier (e.g. "r01"), not the display
name. Use --status to list available IDs.
"""

from __future__ import annotations

import argparse
import getpass
import os
import secrets
import sys
from pathlib import Path

# Add review_app/ to path so submodules import cleanly.
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "review_app"))

try:
    from dotenv import load_dotenv
    load_dotenv(_repo_root / "review_app" / ".env")
except Exception:
    pass

import bcrypt  # noqa: E402

from assignment.assignment_manager import load_manifest, save_manifest  # noqa: E402


MIN_PASSWORD_LENGTH = 12
TEMP_PASSWORD_LENGTH = 16  # ~96 bits of entropy from token_urlsafe alphabet


def _prompt_password(reviewer_id: str, display_name: str) -> str:
    """Prompt for a password twice (confirmation), enforce min length."""
    print(f"\nSetting password for {reviewer_id} ({display_name})")
    print(f"  Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    while True:
        pw1 = getpass.getpass("  New password: ")
        if len(pw1) < MIN_PASSWORD_LENGTH:
            print(
                f"  Password too short ({len(pw1)} chars). "
                f"Need at least {MIN_PASSWORD_LENGTH}."
            )
            continue
        pw2 = getpass.getpass("  Confirm password: ")
        if pw1 != pw2:
            print("  Passwords do not match. Try again.")
            continue
        return pw1


def _generate_temp_password() -> str:
    """Generate a cryptographically random temp password.

    Uses secrets.token_urlsafe which gives base64url alphabet (A-Za-z0-9-_).
    Easier to type than mixed-special-char passwords and still high
    entropy — 16 chars from a 64-char alphabet = ~96 bits.
    """
    # token_urlsafe(n) returns ceil(4n/3) chars; n=12 -> 16 chars
    return secrets.token_urlsafe(12)


def _hash(password: str) -> str:
    """Bcrypt-hash a password. Default cost factor (12) ≈ 50ms verify time
    on commodity hardware — slow enough to deter brute force, fast enough
    to keep login responsive.
    """
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")


def _print_temp_banner(reviewer_id: str, display_name: str, temp: str) -> None:
    """Print a temp password to terminal with a clear banner.

    The banner is intentional — makes the password obvious to the admin
    distributing it and easy to scroll-back-and-clear afterward. The
    banner also reminds the admin not to share via insecure channels.
    """
    print()
    print("=" * 72)
    print(f"Generated temporary password for {reviewer_id} ({display_name}):")
    print()
    print(f"    {temp}")
    print()
    print("Send this to the reviewer via a SECURE channel (1Password share,")
    print("encrypted email, in-person). They will be required to change it on")
    print("first login. After they change it, you will not be able to see")
    print("their new password.")
    print()
    print("Clear your terminal scrollback after distributing the password.")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "reviewer_id", nargs="?", default=None,
        help="Reviewer ID to set password for. Use --status to list IDs.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Apply to every reviewer in the manifest.",
    )
    parser.add_argument(
        "--temp", action="store_true",
        help="Generate a random one-time temporary password and flag the "
             "reviewer for forced password change on first login. The admin "
             "sees the temp once on stdout but never the reviewer's permanent "
             "password. Recommended over manual --all/single mode.",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print which reviewers do/don't have passwords set; no changes.",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    if manifest is None:
        sys.exit(
            "No manifest found at the configured Blob Storage location. "
            "Bootstrap one via the admin page first."
        )

    if args.status:
        print(f"Reviewers in manifest ({len(manifest.reviewers)}):")
        print(f"  {'reviewer_id':<10} {'display_name':<30} {'password':<10} must_change")
        print(f"  {'-' * 70}")
        for r in manifest.reviewers:
            password_status = "set" if r.password_hash else "NOT SET"
            mc = "yes (temp pending)" if r.password_must_change else "no"
            print(f"  {r.reviewer_id:<10} {r.display_name:<30} {password_status:<10} {mc}")
        return

    if args.all:
        for reviewer in manifest.reviewers:
            if args.temp:
                temp = _generate_temp_password()
                reviewer.password_hash = _hash(temp)
                reviewer.password_must_change = True
                _print_temp_banner(reviewer.reviewer_id, reviewer.display_name, temp)
            else:
                password = _prompt_password(reviewer.reviewer_id, reviewer.display_name)
                reviewer.password_hash = _hash(password)
                reviewer.password_must_change = False
                print(f"  ✓ password set for {reviewer.reviewer_id}")
        save_manifest(manifest)
        print(f"\nSaved {len(manifest.reviewers)} password(s) to manifest.")
        return

    if not args.reviewer_id:
        parser.error("reviewer_id is required (or use --all / --status).")

    target = next(
        (r for r in manifest.reviewers if r.reviewer_id == args.reviewer_id), None
    )
    if target is None:
        ids = [r.reviewer_id for r in manifest.reviewers]
        sys.exit(
            f"Reviewer {args.reviewer_id!r} not found. "
            f"Available IDs: {', '.join(ids)}"
        )

    if args.temp:
        temp = _generate_temp_password()
        target.password_hash = _hash(temp)
        target.password_must_change = True
        save_manifest(manifest)
        _print_temp_banner(target.reviewer_id, target.display_name, temp)
    else:
        password = _prompt_password(target.reviewer_id, target.display_name)
        target.password_hash = _hash(password)
        target.password_must_change = False
        save_manifest(manifest)
        print(f"\n  ✓ password set for {target.reviewer_id} ({target.display_name})")


if __name__ == "__main__":
    main()
