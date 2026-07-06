"""
Philter-ucsf invoker.

Runs inside a Python 3.11 sub-venv where philter-ucsf is installed.
The main project venv is Python 3.12 and cannot host philter-ucsf v1.0.3
(the bundled regex pattern files use mid-pattern global flags that 3.11+
re.compile rejects, and the CLI imports the removed `distutils` module).

Usage (called as subprocess by review_app/redaction/philter_runner.py):
    python run_philter.py --input-dir <dir> --output-dir <dir>

Each .txt file in input-dir is redacted (asterisk format, length-preserved)
to a same-named file in output-dir. DATE patterns are dropped from the
philter config at runtime so that timestamps and date strings inside notes
remain readable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

# Philter v1.0.3 ships regex pattern files with mid-pattern global flags
# (e.g. "...(?i)..."), which Python 3.11+ stdlib `re` rejects. The
# third-party `regex` package (an nltk transitive dep) accepts the same
# syntax. Swap in the more permissive parser before philter imports.
try:
    import regex as _regex_module
    import re as _re_module
    _re_module.compile = _regex_module.compile  # type: ignore[assignment]
except ImportError:
    pass


def _build_filtered_config(philter_dir: str, drop_phi_types: set[str]) -> str:
    """Load philter_delta.json, drop entries whose phi_type is in drop_phi_types,
    write filtered config to a temp file, return its path."""
    src_path = os.path.join(philter_dir, "configs", "philter_delta.json")
    with open(src_path) as f:
        patterns = json.load(f)
    filtered = [p for p in patterns if p.get("phi_type") not in drop_phi_types]
    fd, out_path = tempfile.mkstemp(prefix="philter_cfg_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(filtered, f)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"input-dir does not exist: {args.input_dir}", file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)

    import philter_ucsf
    philter_dir = os.path.dirname(philter_ucsf.__file__)
    os.chdir(philter_dir)

    filtered_cfg = _build_filtered_config(philter_dir, drop_phi_types={"DATE"})

    anno_dir = tempfile.mkdtemp(prefix="philter_anno_")
    eval_dir = tempfile.mkdtemp(prefix="philter_eval_")
    fd, coords_path = tempfile.mkstemp(prefix="philter_coords_", suffix=".json")
    os.close(fd)

    config = {
        "verbose": False,
        "run_eval": False,
        "freq_table": False,
        "initials": False,
        # Philter joins paths with `+` (not os.path.join), so directories
        # MUST end with a trailing slash, otherwise filenames concatenate
        # directly onto the dir path and produce non-existent paths.
        "finpath": args.input_dir.rstrip("/") + "/",
        "foutpath": args.output_dir.rstrip("/") + "/",
        "outformat": "asterisk",
        "ucsfformat": False,
        "anno_folder": anno_dir,
        "filters": filtered_cfg,
        "xml": os.path.join(philter_dir, "data", "phi_notes_i2b2.json"),
        "coords": coords_path,
        "eval_out": eval_dir,
        "cachepos": None,
    }

    from philter_ucsf.philter import Philter

    p = Philter(config)
    p.map_coordinates()
    p.transform()
    return 0


if __name__ == "__main__":
    sys.exit(main())
