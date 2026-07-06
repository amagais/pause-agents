"""
Offline case preparation script.

For each hospitalization_id provided, this script:
  1. Loads the pipeline output JSON (from a local output directory)
  2. Extracts the display-friendly source_bundle from the embedded
     pipeline_output.metadata.source_data (no CLIF Parquet files needed)
  3. Extracts atomic claims from the generated note (rule-based, no LLM)
  4. Uploads all three files to Azure Blob Storage

Usage:
    python review_app/scripts/prepare_cases.py \
        --outputs-dir /path/to/icu_pause_outputs/ \
        --hosp-ids CASE_001 CASE_002 CASE_003

    # Or read IDs from a file (one per line):
    python review_app/scripts/prepare_cases.py \
        --outputs-dir /path/to/outputs/ \
        --ids-file case_list.txt

"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Ensure we can import from the pipeline src
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root / "review_app"))

from dotenv import load_dotenv
load_dotenv(_repo_root / "review_app" / ".env")

from storage.blob_client import write_json
from redaction import redact_case_payload  # noqa: E402


# ---------------------------------------------------------------------------
# Claim extraction (rule-based, no LLM)
# ---------------------------------------------------------------------------
#
# Two extraction modes by section shape:
#   * Block-mode (U_unprescribing, U_uncertainty): structured dotphrase blocks
#     of the form ``Header:\n option line\n option line``. The whole block —
#     header + options — becomes one claim, so a reviewer verifies the
#     anticoagulation plan as a unit, not each ☐/☑ line in isolation.
#   * Line-mode (everything else): each non-blank line/sentence is one claim.
#     Section I prose is sentence-split; bulleted/checkbox lines stay whole.
#
# Pure scaffolding (e.g. "[age]yo [sex] with PMH of [PMH]") is filtered by
# placeholder-character ratio. Citation tags ([cite: ...]) and checkbox
# markers ([x], []) are excluded from that ratio.

_BLOCK_MODE_SECTIONS = {"U_unprescribing", "U_uncertainty"}

# A header line: starts with a letter, ends with ":", no leading
# checkbox/bullet/hash. Caps the body length to avoid matching prose lines
# that happen to end with a colon.
_HEADER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 /&\-]{0,40}:\s*$")

# All bracketed segments — used to count "filled vs scaffolding" ratio.
_BRACKET_RE = re.compile(r"\[([^\]]*)\]")

# Sentence boundary inside a single line (used for prose like Section I).
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

_MAX_CLAIMS_PER_SECTION = 10
_PLACEHOLDER_RATIO_THRESHOLD = 0.40


def _is_template_placeholder(content: str) -> bool:
    """True if a ``[...]`` payload is a dotphrase placeholder, not a
    citation tag or checkbox marker."""
    c = content.strip()
    if not c:                                     # "[]"
        return False
    if c.lower() == "x":                          # "[x]"
        return False
    if c.lower().startswith("cite"):              # "[cite: (...)]"
        return False
    if c.startswith("("):                         # "[(timestamp)]" — citation
        return False
    return True


def _is_scaffolding(text: str) -> bool:
    """Drop chunks that are mostly unfilled placeholders."""
    stripped = text.strip()
    if len(stripped) < 4:
        return True
    if _is_unanswered_safety_question(stripped):
        return True
    placeholder_chars = sum(
        len(m.group(0))
        for m in _BRACKET_RE.finditer(stripped)
        if _is_template_placeholder(m.group(1))
    )
    if placeholder_chars / max(len(stripped), 1) > _PLACEHOLDER_RATIO_THRESHOLD:
        return True
    # After stripping placeholders, require at least 2 real words.
    cleaned = _BRACKET_RE.sub(
        lambda m: "" if _is_template_placeholder(m.group(1)) else m.group(0),
        stripped,
    )
    words = re.findall(r"[A-Za-z]{2,}", cleaned)
    return len(words) < 2


# Narrow filter for known E-section safety-question template literals that
# the LLM sometimes emits without filling in an answer (e.g. "Difficult
# airway?" or "Difficult airway? [Y/N]"). The orchestrator's deterministic
# post-processor (a646c58) replaces these in place going forward, but
# briefs already on disk before that fix still carry them — this defensive
# filter lets us re-run prepare_cases.py on existing briefs without a full
# pipeline rerun. Narrow-by-design: matching only the specific safety
# questions avoids false-positive drops on legitimate prose that happens
# to be a question (e.g. an I-section sentence ending in '?').
_UNANSWERED_SAFETY_QUESTION_RE = re.compile(
    r"(?i)^("
    r"Difficult\s+airway"
    r"|Lines?/drains?\s+assessed\s+for\s+removal"
    r")\?\s*(?:\[[^\]]*\])?\s*$"
)


def _is_unanswered_safety_question(text: str) -> bool:
    """True if the chunk is a bare safety question with no Y/N answer.

    Matches: 'Difficult airway?', 'Difficult airway? [Y/N]',
             'Lines/drains assessed for removal?', etc.
    Does NOT match: 'Difficult airway? Yes', 'Difficult airway? ☐',
             'Difficult airway? Yes — tracheostomy in place'.
    """
    return bool(_UNANSWERED_SAFETY_QUESTION_RE.match(text.strip()))


_CHECKBOX_CHARS = "☐☑☒"
# Real checkbox/structured-line prefix only — `[78]` and other placeholders
# starting with `[` deliberately fall through to prose sentence-splitting.
_STRUCTURED_PREFIX_RE = re.compile(r"^(\[\s?[xX]?\s?\]|[☐☑☒#•]|-\s)")


def _split_lines(text: str) -> list[str]:
    """One claim per non-blank line; sentence-split prose lines only.
    A line is treated as structured (kept whole) if it starts with a real
    checkbox/bullet marker OR contains a checkbox char anywhere — that
    keeps things like ``Difficult airway? ☐`` intact instead of splitting
    the question off from its checkbox."""
    out: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        is_structured = (
            bool(_STRUCTURED_PREFIX_RE.match(line))
            or any(c in line for c in _CHECKBOX_CHARS)
        )
        if not is_structured and re.search(r"[.!?]\s", line):
            out.extend(s.strip() for s in _SENT_SPLIT.split(line) if s.strip())
        else:
            out.append(line)
    return out


def _split_blocks(text: str) -> list[str]:
    """Group lines into blocks: each header line + the non-header lines
    that follow it form one chunk. Chunks with no ``Header:`` line fall
    back to sentence-split — they're free-form prose recommendations
    where each sentence is a distinct verifiable claim."""
    raw_blocks: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if _HEADER_RE.match(line.strip()):
            if current:
                raw_blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        raw_blocks.append("\n".join(current).strip())

    out: list[str] = []
    for block in raw_blocks:
        if not block.strip():
            continue
        has_header = any(_HEADER_RE.match(ln.strip()) for ln in block.split("\n"))
        if has_header:
            out.append(block)
        else:
            out.extend(_split_lines(block))
    return out


def extract_claims(output: dict) -> list[dict]:
    """
    Extract atomic verifiable claims from an ICUPauseOutput dict.

    Returns list of {claim_id, section, text}.
    """
    section_order = [
        "I", "C", "U_unprescribing", "P", "A", "U_uncertainty", "S", "E"
    ]
    claims = []

    sections = output.get("sections", {})
    for section_key in section_order:
        content = sections.get(section_key, "")
        if not content or not content.strip():
            continue

        if section_key in _BLOCK_MODE_SECTIONS:
            chunks = _split_blocks(content)
        else:
            chunks = _split_lines(content)

        chunks = [c for c in chunks if not _is_scaffolding(c)]
        chunks = chunks[:_MAX_CLAIMS_PER_SECTION]

        for i, text in enumerate(chunks, 1):
            claims.append({
                "claim_id": f"{section_key}_claim_{i}",
                "section": section_key,
                "text": text,
            })

    # Also include to-do checklist items as claims (high clinical salience)
    for i, item in enumerate(output.get("todo_checklist", [])[:4], 1):
        text = item["text"] if isinstance(item, dict) else str(item)
        claims.append({
            "claim_id": f"todo_claim_{i}",
            "section": "S",
            "text": text,
        })

    return claims


# ---------------------------------------------------------------------------
# Source bundle serialization
# ---------------------------------------------------------------------------

# Parsed from the .log sibling. Two retriever traces we mine:
#   "Note leakage guard (<note_type>): excluded N notes at/after ref=..."
#     — emitted directly by logger.info in retriever._load_notes_for_hospitalization
#   "[<note_type>] Scanned <file>: N rows for hosp_id=..."
#     — emitted via retriever._trace, which prepends [{node}] to the message;
#       node == note_type_key (see retriever.py:128). The standard pipeline
#       logger format is "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
#       (main.py:104), so the per-message bracket is the *last* [...] on the
#       line — `[^\n\[]*` between the node bracket and "Scanned" prevents
#       accidentally capturing the level/logger-name brackets.
# Together they let the reviewer panel distinguish three absence flavors:
# no rows at all vs. all rows post-transfer vs. all rows outside window.
# Tolerant of trace-format drift: missing log → empty dict → renderer
# falls back to a generic absence caption, never a false claim.
_LEAKAGE_RE = re.compile(
    r"Note leakage guard \((?P<nt>\w+)\): excluded (?P<n>\d+) notes at/after ref"
)
_SCANNED_RE = re.compile(
    r"\[(?P<nt>\w*_note)\][^\n\[]*Scanned\s+\S+:\s+(?P<n>\d+) rows for hosp_id="
)


def parse_note_absence_reasons_from_log(log_path: Path) -> dict[str, dict[str, int]]:
    """Extract per-note-type retrieval counts from the .log sibling.

    Returns ``{note_type: {"scanned_total": int, "excluded_leakage": int}}``.
    Missing log, missing fields, parse errors → empty/partial dict; never
    raises. Renderer treats absence of an entry as "no detail available."
    """
    if not log_path.exists():
        return {}
    reasons: dict[str, dict[str, int]] = {}
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return {}

    for m in _LEAKAGE_RE.finditer(text):
        nt = m.group("nt")
        n = int(m.group("n"))
        entry = reasons.setdefault(nt, {})
        entry["excluded_leakage"] = entry.get("excluded_leakage", 0) + n

    for m in _SCANNED_RE.finditer(text):
        nt = m.group("nt")
        n_rows = int(m.group("n"))
        entry = reasons.setdefault(nt, {})
        # Take the max — a note type can be scanned more than once (e.g.
        # cache miss + reload path); the larger count is the true
        # row-cardinality for the hosp_id.
        entry["scanned_total"] = max(entry.get("scanned_total", 0), n_rows)
    return reasons


def build_source_bundle_from_metadata(
    source_data: dict,
    absence_reasons: dict[str, dict[str, int]] | None = None,
    lookback_hours: int = 48,
) -> dict:
    """
    Build a display-friendly source bundle directly from the
    pipeline_output.metadata.source_data dict embedded in the output JSON.
    No CLIF parquet files needed.
    """
    bundle: dict = {}

    # Demographics
    bundle["demographics"] = source_data.get("demographics", {})

    # Notes window — fed to source_renderer so the reviewer panel banners the
    # lookback boundaries above the notes list. reference_dttm comes from
    # demographics (populated by retriever); lookback_hours defaults to the
    # pipeline default (48h) since it's not currently in the brief metadata.
    # Update when orchestrator starts emitting notes_lookback_hours alongside
    # source_data — until then 48 is correct for all production runs.
    bundle["notes_window"] = {
        "reference_dttm": (source_data.get("demographics") or {}).get("reference_dttm"),
        "lookback_hours": lookback_hours,
    }
    bundle["notes_absence_reasons"] = absence_reasons or {}

    # Vitals — stored as {bucketed_trends: [...], recent_raw: [...]} in metadata.
    # Surface BOTH so the reviewer can audit snapshot-style sentences (which
    # the nurse agent composes from recent_raw values the bucketed mean
    # smooths over). The bundle carries a dict; source_renderer handles
    # both shapes for back-compat with legacy briefs that pre-date this
    # change and with the demo page which passes a flat list.
    vitals_raw = source_data.get("vitals", {})
    if isinstance(vitals_raw, dict):
        bundle["vitals_summary"] = {
            "bucketed_trends": vitals_raw.get("bucketed_trends", []) or [],
            "recent_raw": vitals_raw.get("recent_raw", []) or [],
        }
    elif isinstance(vitals_raw, list):
        # Legacy on-disk briefs from before recent_raw was preserved in
        # metadata.source_data. Pass through unchanged; renderer will treat
        # as bucketed_trends. Drop the back-compat branch once all blob
        # cases are regenerated.
        print(
            "  [warn] vitals_summary: legacy list shape; "
            "regenerate brief to surface recent_raw alongside bucketed_trends"
        )
        bundle["vitals_summary"] = vitals_raw
    else:
        bundle["vitals_summary"] = {"bucketed_trends": [], "recent_raw": []}

    # Transfer-exam block: deterministic Section E vitals/neuro/respiratory
    # snapshot the intensivist is instructed to copy verbatim. Populated by
    # serialize_to_json in Phase 3; empty string here is the expected state
    # until B/C ships. Surfacing it in the reviewer panel now so the wiring
    # is ready and reviewers can immediately audit the deterministic block
    # against the rendered E section once B ships.
    bundle["transfer_exam_block"] = source_data.get("transfer_exam_block", "") or ""

    # Labs
    bundle["labs_recent"] = source_data.get("labs", []) or []

    # Meds — stored as {continuous: [...], intermittent: [...]}
    meds = source_data.get("meds", {})
    if isinstance(meds, dict):
        bundle["meds_continuous"] = meds.get("continuous", []) or []
        bundle["meds_intermittent"] = meds.get("intermittent", []) or []
    else:
        bundle["meds_continuous"] = []
        bundle["meds_intermittent"] = []

    # Respiratory
    bundle["respiratory_support"] = source_data.get("respiratory", []) or []

    # Assessments
    bundle["assessments"] = source_data.get("assessments", []) or []

    # Code status
    bundle["code_status"] = source_data.get("code_status", []) or []

    # Diagnoses (ICD codes — no human-readable names in metadata)
    bundle["diagnoses"] = source_data.get("diagnoses", []) or []

    # Microbiology
    bundle["microbiology"] = source_data.get("microbiology", []) or []

    # Procedures
    bundle["procedures"] = source_data.get("procedures", []) or []

    # Clinical notes — normalise to {note_type: [note_dict, …]}
    notes_raw = source_data.get("notes", [])
    if isinstance(notes_raw, dict):
        bundle["clinical_notes"] = notes_raw
    elif isinstance(notes_raw, list):
        # Legacy flat list: group by note_type key
        grouped: dict[str, list[dict]] = {}
        for note in notes_raw:
            ntype = note.get("note_type", "other")
            grouped.setdefault(ntype, []).append(note)
        bundle["clinical_notes"] = grouped
    else:
        bundle["clinical_notes"] = {}

    return bundle


# ---------------------------------------------------------------------------
# Main upload loop
# ---------------------------------------------------------------------------

def prepare_case(
    hosp_id: str,
    outputs_dir: str,
    redact: bool = True,
) -> None:
    print(f"  [{hosp_id}] Loading output JSON…")
    output_path = Path(outputs_dir) / f"{hosp_id}.json"
    if not output_path.exists():
        print(f"  [{hosp_id}] ERROR: output file not found at {output_path}")
        return

    with open(output_path) as f:
        wrapper = json.load(f)

    # Support both the wrapper format {pipeline_output: {...}} and bare ICUPauseOutput
    if "pipeline_output" in wrapper:
        pipeline_output = wrapper["pipeline_output"]
    else:
        pipeline_output = wrapper

    # The current pipeline schema nests the renderable note under
    # pipeline_output["final_output"] (sections, todo_checklist, warnings,
    # qa_issues, generated_at, metadata). Older schemas stored those keys
    # at pipeline_output's top level. The reviewer-app renderer + claim
    # extractor + PHI redactor all read top-level keys, so unwrap to
    # final_output before redact/extract/upload. Falls back to the wrapper
    # itself when there's no nested final_output (legacy cases).
    note_payload = pipeline_output.get("final_output") or pipeline_output

    print(f"  [{hosp_id}] Building source bundle from embedded metadata…")
    # source_data lives under metadata. Schema has migrated: in the
    # current pipeline it's at pipeline_output.final_output.metadata.source_data
    # (i.e. note_payload.metadata.source_data after our final_output unwrap).
    # Older runs put it at pipeline_output.metadata.source_data. Try the
    # nested location first, fall back to the wrapper's metadata.
    source_data = (
        note_payload.get("metadata", {}).get("source_data")
        or pipeline_output.get("metadata", {}).get("source_data", {})
    )
    if not source_data:
        print(f"  [{hosp_id}] WARNING: no source_data found in metadata; source bundle will be empty.")

    # Mine the .log sibling for per-note-type retrieval counts (no log =
    # no detail; absence captions degrade to the generic form). Pilot
    # convention is <hosp_id>.json + <hosp_id>.log in the same directory
    # — see feedback_icu_pause_pilot_output_paths.md.
    log_path = Path(outputs_dir) / f"{hosp_id}.log"
    absence_reasons = parse_note_absence_reasons_from_log(log_path)
    if absence_reasons:
        print(
            f"  [{hosp_id}] Parsed retrieval counts for "
            f"{len(absence_reasons)} note type(s) from {log_path.name}"
        )
    source = build_source_bundle_from_metadata(source_data, absence_reasons=absence_reasons)

    if redact:
        print(f"  [{hosp_id}] Redacting PHI (names + phones; dates preserved)…")
        source, note_payload = redact_case_payload(source, note_payload)

    print(f"  [{hosp_id}] Extracting claims…")
    claims = extract_claims(note_payload)
    print(f"  [{hosp_id}] Extracted {len(claims)} claims.")

    print(f"  [{hosp_id}] Uploading to Azure Blob…")
    write_json(f"cases/{hosp_id}/output.json", note_payload)
    write_json(f"cases/{hosp_id}/source_bundle.json", source)
    write_json(f"cases/{hosp_id}/claims.json", claims)
    print(f"  [{hosp_id}] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and upload ICU-PAUSE review cases to Azure Blob.")
    parser.add_argument("--outputs-dir", required=True, help="Directory containing {hosp_id}.json output files")
    parser.add_argument("--hosp-ids", nargs="*", help="Hospitalization IDs to process")
    parser.add_argument("--ids-file", default=None, help="Text file with one hosp_id per line")
    parser.add_argument(
        "--no-redact",
        action="store_true",
        help="Skip PHI redaction (default: redact names and phones, preserve dates). "
             "Requires ICUPAUSE_REDACT_PYTHON env var to point at a Python 3.11 venv "
             "with philter-ucsf installed when redaction is on.",
    )
    args = parser.parse_args()

    hosp_ids: list[str] = []
    if args.hosp_ids:
        hosp_ids.extend(args.hosp_ids)
    if args.ids_file:
        with open(args.ids_file) as f:
            hosp_ids.extend(line.strip() for line in f if line.strip())

    if not hosp_ids:
        print("No hospitalization IDs provided. Use --hosp-ids or --ids-file.")
        sys.exit(1)

    print(f"Preparing {len(hosp_ids)} case(s){' (redaction OFF)' if args.no_redact else ''}…")
    for hosp_id in hosp_ids:
        prepare_case(
            hosp_id=hosp_id,
            outputs_dir=args.outputs_dir,
            redact=not args.no_redact,
        )

    print("All done.")


if __name__ == "__main__":
    main()
