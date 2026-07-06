"""Monolithic single-LLM baselines (ablation arms 2 & 3).

Both consume the SAME serialized bundle every other arm receives
(``state["patient_context_text"]`` = full structured data + the union of all
notes) and emit an ``ICUPauseOutput``-shaped dict so every downstream metric
applies uniformly. The only thing that differs between the two arms — and from
the full pipeline — is the prompt; data, model, and temperature=0 are held fixed.

    monolith_best_effort : one expert prompt; the model writes the best possible
        all-sections transfer brief in its own organization (free-form). This is
        the headline baseline — "why not just use a strong GPT-5 directly?"
    monolith_templated   : the SAME single call, but handed the exact 8-section
        ICU-PAUSE schema/headers. Contrast with best_effort isolates the value of
        output STRUCTURE alone (vs. role decomposition).
    monolith_guided      : the SAME single call + the 8-section schema + DISTILLED
        per-section instructions sourced from each owning agent's prompt
        (config/prompts/monolith_guided.yaml). The instruction-matched arm:
        contrast with templated isolates instruction richness, and with full
        isolates ARCHITECTURE (single pass vs. routed) holding instructions fixed.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from icu_pause.ablation.monolith_enrichment import build_enrichment_block
from icu_pause.llm.provider import create_llm
from icu_pause.schemas.icu_pause import ICUPauseOutput

logger = logging.getLogger(__name__)

# Canonical 8 ICU-PAUSE sections (key -> label). Kept in sync with
# orchestrator.SECTION_ORDER; defined locally so the prompt carries the
# descriptions without importing orchestrator internals.
SECTION_SCHEMA: list[tuple[str, str]] = [
    ("I", "ICU Admission Reason & Brief ICU Course"),
    ("C", "Code Status / DPOA / Goals of Care / ACP Note"),
    ("U_unprescribing", "Unprescribing & Pertinent High-Risk Medications"),
    ("P", "Pending Tests at Time of Transfer"),
    ("A", "Active Consultants (including Rehab: PT, OT, SLP, Wound Care)"),
    ("U_uncertainty", "Uncertainty Measure / Diagnostic Pause"),
    ("S", "Summary of Major Problems and To-Do's"),
    ("E", "Exam at Transfer, Lines/Drains/Airways & Data Review"),
]

# Generous generation budget so the single call isn't truncated relative to the
# pipeline, where many agents each have their own budget. Fairness lever.
MONOLITH_MAX_TOKENS = 8192


_SECTION_KEYS = [k for k, _ in SECTION_SCHEMA]

# Match a delimiter line like "===SECTION:U_unprescribing===" (tolerant of
# spacing and '=' count). Used to split the templated arm's output instead of
# forcing JSON guided-decoding, which local models (Gemma/vLLM) honor unreliably.
_SECTION_DELIM_RE = re.compile(
    r"={2,}\s*SECTION\s*:\s*([A-Za-z_]+)\s*={2,}", re.IGNORECASE)


def parse_delimited_sections(text: str) -> dict[str, str]:
    """Split delimiter-formatted output into a {section_key: content} dict.

    Returns only recognized keys; empty if the model ignored the format (the
    caller then falls back to scoring the whole brief as one blob).
    """
    parts = _SECTION_DELIM_RE.split(text)
    out: dict[str, str] = {}
    # parts = [pre, key1, body1, key2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].strip()
        # Case-insensitive match back to the canonical key.
        canon = next((k for k in _SECTION_KEYS if k.lower() == key.lower()), None)
        if canon:
            out[canon] = parts[i + 1].strip()
    return out


_SYSTEM = (
    "You are a board-certified intensivist writing an ICU-to-ward transfer brief "
    "for the accepting ward team. You are given the patient's structured ICU data "
    "and clinical notes. Write accurately and concisely. Use ONLY values present in "
    "the provided data; never invent numbers, doses, or findings. Preserve numeric "
    "values (vitals, medication doses, ventilator/FiO2 settings, antibiotic "
    "durations, dialysis status) exactly as they appear in the source."
)

_BEST_EFFORT_USER = (
    "Write the best possible ICU→ward transfer brief for this patient. Cover "
    "everything the ward team needs to safely take over: the reason for ICU "
    "admission and a brief ICU course; current clinical status and active "
    "problems with their to-dos; current and recently stopped high-risk "
    "medications (including vasopressor doses and any unprescribing); ventilator/"
    "oxygenation and dialysis status; lines, drains, and airways; the transfer "
    "exam; pending tests/results; active consultants; code status and goals of "
    "care; and any diagnostic uncertainty. Organize it however you judge best for "
    "a safe handoff.\n\n"
    "=== PATIENT DATA AND CLINICAL NOTES ===\n{bundle}\n"
)

_TEMPLATED_USER = (
    "Write an ICU→ward transfer brief by filling EACH of the eight ICU-PAUSE "
    "sections below. Output EXACTLY in this format — one block per section, using "
    "the exact section keys and the delimiter lines shown, with nothing before "
    "the first delimiter:\n\n"
    "===SECTION:I===\n<content for section I>\n"
    "===SECTION:C===\n<content for section C>\n"
    "===SECTION:U_unprescribing===\n<content>\n"
    "===SECTION:P===\n<content>\n"
    "===SECTION:A===\n<content>\n"
    "===SECTION:U_uncertainty===\n<content>\n"
    "===SECTION:S===\n<content>\n"
    "===SECTION:E===\n<content>\n\n"
    "If a section has no relevant data, write \"No relevant data.\" under it. Do "
    "not merge, reorder, or omit sections.\n\n"
    "Section keys and what each means:\n{schema}\n\n"
    "=== PATIENT DATA AND CLINICAL NOTES ===\n{bundle}\n"
)

# Same delimiter-format output contract as templated, but each section carries
# DISTILLED per-section instructions (from the owning agent's prompt) instead of
# a one-line label. This is the instruction-matched arm. To make its OUTPUT match
# the deployed multiagent note (so the LLM-judge scores both in the regime it was
# validated on), the single call also: (1) adds inline source citations, and
# (2) emits the to-do / warnings / QA / confidence blocks the pipeline produces.
# All from ONE call — these are output requirements, not extra agents. Deterministic
# tool enrichment (DDI, device dwell, lab flags) is injected as {enrichment}.
_GUIDED_USER = (
    "Write an ICU→ward transfer brief by filling EACH of the eight ICU-PAUSE "
    "sections below, following the SECTION INSTRUCTIONS for each.\n\n"
    "CITATIONS: immediately after each clinical assertion, add an inline source "
    "tag in the format (source_type M-DD HH:MM) naming the data item or note it "
    "came from, exactly as it appears in the data (e.g. \"(progress 2-15 08:00)\", "
    "\"(labs 2-15 06:00)\"). Cite only assertions the data supports; never invent a "
    "citation.\n\n"
    "Output EXACTLY in this format — one block per section using the exact keys and "
    "delimiter lines shown, with nothing before the first delimiter:\n\n"
    "===SECTION:I===\n<content for section I>\n"
    "===SECTION:C===\n<content for section C>\n"
    "===SECTION:U_unprescribing===\n<content>\n"
    "===SECTION:P===\n<content>\n"
    "===SECTION:A===\n<content>\n"
    "===SECTION:U_uncertainty===\n<content>\n"
    "===SECTION:S===\n<content>\n"
    "===SECTION:E===\n<content>\n\n"
    "Then add these four blocks, in this order:\n"
    "===TODO===\n<one actionable to-do item per line, prefixed '- ', optionally "
    "tagged with a bucket: '[pre_transfer]', '[ward_ongoing]', or '[discharge]'>\n"
    "===WARNINGS===\n<one safety/clinical warning per line, prefixed '- '; or 'None'>\n"
    "===QA===\n<assertions you are uncertain about or that the ward team should "
    "independently verify, one per line prefixed '- '; or 'None'>\n"
    "===CONFIDENCE===\n<for each section key, a line 'KEY: <0.0-1.0>'>\n\n"
    "If a section has no relevant data, write \"No relevant data.\" under it. Do not "
    "merge, reorder, or omit sections.\n\n"
    "=== SECTION INSTRUCTIONS ===\n{guidance}\n\n"
    "{enrichment}"
    "=== PATIENT DATA AND CLINICAL NOTES ===\n{bundle}\n"
)

# Trailing meta blocks the guided arm emits alongside the 8 sections.
_META_KEYS = {"TODO", "WARNINGS", "QA", "CONFIDENCE"}
# Matches "===SECTION:I===" (captures "I") AND "===TODO===" (captures "TODO").
_BLOCK_DELIM_RE = re.compile(
    r"={2,}\s*(?:SECTION\s*:\s*)?([A-Za-z_]+)\s*={2,}", re.IGNORECASE)


def parse_guided_output(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Split guided output into ({section_key: content}, {META_KEY: block})."""
    parts = _BLOCK_DELIM_RE.split(text)
    sections: dict[str, str] = {}
    meta: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        raw = parts[i].strip()
        body = parts[i + 1].strip()
        canon = next((k for k in _SECTION_KEYS if k.lower() == raw.lower()), None)
        if canon:
            sections[canon] = body
        elif raw.upper() in _META_KEYS:
            meta[raw.upper()] = body
    return sections, meta


def _block_lines(block: str) -> list[str]:
    """One stripped item per non-empty line; drops bullet prefixes and 'None'."""
    out: list[str] = []
    for ln in (block or "").splitlines():
        s = ln.strip().lstrip("-•*").strip()
        if s and s.lower() != "none":
            out.append(s)
    return out


_TODO_BUCKETS = {"pre_transfer", "ward_ongoing", "discharge"}


def _parse_todo(block: str) -> list[dict[str, str]]:
    """ICUPauseOutput requires todo_checklist items as {'bucket','text'} dicts.

    Accepts an optional leading bucket tag like '[pre_transfer] ...'; defaults to
    'ward_ongoing' (render groups by bucket; default keeps it valid + rendered).
    """
    out: list[dict[str, str]] = []
    for s in _block_lines(block):
        bucket = "ward_ongoing"
        m = re.match(r"\[(\w+)\]\s*(.*)", s)
        if m and m.group(1).lower() in _TODO_BUCKETS:
            bucket = m.group(1).lower()
            s = m.group(2).strip()
        if s:
            out.append({"bucket": bucket, "text": s})
    return out


def _parse_confidences(block: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for ln in (block or "").splitlines():
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        k = k.strip()
        canon = next((kk for kk in _SECTION_KEYS if kk.lower() == k.lower()), None)
        if not canon:
            continue
        try:
            out[canon] = float(v.strip())
        except ValueError:
            pass
    return out

GUIDED_PROMPTS_FILENAME = "monolith_guided.yaml"


def load_guided_sections(settings) -> dict[str, str]:
    """Load distilled per-section instructions for the `guided` (instruction-matched)
    monolith arm from ``config/prompts/monolith_guided.yaml``.

    Each section's block is distilled from its OWNING agent's prompt (recorded in
    the YAML's per-section ``source`` field) so the single-call monolith receives
    substantive section guidance comparable to what the pipeline's specialists get
    — isolating instruction richness from architecture. Pipeline plumbing (citation
    registry, inter-agent handoff formatting, site-specific HITL lore) is
    deliberately excluded: it doesn't transfer to a single prompt, and dropping it
    is the fair condensation (it can only help, not handicap, the monolith).
    """
    import yaml

    path = Path(getattr(settings, "prompts_dir", "config/prompts")) / GUIDED_PROMPTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"guided monolith prompts not found at {path}; create it (distilled, "
            "source-mapped per-section instructions) before running monolith_guided")
    data = yaml.safe_load(path.read_text()) or {}
    secs = data.get("sections", {}) or {}
    out = {str(k): str(v).strip() for k, v in secs.items() if v and str(v).strip()}
    if not out:
        raise ValueError(f"no section instructions found in {path}")
    return out


# First-shrink target when the full bundle overflows the model context. Notes are
# truncated first (the bulk + most expendable per token); all structured data is
# preserved so numeric fidelity stays fair. ~115k tokens @4 chars/tok; the retry
# loop shrinks further if this still overflows.
MONOLITH_MAX_BUNDLE_CHARS = 420000


def bundle_to_text(patient_context_text: dict[str, Any],
                   max_chars: int | None = None) -> tuple[str, bool]:
    """Render the shared bundle for the prompt; truncate notes to ``max_chars``.

    Uses the SAME renderer the pipeline's agents use — ``format_data_sections_block``
    (per-domain ``## KEY`` headers, compact JSON, and the ``M-DD HH:MM`` local-dttm
    formatter) — so the monolith's input representation is identical to what each
    agent sees (and what the LLM-judge / reviewer app treat as source, since the
    judge shares this code path). Only the SLICE differs: the monolith gets the full
    bundle; each agent gets its own keys. This keeps the single-vs-multi-agent
    contrast about architecture, not data formatting (raw ISO blob vs. agent view).

    Returns (text, truncated). With ``max_chars=None`` the full bundle is emitted
    (fair to cases that fit). When set, all structured domains are kept intact and
    only the ``notes`` block is trimmed to fit — so a heavy-patient monolith loses
    narrative (its honest limitation) but never silently loses vitals/meds/labs.
    Key order follows serialize_to_json (matches the agents), keeping it byte-stable
    for determinism at temp=0.
    """
    from icu_pause.agents.base import format_data_sections_block

    ctx = dict(patient_context_text or {})
    notes = ctx.pop("notes", None)
    base = format_data_sections_block(ctx)
    truncated = False
    if notes is not None:
        notes_str = format_data_sections_block({"notes": notes})
        if max_chars is not None and len(base) + len(notes_str) > max_chars:
            budget = max(0, max_chars - len(base) - 240)
            notes_str = notes_str[:budget].rstrip() + "\n...[NOTES TRUNCATED TO FIT CONTEXT WINDOW]..."
            truncated = True
        base = base + "\n\n" + notes_str
    if max_chars is not None and len(base) > max_chars:
        base = base[:max_chars].rstrip() + "\n...[BUNDLE TRUNCATED]..."
        truncated = True
    return base, truncated


class MonolithAgent:
    """Single-LLM baseline. ``mode`` selects best_effort vs. section-templated."""

    def __init__(self, settings, mode: str, temperature: float = 0.0):
        if mode not in ("best_effort", "templated", "guided"):
            raise ValueError(f"unknown monolith mode: {mode!r}")
        self.settings = settings
        self.mode = mode
        self.temperature = temperature
        # Instruction-matched arm: load the distilled per-section guidance up front
        # so a missing/empty prompt file fails fast (before any LLM call).
        self.guided_sections: dict[str, str] = (
            load_guided_sections(settings) if mode == "guided" else {})
        # Default temp 0 = deterministic greedy decoding (screening). For a
        # reuse-existing-full comparison the monolith is temp-matched to the
        # production full briefs (e.g. 0.2) so the arms differ only in architecture.
        self.llm = create_llm(
            settings,
            max_tokens_override=MONOLITH_MAX_TOKENS,
            temperature_override=temperature,
        )

    def _guidance_block(self) -> str:
        """Assemble the distilled per-section instructions in canonical order.

        Only sections present in the prompt file are emitted; each is labeled with
        its ICU-PAUSE key + section name so the model maps guidance → output block.
        """
        parts = []
        for k, label in SECTION_SCHEMA:
            g = self.guided_sections.get(k)
            if g:
                parts.append(f"## SECTION {k} — {label}\n{g}")
        return "\n\n".join(parts)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        t0 = datetime.now(timezone.utc)
        ctx = state.get("patient_context_text", {})
        # Guided arm only: compute deterministic clinical enrichment (DDI, device
        # dwell, lab flags) ONCE and inject into the single prompt. Same tools the
        # pipeline agents use, held constant so the single-vs-multi contrast is
        # architecture, not tool access. Best-effort; never blocks generation.
        enrichment_str = ""
        if self.mode == "guided":
            demo = (ctx.get("demographics") or {}) if isinstance(ctx, dict) else {}
            block = build_enrichment_block(
                ctx,
                reference_dttm=demo.get("reference_dttm") or state.get("reference_dttm"),
                icu_admission_dttm=demo.get("icu_admission_dttm"),
                allow_network=getattr(self.settings, "drug_interaction_allow_network", False),
                timeout=getattr(self.settings, "drug_interaction_timeout", 5.0),
            )
            enrichment_str = (block + "\n\n") if block else ""
        # First attempt: full bundle. On context-window overflow (heavy patients
        # whose full bundle exceeds the model context — the routed pipeline never
        # hits this), shrink the bundle (notes first) and retry. A real single-LLM
        # baseline degrades gracefully; the truncation IS the monolith's honest
        # limitation and is recorded in metadata.
        max_chars: int | None = None
        truncated = False
        resp = None
        last_err: Exception | None = None
        for _ in range(6):
            bundle, trunc = bundle_to_text(ctx, max_chars)
            truncated = truncated or trunc
            if self.mode == "templated":
                schema_str = "\n".join(f"  {k}: {label}" for k, label in SECTION_SCHEMA)
                user = _TEMPLATED_USER.format(schema=schema_str, bundle=bundle)
            elif self.mode == "guided":
                user = _GUIDED_USER.format(guidance=self._guidance_block(),
                                           enrichment=enrichment_str, bundle=bundle)
            else:
                user = _BEST_EFFORT_USER.format(bundle=bundle)
            try:
                resp = self.llm.invoke(_SYSTEM, user, response_format=None)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                if "context length" in msg or "maximum context" in msg or "input_tokens" in msg:
                    max_chars = (MONOLITH_MAX_BUNDLE_CHARS if max_chars is None
                                 else int(max_chars * 0.7))
                    truncated = True
                    logger.warning("monolith %s context overflow on %s; retry at max_chars=%s",
                                   self.mode, state.get("hospitalization_id"), max_chars)
                    continue
                raise
        if resp is None:
            raise last_err if last_err else RuntimeError("monolith produced no response")

        text = resp if isinstance(resp, str) else str(resp)
        templated_parse_ok = None
        todo_items: list[dict[str, str]] = []
        warnings_items: list[str] = []
        qa_items: list[str] = []
        confidences: dict[str, float] = {}
        if self.mode == "guided":
            # Guided arm now emits cited sections + TODO/WARNINGS/QA/CONFIDENCE blocks
            # in ONE call, so its OUTPUT matches the deployed multiagent note (the
            # regime the judge was validated on). Parse sections + meta deterministically.
            sections, meta = parse_guided_output(text)
            templated_parse_ok = bool(sections)
            if not sections:
                sections = {"BRIEF": text}  # keep the brief so the case still scores
            todo_items = _parse_todo(meta.get("TODO", ""))
            warnings_items = _block_lines(meta.get("WARNINGS", ""))
            qa_items = _block_lines(meta.get("QA", ""))
            confidences = _parse_confidences(meta.get("CONFIDENCE", ""))
            arm = "monolith_guided"
        elif self.mode == "templated":
            # Free-form generation (no JSON guided-decoding — unreliable on local
            # models); parse the delimited sections deterministically.
            sections = parse_delimited_sections(text)
            templated_parse_ok = bool(sections)
            if not sections:
                sections = {"BRIEF": text}
            arm = "monolith_templated"
        else:
            sections = {"BRIEF": text}
            arm = "monolith_best_effort"

        usage = getattr(self.llm, "last_usage", None)
        wall_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0
        latency_ms = getattr(usage, "latency_ms", 0.0) or wall_ms

        output = ICUPauseOutput(
            hospitalization_id=state.get("hospitalization_id", "unknown"),
            generated_at=datetime.now(timezone.utc).isoformat(),
            sections=sections,
            todo_checklist=todo_items,
            warnings=warnings_items,
            qa_issues=qa_items,
            section_confidences=confidences,
            metadata={
                "arm": arm,
                "monolith_mode": self.mode,
                "monolith_freeform": self.mode == "best_effort",
                "templated_parse_ok": templated_parse_ok,
                "bundle_truncated": truncated,
                "enrichment_injected": bool(enrichment_str),
                # Instruction-matched arm: how much section guidance it carried, and
                # whether that guidance + bundle forced a truncation (the "can't fit
                # rich instructions AND the full record in one call" effect).
                "instruction_chars": (len(self._guidance_block())
                                      if self.mode == "guided" else None),
                "instruction_overflow": (truncated if self.mode == "guided" else None),
                "llm_provider": self.settings.llm_provider,
                "llm_model": self.settings.llm_model,
                "total_input_tokens": getattr(usage, "input_tokens", None),
                "total_output_tokens": getattr(usage, "output_tokens", None),
                "total_latency_ms": round(latency_ms, 1),
                "agent_count": 1,
            },
        )
        return {"icu_pause_output": output.model_dump()}
