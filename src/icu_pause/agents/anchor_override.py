"""Stage E — Deterministic anchor-override post-processing for hybrid_v1.

Reconciles each domain agent's emitted output against the structured anchors
extracted upstream by the per-domain extractor. When an anchor is present
and unambiguous, this layer enforces the anchor as authoritative — replacing
or inserting the value in the agent's output and emitting an
`anchor_override` trace event for audit.

Pre-registered in PRE_REGISTRATION_compression_redesign.md §1.3 (Stage E)
and §1.7 (the `use_anchor_override` ablation flag for hybrid_v1_no_anchor).

Behavior locked by the five unit-test cases in pre-reg §9:
  (a) anchor matches agent       → no override, no trace
  (b) anchor conflicts            → override applied, trace logged
  (c) anchor present, agent omits → insertion, trace logged
  (d) anchor absent               → no-op, trace not logged
  (e) ambiguous parse             → skip override, trace logged with reason

This module is intentionally generic in the anchor schema. The per-agent
anchor bindings are passed in by the caller, so this code does not depend
on the schema-expansion sign-off (pre-reg §1.5) and can be unit-tested
against synthetic inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "AnchorAction",
    "AnchorBinding",
    "AnchorOverrideEvent",
    "ReconciliationResult",
    "apply_anchor_override",
    "is_anchor_absent",
    "is_anchor_ambiguous",
    "is_agent_omitted",
    "normalize_for_compare",
    "values_match",
]


# Matches "2.1", "2.10", "-3", "+4.5", ".001". Non-overlapping search at multiple
# positions returns each numeric token in the string separately.
_NUMERIC_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")


class AnchorAction(str, Enum):
    """Outcome of reconciling one binding."""

    NONE = "none"                          # anchor matches agent; no change
    OVERRIDE = "override"                  # anchor conflicts; agent value replaced
    INSERT = "insert"                      # agent omitted; anchor inserted
    SKIP_AMBIGUOUS = "skip_ambiguous"      # anchor parse ambiguous; agent preserved
    SKIP_ABSENT = "skip_absent"            # anchor absent; agent preserved (no trace)


@dataclass(frozen=True)
class AnchorBinding:
    """One anchor → agent-field binding declared by a domain agent.

    `anchor_path` is a dotted path into the extracted-anchors dict for the
    domain (e.g. "code_status", "active_medications.0.dose").

    `agent_path` is a dotted path into the agent-output dict where the
    corresponding value lives (e.g. "code_status", "medications[0].dose" —
    list indices use bracket form for clarity at the path layer).
    """

    anchor_path: str
    agent_path: str
    # Pre-declared ambiguity markers. If the anchor value (after string
    # normalization) contains any of these tokens, the binding is treated
    # as ambiguous and the agent's value is preserved.
    ambiguous_markers: tuple[str, ...] = (
        "varies",
        "see notes",
        "see chart",
        "range:",
    )


@dataclass
class AnchorOverrideEvent:
    """A single audit-trail record from one binding's reconciliation."""

    action: AnchorAction
    anchor_path: str
    agent_path: str
    anchor_value: Any
    agent_value_before: Any
    agent_value_after: Any
    reason: str = ""

    def to_trace(self) -> dict[str, Any]:
        """Serialize to the trace-event payload schema."""
        return {
            "type": _TRACE_TYPE_BY_ACTION[self.action],
            "anchor_path": self.anchor_path,
            "agent_path": self.agent_path,
            "anchor_value": self.anchor_value,
            "agent_value_before": self.agent_value_before,
            "agent_value_after": self.agent_value_after,
            "reason": self.reason,
        }


_TRACE_TYPE_BY_ACTION: dict[AnchorAction, str] = {
    AnchorAction.OVERRIDE: "anchor_override",
    AnchorAction.INSERT: "anchor_override_insert",
    AnchorAction.SKIP_AMBIGUOUS: "anchor_override_skipped_ambiguous",
    # NONE and SKIP_ABSENT do not emit trace events; they exist as actions
    # for in-process accounting only.
}


@dataclass
class ReconciliationResult:
    corrected_output: dict[str, Any]
    events: list[AnchorOverrideEvent] = field(default_factory=list)

    @property
    def overrides_applied(self) -> int:
        return sum(1 for e in self.events if e.action == AnchorAction.OVERRIDE)

    @property
    def inserts_applied(self) -> int:
        return sum(1 for e in self.events if e.action == AnchorAction.INSERT)

    @property
    def skipped_ambiguous(self) -> int:
        return sum(1 for e in self.events if e.action == AnchorAction.SKIP_AMBIGUOUS)


# --------------------------------------------------------------------------- #
# Predicates for the four "is this binding actionable" decisions             #
# --------------------------------------------------------------------------- #

_NOT_DOCUMENTED_SENTINELS = frozenset(
    {"not documented", "not_documented", "nd", "unknown", "n/a", "na"}
)


def is_anchor_absent(value: Any) -> bool:
    """True iff the anchor reports the field is not documented."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NOT_DOCUMENTED_SENTINELS
    return False


def is_anchor_ambiguous(value: Any, markers: tuple[str, ...]) -> bool:
    """True iff the anchor value is unsafe for deterministic override.

    Pre-reg §9(e): ambiguous parses must skip override and log a trace event,
    never silent-fail.

    Three triggers:
    - Configurable text markers ("varies", "see notes", "range:", etc.).
    - Multiple numeric tokens in the value (e.g., "2.1-2.4", "Cr 2.1 on 1/17"):
      we cannot unambiguously decide which number to compare against the agent's
      value, so we skip rather than risk replacing a correct agent value with
      a range string. Documented limitation: this also flags clean values that
      happen to have a date or other secondary number in them.
    - Numerics themselves are never ambiguous (covered by the single-token check
      in ``_try_extract_single_numeric``).
    """
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    if not normalized:
        return False
    if any(marker in normalized for marker in markers):
        return True
    # Multi-numeric values are ambiguous for the same reason ranges are.
    if len(_NUMERIC_PATTERN.findall(normalized)) >= 2:
        return True
    return False


def is_agent_omitted(value: Any) -> bool:
    """True iff the agent did not supply a usable value for the field."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def normalize_for_compare(value: Any) -> str:
    """Whitespace-collapsed, case-folded comparison key.

    Used as the string-compare path in ``values_match``. NOT used directly
    for anchor reconciliation — call ``values_match`` instead so that
    numerics get the dedicated single-numeric+unit comparison.
    """
    if value is None:
        return ""
    s = str(value).strip().lower()
    return " ".join(s.split())


def _try_extract_single_numeric(value: Any) -> tuple[float, str] | None:
    """Extract ``(numeric, unit)`` iff the value contains exactly one numeric
    token. Returns ``None`` otherwise.

    Examples (input → output):
        2.1                       → (2.1, "")
        "2.1"                     → (2.1, "")
        "2.10"                    → (2.1, "")
        "Cr 2.1 mg/dL"            → (2.1, "cr mg/dl")
        "2.1 mg/dL"               → (2.1, "mg/dl")
        "DNR"                     → None  (no numeric token)
        "2.1-2.4"                 → None  (multiple numeric tokens)
        "Cr 2.1 mg/dL on 1/17"    → None  (multiple numeric tokens)

    Unit derivation: everything around the matched numeric in the original
    string, joined and lowered. This is intentionally crude — clinically
    meaningful unit conflict is caught by string-inequality in the unit slot
    (e.g., "mg/dl" ≠ "mcg/dl"); fine-grained unit aliasing is out of scope
    for v1 per the expert's "nothing more" guidance.
    """
    if isinstance(value, bool):
        return None  # bool subclasses int; intentionally not treated as numeric
    if isinstance(value, (int, float)):
        return (float(value), "")
    if not isinstance(value, str):
        return None
    matches = _NUMERIC_PATTERN.findall(value)
    if len(matches) != 1:
        return None
    try:
        num = float(matches[0])
    except (ValueError, OverflowError):
        return None
    m = _NUMERIC_PATTERN.search(value)
    if m is None:
        return None
    rest = (value[: m.start()] + value[m.end():]).strip().lower()
    rest = " ".join(rest.split())  # collapse internal whitespace
    return (num, rest)


def _floats_close(a: float, b: float) -> bool:
    """Compare floats with absolute + relative tolerance.

    The relative component (1e-6) catches benign precision-display differences
    (e.g., "2.1" vs "2.10"). The absolute floor (1e-9) avoids division-by-zero
    edge cases when values are near zero.
    """
    return abs(a - b) <= max(1e-9, 1e-6 * max(abs(a), abs(b)))


def values_match(a: Any, b: Any) -> bool:
    """Pre-reg-locked match predicate (expert 2026-06-01 second-pass).

    Behavior:
    - If BOTH values parse as a single numeric token (with optional unit):
      compare floats with floating-point tolerance. If both also carry a
      non-empty unit AND those units differ → conflict (mg/dL vs mcg/dL is
      a clinical error, not cosmetic). If only one carries a unit, the
      numeric equality wins — the agent's labeled output is preserved
      ("Cr 2.1 mg/dL" agent vs "2.1" anchor → match, no override).
    - Otherwise: case-insensitive trimmed string compare.

    Nothing else is normalized — no label-prefix stripping, no aliasing.
    Anything looser risks silent suppression of real conflicts.
    """
    na = _try_extract_single_numeric(a)
    nb = _try_extract_single_numeric(b)
    if na is not None and nb is not None:
        if not _floats_close(na[0], nb[0]):
            return False
        if na[1] and nb[1] and na[1] != nb[1]:
            return False
        return True
    return normalize_for_compare(a) == normalize_for_compare(b)


# --------------------------------------------------------------------------- #
# Path traversal                                                             #
# --------------------------------------------------------------------------- #


def _resolve_path(data: dict[str, Any] | None, path: str) -> Any:
    """Read a value at a dotted path; return None if any segment is missing.

    Supports bracket-form list indexing: "foo[0].bar".
    """
    if data is None:
        return None
    cur: Any = data
    for seg in _split_path(path):
        if isinstance(seg, int):
            if not isinstance(cur, list) or seg >= len(cur):
                return None
            cur = cur[seg]
        else:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(seg)
            if cur is None:
                return None
    return cur


def _write_path(data: dict[str, Any], path: str, value: Any) -> None:
    """Write a value at a dotted path; create intermediate dicts as needed.

    Bracket-form list indexing supports replacement at an existing index but
    does NOT extend lists. Stage E should only insert into pre-existing
    agent-output structures.
    """
    segments = _split_path(path)
    cur: Any = data
    for seg in segments[:-1]:
        if isinstance(seg, int):
            if not isinstance(cur, list) or seg >= len(cur):
                raise KeyError(f"list index {seg} out of range while writing {path!r}")
            cur = cur[seg]
        else:
            if not isinstance(cur, dict):
                raise TypeError(f"cannot traverse non-dict at segment {seg!r} of {path!r}")
            if seg not in cur or cur[seg] is None or not isinstance(cur[seg], (dict, list)):
                cur[seg] = {}
            cur = cur[seg]
    final = segments[-1]
    if isinstance(final, int):
        if not isinstance(cur, list) or final >= len(cur):
            raise KeyError(f"list index {final} out of range while writing {path!r}")
        cur[final] = value
    else:
        if not isinstance(cur, dict):
            raise TypeError(f"cannot write to non-dict parent of {final!r}")
        cur[final] = value


def _split_path(path: str) -> list[str | int]:
    out: list[str | int] = []
    for raw_seg in path.split("."):
        cur = raw_seg
        while "[" in cur:
            head, _, rest = cur.partition("[")
            idx_str, _, tail = rest.partition("]")
            if head:
                out.append(head)
            try:
                out.append(int(idx_str))
            except ValueError as exc:
                raise ValueError(f"bad list index {idx_str!r} in path {path!r}") from exc
            cur = tail
        if cur:
            out.append(cur)
    return out


# --------------------------------------------------------------------------- #
# Public API                                                                 #
# --------------------------------------------------------------------------- #


def apply_anchor_override(
    agent_output: dict[str, Any],
    extracted_anchors: dict[str, Any] | None,
    bindings: list[AnchorBinding],
) -> ReconciliationResult:
    """Reconcile one domain agent's output against its extracted anchors.

    Returns a new ReconciliationResult; does not mutate the input
    `agent_output` (a deep copy is performed at the path layer).

    Pre-reg §1.3 Stage E behavior, locked by §9 unit-test cases:
        (a) match           → AnchorAction.NONE, no event emitted
        (b) conflict        → AnchorAction.OVERRIDE, event emitted
        (c) agent omitted   → AnchorAction.INSERT, event emitted
        (d) anchor absent   → AnchorAction.SKIP_ABSENT, no event emitted
        (e) ambiguous parse → AnchorAction.SKIP_AMBIGUOUS, event emitted
    """
    # Defensive deep copy at the dict layer; this is sufficient for the
    # JSON-shaped agent outputs we work with.
    import copy

    corrected = copy.deepcopy(agent_output)
    events: list[AnchorOverrideEvent] = []

    for binding in bindings:
        anchor_value = _resolve_path(extracted_anchors, binding.anchor_path)
        agent_value = _resolve_path(corrected, binding.agent_path)

        # (d) anchor absent → no-op, no trace event
        if is_anchor_absent(anchor_value):
            continue

        # (e) ambiguous parse → skip, emit trace event with reason
        if is_anchor_ambiguous(anchor_value, binding.ambiguous_markers):
            events.append(
                AnchorOverrideEvent(
                    action=AnchorAction.SKIP_AMBIGUOUS,
                    anchor_path=binding.anchor_path,
                    agent_path=binding.agent_path,
                    anchor_value=anchor_value,
                    agent_value_before=agent_value,
                    agent_value_after=agent_value,
                    reason="anchor value contains an ambiguity marker; agent value preserved",
                )
            )
            continue

        # (c) agent omitted → insert anchor, emit trace event
        if is_agent_omitted(agent_value):
            _write_path(corrected, binding.agent_path, anchor_value)
            events.append(
                AnchorOverrideEvent(
                    action=AnchorAction.INSERT,
                    anchor_path=binding.anchor_path,
                    agent_path=binding.agent_path,
                    anchor_value=anchor_value,
                    agent_value_before=agent_value,
                    agent_value_after=anchor_value,
                    reason="agent omitted the field; anchor inserted",
                )
            )
            continue

        # (a) match → no-op, no trace event
        if values_match(anchor_value, agent_value):
            continue

        # (b) conflict → override, emit trace event
        _write_path(corrected, binding.agent_path, anchor_value)
        events.append(
            AnchorOverrideEvent(
                action=AnchorAction.OVERRIDE,
                anchor_path=binding.anchor_path,
                agent_path=binding.agent_path,
                anchor_value=anchor_value,
                agent_value_before=agent_value,
                agent_value_after=anchor_value,
                reason="anchor conflicts with agent value; anchor authoritative",
            )
        )

    return ReconciliationResult(corrected_output=corrected, events=events)
