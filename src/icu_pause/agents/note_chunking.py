"""Token-budget chunking for fusion-mode note summarization (substudy only).

This module exists to keep the CR-DSF Note Interpreter (N1) within the served
context window on long-LOS cases. It is imported ONLY by the experimental
fusion modes (cr_dsf / hybrid_v1 via interpreter.py); the locked early_fusion
production path never loads it (see graph/workflow.py:499-515).

Root cause it addresses: NoteInterpreterAgent concatenates all six domains'
routed notes into a single LLM call. On long cases the aggregate exceeds the
window (observed: 126,977 input + 4,096 output > 131,072), the call 400s, and
the existing except-fallback silently zeroes every domain summary -> a
complete-looking brief with the entire notes stream missing.

Design (v1): DOMAIN-LEVEL greedy bin-packing.
  - Common case (everything fits one call): a single chunk -> behavior is
    byte-identical to the pre-change single-call interpreter. No perturbation.
  - Overflow case: domains are split across multiple calls. Because each domain
    lives in exactly one chunk, per-domain summaries are disjoint and merge by
    plain dict-union -- NO second "merge" LLM call, NO cross-chunk coreference
    loss between domains.
  - Pathological case (one domain's notes alone exceed the budget): that domain
    gets its own chunk and the caller truncates its notes with an EXPLICIT
    warning (never a silent drop). Within-domain map-reduce is a documented
    v1.1 enhancement.

Token estimation is deliberately conservative (over-counts) so we stay under
the window without needing the served model's exact tokenizer at call time.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Conservative chars-per-token. English clinical text runs ~3.5-4.5 chars/token;
# 3.0 intentionally OVER-estimates tokens so the budget check errs toward safety.
_CHARS_PER_TOKEN = 3.0


def estimate_tokens(text: str) -> int:
    """Conservative (over-)estimate of a string's token count."""
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def char_budget(max_input_tokens: int) -> int:
    """Approximate character budget for a token budget (used for truncation)."""
    return int(max_input_tokens * _CHARS_PER_TOKEN)


def plan_domain_chunks(
    domain_token_counts: dict[str, int],
    max_input_tokens: int,
) -> list[list[str]]:
    """Greedy bin-pack domains into chunks each <= max_input_tokens.

    Args:
        domain_token_counts: ordered {domain_name: estimated_tokens}. Insertion
            order is preserved in the output (callers rely on stable ordering).
        max_input_tokens: per-call input-token budget (window minus reserved
            output budget, prompt overhead, and safety margin).

    Returns:
        List of chunks; each chunk is a list of domain names. A domain whose own
        token count exceeds the budget is returned alone in its own chunk so the
        caller can truncate it explicitly.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for domain, tok in domain_token_counts.items():
        if tok > max_input_tokens:
            # Oversized single domain: flush whatever is buffered, then give it
            # its own chunk for the caller to truncate-with-warning.
            if current:
                chunks.append(current)
                current, current_tokens = [], 0
            chunks.append([domain])
            continue

        if current and current_tokens + tok > max_input_tokens:
            chunks.append(current)
            current, current_tokens = [], 0

        current.append(domain)
        current_tokens += tok

    if current:
        chunks.append(current)

    return chunks
