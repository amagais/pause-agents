"""LLM provider abstraction supporting Anthropic, OpenAI, and local models."""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional process-wide LLM serialization
# ---------------------------------------------------------------------------
#
# LangGraph fan-out runs domain agents in parallel, which on Azure deployments
# with tight per-deployment concurrency or TPM caps surfaces as
# APIConnectionError storms (multiple agents retrying simultaneously). Setting
# ICUPAUSE_SERIALIZE_LLM=1 forces all .invoke() calls in the process through
# a single mutex — the graph structure stays parallel but only one LLM call
# is in-flight at a time. Loses the wall-clock benefit of fan-out, but works
# within strict provider rate limits without requiring quota increases or
# graph rewrites.
#
# Toggle with env var at run time; no config-file change needed.
_SERIALIZE_LLM = os.environ.get("ICUPAUSE_SERIALIZE_LLM", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
_LLM_LOCK = threading.Lock() if _SERIALIZE_LLM else None
if _SERIALIZE_LLM:
    logger.info(
        "ICUPAUSE_SERIALIZE_LLM=1 — all LLM calls will be serialized via a "
        "process-wide lock. Expect slower wall-clock per case."
    )


@contextmanager
def _serialize_if_enabled():
    """Context manager that holds the LLM lock when serialization is enabled."""
    if _LLM_LOCK is None:
        yield
        return
    with _LLM_LOCK:
        yield

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BASE_DELAY_S = 2.0  # 2s, 8s with exponential backoff
_BACKOFF_FACTOR = 4.0
_JITTER_MAX_S = 1.0


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is transient and worth retrying."""
    # OpenAI SDK exceptions (also used by Azure and local via openai client)
    try:
        from openai import (
            APIConnectionError as OAIConnErr,
            APITimeoutError as OAITimeout,
            InternalServerError as OAIInternal,
            RateLimitError as OAIRateLimit,
        )
        if isinstance(exc, (OAIConnErr, OAITimeout, OAIInternal, OAIRateLimit)):
            return True
    except ImportError:
        pass

    # Anthropic SDK exceptions
    try:
        from anthropic import (
            APIConnectionError as AntConnErr,
            APITimeoutError as AntTimeout,
            InternalServerError as AntInternal,
            RateLimitError as AntRateLimit,
        )
        if isinstance(exc, (AntConnErr, AntTimeout, AntInternal, AntRateLimit)):
            return True
    except ImportError:
        pass

    return False


def _retry_with_backoff(func, *, provider_name: str = "llm"):
    """Call func() with retry on transient API errors.

    Non-retryable errors (auth, bad request, etc.) are raised immediately.
    When ICUPAUSE_SERIALIZE_LLM=1, the lock is held across the entire
    func() call (and across retries) so only one LLM request is in flight
    at a time — necessary on tight Azure quotas where parallel fan-out
    triggers APIConnectionError storms.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with _serialize_if_enabled():
                return func()
        except Exception as exc:
            if not _is_retryable(exc) or attempt == _MAX_RETRIES:
                if attempt > 1:
                    logger.error(
                        "%s: failed after %d attempts: %s",
                        provider_name, attempt, exc,
                    )
                raise
            delay = _BASE_DELAY_S * (_BACKOFF_FACTOR ** (attempt - 1))
            jitter = random.uniform(0, _JITTER_MAX_S)
            total_delay = delay + jitter
            logger.warning(
                "%s: attempt %d/%d failed (%s: %s), retrying in %.1fs",
                provider_name, attempt, _MAX_RETRIES,
                type(exc).__name__, exc, total_delay,
            )
            last_exc = exc
            time.sleep(total_delay)
    raise last_exc  # unreachable, but satisfies type checker


@dataclass
class LLMUsage:
    """Token usage and latency for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""


def _strip_thinking_tags(text: str) -> str:
    """Strip reasoning blocks from model output.

    Handles two families:
    - <think>...</think> — QwQ, DeepSeek-R1
    - <unused\\d+>...<unused\\d+> — MedGemma 1.5 (Gemma3 thinking mode emits unused94/95)
    """
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<unused\d+>.*?<unused\d+>", "", text, flags=re.DOTALL)
    return text.strip()


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from LLM output.

    Handles fences that appear after preamble text (common with GPT-4.1),
    not just fences at the start of the string.
    """
    cleaned = text.strip()
    if "```" in cleaned:
        lines = cleaned.split("\n")
        json_lines = []
        inside = False
        for line in lines:
            if line.strip().startswith("```") and not inside:
                inside = True
                continue
            if line.strip() == "```" and inside:
                break
            if inside:
                json_lines.append(line)
        if json_lines:
            cleaned = "\n".join(json_lines)
    return cleaned


def _clean_llm_output(text: str) -> str:
    """Clean LLM output: strip thinking tags, then code fences."""
    cleaned = _strip_thinking_tags(text)
    cleaned = _strip_code_fences(cleaned)
    return cleaned


def _pydantic_to_openai_schema(model: Type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to an OpenAI-compatible JSON schema for structured output.

    OpenAI strict mode requires:
    - All properties listed in "required"
    - No "default" values
    - "additionalProperties": false on all objects
    - No "title" fields
    """
    schema = model.model_json_schema()

    def _clean(obj: dict) -> dict:
        """Recursively enforce strict-mode constraints."""
        if "properties" in obj:
            for prop in obj["properties"].values():
                prop.pop("default", None)
                if isinstance(prop, dict):
                    _clean(prop)
            obj["required"] = list(obj["properties"].keys())
            obj["additionalProperties"] = False
        if "items" in obj and isinstance(obj["items"], dict):
            _clean(obj["items"])
        if "$defs" in obj:
            for defn in obj["$defs"].values():
                _clean(defn)
        obj.pop("title", None)
        return obj

    return _clean(schema)


class BaseLLM(ABC):
    """Abstract base for LLM providers."""

    last_usage: LLMUsage

    @abstractmethod
    def invoke(
        self,
        system: str,
        user: str,
        response_format: Type[BaseModel] | None = None,
    ) -> Any:
        """Send a prompt and return the response.

        After each call, ``self.last_usage`` is populated with token counts
        and latency for the most recent invocation.

        Args:
            system: System prompt.
            user: User message.
            response_format: If provided, parse response into this Pydantic model.
        """
        ...


class AnthropicLLM(BaseLLM):
    """Anthropic Claude provider."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.2, max_tokens: int = 4096):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_usage = LLMUsage()

    def invoke(self, system: str, user: str, response_format=None) -> Any:
        t0 = time.perf_counter()

        if response_format:
            # Use tool_use for structured output — Anthropic's recommended approach
            schema = response_format.model_json_schema()
            # Remove unsupported keys
            schema.pop("title", None)
            schema.pop("$defs", None)

            def _call():
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    tools=[{
                        "name": "structured_output",
                        "description": "Return the structured output.",
                        "input_schema": schema,
                    }],
                    tool_choice={"type": "tool", "name": "structured_output"},
                )

            resp = _retry_with_backoff(_call, provider_name=f"anthropic/{self.model}")
            elapsed_ms = (time.perf_counter() - t0) * 1000

            self.last_usage = LLMUsage(
                input_tokens=getattr(resp.usage, "input_tokens", 0),
                output_tokens=getattr(resp.usage, "output_tokens", 0),
                latency_ms=elapsed_ms,
                model=self.model,
            )

            # Extract tool call input as structured data
            for block in resp.content:
                if block.type == "tool_use":
                    return response_format.model_validate(block.input)

            # Fallback: if no tool_use block, try text parsing
            logger.warning("anthropic: no tool_use block in response, falling back to text parsing")
            text = next((b.text for b in resp.content if hasattr(b, "text")), "")
            return self._parse_json_response(text, response_format)
        else:
            def _call():
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )

            resp = _retry_with_backoff(_call, provider_name=f"anthropic/{self.model}")
            elapsed_ms = (time.perf_counter() - t0) * 1000

            self.last_usage = LLMUsage(
                input_tokens=getattr(resp.usage, "input_tokens", 0),
                output_tokens=getattr(resp.usage, "output_tokens", 0),
                latency_ms=elapsed_ms,
                model=self.model,
            )

            return resp.content[0].text

    @staticmethod
    def _parse_json_response(text: str, model: Type[BaseModel]) -> BaseModel:
        """Extract JSON from response text, handling markdown code blocks."""
        cleaned = _clean_llm_output(text)
        return model.model_validate_json(cleaned)


class OpenAILLM(BaseLLM):
    """OpenAI provider."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.2, max_tokens: int = 4096):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_usage = LLMUsage()

    def invoke(self, system: str, user: str, response_format=None) -> Any:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if response_format:
            # Use JSON schema mode for strict schema enforcement
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": _pydantic_to_openai_schema(response_format),
                    "strict": True,
                },
            }

        t0 = time.perf_counter()
        resp = _retry_with_backoff(
            lambda: self.client.chat.completions.create(**kwargs),
            provider_name=f"openai/{self.model}",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        usage = getattr(resp, "usage", None)
        self.last_usage = LLMUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=elapsed_ms,
            model=self.model,
        )

        msg = resp.choices[0].message
        text = msg.content or getattr(msg, "reasoning_content", None) or ""
        if response_format:
            return response_format.model_validate_json(_clean_llm_output(text))
        return text


class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI provider (GPT-4o, o3-mini, etc.)."""

    def __init__(self, model: str, api_key: str, endpoint: str,
                 api_version: str = "2024-12-01-preview",
                 temperature: float = 0.2, max_tokens: int = 4096,
                 is_reasoning: bool = False,
                 uses_max_completion_tokens: bool = False):
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.model = model  # Azure deployment name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_reasoning = is_reasoning  # o-series: no temp, no response_format
        self.uses_max_completion_tokens = uses_max_completion_tokens or is_reasoning
        self.last_usage = LLMUsage()

    def invoke(self, system: str, user: str, response_format=None) -> Any:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if self.is_reasoning:
            # o-series: no temperature, no response_format
            kwargs["max_completion_tokens"] = self.max_tokens
        elif self.uses_max_completion_tokens:
            # gpt-5.x, gpt-4.1: use max_completion_tokens but support temp + response_format
            kwargs["max_completion_tokens"] = self.max_tokens
            kwargs["temperature"] = self.temperature
        else:
            kwargs["temperature"] = self.temperature
            kwargs["max_tokens"] = self.max_tokens

        # Structured output: JSON schema mode for models that support it.
        # o-series (reasoning) doesn't support response_format at all.
        # gpt-4o supports json_schema. gpt-4.1 and gpt-5.x: use json_schema if available.
        if response_format and not self.is_reasoning:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": _pydantic_to_openai_schema(response_format),
                    "strict": True,
                },
            }

        t0 = time.perf_counter()
        resp = _retry_with_backoff(
            lambda: self.client.chat.completions.create(**kwargs),
            provider_name=f"azure/{self.model}",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        usage = getattr(resp, "usage", None)
        self.last_usage = LLMUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=elapsed_ms,
            model=self.model,
        )

        msg = resp.choices[0].message
        text = msg.content or getattr(msg, "reasoning_content", None) or ""
        if response_format:
            return response_format.model_validate_json(_clean_llm_output(text))
        return text


class LocalLLM(BaseLLM):
    """Local model via OpenAI-compatible API (vLLM, Ollama, etc.)."""

    def __init__(self, model: str, base_url: str, temperature: float = 0.2,
                 max_tokens: int = 4096, num_ctx: int = 32768,
                 backend: str = "ollama"):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.backend = backend  # "ollama" or "vllm"
        self.last_usage = LLMUsage()

    def invoke(self, system: str, user: str, response_format=None) -> Any:
        t0 = time.perf_counter()

        # Mistral/Mixtral chat templates reject a separate ``system`` turn
        # ("Conversation roles must alternate user/assistant/..."). Opt-in via
        # env to fold the system prompt into the first user turn (a single
        # user message), which is the canonical Mistral usage. Default OFF, so
        # models that accept a system role (Gemma/Qwen/DeepSeek) are unchanged.
        if os.getenv("ICUPAUSE_LOCAL_MERGE_SYSTEM") == "1":
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Ollama needs num_ctx passed via extra_body to set context window.
        # vLLM handles context window at server startup (--max-model-len).
        if self.backend == "ollama":
            kwargs["extra_body"] = {
                "options": {"num_ctx": self.num_ctx},
                "num_ctx": self.num_ctx,
            }
        # vLLM thinking-model control: Qwen3.x emits a <think> chain-of-thought
        # by default, which (a) leaks into content unless --reasoning-parser is
        # set and (b) consumes the per-agent output budget so the JSON answer is
        # never emitted (content=''), failing model_validate_json. Opt-in disable
        # via env so non-thinking models (Gemma/MedGemma) are unaffected and
        # existing results don't change. Confirmed 2026-06-07: reasoning-parser
        # alone is insufficient (budget exhaustion); thinking must be OFF.
        elif self.backend == "vllm" and os.getenv("ICUPAUSE_LOCAL_DISABLE_THINKING") == "1":
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        # NOTE: local response_format enforcement is DELIBERATELY OFF.
        # Tried twice (2026-06-06): (1) all-schemas — open-dict schemas truncated;
        # (2) allowlist-gated to fixed-field early_fusion schemas — STILL FAILED
        # the gate (scripts/smoke_enforcement_gate.sh): vLLM guided decoding
        # suppresses EOS and rambles even on fixed-field schemas (output tokens
        # +36–50%, truncation on 5/5 cases, no quality gain — final brief size
        # unchanged). vLLM logs showed NO fallback, so guided decoding engaged
        # and misbehaved. Conclusion: do NOT pass response_format on the local
        # path; the cross-model parity fix belongs at the parsing/prompt layer.
        # See memory: project_icu_pause_local_provider_structured_output.

        resp = _retry_with_backoff(
            lambda: self.client.chat.completions.create(**kwargs),
            provider_name=f"local/{self.model}",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        usage = getattr(resp, "usage", None)
        self.last_usage = LLMUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            latency_ms=elapsed_ms,
            model=self.model,
        )

        msg = resp.choices[0].message
        text = msg.content or getattr(msg, "reasoning_content", None) or ""
        if response_format:
            return response_format.model_validate_json(_clean_llm_output(text))
        return text


def create_llm(
    settings,
    max_tokens_override: int | None = None,
    temperature_override: float | None = None,
) -> BaseLLM:
    """Factory: create the LLM backend from application settings.

    Args:
        settings: Application settings.
        max_tokens_override: If provided, override the default max_tokens
            from settings (used for per-agent token budgets).
        temperature_override: If provided, override the default temperature
            from settings. Used by hybrid_v1 compression-stage callers
            (extractors) per pre-reg §1.4 lock (temperature 0.0 for
            structured-output stability). Domain agents do NOT pass this
            and retain the production temperature unchanged.
    """
    provider = settings.llm_provider
    max_tokens = max_tokens_override or settings.llm_max_tokens
    temperature = (
        temperature_override
        if temperature_override is not None
        else settings.llm_temperature
    )

    if provider == "anthropic":
        return AnthropicLLM(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "openai":
        return OpenAILLM(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "azure":
        # Reasoning models: use max_completion_tokens, no temperature, no response_format
        # o-series (o1, o3, o4) are true reasoning models
        # gpt-5.x and gpt-4.1 use max_completion_tokens but DO support temperature + response_format
        _STRICT_REASONING = ["o3", "o1", "o4"]
        _MAX_COMPLETION_TOKENS = ["o3", "o1", "o4", "gpt-5", "gpt-4.1"]
        model_lower = settings.llm_model.lower()
        is_reasoning = any(r in model_lower for r in _STRICT_REASONING)
        uses_max_completion_tokens = any(r in model_lower for r in _MAX_COMPLETION_TOKENS)
        return AzureOpenAILLM(
            model=settings.llm_model,
            api_key=settings.azure_api_key,
            endpoint=settings.azure_endpoint,
            api_version=settings.azure_api_version,
            temperature=temperature,
            max_tokens=max_tokens,
            is_reasoning=is_reasoning,
            uses_max_completion_tokens=uses_max_completion_tokens,
        )
    elif provider == "local":
        return LocalLLM(
            model=settings.llm_model,
            base_url=settings.local_llm_url,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=getattr(settings, "llm_context_window", 32768),
            backend=getattr(settings, "local_llm_backend", "ollama"),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
