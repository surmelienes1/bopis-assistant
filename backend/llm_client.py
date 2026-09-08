"""
backend/llm_client.py

Thin provider wrapper for OpenAI-compatible structured-output chat
completions, for the BOPIS prototype.

This module owns provider configuration, client construction, retry/
timeout policy, and response normalization — nothing else. Candidate
retrieval, prompt construction, ranking, and JSON-schema validation
belong to candidates.py and the ai_ranking.py / ranking_validation.py
modules that call this one; this file has no product knowledge and
does not know what a "substitution" is.

Unlike the Watchtower reference this project follows the conventions
of, this module deliberately does NOT add tool calling, streaming, or
agent orchestration. The BOPIS substitution-ranking use case is a
single structured-text call per item — a completion, not a
conversation — so the extra machinery the reference needed for a
tool-calling agent loop would be unused complexity here, not a feature.

Two exception types, kept as independent siblings of Exception (one is
NOT a subclass of the other) so a caller catching one can never
accidentally also catch the other:

    LLMClientError      Configuration problems, malformed requests, and
                         response-format failures. Never retried. This
                         signals a bug or misconfiguration — callers
                         should let it propagate, not swallow it.

    LLMUnavailableError Raised only after a transient provider failure
                         (connection/network error, or HTTP 5xx) has
                         been retried once and failed again. This is
                         the EXACT exception ai_ranking.py should catch
                         to switch to deterministic fallback ranking —
                         nothing else raised by this module means "the
                         provider is down," so nothing else should
                         trigger that fallback path.
"""

from __future__ import annotations

import os
import threading
from typing import Any

# `openai` is a hard dependency of this module regardless of which
# provider branch is active: the plain fallback imports OpenAI
# directly, and SAP GenAI Hub's `gen_ai_hub.proxy.native.openai.OpenAI`
# is itself a thin wrapper around the same SDK, raising the same
# `openai.*` exception types. Provider SDKs that are genuinely
# optional depending on which branch is configured (ai_core_sdk,
# gen_ai_hub) stay lazily imported inside their builder functions
# below, per "keep provider-specific SDK details inside this module."
import openai


class LLMClientError(Exception):
    """Configuration or response-format error. Never retried, never to
    be treated as 'provider unavailable' by a caller's fallback logic."""


class LLMUnavailableError(Exception):
    """Provider unreachable after one retry of a transient failure.
    Callers (specifically ai_ranking.py) should catch this exact type
    to fall back to deterministic ranking."""


# Set AICORE_CLIENT_ID to use SAP AI Core / GenAI Hub; otherwise fall
# back to a plain OpenAI-compatible provider. Mirrors the Watchtower
# reference's provider-selection convention.
_REQUIRED_AICORE_VARS = (
    "AICORE_CLIENT_ID",
    "AICORE_AUTH_URL",
    "AICORE_BASE_URL",
    "AICORE_RESOURCE_GROUP",
)

# Low temperature: this call is used for structured/ranking output,
# where we want the model to consistently pick the best-supported
# answer rather than explore stylistic variety.
_STRUCTURED_OUTPUT_TEMPERATURE = 0.1

# Bounded per-request timeout, applied to the chat-completion call
# itself (not to client construction — see call_structured()'s
# docstring for why that's out of scope for the retry below).
_REQUEST_TIMEOUT_SECONDS = 20.0

# Exactly one retry means two total attempts.
_MAX_ATTEMPTS = 2

_client_cache: Any = None  # Lazy singleton.
_client_lock = threading.Lock()  # Protects first-time construction.


# ----------------------------------------------------------------------
# Configuration / client construction
# ----------------------------------------------------------------------


def _resolve_client_secret() -> str | None:
    return os.environ.get("AICORE_CLIENT_SECRET")


def _use_sap_ai_core() -> bool:
    """Return whether SAP AI Core / GenAI Hub is selected, via AICORE_CLIENT_ID."""
    return bool(os.environ.get("AICORE_CLIENT_ID"))


def _check_required_env() -> None:
    """Validate configuration before touching any provider SDK.

    Raises LLMClientError naming only the missing variable NAMES, never
    values — this module never logs or includes secret values in any
    exception message.
    """
    if _use_sap_ai_core():
        missing = [v for v in _REQUIRED_AICORE_VARS if not os.environ.get(v)]
        if not _resolve_client_secret():
            missing.append("AICORE_CLIENT_SECRET")
        if not os.environ.get("LLM_MODEL"):
            missing.append("LLM_MODEL")
        if missing:
            raise LLMClientError(
                "Missing required SAP AI Core / GenAI Hub configuration in "
                f"environment: {', '.join(missing)}. Set these (see "
                ".env.example) before calling llm_client.call_structured()."
            )
    else:
        missing = [] if os.environ.get("OPENAI_API_KEY") else ["OPENAI_API_KEY"]
        if not os.environ.get("LLM_MODEL"):
            missing.append("LLM_MODEL")
        if missing:
            raise LLMClientError(
                "Missing required configuration in environment: "
                f"{', '.join(missing)}. Set AICORE_CLIENT_ID (and its SAP "
                "AI Core siblings) to use GenAI Hub, or OPENAI_API_KEY to "
                "use a plain OpenAI-compatible provider instead. See "
                ".env.example before calling llm_client.call_structured()."
            )


def _build_sap_ai_core_client():
    """Build the OpenAI-compatible SAP GenAI Hub client:
    AICoreV2Client -> GenAIHubProxyClient -> OpenAI."""
    try:
        from ai_core_sdk.ai_core_v2_client import AICoreV2Client
        from gen_ai_hub.proxy import GenAIHubProxyClient
        from gen_ai_hub.proxy.native.openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment issue, not logic
        raise LLMClientError(
            "SAP GenAI Hub SDK packages not installed "
            "(ai-core-sdk, generative-ai-hub-sdk). "
            f"Import failed: {exc}"
        ) from exc

    try:
        ai_core_client = AICoreV2Client(
            client_id=os.environ["AICORE_CLIENT_ID"],
            client_secret=_resolve_client_secret(),
            auth_url=os.environ["AICORE_AUTH_URL"],
            base_url=os.environ["AICORE_BASE_URL"],
            resource_group=os.environ["AICORE_RESOURCE_GROUP"],
        )
        proxy_client = GenAIHubProxyClient(ai_core_client=ai_core_client)
        return OpenAI(proxy_client=proxy_client)
    except LLMClientError:
        raise
    except Exception as exc:  # pragma: no cover - network/auth failure
        raise LLMClientError(
            f"Failed to initialize SAP GenAI Hub client: {exc}"
        ) from exc


def _build_plain_openai_client():
    """Build the plain OpenAI-compatible fallback client, used only when
    AICORE_CLIENT_ID is not configured."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment issue, not logic
        raise LLMClientError(
            "openai package not installed. Install it, or set "
            "AICORE_CLIENT_ID (and its SAP AI Core siblings) to use SAP "
            f"GenAI Hub instead. Import failed: {exc}"
        ) from exc

    try:
        return OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
    except Exception as exc:  # pragma: no cover - network/auth failure
        raise LLMClientError(
            f"Failed to initialize OpenAI-compatible client: {exc}"
        ) from exc


def _build_client():
    """Build and cache the configured OpenAI-compatible client.

    `_client_cache` is shared, mutable module state read and written
    from synchronous FastAPI route handlers running in a worker thread
    pool rather than strictly sequentially — same situation as db.py's
    module-level dicts. `_client_lock` makes first-time construction
    atomic; every call after that is a lock-free read of an already-set
    cache, so steady-state request handling pays no locking cost.
    Client construction itself is NOT part of call_structured()'s retry
    loop — a construction failure is a configuration/auth problem, not
    a per-request transient one, so it surfaces immediately as
    LLMClientError rather than being retried.
    """
    global _client_cache
    if _client_cache is not None:
        return _client_cache

    with _client_lock:
        if _client_cache is not None:  # another thread may have built it while we waited for the lock
            return _client_cache

        _check_required_env()
        _client_cache = (
            _build_sap_ai_core_client() if _use_sap_ai_core() else _build_plain_openai_client()
        )
        return _client_cache


def _model_name() -> str:
    model = os.environ.get("LLM_MODEL")
    if model:
        return model
    raise LLMClientError("LLM_MODEL is required.")  # pragma: no cover - caught earlier by _check_required_env


# ----------------------------------------------------------------------
# Transient-failure classification
# ----------------------------------------------------------------------


def _is_transient_provider_error(exc: Exception) -> bool:
    """Transient == worth exactly one retry: a connection/network
    failure, or an HTTP 5xx from the provider.

    Deliberately excludes everything else — bad request (400), auth
    (401/403), not-found (404), conflict (409), unprocessable (422),
    and rate-limit (429) are all either a bug in how we're calling the
    provider or a provider-side rejection that retrying the exact same
    request won't fix. Retrying those would waste the timeout budget
    on a call that's going to fail the same way twice, and would risk
    masking a real configuration bug as a flaky-network blip.
    """
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return False


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def call_structured(system_prompt: str, user_content: str, max_tokens: int = 1000) -> str:
    """Call the configured LLM with a system/user message pair and
    return the raw text of the first response choice.

    This is the ONLY function the ranking layer should call. It builds
    a minimal, non-streaming, non-tool-calling chat-completion request
    — model, one system message, one user message, low temperature,
    the caller's `max_tokens` — and returns plain text. Parsing that
    text as JSON, validating it against a schema, and deciding what to
    do if it's malformed-but-well-formed-JSON are ai_ranking.py's and
    ranking_validation.py's job, not this function's: this layer only
    knows "did the provider give me text back," not "was the text a
    valid ranking."

    Retry policy: up to `_MAX_ATTEMPTS` (2) attempts at the completion
    call itself. A transient failure (connection/network error, or
    HTTP 5xx — see _is_transient_provider_error) on the first attempt
    is retried once; a second transient failure raises
    LLMUnavailableError with the original exception preserved as
    `__cause__`. A non-transient failure (malformed request, auth
    error, 4xx) is NOT retried — it's converted directly to
    LLMClientError and raised immediately, since retrying it can't
    help and retrying would delay surfacing what is very likely a bug.

    Raises:
        LLMClientError: configuration is missing/invalid, the request
            was malformed, the provider rejected it non-transiently, or
            the provider returned no usable textual content.
        LLMUnavailableError: the provider failed transiently on both
            attempts. Callers should catch this exact exception to
            switch to deterministic fallback ranking.
    """
    client = _build_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    last_transient_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            raw = client.chat.completions.create(
                model=_model_name(),
                messages=messages,
                temperature=_STRUCTURED_OUTPUT_TEMPERATURE,
                max_tokens=max_tokens,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            if _is_transient_provider_error(exc):
                last_transient_exc = exc
                continue  # exactly one retry, per _MAX_ATTEMPTS
            raise LLMClientError(f"LLM call failed: {exc}") from exc
        else:
            return _extract_text(raw)

    # Both attempts failed transiently.
    raise LLMUnavailableError(
        f"LLM provider unavailable after {_MAX_ATTEMPTS - 1} retry "
        "(connection/network or HTTP 5xx failure on every attempt)."
    ) from last_transient_exc


def _extract_text(raw: Any) -> str:
    """Pull the raw textual content out of the first response choice.

    Raises LLMClientError — NOT LLMUnavailableError — if the provider
    responded successfully at the transport level but didn't hand back
    usable text (no choices, or a None/non-string/blank content field).
    That's a response-format problem, not a transient outage: retrying
    the identical request wouldn't reliably fix it, and silently
    returning None or an empty string here would let a caller mistake
    "the provider said nothing" for "the provider said something
    empty," which candidates/ranking code must never be allowed to
    treat as a valid (if uninformative) answer.
    """
    try:
        content = raw.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMClientError("Provider response contained no usable choices.") from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMClientError(
            f"Provider returned no usable textual content (content={content!r})."
        )
    return content