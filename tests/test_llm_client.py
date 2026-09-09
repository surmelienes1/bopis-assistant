"""
tests/test_llm_client.py

backend/llm_client.py's own docstring draws a hard line between two
exception types that must never be confused: LLMClientError (config/
format problems, never retried) and LLMUnavailableError (raised only
after a transient failure has been retried once and failed again --
"the EXACT exception ai_ranking.py should catch"). This file tests
that contract directly, plus the transient-vs-not classification and
the environment-variable validation that gates client construction.

None of these tests make a real network call. `call_structured()`'s
tests monkeypatch `llm_client._build_client` to return a fake client
object whose `.chat.completions.create` is a plain Python callable we
control -- this tests call_structured()'s retry/error-translation logic
in isolation from openai's actual HTTP transport, which is exactly the
boundary this module exists to own (see its own docstring: "candidate
retrieval, prompt construction, ranking... belong to
[other modules]... this file has no product knowledge").

A `_reset_client_cache` autouse fixture guards every test in this file
against the module-level `_client_cache` singleton leaking between
tests (see llm_client.py's own docstring on why that cache exists).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from backend import llm_client


@pytest.fixture(autouse=True)
def _reset_client_cache(monkeypatch):
    """Every test gets a clean singleton and a clean provider-relevant
    environment, regardless of what earlier tests (or the real .env
    loaded at import time) set.
    """
    monkeypatch.setattr(llm_client, "_client_cache", None)
    for var in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "LLM_MODEL",
        "AICORE_CLIENT_ID",
        "AICORE_CLIENT_SECRET",
        "AICORE_AUTH_URL",
        "AICORE_BASE_URL",
        "AICORE_RESOURCE_GROUP",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def _fake_response(content) -> SimpleNamespace:
    """Build a minimal stand-in for openai's ChatCompletion response
    shape: raw.choices[0].message.content.
    """
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _api_connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "https://example.com"))


def _api_status_error(status_code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(status_code, request=request)
    return openai.APIStatusError("boom", response=response, body=None)


# ----------------------------------------------------------------------
# _is_transient_provider_error
# ----------------------------------------------------------------------


class TestIsTransientProviderError:
    def test_connection_error_is_transient(self):
        assert llm_client._is_transient_provider_error(_api_connection_error()) is True

    @pytest.mark.parametrize("status", [500, 502, 503, 599])
    def test_5xx_status_is_transient(self, status):
        assert llm_client._is_transient_provider_error(_api_status_error(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 429])
    def test_4xx_status_is_not_transient(self, status):
        assert llm_client._is_transient_provider_error(_api_status_error(status)) is False

    def test_arbitrary_exception_is_not_transient(self):
        assert llm_client._is_transient_provider_error(ValueError("nope")) is False


# ----------------------------------------------------------------------
# _use_sap_ai_core / _check_required_env
# ----------------------------------------------------------------------


class TestUseSapAiCore:
    def test_false_when_unset(self, monkeypatch):
        assert llm_client._use_sap_ai_core() is False

    def test_true_when_client_id_set(self, monkeypatch):
        monkeypatch.setenv("AICORE_CLIENT_ID", "abc")
        assert llm_client._use_sap_ai_core() is True


class TestCheckRequiredEnvPlainOpenAI:
    def test_raises_when_api_key_missing(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        with pytest.raises(llm_client.LLMClientError, match="OPENAI_API_KEY"):
            llm_client._check_required_env()

    def test_raises_when_model_missing(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with pytest.raises(llm_client.LLMClientError, match="LLM_MODEL"):
            llm_client._check_required_env()

    def test_passes_when_both_present(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        llm_client._check_required_env()  # must not raise

    def test_error_message_never_includes_secret_value(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        with pytest.raises(llm_client.LLMClientError) as exc_info:
            llm_client._check_required_env()
        # Only the variable NAME should appear, never a value -- there's
        # no value to leak here since the var is unset, but this also
        # guards the message format itself never echoes os.environ.
        assert "sk-" not in str(exc_info.value)


class TestCheckRequiredEnvSapAiCore:
    def _set_all_but(self, monkeypatch, missing: set[str]):
        values = {
            "AICORE_CLIENT_ID": "client-id",
            "AICORE_CLIENT_SECRET": "secret",
            "AICORE_AUTH_URL": "https://auth.example.com",
            "AICORE_BASE_URL": "https://base.example.com",
            "AICORE_RESOURCE_GROUP": "default",
            "LLM_MODEL": "gpt-4o-mini",
        }
        for key, value in values.items():
            if key not in missing:
                monkeypatch.setenv(key, value)

    def test_passes_when_fully_configured(self, monkeypatch):
        self._set_all_but(monkeypatch, missing=set())
        llm_client._check_required_env()  # must not raise

    def test_raises_when_client_secret_missing(self, monkeypatch):
        self._set_all_but(monkeypatch, missing={"AICORE_CLIENT_SECRET"})
        with pytest.raises(llm_client.LLMClientError, match="AICORE_CLIENT_SECRET"):
            llm_client._check_required_env()

    def test_raises_when_auth_url_missing(self, monkeypatch):
        self._set_all_but(monkeypatch, missing={"AICORE_AUTH_URL"})
        with pytest.raises(llm_client.LLMClientError, match="AICORE_AUTH_URL"):
            llm_client._check_required_env()

    def test_raises_when_model_missing(self, monkeypatch):
        self._set_all_but(monkeypatch, missing={"LLM_MODEL"})
        with pytest.raises(llm_client.LLMClientError, match="LLM_MODEL"):
            llm_client._check_required_env()

    def test_sap_branch_taken_over_plain_openai_when_client_id_set(self, monkeypatch):
        # OPENAI_API_KEY absent entirely -- if the plain-OpenAI branch
        # were mistakenly checked instead, this would raise mentioning
        # OPENAI_API_KEY. It must raise about the AICORE side instead.
        monkeypatch.setenv("AICORE_CLIENT_ID", "client-id")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        with pytest.raises(llm_client.LLMClientError, match="AICORE_AUTH_URL"):
            llm_client._check_required_env()


# ----------------------------------------------------------------------
# _extract_text
# ----------------------------------------------------------------------


class TestExtractText:
    def test_extracts_valid_content(self):
        raw = _fake_response("hello world")
        assert llm_client._extract_text(raw) == "hello world"

    def test_no_choices_raises_client_error(self):
        raw = SimpleNamespace(choices=[])
        with pytest.raises(llm_client.LLMClientError, match="no usable choices"):
            llm_client._extract_text(raw)

    def test_missing_choices_attribute_raises_client_error(self):
        raw = SimpleNamespace()
        with pytest.raises(llm_client.LLMClientError, match="no usable choices"):
            llm_client._extract_text(raw)

    def test_none_content_raises_client_error(self):
        raw = _fake_response(None)
        with pytest.raises(llm_client.LLMClientError, match="no usable textual content"):
            llm_client._extract_text(raw)

    def test_blank_content_raises_client_error(self):
        raw = _fake_response("   ")
        with pytest.raises(llm_client.LLMClientError, match="no usable textual content"):
            llm_client._extract_text(raw)

    def test_non_string_content_raises_client_error(self):
        raw = _fake_response(12345)
        with pytest.raises(llm_client.LLMClientError):
            llm_client._extract_text(raw)


# ----------------------------------------------------------------------
# call_structured -- retry policy and exception translation.
# ----------------------------------------------------------------------


class TestCallStructured:
    def _fake_client(self, create_mock: MagicMock) -> SimpleNamespace:
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )

    def test_success_on_first_attempt(self, monkeypatch):
        create_mock = MagicMock(return_value=_fake_response("ranked output"))
        monkeypatch.setattr(
            llm_client, "_build_client", lambda: self._fake_client(create_mock)
        )
        monkeypatch.setattr(llm_client, "_model_name", lambda: "test-model")

        result = llm_client.call_structured("system", "user", max_tokens=123)

        assert result == "ranked output"
        assert create_mock.call_count == 1
        _, kwargs = create_mock.call_args
        assert kwargs["model"] == "test-model"
        assert kwargs["max_tokens"] == 123
        assert kwargs["temperature"] == llm_client._STRUCTURED_OUTPUT_TEMPERATURE
        assert kwargs["timeout"] == llm_client._REQUEST_TIMEOUT_SECONDS
        assert kwargs["messages"] == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

    def test_transient_failure_then_success_retries_once(self, monkeypatch):
        create_mock = MagicMock(
            side_effect=[_api_connection_error(), _fake_response("ok after retry")]
        )
        monkeypatch.setattr(
            llm_client, "_build_client", lambda: self._fake_client(create_mock)
        )
        monkeypatch.setattr(llm_client, "_model_name", lambda: "test-model")

        result = llm_client.call_structured("system", "user")

        assert result == "ok after retry"
        assert create_mock.call_count == 2

    def test_transient_failure_twice_raises_unavailable(self, monkeypatch):
        exc1 = _api_connection_error()
        exc2 = _api_connection_error()
        create_mock = MagicMock(side_effect=[exc1, exc2])
        monkeypatch.setattr(
            llm_client, "_build_client", lambda: self._fake_client(create_mock)
        )
        monkeypatch.setattr(llm_client, "_model_name", lambda: "test-model")

        with pytest.raises(llm_client.LLMUnavailableError) as exc_info:
            llm_client.call_structured("system", "user")

        assert create_mock.call_count == 2
        # The original exception is preserved as __cause__, per the
        # module docstring's stated contract.
        assert exc_info.value.__cause__ is exc2

    def test_5xx_status_error_is_treated_as_transient_and_retried(self, monkeypatch):
        create_mock = MagicMock(
            side_effect=[_api_status_error(503), _fake_response("recovered")]
        )
        monkeypatch.setattr(
            llm_client, "_build_client", lambda: self._fake_client(create_mock)
        )
        monkeypatch.setattr(llm_client, "_model_name", lambda: "test-model")

        result = llm_client.call_structured("system", "user")

        assert result == "recovered"
        assert create_mock.call_count == 2

    def test_4xx_status_error_is_not_retried_and_raises_client_error(self, monkeypatch):
        create_mock = MagicMock(side_effect=[_api_status_error(400)])
        monkeypatch.setattr(
            llm_client, "_build_client", lambda: self._fake_client(create_mock)
        )
        monkeypatch.setattr(llm_client, "_model_name", lambda: "test-model")

        with pytest.raises(llm_client.LLMClientError):
            llm_client.call_structured("system", "user")

        # Non-transient failures must NOT be retried.
        assert create_mock.call_count == 1

    def test_arbitrary_exception_is_not_retried_and_becomes_client_error(self, monkeypatch):
        create_mock = MagicMock(side_effect=[RuntimeError("weird sdk bug")])
        monkeypatch.setattr(
            llm_client, "_build_client", lambda: self._fake_client(create_mock)
        )
        monkeypatch.setattr(llm_client, "_model_name", lambda: "test-model")

        with pytest.raises(llm_client.LLMClientError, match="weird sdk bug"):
            llm_client.call_structured("system", "user")
        assert create_mock.call_count == 1

    def test_missing_config_raises_before_any_network_call(self, monkeypatch):
        # No env vars set at all (autouse fixture cleared them) and
        # _build_client is NOT monkeypatched here -- this exercises the
        # real _build_client -> _check_required_env path end to end.
        with pytest.raises(llm_client.LLMClientError, match="OPENAI_API_KEY"):
            llm_client.call_structured("system", "user")

    def test_client_error_from_extract_text_propagates_and_is_not_retried(self, monkeypatch):
        # A malformed-but-successful response (no usable content) must
        # surface as LLMClientError, and must NOT trigger a retry --
        # retrying an identical malformed response can't help.
        create_mock = MagicMock(return_value=_fake_response(None))
        monkeypatch.setattr(
            llm_client, "_build_client", lambda: self._fake_client(create_mock)
        )
        monkeypatch.setattr(llm_client, "_model_name", lambda: "test-model")

        with pytest.raises(llm_client.LLMClientError, match="no usable textual content"):
            llm_client.call_structured("system", "user")
        assert create_mock.call_count == 1


# ----------------------------------------------------------------------
# _build_client caching behavior
# ----------------------------------------------------------------------


class TestBuildClientCaching:
    def test_returns_cached_instance_without_rebuilding(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(llm_client, "_client_cache", sentinel)
        assert llm_client._build_client() is sentinel

    def test_builds_plain_openai_client_when_not_using_sap(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        sentinel = object()
        monkeypatch.setattr(llm_client, "_build_plain_openai_client", lambda: sentinel)
        result = llm_client._build_client()
        assert result is sentinel
        # Second call must reuse the cache, not call the builder again.
        calls = []
        monkeypatch.setattr(
            llm_client,
            "_build_plain_openai_client",
            lambda: calls.append(1) or sentinel,
        )
        llm_client._build_client()
        assert calls == []

    def test_build_plain_openai_client_returns_a_real_openai_client(self, monkeypatch):
        # Exercises the real (non-monkeypatched) _build_plain_openai_client
        # body: constructing an OpenAI() instance performs no network
        # I/O, so this is safe to run for real and confirms the
        # function's actual wiring (api_key / base_url plumbing), not
        # just a mocked stand-in.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        client = llm_client._build_plain_openai_client()
        assert client.api_key == "sk-test-key"

    def test_build_plain_openai_client_passes_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.example.com/v1")
        client = llm_client._build_plain_openai_client()
        assert str(client.base_url) == "https://custom.example.com/v1/"

    def test_builds_sap_client_when_aicore_configured(self, monkeypatch):
        monkeypatch.setenv("AICORE_CLIENT_ID", "id")
        monkeypatch.setenv("AICORE_CLIENT_SECRET", "secret")
        monkeypatch.setenv("AICORE_AUTH_URL", "https://auth.example.com")
        monkeypatch.setenv("AICORE_BASE_URL", "https://base.example.com")
        monkeypatch.setenv("AICORE_RESOURCE_GROUP", "default")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
        sentinel = object()
        monkeypatch.setattr(llm_client, "_build_sap_ai_core_client", lambda: sentinel)
        assert llm_client._build_client() is sentinel
