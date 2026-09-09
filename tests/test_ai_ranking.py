"""
tests/test_ai_ranking.py

ai_ranking.py's own docstring is explicit about its contract:
"PRODUCES UNTRUSTED OUTPUT ONLY." This file tests rank_candidates()'s
job -- calling the LLM and turning its raw text into either a parsed
dict or None -- entirely by mocking backend.llm_client.call_structured,
never by making a real network call. Whether that parsed dict is
actually TRUSTWORTHY (right shape, real product_ids, valid scores) is
ranking_validation.py's job and is tested in test_ranking_validation.py,
not here -- this file only tests "did rank_candidates() correctly turn
provider output into {dict output} or {None}."

Also covered: the module's documented exception-handling boundary --
LLMUnavailableError is caught and becomes None (the fallback signal),
but a genuine LLMClientError is NOT caught and must propagate, so a
config bug is never silently reinterpreted as "the LLM had nothing to
say" (see the module's own docstring, last paragraph).
"""

from __future__ import annotations

import json

import pytest

from backend import ai_ranking
from backend.llm_client import LLMClientError, LLMUnavailableError


ORDER_ITEM = {
    "product_id": "SKU-COKE-ZERO-150",
    "name": "Coca-Cola Zero 1.5L",
    "brand": "Coca-Cola",
    "size": 1.5,
    "unit": "L",
    "price": 2.49,
}

CANDIDATES = [
    {
        "product_id": "SKU-COKE-ZERO-100",
        "name": "Coca-Cola Zero 1L",
        "brand": "Coca-Cola",
        "size": 1.0,
        "unit": "L",
        "price": 1.99,
    },
    {
        "product_id": "SKU-SPRITE-150",
        "name": "Sprite 1.5L",
        "brand": "Coca-Cola",
        "size": 1.5,
        "unit": "L",
        "price": 2.39,
    },
]


# ----------------------------------------------------------------------
# No candidates -> short-circuit, never calls the LLM at all.
# ----------------------------------------------------------------------


class TestNoCandidatesShortCircuit:
    def test_empty_candidate_list_returns_none_without_calling_llm(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            ai_ranking,
            "call_structured",
            lambda *a, **kw: called.append(1) or "{}",
        )
        result = ai_ranking.rank_candidates(ORDER_ITEM, [])
        assert result is None
        assert called == []


# ----------------------------------------------------------------------
# Successful, well-formed responses.
# ----------------------------------------------------------------------


class TestSuccessfulResponses:
    def test_valid_json_object_is_parsed_and_returned(self, monkeypatch):
        payload = {
            "recommendations": [
                {"product_id": "SKU-COKE-ZERO-100", "score": 0.9, "reason": "Same brand, closest size."}
            ]
        }
        monkeypatch.setattr(
            ai_ranking, "call_structured", lambda *a, **kw: json.dumps(payload)
        )
        result = ai_ranking.rank_candidates(ORDER_ITEM, CANDIDATES)
        assert result == payload

    def test_strips_a_full_response_markdown_fence(self, monkeypatch):
        payload = {"recommendations": [{"product_id": "SKU-SPRITE-150", "score": 0.5, "reason": "x"}]}
        fenced = f"```json\n{json.dumps(payload)}\n```"
        monkeypatch.setattr(ai_ranking, "call_structured", lambda *a, **kw: fenced)
        result = ai_ranking.rank_candidates(ORDER_ITEM, CANDIDATES)
        assert result == payload

    def test_strips_fence_without_json_language_tag(self, monkeypatch):
        payload = {"recommendations": []}
        fenced = f"```\n{json.dumps(payload)}\n```"
        monkeypatch.setattr(ai_ranking, "call_structured", lambda *a, **kw: fenced)
        result = ai_ranking.rank_candidates(ORDER_ITEM, CANDIDATES)
        assert result == payload

    def test_passes_max_tokens_and_prompt_to_call_structured(self, monkeypatch):
        captured = {}

        def fake_call_structured(system_prompt, user_content, max_tokens):
            captured["system_prompt"] = system_prompt
            captured["user_content"] = user_content
            captured["max_tokens"] = max_tokens
            return "{}"

        monkeypatch.setattr(ai_ranking, "call_structured", fake_call_structured)
        ai_ranking.rank_candidates(ORDER_ITEM, CANDIDATES)

        assert captured["max_tokens"] == ai_ranking._MAX_TOKENS
        # The user content must be exactly the order_item/candidates the
        # caller passed, round-tripped through JSON -- ai_ranking.py must
        # never add, remove, or re-derive candidates (see module
        # docstring's "INPUT side" boundary).
        decoded = json.loads(captured["user_content"])
        assert decoded == {"order_item": ORDER_ITEM, "candidates": CANDIDATES}
        assert "only rank" in captured["system_prompt"].lower() or "may only rank" in captured["system_prompt"].lower()


# ----------------------------------------------------------------------
# Malformed-but-syntactically-parseable responses -> None.
# ----------------------------------------------------------------------


class TestMalformedResponsesBecomeNone:
    def test_unparseable_json_returns_none(self, monkeypatch):
        monkeypatch.setattr(ai_ranking, "call_structured", lambda *a, **kw: "not json at all {{{")
        assert ai_ranking.rank_candidates(ORDER_ITEM, CANDIDATES) is None

    def test_bare_json_list_returns_none(self, monkeypatch):
        monkeypatch.setattr(ai_ranking, "call_structured", lambda *a, **kw: "[1, 2, 3]")
        assert ai_ranking.rank_candidates(ORDER_ITEM, CANDIDATES) is None

    def test_bare_json_string_returns_none(self, monkeypatch):
        monkeypatch.setattr(ai_ranking, "call_structured", lambda *a, **kw: '"just a string"')
        assert ai_ranking.rank_candidates(ORDER_ITEM, CANDIDATES) is None

    def test_bare_json_number_returns_none(self, monkeypatch):
        monkeypatch.setattr(ai_ranking, "call_structured", lambda *a, **kw: "42")
        assert ai_ranking.rank_candidates(ORDER_ITEM, CANDIDATES) is None

    def test_empty_string_returns_none(self, monkeypatch):
        monkeypatch.setattr(ai_ranking, "call_structured", lambda *a, **kw: "")
        assert ai_ranking.rank_candidates(ORDER_ITEM, CANDIDATES) is None

    def test_fence_like_pattern_in_the_middle_of_malformed_text_not_unwrapped(self, monkeypatch):
        # Per _strip_markdown_fence's own docstring: only a fence that
        # wraps the ENTIRE response is stripped. A fence-like fragment
        # inside otherwise-broken text is left alone, and the resulting
        # JSONDecodeError becomes None -- not a guess at which fragment
        # was meant to be the payload.
        text = 'Here is my answer: ```json\n{"recommendations": []}\n``` -- hope that helps!'
        monkeypatch.setattr(ai_ranking, "call_structured", lambda *a, **kw: text)
        assert ai_ranking.rank_candidates(ORDER_ITEM, CANDIDATES) is None


# ----------------------------------------------------------------------
# Exception-handling boundary.
# ----------------------------------------------------------------------


class TestExceptionBoundary:
    def test_llm_unavailable_error_becomes_none(self, monkeypatch):
        def raise_unavailable(*a, **kw):
            raise LLMUnavailableError("provider down")

        monkeypatch.setattr(ai_ranking, "call_structured", raise_unavailable)
        assert ai_ranking.rank_candidates(ORDER_ITEM, CANDIDATES) is None

    def test_llm_client_error_propagates_uncaught(self, monkeypatch):
        def raise_client_error(*a, **kw):
            raise LLMClientError("missing config")

        monkeypatch.setattr(ai_ranking, "call_structured", raise_client_error)
        with pytest.raises(LLMClientError, match="missing config"):
            ai_ranking.rank_candidates(ORDER_ITEM, CANDIDATES)

    def test_arbitrary_exception_also_propagates_uncaught(self, monkeypatch):
        # rank_candidates() only catches LLMUnavailableError and JSON
        # parse failures -- anything else (a genuine bug) must surface.
        def raise_weird(*a, **kw):
            raise RuntimeError("totally unexpected")

        monkeypatch.setattr(ai_ranking, "call_structured", raise_weird)
        with pytest.raises(RuntimeError, match="totally unexpected"):
            ai_ranking.rank_candidates(ORDER_ITEM, CANDIDATES)


# ----------------------------------------------------------------------
# _strip_markdown_fence unit-level behavior.
# ----------------------------------------------------------------------


class TestStripMarkdownFence:
    def test_no_fence_returns_stripped_text_unchanged(self):
        assert ai_ranking._strip_markdown_fence('  {"a": 1}  ') == '{"a": 1}'

    def test_fence_with_json_tag_stripped(self):
        assert ai_ranking._strip_markdown_fence('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_fence_without_tag_stripped(self):
        assert ai_ranking._strip_markdown_fence('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_fence_is_case_insensitive(self):
        assert ai_ranking._strip_markdown_fence('```JSON\n{"a": 1}\n```') == '{"a": 1}'

    def test_partial_fence_not_stripped(self):
        text = 'prefix ```json\n{"a": 1}\n``` suffix'
        assert ai_ranking._strip_markdown_fence(text) == text
