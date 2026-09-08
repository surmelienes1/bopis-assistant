"""
backend/ranking_validation.py

THIS MODULE IS THE ENFORCEMENT POINT for the invariant that model
output may never introduce a product outside the trusted deterministic
candidate set. If a hallucinated product ID ever reaches the UI, this
validator or its caller has violated the architecture.

The three-file pipeline this module completes:

    candidates.py        defines the trusted candidate universe —
                          nothing downstream may introduce a
                          product_id that didn't come from it.
    ai_ranking.py         produces UNTRUSTED model output over that
                          universe — a ranking and a one-line reason
                          per candidate, or None.
    ranking_validation.py (this file) converts that untrusted output
                          into either the SAME output unchanged (every
                          check passed) or a fully independent,
                          LLM-free deterministic ranking (any check
                          failed). There is no third outcome and no
                          partial acceptance.

The key invariant, stated plainly: NO PRODUCT ID FROM THE MODEL MAY
REACH THE UI UNLESS THAT PRODUCT ID EXISTS IN THE TRUSTED `candidates`
LIST PASSED INTO THIS FUNCTION. Every other check below (schema shape,
score range, non-empty reason) exists to keep a well-formed-looking
but broken response from reaching the associate — but the product-id
check is the one this whole module exists for, and it is checked
before anything is ever returned.

Rejection is total, not partial: if ANY recommendation in an otherwise
mostly-fine response fails ANY check, the ENTIRE response is discarded
in favor of deterministic fallback — this function never strips out
just the bad entry and ships the rest. Partial acceptance would mean
the trust boundary has a "except when most of it looks fine" clause,
which is exactly the kind of boundary that erodes in practice.

Dependency boundaries: this module (like ai_ranking.py) does not
import db, does not call FastAPI, does not modify inventory or orders,
does not commit substitutions, does not retrieve additional candidates,
and makes no customer-safety assumption beyond what's present in the
`candidates` and `original` dicts it was given. It also never calls
the LLM — the only LLM call in this codebase is
llm_client.call_structured(), reached exclusively through
ai_ranking.rank_candidates().
"""

from __future__ import annotations

import math
from typing import Any

# ----------------------------------------------------------------------
# Trust boundary: which product_ids are allowed to appear in the
# returned recommendations, under either outcome (validated AI output
# or fallback).
# ----------------------------------------------------------------------


def _candidate_product_id(candidate: Any) -> str | None:
    """Extract a well-formed product_id from one candidate record, or
    None if the record is malformed.

    Shared by the trusted-id-set builder and fallback ranking below so
    a malformed candidate record is skipped the same way in both
    places, rather than crashing one code path while being silently
    tolerated by the other. candidates.py should never actually emit a
    malformed record — this function does not assume its caller is
    bug-free anyway.
    """
    if not isinstance(candidate, dict):
        return None
    product_id = candidate.get("product_id")
    return product_id if isinstance(product_id, str) and product_id else None


def _trusted_candidate_ids(candidates: list[dict]) -> set[str]:
    """The complete, trusted set of ids a recommendation is allowed to
    reference — built defensively so one malformed candidate record
    can't crash validation for every other, legitimately-formed one.
    """
    trusted: set[str] = set()
    for candidate in candidates:
        product_id = _candidate_product_id(candidate)
        if product_id is not None:
            trusted.add(product_id)
    return trusted


# ----------------------------------------------------------------------
# AI-response validation. Returns the validated recommendation list,
# in the MODEL'S OWN ORDER, or None if any check fails anywhere.
# ----------------------------------------------------------------------


def _is_valid_score(score: Any) -> bool:
    """True only for a real, finite int/float in [0.0, 1.0].

    Explicitly rejects:
      - bool: a bool IS an int in Python (isinstance(True, int) is
        True), but a True/False "score" is never a legitimate model
        output here, so it's excluded before the isinstance(..., int)
        check would otherwise let it slip through.
      - numeric strings like "0.9": the schema calls for a number, and
        silently coercing a string would hide a model that isn't
        actually following the requested output shape.
      - NaN / inf: math.isfinite() makes the exclusion explicit rather
        than relying on the fact that `float('nan') <= 1.0` happens to
        already be False.
    """
    if isinstance(score, bool):
        return False
    if not isinstance(score, (int, float)):
        return False
    if not math.isfinite(score):
        return False
    return 0.0 <= score <= 1.0


def _validate_single_recommendation(rec: Any, trusted_ids: set[str]) -> dict | None:
    """Validate one recommendation record. Returns a clean {product_id,
    score, reason} dict on success, or None on ANY failure — including
    the one check this whole module exists for for: `product_id` must
    be a member of `trusted_ids`.
    """
    if not isinstance(rec, dict):
        return None

    product_id = rec.get("product_id")
    if not isinstance(product_id, str) or not product_id:
        return None
    if product_id not in trusted_ids:
        return None

    score = rec.get("score")
    if not _is_valid_score(score):
        return None

    reason = rec.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return None

    return {"product_id": product_id, "score": float(score), "reason": reason}


def _validate_ai_response(raw: dict | None, trusted_ids: set[str]) -> list[dict] | None:
    """Return the AI's recommendations, unchanged in the model's own
    order, ONLY if every single one survives validation. Any single
    failure discards the WHOLE response — see the module docstring for
    why partial acceptance isn't an option here.

    An empty (but well-typed) `recommendations` list is also treated
    as a failure: with a non-empty trusted candidate set on hand (the
    only case this is ever called for — see rank_candidates()'s point
    1 and this module's caller), a response that recommends nothing
    doesn't serve the associate any better than a malformed one would,
    and deterministic fallback can still offer a real ranked list.
    """
    if not isinstance(raw, dict):
        return None

    recommendations = raw.get("recommendations")
    if not isinstance(recommendations, list) or not recommendations:
        return None

    validated: list[dict] = []
    for rec in recommendations:
        checked = _validate_single_recommendation(rec, trusted_ids)
        if checked is None:
            return None
        validated.append(checked)

    return validated


# ----------------------------------------------------------------------
# Deterministic, LLM-free fallback ranking.
# ----------------------------------------------------------------------

# unit (lowercased) -> (physical-quantity family, multiplier to that
# family's base unit). Base units: liters for volume, grams for weight.
#
# Product.size is already a typed float and Product.unit a separate
# string (see models.py) — this catalog never hands us a compound
# string like "6x330ml" to parse; a multi-pack's size is already
# expressed as its total volume (SKU-COKE-ZERO-6X330 stores
# size=1.98, unit="L"). What DOES still need normalizing is comparing
# across differently-scaled units of the SAME physical quantity
# (1L vs 500ml) and correctly refusing to compare across DIFFERENT
# quantities (a volume in liters against a weight in grams) instead of
# silently treating their raw numbers as commensurate.
_UNIT_TO_BASE: dict[str, tuple[str, float]] = {
    "l": ("volume", 1.0),
    "ml": ("volume", 0.001),
    "kg": ("weight", 1000.0),
    "g": ("weight", 1.0),
}

# Returned whenever two sizes (or two prices) can't be meaningfully
# compared — unrecognized unit, missing/non-numeric field, or units
# from different physical-quantity families. Using +inf rather than a
# large finite number means "incomparable" always sorts strictly after
# every genuinely comparable candidate, with no arbitrary magnitude to
# tune or accidentally undersize.
_INCOMPARABLE_DISTANCE = math.inf


def _normalize_size(size: Any, unit: Any) -> tuple[str, float] | None:
    """Return (family, value-in-base-unit) for one product's size, or
    None if `unit` isn't one of the handful this prototype's catalog
    actually uses, or `size` isn't a real number.
    """
    if not isinstance(size, (int, float)) or isinstance(size, bool):
        return None
    if not isinstance(unit, str):
        return None
    entry = _UNIT_TO_BASE.get(unit.strip().lower())
    if entry is None:
        return None
    family, multiplier = entry
    return family, float(size) * multiplier


def _size_distance(candidate: dict, original: dict) -> float:
    """Absolute size difference in a shared base unit, or
    _INCOMPARABLE_DISTANCE if the two sizes aren't in the same
    physical-quantity family (or either is missing/malformed).
    """
    norm_candidate = _normalize_size(candidate.get("size"), candidate.get("unit"))
    norm_original = _normalize_size(original.get("size"), original.get("unit"))
    if norm_candidate is None or norm_original is None:
        return _INCOMPARABLE_DISTANCE

    family_candidate, value_candidate = norm_candidate
    family_original, value_original = norm_original
    if family_candidate != family_original:
        return _INCOMPARABLE_DISTANCE

    return abs(value_candidate - value_original)


def _price_distance(candidate: dict, original: dict) -> float:
    """Absolute price difference, or _INCOMPARABLE_DISTANCE if either
    price is missing or not a real number."""
    price_candidate = candidate.get("price")
    price_original = original.get("price")
    if not isinstance(price_candidate, (int, float)) or isinstance(price_candidate, bool):
        return _INCOMPARABLE_DISTANCE
    if not isinstance(price_original, (int, float)) or isinstance(price_original, bool):
        return _INCOMPARABLE_DISTANCE
    return abs(float(price_candidate) - float(price_original))


_FALLBACK_REASON = "Same category, ranked by brand, size, and price similarity (fallback ranking)."


def _fallback_rank(candidates: list[dict], original: dict) -> list[dict]:
    """Deterministic, LLM-free ranking — used whenever the model's
    output is missing or fails any validation check.

    Sort priority, applied via one tuple key so Python's stable sort
    handles tie-breaking automatically in the stated order:
        1. Same brand as `original` first.
        2. Smaller size difference next (see _size_distance).
        3. Smaller absolute price difference last.

    Scores are reciprocal rank (1 / (position + 1)): a deterministic
    value in (0, 1] that reflects the ordering this function already
    computed, and nothing more. This is NOT a calibrated confidence
    estimate — it exists only so the UI's score field always holds a
    valid, meaningfully-ordered number, without this module pretending
    to know how likely a fallback pick is to satisfy the customer.
    """
    original_brand = original.get("brand")
    well_formed = [c for c in candidates if _candidate_product_id(c) is not None]

    def sort_key(candidate: dict) -> tuple[bool, float, float]:
        same_brand = original_brand is not None and candidate.get("brand") == original_brand
        return (
            not same_brand,  # False (same brand) sorts before True
            _size_distance(candidate, original),
            _price_distance(candidate, original),
        )

    ranked = sorted(well_formed, key=sort_key)

    return [
        {
            "product_id": _candidate_product_id(candidate),
            "score": round(1.0 / (index + 1), 2),
            "reason": _FALLBACK_REASON,
        }
        for index, candidate in enumerate(ranked)
    ]


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def validate_recommendations(
    raw: dict | None,
    candidates: list[dict],
    original: dict,
) -> list[dict]:
    """Convert untrusted AI output into UI-safe recommendations, or
    deterministic fallback output. This is the single call site any
    caller needs — it never raises for a bad `raw` value; "the model
    got it wrong" is an expected, handled case, not an error.

    `raw` is ai_ranking.rank_candidates()'s return value, unmodified —
    None, or a dict that may or may not match the expected shape.
    `candidates` is candidates.get_substitution_candidates()'s return
    value for this same (product_id, store_id) call — the trusted
    universe both the AI response and the fallback are checked/ranked
    against. `original` is the product being substituted, in the same
    {product_id, name, brand, size, unit, price} shape as a candidate,
    used only by the fallback path's brand/size/price comparisons.

    Returns a list of {product_id, score, reason} dicts. The caller
    (a later API/UI layer) never needs to know whether the result came
    from the model or from fallback — both paths return the identical
    shape, and every product_id in the result is guaranteed to be a
    member of `candidates`.
    """
    trusted_ids = _trusted_candidate_ids(candidates)

    accepted = _validate_ai_response(raw, trusted_ids)
    if accepted is not None:
        return accepted

    return _fallback_rank(candidates, original)