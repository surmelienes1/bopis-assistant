"""
tests/test_ranking_validation.py

ranking_validation.py's own docstring names the one invariant this
whole module exists to enforce: "NO PRODUCT ID FROM THE MODEL MAY REACH
THE UI UNLESS THAT PRODUCT ID EXISTS IN THE TRUSTED `candidates` LIST."
Every test class below maps to one piece of that contract:

  - TestValidateRecommendationsAcceptsGoodResponses /
    TestValidateRecommendationsRejectsBadResponses: the core
    accept-or-fall-back behavior, including the "reject the WHOLE
    response if ANY entry fails ANY check" rule -- there is no partial
    acceptance, by design.
  - TestIsValidScore: the score-range/type edge cases the module's own
    docstring calls out explicitly (bool-is-not-a-valid-score, NaN/inf,
    numeric strings).
  - TestFallbackRank: the deterministic, LLM-free ranking used whenever
    validation fails, including its brand/size/price sort key and the
    unit-normalization logic that refuses to compare a volume against a
    weight.
  - TestValidateRecommendationsIntegration: the full function against
    the real seed data's worked example (Coca-Cola Zero 1.5L @ STORE-1),
    fed through candidates.py for a realistic `candidates`/`original`
    shape rather than a hand-built one.
"""

from __future__ import annotations

import math

import pytest

from backend import ranking_validation as rv


CANDIDATES = [
    {"product_id": "A", "name": "A", "brand": "BrandX", "size": 1.0, "unit": "L", "price": 2.00},
    {"product_id": "B", "name": "B", "brand": "BrandY", "size": 1.5, "unit": "L", "price": 2.50},
]

ORIGINAL = {"product_id": "O", "name": "O", "brand": "BrandX", "size": 1.0, "unit": "L", "price": 2.00}


# ----------------------------------------------------------------------
# validate_recommendations: accepting good AI output
# ----------------------------------------------------------------------


class TestValidateRecommendationsAcceptsGoodResponses:
    def test_fully_valid_response_is_returned_unchanged_in_model_order(self):
        raw = {
            "recommendations": [
                {"product_id": "B", "score": 0.4, "reason": "Different brand but closest size."},
                {"product_id": "A", "score": 0.9, "reason": "Exact brand and price match."},
            ]
        }
        result = rv.validate_recommendations(raw, CANDIDATES, ORIGINAL)
        assert result == [
            {"product_id": "B", "score": 0.4, "reason": "Different brand but closest size."},
            {"product_id": "A", "score": 0.9, "reason": "Exact brand and price match."},
        ]

    def test_single_valid_recommendation_is_accepted(self):
        raw = {"recommendations": [{"product_id": "A", "score": 1.0, "reason": "Perfect match."}]}
        result = rv.validate_recommendations(raw, CANDIDATES, ORIGINAL)
        assert result == [{"product_id": "A", "score": 1.0, "reason": "Perfect match."}]

    def test_boundary_scores_0_and_1_are_valid(self):
        raw = {
            "recommendations": [
                {"product_id": "A", "score": 0.0, "reason": "low"},
                {"product_id": "B", "score": 1.0, "reason": "high"},
            ]
        }
        result = rv.validate_recommendations(raw, CANDIDATES, ORIGINAL)
        assert [r["product_id"] for r in result] == ["A", "B"]

    def test_integer_score_is_coerced_to_float(self):
        raw = {"recommendations": [{"product_id": "A", "score": 1, "reason": "x"}]}
        result = rv.validate_recommendations(raw, CANDIDATES, ORIGINAL)
        assert result[0]["score"] == 1.0
        assert isinstance(result[0]["score"], float)


# ----------------------------------------------------------------------
# validate_recommendations: rejecting bad AI output -> fallback
# ----------------------------------------------------------------------


class TestValidateRecommendationsRejectsBadResponses:
    def test_none_raw_falls_back(self):
        result = rv.validate_recommendations(None, CANDIDATES, ORIGINAL)
        assert {r["product_id"] for r in result} == {"A", "B"}
        assert all("fallback" in r["reason"] for r in result)

    def test_non_dict_raw_falls_back(self):
        result = rv.validate_recommendations(["not", "a", "dict"], CANDIDATES, ORIGINAL)
        assert {r["product_id"] for r in result} == {"A", "B"}

    def test_missing_recommendations_key_falls_back(self):
        result = rv.validate_recommendations({"oops": []}, CANDIDATES, ORIGINAL)
        assert {r["product_id"] for r in result} == {"A", "B"}

    def test_recommendations_not_a_list_falls_back(self):
        result = rv.validate_recommendations(
            {"recommendations": "A"}, CANDIDATES, ORIGINAL
        )
        assert {r["product_id"] for r in result} == {"A", "B"}

    def test_empty_recommendations_list_falls_back(self):
        # An empty-but-well-typed list is explicitly treated as failure
        # per the module's own docstring -- "a response that recommends
        # nothing doesn't serve the associate any better than a
        # malformed one would."
        result = rv.validate_recommendations({"recommendations": []}, CANDIDATES, ORIGINAL)
        assert {r["product_id"] for r in result} == {"A", "B"}

    def test_hallucinated_product_id_rejects_entire_response(self):
        # THE core invariant this module exists for.
        raw = {
            "recommendations": [
                {"product_id": "A", "score": 0.9, "reason": "good match"},
                {"product_id": "HALLUCINATED-SKU", "score": 0.5, "reason": "made up"},
            ]
        }
        result = rv.validate_recommendations(raw, CANDIDATES, ORIGINAL)
        # The WHOLE response is discarded, including the otherwise-valid
        # "A" entry -- fallback ranking is used instead.
        assert all("fallback" in r["reason"] for r in result)
        assert {r["product_id"] for r in result} == {"A", "B"}

    def test_missing_product_id_field_rejects_entire_response(self):
        raw = {"recommendations": [{"score": 0.9, "reason": "no id given"}]}
        result = rv.validate_recommendations(raw, CANDIDATES, ORIGINAL)
        assert all("fallback" in r["reason"] for r in result)

    def test_non_string_product_id_rejects_entire_response(self):
        raw = {"recommendations": [{"product_id": 123, "score": 0.9, "reason": "x"}]}
        result = rv.validate_recommendations(raw, CANDIDATES, ORIGINAL)
        assert all("fallback" in r["reason"] for r in result)

    def test_empty_string_product_id_rejects_entire_response(self):
        raw = {"recommendations": [{"product_id": "", "score": 0.9, "reason": "x"}]}
        result = rv.validate_recommendations(raw, CANDIDATES, ORIGINAL)
        assert all("fallback" in r["reason"] for r in result)

    def test_score_out_of_range_rejects_entire_response(self):
        raw = {
            "recommendations": [
                {"product_id": "A", "score": 1.5, "reason": "too high"},
            ]
        }
        result = rv.validate_recommendations(raw, CANDIDATES, ORIGINAL)
        assert all("fallback" in r["reason"] for r in result)

    def test_missing_reason_rejects_entire_response(self):
        raw = {"recommendations": [{"product_id": "A", "score": 0.9}]}
        result = rv.validate_recommendations(raw, CANDIDATES, ORIGINAL)
        assert all("fallback" in r["reason"] for r in result)

    def test_blank_reason_rejects_entire_response(self):
        raw = {"recommendations": [{"product_id": "A", "score": 0.9, "reason": "   "}]}
        result = rv.validate_recommendations(raw, CANDIDATES, ORIGINAL)
        assert all("fallback" in r["reason"] for r in result)

    def test_one_bad_entry_discards_otherwise_fully_valid_response(self):
        # No partial acceptance: 9 good entries + 1 bad entry must
        # discard ALL of it, not strip the bad one and keep the rest.
        raw = {
            "recommendations": [
                {"product_id": "A", "score": 0.9, "reason": "good"},
                {"product_id": "B", "score": 0.8, "reason": "good"},
                {"product_id": "B", "score": -0.1, "reason": "bad score, same id repeated"},
            ]
        }
        result = rv.validate_recommendations(raw, CANDIDATES, ORIGINAL)
        assert all("fallback" in r["reason"] for r in result)

    def test_recommendation_entry_not_a_dict_rejects_entire_response(self):
        raw = {"recommendations": ["A"]}
        result = rv.validate_recommendations(raw, CANDIDATES, ORIGINAL)
        assert all("fallback" in r["reason"] for r in result)


# ----------------------------------------------------------------------
# _is_valid_score edge cases
# ----------------------------------------------------------------------


class TestIsValidScore:
    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0, 0, 1])
    def test_valid_scores(self, score):
        assert rv._is_valid_score(score) is True

    @pytest.mark.parametrize("score", [-0.01, 1.01, -1, 2])
    def test_out_of_range_scores(self, score):
        assert rv._is_valid_score(score) is False

    def test_bool_true_is_rejected_despite_being_an_int_subclass(self):
        # isinstance(True, int) is True in Python -- the module
        # explicitly guards against this footgun.
        assert rv._is_valid_score(True) is False

    def test_bool_false_is_rejected(self):
        assert rv._is_valid_score(False) is False

    def test_numeric_string_is_rejected(self):
        assert rv._is_valid_score("0.9") is False

    def test_none_is_rejected(self):
        assert rv._is_valid_score(None) is False

    def test_nan_is_rejected(self):
        assert rv._is_valid_score(float("nan")) is False

    def test_positive_infinity_is_rejected(self):
        assert rv._is_valid_score(float("inf")) is False

    def test_negative_infinity_is_rejected(self):
        assert rv._is_valid_score(float("-inf")) is False

    def test_list_is_rejected(self):
        assert rv._is_valid_score([0.5]) is False


# ----------------------------------------------------------------------
# Deterministic fallback ranking
# ----------------------------------------------------------------------


class TestFallbackRank:
    def test_same_brand_sorts_before_different_brand(self):
        candidates = [
            {"product_id": "DIFF", "brand": "Other", "size": 1.0, "unit": "L", "price": 2.00},
            {"product_id": "SAME", "brand": "BrandX", "size": 1.0, "unit": "L", "price": 2.00},
        ]
        result = rv._fallback_rank(candidates, ORIGINAL)
        assert [r["product_id"] for r in result] == ["SAME", "DIFF"]

    def test_smaller_size_difference_wins_within_same_brand_tier(self):
        candidates = [
            {"product_id": "FAR", "brand": "Other", "size": 5.0, "unit": "L", "price": 2.00},
            {"product_id": "NEAR", "brand": "Other", "size": 1.1, "unit": "L", "price": 2.00},
        ]
        result = rv._fallback_rank(candidates, ORIGINAL)
        assert [r["product_id"] for r in result] == ["NEAR", "FAR"]

    def test_price_breaks_ties_after_brand_and_size(self):
        candidates = [
            {"product_id": "FAR_PRICE", "brand": "Other", "size": 1.0, "unit": "L", "price": 5.00},
            {"product_id": "NEAR_PRICE", "brand": "Other", "size": 1.0, "unit": "L", "price": 2.05},
        ]
        result = rv._fallback_rank(candidates, ORIGINAL)
        assert [r["product_id"] for r in result] == ["NEAR_PRICE", "FAR_PRICE"]

    def test_scores_are_reciprocal_rank(self):
        candidates = [
            {"product_id": "FIRST", "brand": "BrandX", "size": 1.0, "unit": "L", "price": 2.00},
            {"product_id": "SECOND", "brand": "Other", "size": 1.0, "unit": "L", "price": 2.00},
            {"product_id": "THIRD", "brand": "Other", "size": 9.0, "unit": "L", "price": 2.00},
        ]
        result = rv._fallback_rank(candidates, ORIGINAL)
        scores = {r["product_id"]: r["score"] for r in result}
        assert scores["FIRST"] == 1.0
        assert scores["SECOND"] == 0.5
        assert scores["THIRD"] == round(1 / 3, 2)

    def test_all_fallback_reasons_are_the_stated_constant(self):
        candidates = [{"product_id": "A", "brand": "BrandX", "size": 1.0, "unit": "L", "price": 2.00}]
        result = rv._fallback_rank(candidates, ORIGINAL)
        assert result[0]["reason"] == rv._FALLBACK_REASON

    def test_malformed_candidate_records_are_skipped_not_crashed_on(self):
        candidates = [
            {"product_id": "GOOD", "brand": "BrandX", "size": 1.0, "unit": "L", "price": 2.00},
            {"no_product_id": True},
            "not even a dict",
        ]
        result = rv._fallback_rank(candidates, ORIGINAL)
        assert [r["product_id"] for r in result] == ["GOOD"]

    def test_unrecognized_unit_is_incomparable_and_sorts_last(self):
        candidates = [
            {"product_id": "WEIRD_UNIT", "brand": "Other", "size": 1.0, "unit": "dozen", "price": 2.00},
            {"product_id": "NORMAL", "brand": "Other", "size": 3.0, "unit": "L", "price": 2.00},
        ]
        result = rv._fallback_rank(candidates, ORIGINAL)
        # NORMAL (comparable, size diff 2.0L) must outrank WEIRD_UNIT
        # (incomparable -> distance +inf) despite NORMAL's larger raw
        # size difference in absolute terms.
        assert [r["product_id"] for r in result] == ["NORMAL", "WEIRD_UNIT"]

    def test_cross_family_units_are_never_compared(self):
        # original is 1.0 L (volume); a candidate measured in grams
        # (weight) must never be treated as commensurate just because
        # the raw numbers happen to be close.
        candidates = [
            {"product_id": "WEIGHT", "brand": "Other", "size": 1.0, "unit": "g", "price": 2.00},
            {"product_id": "VOLUME", "brand": "Other", "size": 2.0, "unit": "L", "price": 2.00},
        ]
        result = rv._fallback_rank(candidates, ORIGINAL)
        # VOLUME is comparable (1.0L diff); WEIGHT is cross-family
        # incomparable (+inf) and must rank behind it despite its
        # smaller raw number.
        assert [r["product_id"] for r in result] == ["VOLUME", "WEIGHT"]

    def test_ml_and_l_are_correctly_normalized_to_same_base_unit(self):
        # original size=1.0 unit="L" (1000ml). A 900ml candidate should
        # be judged as 100ml away, not treated as incomparable or as
        # "900 units away" from a raw-number mismatch.
        candidates = [
            {"product_id": "CLOSE_ML", "brand": "Other", "size": 900.0, "unit": "ml", "price": 2.00},
            {"product_id": "FAR_L", "brand": "Other", "size": 5.0, "unit": "L", "price": 2.00},
        ]
        result = rv._fallback_rank(candidates, ORIGINAL)
        assert [r["product_id"] for r in result] == ["CLOSE_ML", "FAR_L"]

    def test_kg_and_g_are_correctly_normalized(self):
        weight_original = {"product_id": "O", "brand": "BrandX", "size": 500.0, "unit": "g", "price": 2.00}
        candidates = [
            {"product_id": "CLOSE_KG", "brand": "Other", "size": 0.55, "unit": "kg", "price": 2.00},  # 550g
            {"product_id": "FAR_KG", "brand": "Other", "size": 5.0, "unit": "kg", "price": 2.00},  # 5000g
        ]
        result = rv._fallback_rank(candidates, weight_original)
        assert [r["product_id"] for r in result] == ["CLOSE_KG", "FAR_KG"]

    def test_missing_size_or_unit_is_incomparable(self):
        candidates = [
            {"product_id": "NO_SIZE", "brand": "Other", "unit": "L", "price": 2.00},
            {"product_id": "HAS_BOTH", "brand": "Other", "size": 9.0, "unit": "L", "price": 2.00},
        ]
        result = rv._fallback_rank(candidates, ORIGINAL)
        assert [r["product_id"] for r in result] == ["HAS_BOTH", "NO_SIZE"]

    def test_missing_price_is_incomparable_and_sorts_after_priced_candidates(self):
        candidates = [
            {"product_id": "NO_PRICE", "brand": "Other", "size": 1.0, "unit": "L"},
            {"product_id": "HAS_PRICE", "brand": "Other", "size": 1.0, "unit": "L", "price": 100.00},
        ]
        result = rv._fallback_rank(candidates, ORIGINAL)
        assert [r["product_id"] for r in result] == ["HAS_PRICE", "NO_PRICE"]

    def test_non_string_unit_on_candidate_is_incomparable(self):
        # Covers _normalize_size's `not isinstance(unit, str)` branch
        # directly -- a unit that isn't a string at all (not just an
        # unrecognized one), e.g. a `None` left by a missing key.
        candidates = [
            {"product_id": "BAD_UNIT", "brand": "Other", "size": 1.0, "unit": None, "price": 2.00},
            {"product_id": "GOOD_UNIT", "brand": "Other", "size": 9.0, "unit": "L", "price": 2.00},
        ]
        result = rv._fallback_rank(candidates, ORIGINAL)
        assert [r["product_id"] for r in result] == ["GOOD_UNIT", "BAD_UNIT"]

    def test_invalid_original_price_makes_every_candidate_incomparable_on_price(self):
        # Covers _price_distance's check of `price_original`, not just
        # `price_candidate` -- an original with a non-numeric price
        # must fall back to brand/size ordering only, never crash.
        bad_price_original = {"product_id": "O", "brand": "BrandX", "size": 1.0, "unit": "L", "price": "N/A"}
        candidates = [
            {"product_id": "A", "brand": "BrandX", "size": 1.0, "unit": "L", "price": 2.00},
        ]
        result = rv._fallback_rank(candidates, bad_price_original)
        assert [r["product_id"] for r in result] == ["A"]

    def test_no_original_brand_never_matches_by_accident(self):
        # If `original` itself has no brand field, `same_brand` should
        # never spuriously match a candidate whose brand is also None/
        # missing -- the sort key guards this with
        # `original_brand is not None`.
        original_no_brand = {"product_id": "O", "size": 1.0, "unit": "L", "price": 2.00}
        candidates = [
            {"product_id": "ALSO_NO_BRAND", "size": 1.0, "unit": "L", "price": 2.00},
            {"product_id": "HAS_BRAND", "brand": "X", "size": 9.0, "unit": "L", "price": 2.00},
        ]
        result = rv._fallback_rank(candidates, original_no_brand)
        # Neither is treated as "same brand" as the (brandless)
        # original, so size distance alone decides: ALSO_NO_BRAND (0.0
        # diff) still wins on size, not because of a false brand match.
        assert [r["product_id"] for r in result] == ["ALSO_NO_BRAND", "HAS_BRAND"]


# ----------------------------------------------------------------------
# Integration: validate_recommendations against real seed-data-shaped
# candidates (via candidates.py), not just hand-built fixtures.
# ----------------------------------------------------------------------


class TestValidateRecommendationsIntegration:
    def test_fallback_against_real_coke_zero_candidates(self, fresh_db):
        from backend import candidates as candidates_module
        from backend import db

        candidate_list = candidates_module.get_substitution_candidates(
            "SKU-COKE-ZERO-150", "STORE-1"
        )
        original_product = db.get_product("SKU-COKE-ZERO-150")
        original_dict = {
            "product_id": original_product.id,
            "name": original_product.name,
            "brand": original_product.brand,
            "size": original_product.size,
            "unit": original_product.unit,
            "price": original_product.price,
        }

        result = rv.validate_recommendations(None, candidate_list, original_dict)

        result_ids = {r["product_id"] for r in result}
        candidate_ids = {c["product_id"] for c in candidate_list}
        assert result_ids == candidate_ids
        # Same-brand candidates (Coca-Cola: Zero 1L, Original 1.5L,
        # Sprite 1.5L, Fanta Orange 1.5L) must all outrank the
        # different-brand ones (Pepsi Max, 7Up, Pepsi Regular) -- the
        # seed catalog now has three different-brand sodas in this
        # price band, not just Pepsi Max, so we assert the brand-tier
        # boundary itself rather than pinning one candidate to the
        # very last slot.
        ranked_ids = [r["product_id"] for r in result]
        same_brand_ids = {
            "SKU-COKE-ZERO-100",
            "SKU-COKE-ORIG-150",
            "SKU-SPRITE-150",
            "SKU-FANTA-ORANGE-150",
        }
        different_brand_ids = {"SKU-PEPSI-MAX-150", "SKU-7UP-150", "SKU-PEPSI-REG-150"}
        assert same_brand_ids | different_brand_ids == set(ranked_ids)
        last_same_brand_index = max(ranked_ids.index(pid) for pid in same_brand_ids)
        first_different_brand_index = min(
            ranked_ids.index(pid) for pid in different_brand_ids
        )
        assert last_same_brand_index < first_different_brand_index

    def test_hallucinated_id_against_real_candidates_falls_back(self, fresh_db):
        from backend import candidates as candidates_module
        from backend import db

        candidate_list = candidates_module.get_substitution_candidates(
            "SKU-COKE-ZERO-150", "STORE-1"
        )
        original_product = db.get_product("SKU-COKE-ZERO-150")
        original_dict = {
            "product_id": original_product.id,
            "name": original_product.name,
            "brand": original_product.brand,
            "size": original_product.size,
            "unit": original_product.unit,
            "price": original_product.price,
        }
        # A real product_id, but one that is NOT a member of THIS
        # item's trusted candidate set (it's a dairy product).
        raw = {
            "recommendations": [
                {"product_id": "SKU-MILK-WHOLE-1L", "score": 0.9, "reason": "hallucinated cross-category pick"}
            ]
        }
        result = rv.validate_recommendations(raw, candidate_list, original_dict)
        assert all("fallback" in r["reason"] for r in result)
        assert {r["product_id"] for r in result} == {c["product_id"] for c in candidate_list}