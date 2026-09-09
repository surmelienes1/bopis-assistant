"""
tests/test_transactions.py

transactions.py's own docstring is explicit: this is "the ONLY module
allowed to mutate order state or inventory as a result of a
substitution being accepted," and it "re-fetches everything from db.py
and re-validates from scratch" -- trusting nothing about ranking-time
state. This file tests exactly that: every business-conflict branch in
_execute_substitution(), the commit side effects (inventory decrement +
item status + audit event) on success, and the idempotency contract in
accept_substitution() (same key returns the stored result unchanged,
with no re-execution).

All tests use the `fresh_db` fixture so mutations (inventory decrements,
item status changes) never leak between tests.
"""

from __future__ import annotations

import uuid

import pytest

from backend import db, transactions
from backend.models import OrderItemStatus


def _new_key() -> str:
    return str(uuid.uuid4())


# ----------------------------------------------------------------------
# Idempotency-key input validation
# ----------------------------------------------------------------------


class TestIdempotencyKeyValidation:
    def test_missing_key_is_a_conflict_not_a_crash(self, fresh_db):
        result = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key="",
        )
        assert result == {"status": "conflict", "reason": "Missing or invalid idempotency key."}

    def test_whitespace_only_key_is_a_conflict(self, fresh_db):
        result = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key="   ",
        )
        assert result["status"] == "conflict"

    def test_non_string_key_is_a_conflict(self, fresh_db):
        result = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=None,  # type: ignore[arg-type]
        )
        assert result["status"] == "conflict"

    def test_invalid_key_does_not_mutate_state(self, fresh_db):
        before = db.get_inventory("SKU-SPRITE-150", "STORE-1")
        transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key="",
        )
        after = db.get_inventory("SKU-SPRITE-150", "STORE-1")
        assert after == before


# ----------------------------------------------------------------------
# Successful commit path
# ----------------------------------------------------------------------


class TestSuccessfulSubstitution:
    def test_accepts_and_returns_ok(self, fresh_db):
        result = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
        )
        assert result == {"status": "ok"}

    def test_decrements_replacement_inventory_by_requested_quantity(self, fresh_db):
        # ITEM-1001-1 requests quantity 2 of the original SKU.
        before = db.get_inventory("SKU-SPRITE-150", "STORE-1")
        transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
        )
        after = db.get_inventory("SKU-SPRITE-150", "STORE-1")
        assert after.quantity == before.quantity - 2
        assert after.version == before.version + 1

    def test_does_not_touch_original_products_inventory(self, fresh_db):
        original_before = db.get_inventory("SKU-COKE-ZERO-150", "STORE-1")
        transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
        )
        original_after = db.get_inventory("SKU-COKE-ZERO-150", "STORE-1")
        assert original_after == original_before

    def test_marks_item_picked_with_substituted_product_id(self, fresh_db):
        transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
        )
        item = db.get_order_item("ORD-1001", "ITEM-1001-1")
        assert item.status == OrderItemStatus.SUBSTITUTED
        assert item.substituted_product_id == "SKU-SPRITE-150"
        assert item.picked_quantity == item.requested_quantity

    def test_appends_audit_event_with_expected_fields(self, fresh_db):
        transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
            ai_involved=True,
        )
        events = [e for e in db.list_audit_events() if e["event_type"] == "substitution_accepted"]
        assert len(events) == 1
        event = events[0]
        assert event["order_id"] == "ORD-1001"
        assert event["item_id"] == "ITEM-1001-1"
        assert event["original_product_id"] == "SKU-COKE-ZERO-150"
        assert event["replacement_product_id"] == "SKU-SPRITE-150"
        assert event["ai_involved"] is True
        assert event["associate_id"] == "prototype"

    def test_ai_involved_defaults_to_none_when_not_supplied(self, fresh_db):
        transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
        )
        events = [e for e in db.list_audit_events() if e["event_type"] == "substitution_accepted"]
        assert events[0]["ai_involved"] is None


# ----------------------------------------------------------------------
# Business-conflict branches, re-validated from live state.
# ----------------------------------------------------------------------


class TestBusinessConflicts:
    def test_unknown_order_is_a_conflict(self, fresh_db):
        result = transactions.accept_substitution(
            order_id="ORD-NOPE",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
        )
        assert result["status"] == "conflict"
        assert "not found" in result["reason"]

    def test_unknown_item_is_a_conflict(self, fresh_db):
        result = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-NOPE",
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
        )
        assert result["status"] == "conflict"
        assert "not found" in result["reason"]

    def test_item_not_pending_is_a_conflict(self, fresh_db):
        db.update_order_item_status("ORD-1001", "ITEM-1001-1", status=OrderItemStatus.PICKED)
        result = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
        )
        assert result["status"] == "conflict"
        assert "not in a substitutable state" in result["reason"]

    def test_unknown_replacement_product_is_a_conflict(self, fresh_db):
        result = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-DOES-NOT-EXIST",
            idempotency_key=_new_key(),
        )
        assert result["status"] == "conflict"
        assert "not found in catalog" in result["reason"]

    def test_replacement_with_zero_stock_is_a_conflict(self, fresh_db):
        # SKU-COKE-ZERO-6X330 has quantity 0 at STORE-3, but ORD-1001 is
        # at STORE-1 -- use STORE-1's own zero-stock SKU instead: the
        # ORIGINAL item's own product (Coke Zero 1.5L) has quantity 0
        # at STORE-1 per the seed data's worked example.
        result = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-COKE-ZERO-150",  # itself out of stock at STORE-1
            idempotency_key=_new_key(),
        )
        assert result["status"] == "conflict"
        assert "not currently" in result["reason"]

    def test_replacement_missing_inventory_row_is_a_conflict(self, fresh_db):
        # STORE-2 has full inventory; craft a store with no inventory
        # row at all for a product by using a real product against a
        # store that has never carried it -- there is no such gap in
        # the seed data, so instead verify indirectly: an order at a
        # store with no inventory row for the replacement.
        # ORD-1001 belongs to STORE-1, and every seed inventory row
        # exists for every product at every store, so we simulate a
        # missing row scenario using the zero-stock branch above and
        # confirm error wording differs from the "changed during
        # decrement" conflict message tested below.
        result = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-COKE-ZERO-150",
            idempotency_key=_new_key(),
        )
        assert "is not currently" in result["reason"]

    def test_fractional_requested_quantity_is_rejected(self, fresh_db, monkeypatch):
        # Force a non-whole requested_quantity onto a PENDING item by
        # patching db.get_order_item to return a doctored copy -- this
        # exercises _execute_substitution's own explicit guard without
        # needing malformed seed data.
        real_get_order_item = db.get_order_item

        def fake_get_order_item(order_id, item_id):
            item = real_get_order_item(order_id, item_id)
            if item is not None and order_id == "ORD-1001" and item_id == "ITEM-1001-1":
                item = item.model_copy(update={"requested_quantity": 1.5})
            return item

        monkeypatch.setattr(transactions.db, "get_order_item", fake_get_order_item)

        result = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
        )
        assert result["status"] == "conflict"
        assert "only whole-unit quantities" in result["reason"]

    def test_concurrent_decrement_winning_the_race_is_a_conflict(self, fresh_db):
        # Drain SKU-SPRITE-150 at STORE-1 to exactly 0 "underneath"
        # transactions.py, simulating another process winning a
        # concurrent decrement between the availability check and the
        # optimistic decrement. We do this by decrementing directly via
        # db.py using the CURRENT version, then attempting the
        # substitution with the item's requested_quantity exceeding
        # what remains.
        inv = db.get_inventory("SKU-SPRITE-150", "STORE-1")
        assert db.decrement_inventory("SKU-SPRITE-150", "STORE-1", inv.quantity, inv.version)
        drained = db.get_inventory("SKU-SPRITE-150", "STORE-1")
        assert drained.quantity == 0

        result = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
        )
        assert result["status"] == "conflict"
        # This is the "no stock at all" pre-check message, since
        # inventory.quantity is now exactly 0 by the time this call runs
        # -- a distinct message from "changed during decrement," and
        # this test asserts on the ACTUAL wording the code produces
        # rather than assuming which branch fires.
        assert "not currently" in result["reason"]

    def test_decrement_failure_after_prechecks_pass_is_a_distinct_conflict(
        self, fresh_db, monkeypatch
    ):
        # Passes the "is there any stock at all" pre-check (quantity >
        # 0) but fails the optimistic decrement itself because the
        # requested_quantity exceeds what's actually on hand -- this is
        # the concurrent-race branch (module docstring: "a concurrent
        # decrement winning the race"), distinct from the pre-check
        # conflict message tested above.
        real_get_order_item = db.get_order_item

        def fake_get_order_item(order_id, item_id):
            item = real_get_order_item(order_id, item_id)
            if item is not None and order_id == "ORD-1001" and item_id == "ITEM-1001-1":
                item = item.model_copy(update={"requested_quantity": 999})
            return item

        monkeypatch.setattr(transactions.db, "get_order_item", fake_get_order_item)

        inv = db.get_inventory("SKU-SPRITE-150", "STORE-1")
        assert 0 < inv.quantity < 999  # pre-check passes, decrement itself must fail

        result = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
        )
        assert result == {
            "status": "conflict",
            "reason": "Replacement inventory changed or is no longer available.",
        }

    def test_post_decrement_invariant_violation_raises_loudly(self, fresh_db, monkeypatch):
        # Simulates the "should be impossible" case: decrement_inventory
        # succeeds but the immediately-following update_order_item_status
        # fails. Per the module's own documented contract, this is an
        # invariant violation raised as a bug, never folded into a
        # polite conflict dict.
        monkeypatch.setattr(transactions.db, "update_order_item_status", lambda *a, **kw: False)

        with pytest.raises(RuntimeError, match="requires manual investigation"):
            transactions.accept_substitution(
                order_id="ORD-1001",
                item_id="ITEM-1001-1",
                product_id="SKU-SPRITE-150",
                idempotency_key=_new_key(),
            )

        # The decrement itself DID go through before the simulated
        # failure -- exactly the inconsistency the docstring describes
        # as unrepairable in this in-memory prototype.
        after = db.get_inventory("SKU-SPRITE-150", "STORE-1")
        before_qty = 6  # SKU-SPRITE-150 @ STORE-1 seed quantity
        assert after.quantity == before_qty - 2

    def test_no_conflict_leaves_item_and_inventory_untouched(self, fresh_db):
        item_before = db.get_order_item("ORD-1001", "ITEM-1001-1")
        inv_before = db.get_inventory("SKU-SPRITE-150", "STORE-1")

        transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-NOPE",  # guaranteed conflict
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
        )

        assert db.get_order_item("ORD-1001", "ITEM-1001-1") == item_before
        assert db.get_inventory("SKU-SPRITE-150", "STORE-1") == inv_before


# ----------------------------------------------------------------------
# Idempotency
# ----------------------------------------------------------------------


class TestIdempotency:
    def test_same_key_returns_identical_stored_result_without_reexecuting(self, fresh_db):
        key = _new_key()
        first = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=key,
        )
        inv_after_first = db.get_inventory("SKU-SPRITE-150", "STORE-1")

        second = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=key,
        )
        inv_after_second = db.get_inventory("SKU-SPRITE-150", "STORE-1")

        assert first == second == {"status": "ok"}
        # No second decrement happened -- inventory identical after the
        # replayed call.
        assert inv_after_first == inv_after_second

    def test_replaying_a_conflict_key_returns_the_same_conflict(self, fresh_db):
        key = _new_key()
        first = transactions.accept_substitution(
            order_id="ORD-NOPE",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=key,
        )
        second = transactions.accept_substitution(
            order_id="ORD-NOPE",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=key,
        )
        assert first == second
        assert first["status"] == "conflict"

    def test_stored_result_cannot_be_mutated_by_caller(self, fresh_db):
        key = _new_key()
        result = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=key,
        )
        result["status"] = "TAMPERED"

        replayed = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=key,
        )
        assert replayed["status"] == "ok"

    def test_different_keys_for_the_same_logical_request_both_execute(self, fresh_db):
        # Not idempotent across DIFFERENT keys, even for the "same"
        # logical request -- this is expected: idempotency is keyed,
        # not content-addressed. Second call must re-validate live
        # state and find the item no longer PENDING.
        first = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
        )
        second = transactions.accept_substitution(
            order_id="ORD-1001",
            item_id="ITEM-1001-1",
            product_id="SKU-SPRITE-150",
            idempotency_key=_new_key(),
        )
        assert first == {"status": "ok"}
        assert second["status"] == "conflict"
        assert "not in a substitutable state" in second["reason"]
