"""
tests/test_models.py

backend/models.py is schema-only (see its own docstring: "no retrieval,
no filtering, no ranking, no persistence"). So this file tests exactly
that surface:

  - Product/Inventory/Order/OrderItem/AIRecommendation accept valid data
    and reject invalid data (Pydantic does the heavy lifting; we're
    confirming the schema is wired the way the rest of the codebase
    assumes -- e.g. that `size` is numeric, not that Pydantic works).
  - Default values the rest of the codebase relies on implicitly
    (OrderItem.status defaults to PENDING, Order.status defaults to
    ASSIGNED, Inventory.reserved_quantity defaults to 0 and is a stated,
    unused scope exclusion -- see models.py's own docstring).
  - RESOLVED_ITEM_STATUSES, the one piece of actual "policy" living in
    this otherwise decision-free file: it must contain exactly
    {PICKED, SUBSTITUTED, UNAVAILABLE} and nothing else, since
    api.py's completion gate (FR-7) and transactions.py both trust this
    set completely.

No db.py, candidates.py, or api.py imports here -- this file only
imports backend.models, on purpose, to keep it a true unit test of the
schema layer in isolation.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.models import (
    RESOLVED_ITEM_STATUSES,
    AIRecommendation,
    Inventory,
    Order,
    OrderItem,
    OrderItemStatus,
    OrderStatus,
    Product,
)


# ----------------------------------------------------------------------
# Product
# ----------------------------------------------------------------------


class TestProduct:
    def test_valid_product_round_trips(self):
        p = Product(
            id="SKU-TEST-1",
            name="Test Widget",
            brand="Acme",
            category="widgets",
            size=1.0,
            unit="L",
            price=1.99,
        )
        assert p.id == "SKU-TEST-1"
        # Defaults for optional collection fields.
        assert p.attributes == {}
        assert p.allergens == []

    def test_allergens_and_attributes_round_trip(self):
        p = Product(
            id="SKU-TEST-2",
            name="Milk",
            brand="Alpenfrisch",
            category="dairy",
            size=1.0,
            unit="L",
            price=1.09,
            attributes={"fat_content": "3.5%"},
            allergens=["milk"],
        )
        assert p.allergens == ["milk"]
        assert p.attributes["fat_content"] == "3.5%"

    @pytest.mark.parametrize("missing_field", ["id", "name", "brand", "category", "size", "unit", "price"])
    def test_missing_required_field_rejected(self, missing_field):
        data = dict(
            id="SKU-X",
            name="X",
            brand="X",
            category="x",
            size=1.0,
            unit="L",
            price=1.0,
        )
        del data[missing_field]
        with pytest.raises(ValidationError):
            Product(**data)

    def test_non_numeric_size_rejected(self):
        with pytest.raises(ValidationError):
            Product(
                id="SKU-X",
                name="X",
                brand="X",
                category="x",
                size="one liter",  # not coercible to float
                unit="L",
                price=1.0,
            )


# ----------------------------------------------------------------------
# Inventory
# ----------------------------------------------------------------------


class TestInventory:
    def test_valid_inventory_round_trips(self):
        inv = Inventory(
            product_id="SKU-1",
            store_id="STORE-1",
            quantity=5,
            version=1,
            updated_at=datetime.now(timezone.utc),
        )
        assert inv.quantity == 5
        # Scope-excluded field: present for schema fidelity, defaults
        # to 0, and nothing in the codebase ever sets it otherwise.
        assert inv.reserved_quantity == 0

    def test_reserved_quantity_can_be_set_explicitly(self):
        inv = Inventory(
            product_id="SKU-1",
            store_id="STORE-1",
            quantity=5,
            reserved_quantity=2,
            version=1,
            updated_at=datetime.now(timezone.utc),
        )
        assert inv.reserved_quantity == 2

    def test_missing_version_rejected(self):
        with pytest.raises(ValidationError):
            Inventory(
                product_id="SKU-1",
                store_id="STORE-1",
                quantity=5,
                updated_at=datetime.now(timezone.utc),
            )


# ----------------------------------------------------------------------
# OrderItem / Order
# ----------------------------------------------------------------------


class TestOrderItem:
    def test_defaults(self):
        item = OrderItem(
            id="ITEM-1",
            order_id="ORD-1",
            product_id="SKU-1",
            requested_quantity=1,
        )
        assert item.status == OrderItemStatus.PENDING
        assert item.picked_quantity == 0
        assert item.substituted_product_id is None
        assert item.exception_reason is None

    def test_explicit_status_and_substitution_fields(self):
        item = OrderItem(
            id="ITEM-1",
            order_id="ORD-1",
            product_id="SKU-1",
            requested_quantity=1,
            picked_quantity=1,
            status=OrderItemStatus.SUBSTITUTED,
            substituted_product_id="SKU-2",
        )
        assert item.status == OrderItemStatus.SUBSTITUTED
        assert item.substituted_product_id == "SKU-2"


class TestOrder:
    def _order_kwargs(self, **overrides):
        now = datetime.now(timezone.utc)
        base = dict(
            id="ORD-1",
            customer_id="CUST-1",
            store_id="STORE-1",
            pickup_deadline=now,
            created_at=now,
            updated_at=now,
        )
        base.update(overrides)
        return base

    def test_defaults_status_and_items(self):
        order = Order(**self._order_kwargs())
        assert order.status == OrderStatus.ASSIGNED
        assert order.items == []
        assert order.version == 1

    def test_embedded_items_round_trip(self):
        item = OrderItem(
            id="ITEM-1", order_id="ORD-1", product_id="SKU-1", requested_quantity=2
        )
        order = Order(**self._order_kwargs(items=[item]))
        assert len(order.items) == 1
        assert order.items[0].product_id == "SKU-1"

    def test_invalid_status_string_rejected(self):
        with pytest.raises(ValidationError):
            Order(**self._order_kwargs(status="NOT_A_REAL_STATUS"))


# ----------------------------------------------------------------------
# AIRecommendation
# ----------------------------------------------------------------------


class TestAIRecommendation:
    def test_valid_round_trip(self):
        rec = AIRecommendation(
            id="REC-1",
            order_id="ORD-1",
            item_id="ITEM-1",
            model="gpt-4o-mini",
            model_version="2024-08-01",
            candidate_ids=["SKU-2", "SKU-3"],
            recommended_id="SKU-2",
            score=0.87,
            reason="Same brand, closest size.",
            prompt_version="v1",
            accepted=True,
            created_at=datetime.now(timezone.utc),
        )
        assert rec.accepted is True
        assert rec.candidate_ids == ["SKU-2", "SKU-3"]

    def test_score_must_be_numeric(self):
        with pytest.raises(ValidationError):
            AIRecommendation(
                id="REC-1",
                order_id="ORD-1",
                item_id="ITEM-1",
                model="m",
                model_version="v",
                candidate_ids=[],
                recommended_id="SKU-2",
                score="high",  # not coercible to float
                reason="x",
                prompt_version="v1",
                accepted=True,
                created_at=datetime.now(timezone.utc),
            )


# ----------------------------------------------------------------------
# RESOLVED_ITEM_STATUSES -- the one policy constant this file defines.
# ----------------------------------------------------------------------


class TestResolvedItemStatuses:
    def test_contains_exactly_picked_substituted_unavailable(self):
        assert RESOLVED_ITEM_STATUSES == {
            OrderItemStatus.PICKED,
            OrderItemStatus.SUBSTITUTED,
            OrderItemStatus.UNAVAILABLE,
        }

    def test_pending_is_not_resolved(self):
        assert OrderItemStatus.PENDING not in RESOLVED_ITEM_STATUSES

    def test_skipped_is_not_resolved(self):
        # models.py's own comment is explicit that SKIPPED is
        # deliberately NOT given a meaning in this prompt's scope --
        # if a future change adds SKIPPED to this set, that's a
        # meaningful policy change this test should force a conscious
        # decision about, not something that slips in silently.
        assert OrderItemStatus.SKIPPED not in RESOLVED_ITEM_STATUSES

    def test_is_a_frozenset(self):
        assert isinstance(RESOLVED_ITEM_STATUSES, frozenset)
