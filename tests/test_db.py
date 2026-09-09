"""
tests/test_db.py

backend/db.py is the single source of truth for "what is the state
right now" and "apply this already-decided state change" (see its own
module docstring). This file tests, against the real seed data via the
`fresh_db` fixture (see conftest.py):

  - init_db() loading behavior, including that it fully REPLACES state
    rather than merging (so calling it twice, as tests do implicitly
    via the fixture, is safe).
  - _ensure_initialized()'s guard against use-before-init.
  - Every read function returns a COPY, never the stored object -- the
    module docstring is explicit that this is what stops a caller from
    corrupting in-memory state by mutating a returned object in place.
  - decrement_inventory()'s optimistic-locking contract: succeeds only
    when quantity is sufficient AND version matches; returns False
    (never raises) on a stale version or an unknown row.
  - update_order_item_status()'s status-transition side effects
    (picked_quantity set to requested_quantity on PICKED/SUBSTITUTED,
    substituted_product_id / exception_reason only applied when given).
  - update_order_status(), append_audit_event()/list_audit_events(),
    store_exists()/get_store_distances()/get_store_inventory_confidence(),
    and the list_* / get_* read paths, including filtering behavior.

Every test in this class-based layout uses the `fresh_db` fixture so
each test starts from a pristine copy of the real seed JSON and can
never leak a mutation into another test.
"""

from __future__ import annotations

import pytest

from backend.models import Inventory, Order, OrderItemStatus, OrderStatus, Product


# ----------------------------------------------------------------------
# Initialization / guard behavior
# ----------------------------------------------------------------------


class TestInitAndGuard:
    def test_uninitialized_module_raises(self, monkeypatch):
        import backend.db as db_module

        # Simulate a fresh, never-initialized module without touching
        # any other test's state: flip the private flag directly, then
        # restore it. Using monkeypatch guarantees restoration even if
        # the assertion fails.
        monkeypatch.setattr(db_module, "_initialized", False)
        with pytest.raises(RuntimeError, match="init_db"):
            db_module.get_product("anything")

    def test_init_db_loads_all_four_seed_files(self, fresh_db):
        assert len(fresh_db.list_products()) == 59
        assert len(fresh_db.list_orders()) == 13
        assert len(fresh_db.list_stores()) == 5
        assert fresh_db.list_audit_events() == []

    def test_init_db_replaces_rather_than_merges(self, fresh_db, seed_data_dir):
        # Mutate state, then re-init from the SAME seed dir. If init_db
        # merged instead of replacing, the mutation would survive.
        inv = fresh_db.get_inventory("SKU-COKE-ZERO-100", "STORE-1")
        assert fresh_db.decrement_inventory(
            "SKU-COKE-ZERO-100", "STORE-1", 1, inv.version
        )
        decremented = fresh_db.get_inventory("SKU-COKE-ZERO-100", "STORE-1")
        assert decremented.quantity == inv.quantity - 1

        fresh_db.init_db(seed_data_dir)

        reset = fresh_db.get_inventory("SKU-COKE-ZERO-100", "STORE-1")
        assert reset.quantity == inv.quantity
        assert reset.version == inv.version

    def test_init_db_clears_audit_log(self, fresh_db, seed_data_dir):
        fresh_db.append_audit_event({"event_type": "test_event"})
        assert len(fresh_db.list_audit_events()) == 1
        fresh_db.init_db(seed_data_dir)
        assert fresh_db.list_audit_events() == []


# ----------------------------------------------------------------------
# Read functions return copies, not live references
# ----------------------------------------------------------------------


class TestReadsReturnCopies:
    def test_get_product_returns_a_copy(self, fresh_db):
        p1 = fresh_db.get_product("SKU-COKE-ZERO-150")
        p1.name = "TAMPERED"
        p2 = fresh_db.get_product("SKU-COKE-ZERO-150")
        assert p2.name == "Coca-Cola Zero 1.5L"

    def test_get_order_returns_a_deep_copy(self, fresh_db):
        order = fresh_db.get_order("ORD-1001")
        order.items[0].status = OrderItemStatus.PICKED  # mutate nested list item
        order.items.append(
            order.items[0].model_copy(update={"id": "FORGED-ITEM"})
        )  # mutate the list itself

        fresh_order = fresh_db.get_order("ORD-1001")
        assert fresh_order.items[0].status == OrderItemStatus.PENDING
        assert all(item.id != "FORGED-ITEM" for item in fresh_order.items)

    def test_get_inventory_returns_a_copy(self, fresh_db):
        inv = fresh_db.get_inventory("SKU-BANANA-1KG", "STORE-1")
        inv.quantity = 999
        fresh_inv = fresh_db.get_inventory("SKU-BANANA-1KG", "STORE-1")
        assert fresh_inv.quantity != 999

    def test_list_products_returns_copies(self, fresh_db):
        products = fresh_db.list_products()
        products[0].name = "TAMPERED"
        assert fresh_db.get_product(products[0].id).name != "TAMPERED"

    def test_get_store_distances_returns_a_copy(self, fresh_db):
        distances = fresh_db.get_store_distances("STORE-1")
        distances["STORE-2"] = -1.0
        fresh_distances = fresh_db.get_store_distances("STORE-1")
        assert fresh_distances["STORE-2"] != -1.0


# ----------------------------------------------------------------------
# get_product / get_order / get_order_item / get_inventory: not-found
# ----------------------------------------------------------------------


class TestGetters:
    def test_get_product_unknown_returns_none(self, fresh_db):
        assert fresh_db.get_product("SKU-DOES-NOT-EXIST") is None

    def test_get_order_unknown_returns_none(self, fresh_db):
        assert fresh_db.get_order("ORD-DOES-NOT-EXIST") is None

    def test_get_order_item_unknown_order_returns_none(self, fresh_db):
        assert fresh_db.get_order_item("ORD-DOES-NOT-EXIST", "ITEM-1001-1") is None

    def test_get_order_item_unknown_item_on_real_order_returns_none(self, fresh_db):
        assert fresh_db.get_order_item("ORD-1001", "ITEM-DOES-NOT-EXIST") is None

    def test_get_order_item_found(self, fresh_db):
        item = fresh_db.get_order_item("ORD-1001", "ITEM-1001-1")
        assert item is not None
        assert item.product_id == "SKU-COKE-ZERO-150"

    def test_get_inventory_unknown_pair_returns_none(self, fresh_db):
        assert fresh_db.get_inventory("SKU-COKE-ZERO-150", "STORE-DOES-NOT-EXIST") is None

    def test_get_inventory_known_pair(self, fresh_db):
        inv = fresh_db.get_inventory("SKU-COKE-ZERO-150", "STORE-1")
        assert inv.quantity == 0  # the seeded "out of stock" worked example


# ----------------------------------------------------------------------
# list_products_by_category
# ----------------------------------------------------------------------


class TestListProductsByCategory:
    def test_returns_only_matching_category(self, fresh_db):
        sodas = fresh_db.list_products_by_category("soda")
        assert {p.id for p in sodas} == {
            "SKU-COKE-ZERO-150",
            "SKU-COKE-ZERO-100",
            "SKU-COKE-ZERO-6X330",
            "SKU-COKE-ZERO-2L",
            "SKU-COKE-ORIG-150",
            "SKU-PEPSI-MAX-150",
            "SKU-PEPSI-REG-150",
            "SKU-SPRITE-150",
            "SKU-FANTA-ORANGE-150",
            "SKU-7UP-150",
        }

    def test_excludes_given_id(self, fresh_db):
        sodas = fresh_db.list_products_by_category(
            "soda", exclude_product_id="SKU-COKE-ZERO-150"
        )
        assert "SKU-COKE-ZERO-150" not in {p.id for p in sodas}

    def test_unknown_category_returns_empty(self, fresh_db):
        assert fresh_db.list_products_by_category("nonexistent-category") == []


# ----------------------------------------------------------------------
# list_orders filtering
# ----------------------------------------------------------------------


class TestListOrders:
    def test_no_filter_returns_all(self, fresh_db):
        assert len(fresh_db.list_orders()) == 13

    def test_filter_by_store_id(self, fresh_db):
        orders = fresh_db.list_orders(store_id="STORE-1")
        assert {o.id for o in orders} == {
            "ORD-1001",
            "ORD-1002",
            "ORD-1005",
            "ORD-1008",
            "ORD-1011",
        }

    def test_filter_by_store_id_no_match(self, fresh_db):
        assert fresh_db.list_orders(store_id="STORE-9") == []

    def test_filter_by_status(self, fresh_db):
        orders = fresh_db.list_orders(status=OrderStatus.ASSIGNED)
        assert len(orders) == 6

    def test_filter_by_status_no_match(self, fresh_db):
        # The seed data now has at least one order in every status it
        # exercises (ASSIGNED, PICKING, READY, COMPLETED, CANCELLED,
        # CREATED, PARTIALLY_PICKED, EXCEPTION), so a pure status filter
        # can no longer be guaranteed empty. Combine a real, in-use
        # status with a store_id that doesn't carry it (the only READY
        # order, ORD-1008, belongs to STORE-1) to keep this a genuine
        # "filter matches nothing" case without depending on unused
        # enum members.
        assert (
            fresh_db.list_orders(store_id="STORE-2", status=OrderStatus.READY) == []
        )

    def test_filter_by_store_and_status_combined(self, fresh_db):
        orders = fresh_db.list_orders(store_id="STORE-1", status=OrderStatus.ASSIGNED)
        assert len(orders) == 2


# ----------------------------------------------------------------------
# decrement_inventory: the optimistic-locking write path
# ----------------------------------------------------------------------


class TestDecrementInventory:
    def test_successful_decrement_updates_quantity_and_bumps_version(self, fresh_db):
        before = fresh_db.get_inventory("SKU-MILK-WHOLE-1L", "STORE-1")
        ok = fresh_db.decrement_inventory(
            "SKU-MILK-WHOLE-1L", "STORE-1", 3, before.version
        )
        assert ok is True

        after = fresh_db.get_inventory("SKU-MILK-WHOLE-1L", "STORE-1")
        assert after.quantity == before.quantity - 3
        assert after.version == before.version + 1

    def test_decrement_to_exactly_zero_succeeds(self, fresh_db):
        before = fresh_db.get_inventory("SKU-COKE-ZERO-6X330", "STORE-3")
        assert before.quantity == 0
        # Requesting 0 units against 0 stock: qty(0) <= quantity(0) holds.
        ok = fresh_db.decrement_inventory(
            "SKU-COKE-ZERO-6X330", "STORE-3", 0, before.version
        )
        assert ok is True

    def test_insufficient_stock_fails_without_raising(self, fresh_db):
        before = fresh_db.get_inventory("SKU-BANANA-1KG", "STORE-3")  # quantity 12
        ok = fresh_db.decrement_inventory(
            "SKU-BANANA-1KG", "STORE-3", before.quantity + 1, before.version
        )
        assert ok is False
        # State must be unchanged after a failed decrement.
        unchanged = fresh_db.get_inventory("SKU-BANANA-1KG", "STORE-3")
        assert unchanged.quantity == before.quantity
        assert unchanged.version == before.version

    def test_stale_version_fails_without_raising(self, fresh_db):
        before = fresh_db.get_inventory("SKU-APPLE-1KG", "STORE-1")
        stale_version = before.version - 1 if before.version > 0 else before.version + 1
        ok = fresh_db.decrement_inventory(
            "SKU-APPLE-1KG", "STORE-1", 1, stale_version
        )
        assert ok is False
        unchanged = fresh_db.get_inventory("SKU-APPLE-1KG", "STORE-1")
        assert unchanged.quantity == before.quantity

    def test_unknown_product_store_pair_returns_false(self, fresh_db):
        ok = fresh_db.decrement_inventory("SKU-DOES-NOT-EXIST", "STORE-1", 1, 1)
        assert ok is False

    def test_second_decrement_must_use_new_version(self, fresh_db):
        before = fresh_db.get_inventory("SKU-TOMATO-1KG", "STORE-1")
        assert fresh_db.decrement_inventory(
            "SKU-TOMATO-1KG", "STORE-1", 1, before.version
        )
        # Reusing the now-stale `before.version` a second time must fail
        # -- this is exactly the "someone else won the race" scenario
        # transactions.py depends on to detect a concurrent winner.
        ok_again = fresh_db.decrement_inventory(
            "SKU-TOMATO-1KG", "STORE-1", 1, before.version
        )
        assert ok_again is False


# ----------------------------------------------------------------------
# update_order_item_status
# ----------------------------------------------------------------------


class TestUpdateOrderItemStatus:
    def test_unknown_order_returns_false(self, fresh_db):
        ok = fresh_db.update_order_item_status(
            "ORD-DOES-NOT-EXIST", "ITEM-1001-1", status=OrderItemStatus.PICKED
        )
        assert ok is False

    def test_unknown_item_on_real_order_returns_false(self, fresh_db):
        ok = fresh_db.update_order_item_status(
            "ORD-1001", "ITEM-DOES-NOT-EXIST", status=OrderItemStatus.PICKED
        )
        assert ok is False

    def test_picked_sets_picked_quantity_to_requested_quantity(self, fresh_db):
        item_before = fresh_db.get_order_item("ORD-1001", "ITEM-1001-2")  # qty 1
        ok = fresh_db.update_order_item_status(
            "ORD-1001", "ITEM-1001-2", status=OrderItemStatus.PICKED
        )
        assert ok is True
        item_after = fresh_db.get_order_item("ORD-1001", "ITEM-1001-2")
        assert item_after.status == OrderItemStatus.PICKED
        assert item_after.picked_quantity == item_before.requested_quantity

    def test_substituted_sets_picked_quantity_and_substituted_product_id(self, fresh_db):
        ok = fresh_db.update_order_item_status(
            "ORD-1001",
            "ITEM-1001-1",
            status=OrderItemStatus.SUBSTITUTED,
            substituted_product_id="SKU-SPRITE-150",
        )
        assert ok is True
        item = fresh_db.get_order_item("ORD-1001", "ITEM-1001-1")
        assert item.status == OrderItemStatus.SUBSTITUTED
        assert item.substituted_product_id == "SKU-SPRITE-150"
        assert item.picked_quantity == item.requested_quantity

    def test_unavailable_does_not_set_picked_quantity(self, fresh_db):
        ok = fresh_db.update_order_item_status(
            "ORD-1001",
            "ITEM-1001-3",
            status=OrderItemStatus.UNAVAILABLE,
            exception_reason="No valid substitutes.",
        )
        assert ok is True
        item = fresh_db.get_order_item("ORD-1001", "ITEM-1001-3")
        assert item.status == OrderItemStatus.UNAVAILABLE
        assert item.picked_quantity == 0
        assert item.exception_reason == "No valid substitutes."

    def test_omitted_substituted_product_id_does_not_clear_existing_value(self, fresh_db):
        # First set it...
        fresh_db.update_order_item_status(
            "ORD-1001",
            "ITEM-1001-1",
            status=OrderItemStatus.SUBSTITUTED,
            substituted_product_id="SKU-SPRITE-150",
        )
        # ...then call again without passing substituted_product_id.
        # update_order_item_status only writes fields explicitly passed
        # (non-None), so the previously-set value must survive.
        fresh_db.update_order_item_status(
            "ORD-1001", "ITEM-1001-1", status=OrderItemStatus.SUBSTITUTED
        )
        item = fresh_db.get_order_item("ORD-1001", "ITEM-1001-1")
        assert item.substituted_product_id == "SKU-SPRITE-150"

    def test_updates_order_updated_at(self, fresh_db):
        order_before = fresh_db.get_order("ORD-1001")
        fresh_db.update_order_item_status(
            "ORD-1001", "ITEM-1001-1", status=OrderItemStatus.PICKED
        )
        order_after = fresh_db.get_order("ORD-1001")
        assert order_after.updated_at >= order_before.updated_at


# ----------------------------------------------------------------------
# update_order_status
# ----------------------------------------------------------------------


class TestUpdateOrderStatus:
    def test_unknown_order_returns_false(self, fresh_db):
        assert fresh_db.update_order_status("ORD-DOES-NOT-EXIST", OrderStatus.READY) is False

    def test_updates_status_and_timestamp(self, fresh_db):
        before = fresh_db.get_order("ORD-1001")
        ok = fresh_db.update_order_status("ORD-1001", OrderStatus.READY)
        assert ok is True
        after = fresh_db.get_order("ORD-1001")
        assert after.status == OrderStatus.READY
        assert after.updated_at >= before.updated_at

    def test_does_not_perform_transition_validation(self, fresh_db):
        # By design (see db.py's own docstring), this function applies
        # ANY status its caller passes -- it is not this layer's job to
        # decide whether the transition makes business sense. Jumping
        # straight to COMPLETED with pending items is nonsensical
        # business-wise but must still succeed at the db layer.
        ok = fresh_db.update_order_status("ORD-1001", OrderStatus.COMPLETED)
        assert ok is True
        assert fresh_db.get_order("ORD-1001").status == OrderStatus.COMPLETED


# ----------------------------------------------------------------------
# Audit log
# ----------------------------------------------------------------------


class TestAuditLog:
    def test_starts_empty(self, fresh_db):
        assert fresh_db.list_audit_events() == []

    def test_append_and_list(self, fresh_db):
        fresh_db.append_audit_event({"event_type": "test_event", "foo": "bar"})
        events = fresh_db.list_audit_events()
        assert len(events) == 1
        assert events[0]["event_type"] == "test_event"
        assert events[0]["foo"] == "bar"

    def test_list_returns_a_copy_of_the_list(self, fresh_db):
        fresh_db.append_audit_event({"event_type": "e1"})
        events = fresh_db.list_audit_events()
        events.append({"event_type": "forged"})
        assert len(fresh_db.list_audit_events()) == 1

    def test_preserves_insertion_order(self, fresh_db):
        fresh_db.append_audit_event({"event_type": "first"})
        fresh_db.append_audit_event({"event_type": "second"})
        events = fresh_db.list_audit_events()
        assert [e["event_type"] for e in events] == ["first", "second"]


# ----------------------------------------------------------------------
# Store metadata: store_exists / get_store_distances /
# get_store_inventory_confidence / list_stores
# ----------------------------------------------------------------------


class TestStoreMetadata:
    def test_store_exists_true_for_known_store(self, fresh_db):
        assert fresh_db.store_exists("STORE-1") is True

    def test_store_exists_false_for_unknown_store(self, fresh_db):
        assert fresh_db.store_exists("STORE-999") is False

    def test_get_store_distances_excludes_self(self, fresh_db):
        distances = fresh_db.get_store_distances("STORE-1")
        assert "STORE-1" not in distances
        assert set(distances) == {"STORE-2", "STORE-3", "STORE-4", "STORE-5"}

    def test_get_store_distances_unknown_store_returns_none(self, fresh_db):
        assert fresh_db.get_store_distances("STORE-999") is None

    def test_get_store_distances_values_match_seed(self, fresh_db, seed_json):
        stores = {s["store_id"]: s for s in seed_json("stores")}
        distances = fresh_db.get_store_distances("STORE-2")
        assert distances == stores["STORE-2"]["distances_km"]

    def test_get_store_inventory_confidence_known(self, fresh_db):
        assert fresh_db.get_store_inventory_confidence("STORE-1") == 0.95

    def test_get_store_inventory_confidence_unknown_returns_none(self, fresh_db):
        assert fresh_db.get_store_inventory_confidence("STORE-999") is None

    def test_list_stores_shape_excludes_scoring_internals(self, fresh_db):
        stores = fresh_db.list_stores()
        assert len(stores) == 5
        for store in stores:
            assert set(store) == {"store_id", "name"}

    def test_list_stores_contains_expected_ids(self, fresh_db):
        ids = {s["store_id"] for s in fresh_db.list_stores()}
        assert ids == {"STORE-1", "STORE-2", "STORE-3", "STORE-4", "STORE-5"}