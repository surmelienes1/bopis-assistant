"""
backend/db.py

In-memory data-access layer for the BOPIS prototype.

Loads the four seed JSON files (products, inventory, orders, stores)
into module-level dicts at startup and exposes retrieval plus a small,
closed set of mutation functions. Ranking, filtering, and substitution
policy do NOT belong here — see candidates.py and ai_ranking.py (later
prompts). This file answers "what is the state right now" and "apply
this one already-decided state change" — nothing decides anything here.

stores.json (Prompt 10) is a small, separate store-metadata seed file
— NOT a redesign of the inventory schema. Inventory rows still carry
exactly the fields they always have (product_id, store_id, quantity,
reserved_quantity, version, updated_at); nothing about "where is this
store" or "how far apart are two stores" belongs on a per-product-per-
store row, and duplicating a distance value across every inventory row
for the same store pair would be redundant, error-prone seed data. A
handful of stores, each carrying its distances to every other store
plus one deterministic inventory-trust value, is the smallest
structure that represents that data once. See get_store_distances()
and get_store_inventory_confidence() below for the read API, and
api.py's GET /inventory/nearby for the only caller.

No persistence across restarts — deliberately, same scope exclusion as
the Watchtower project's db.py. A restart resets to the seed JSON.

Concurrency note: each function below is individually lock-protected,
which is enough to make any single read or single write atomic. It does
NOT make a multi-step operation atomic across two calls — e.g.
decrementing inventory and then marking an item substituted are two
separate lock acquisitions, not one transaction. transactions.py (a
later prompt) is responsible for sequencing those calls and treating a
failure partway through as something to report and reconcile, not for
making the underlying db.py calls themselves span a single lock. A real
database transaction would close this gap; a Python-level lock across
two calls only would not, without holding the lock for the whole
sequence and risking deadlocks as this file grows — not worth it for an
in-memory single-process demo.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.models import (
    Inventory,
    Order,
    OrderItem,
    OrderItemStatus,
    OrderStatus,
    Product,
)

# ----------------------------------------------------------------------
# Module-level state
# ----------------------------------------------------------------------

_PRODUCTS: dict[str, Product] = {}
_INVENTORY: dict[tuple[str, str], Inventory] = {}
_ORDERS: dict[str, Order] = {}
_AUDIT_LOG: list[dict[str, Any]] = []

# store_id -> {"name": str, "inventory_confidence": float,
#              "distances_km": {other_store_id: float, ...}}
# Deterministic seed metadata only (Prompt 10, design doc §6.3) — see
# stores.json and the module docstring above. Plain dicts, not a
# Pydantic model: this data has no mutation path and no business logic
# of its own (unlike Product/Inventory/Order), just lookups, so a
# model class would add ceremony without adding safety.
_STORES: dict[str, dict[str, Any]] = {}

# All of the above is shared, mutable state read and written from
# synchronous FastAPI route handlers, which run in a worker thread pool
# rather than strictly sequentially (same situation as Watchtower's
# db.py). A single process-wide lock is coarser than necessary but far
# easier to reason about under time pressure than per-row locking, and
# the dataset here is small enough that lock contention is a non-issue
# for a single-user demo. A production system would want per-row
# locking or would delegate concurrency control to a real database
# entirely, as decrement_inventory's optimistic-version check already
# anticipates.
_DB_LOCK = threading.Lock()

_initialized = False


# ----------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------


def init_db(data_dir: str | Path) -> None:
    """Load products.json, inventory.json, orders.json, and stores.json
    into memory.

    Called once at process startup (see api.py's lifespan). Safe to call
    again — e.g. in tests, to reset state — since it fully replaces the
    module dicts rather than merging into them.
    """
    global _initialized
    data_dir = Path(data_dir)

    with _DB_LOCK:
        products = json.loads((data_dir / "products.json").read_text())
        inventory = json.loads((data_dir / "inventory.json").read_text())
        orders = json.loads((data_dir / "orders.json").read_text())
        stores = json.loads((data_dir / "stores.json").read_text())

        _PRODUCTS.clear()
        _PRODUCTS.update({p["id"]: Product(**p) for p in products})

        _INVENTORY.clear()
        for row in inventory:
            inv = Inventory(**row)
            _INVENTORY[(inv.product_id, inv.store_id)] = inv

        _ORDERS.clear()
        _ORDERS.update({o["id"]: Order(**o) for o in orders})

        _STORES.clear()
        _STORES.update({s["store_id"]: s for s in stores})

        _AUDIT_LOG.clear()
        _initialized = True


def _ensure_initialized() -> None:
    if not _initialized:
        raise RuntimeError("db.init_db() must be called before use.")


# ----------------------------------------------------------------------
# Read functions — always return copies, never the stored objects
# themselves, so a caller can't mutate our in-memory state by editing a
# returned object's fields or lists in place. All mutation goes through
# the write functions below, on purpose.
# ----------------------------------------------------------------------


def get_product(product_id: str) -> Product | None:
    _ensure_initialized()
    with _DB_LOCK:
        product = _PRODUCTS.get(product_id)
        return product.model_copy() if product is not None else None


def get_order(order_id: str) -> Order | None:
    _ensure_initialized()
    with _DB_LOCK:
        order = _ORDERS.get(order_id)
        return order.model_copy(deep=True) if order is not None else None


def get_order_item(order_id: str, item_id: str) -> OrderItem | None:
    _ensure_initialized()
    with _DB_LOCK:
        order = _ORDERS.get(order_id)
        if order is None:
            return None
        for item in order.items:
            if item.id == item_id:
                return item.model_copy()
        return None


def get_inventory(product_id: str, store_id: str) -> Inventory | None:
    _ensure_initialized()
    with _DB_LOCK:
        inv = _INVENTORY.get((product_id, store_id))
        return inv.model_copy() if inv is not None else None


def store_exists(store_id: str) -> bool:
    """True if `store_id` is a known store in the store-metadata seed
    data (stores.json). The only "does this store exist" check this
    prototype has — there is no Store table/model, only this metadata
    dict and the store_id strings already used inside Inventory/Order.
    Used for 404 handling on store-scoped endpoints (Prompt 10's
    GET /inventory/nearby and POST .../shelf-report).
    """
    _ensure_initialized()
    with _DB_LOCK:
        return store_id in _STORES


def get_store_distances(store_id: str) -> dict[str, float] | None:
    """Return `store_id`'s deterministic seed distances (km) to every
    OTHER known store, or None if `store_id` itself isn't known.

    Never includes `store_id` itself — a store's distance to itself is
    undefined here, not zero, since nothing downstream should ever be
    able to treat "the current store" as one of its own nearby-store
    candidates. Returns a copy so a caller can't mutate our seed state
    by editing the dict it got back.
    """
    _ensure_initialized()
    with _DB_LOCK:
        store = _STORES.get(store_id)
        return dict(store["distances_km"]) if store is not None else None


def get_store_inventory_confidence(store_id: str) -> float | None:
    """Return the deterministic `inventory_confidence` seed value for
    `store_id`, or None if `store_id` isn't known.

    A fixed, seed-data stand-in for a real per-store signal a
    production system might have (e.g. RFID-tracked vs. manual-count
    stockrooms) — never computed, inferred, or updated at runtime.
    """
    _ensure_initialized()
    with _DB_LOCK:
        store = _STORES.get(store_id)
        return store["inventory_confidence"] if store is not None else None


def list_products_by_category(
    category: str, exclude_product_id: str | None = None
) -> list[Product]:
    """All catalog products in `category`, optionally excluding one id.

    Returns Product records only — no inventory join here. Callers that
    need "in stock at this store" (candidates.py) do that filtering
    themselves via get_inventory per candidate, so this stays reusable
    outside the substitution flow (e.g. for a plain category browse).
    """
    _ensure_initialized()
    with _DB_LOCK:
        return [
            p.model_copy()
            for p in _PRODUCTS.values()
            if p.category == category and p.id != exclude_product_id
        ]


# ----------------------------------------------------------------------
# Write functions — the ONLY ways this module's state may change.
# Every one returns a bool/None rather than raising on an expected
# failure (stale version, missing id) so callers can distinguish
# "the world changed under you" from "this is a bug."
# ----------------------------------------------------------------------


def decrement_inventory(
    product_id: str, store_id: str, qty: int, expected_version: int
) -> bool:
    """Optimistically decrement stock by `qty`, guarded by `expected_version`.

    Mirrors the SQL pattern from the design doc's §7.2:

        UPDATE inventory SET quantity = quantity - ?, version = version + 1
         WHERE product_id = ? AND store_id = ? AND quantity >= ? AND version = ?

    Returns False (never raises) if the row doesn't exist, doesn't have
    enough stock, or `expected_version` is stale — the caller
    (transactions.py) turns a False here into a "someone else took it,
    refresh and re-decide" exception state rather than assuming a bug.

    `qty` is typed as int, matching Inventory.quantity. Substitutable
    products in this prototype (soda, packaged grocery) are always
    requested in whole units; produce items sold by weight never go
    through the substitution/accept path in this data set, so the
    int/float mismatch with OrderItem.requested_quantity (a float, to
    accommodate weight-based items elsewhere) never actually surfaces
    here. Worth saying out loud if asked, not worth solving tonight.
    """
    _ensure_initialized()
    with _DB_LOCK:
        inv = _INVENTORY.get((product_id, store_id))
        if inv is None:
            return False
        if inv.quantity < qty or inv.version != expected_version:
            return False
        _INVENTORY[(product_id, store_id)] = inv.model_copy(
            update={
                "quantity": inv.quantity - qty,
                "version": inv.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        return True


def update_order_item_status(
    order_id: str,
    item_id: str,
    status: OrderItemStatus,
    substituted_product_id: str | None = None,
    exception_reason: str | None = None,
) -> bool:
    """Update one item's status in place within its parent order.

    Returns False if the order or item doesn't exist, rather than
    raising — consistent with decrement_inventory's "expected failure,
    not a bug" signaling. When status moves to PICKED or SUBSTITUTED,
    picked_quantity is set to the item's requested_quantity: this
    prototype doesn't model partial picks within a single item line.
    """
    _ensure_initialized()
    with _DB_LOCK:
        order = _ORDERS.get(order_id)
        if order is None:
            return False
        for idx, item in enumerate(order.items):
            if item.id != item_id:
                continue
            updates: dict[str, Any] = {"status": status}
            if substituted_product_id is not None:
                updates["substituted_product_id"] = substituted_product_id
            if exception_reason is not None:
                updates["exception_reason"] = exception_reason
            if status in (OrderItemStatus.PICKED, OrderItemStatus.SUBSTITUTED):
                updates["picked_quantity"] = item.requested_quantity
            order.items[idx] = item.model_copy(update=updates)
            order.updated_at = datetime.now(timezone.utc)
            return True
        return False


def update_order_status(order_id: str, status: OrderStatus) -> bool:
    """Update one order's top-level status.

    Added for api.py's POST /orders/{order_id}/complete endpoint: none
    of the write functions above touch Order.status, and the API layer
    must not reach into _ORDERS directly to change it (see this
    module's own docstring — write functions here are the ONLY way
    state may change). Returns False, rather than raising, if the
    order doesn't exist — consistent with update_order_item_status's
    "expected failure, not a bug" signaling.

    This function does not decide WHETHER a transition is valid (e.g.
    "every item must be PICKED or SUBSTITUTED before moving to READY")
    — that business rule belongs to the caller, which evaluates it
    against the existing OrderItem/RESOLVED_ITEM_STATUSES definitions
    in models.py before ever calling this function. This function only
    applies a status value its caller has already decided is correct.
    """
    _ensure_initialized()
    with _DB_LOCK:
        order = _ORDERS.get(order_id)
        if order is None:
            return False
        _ORDERS[order_id] = order.model_copy(
            update={"status": status, "updated_at": datetime.now(timezone.utc)}
        )
        return True


def append_audit_event(event: dict[str, Any]) -> None:
    """Append one audit event to the in-memory log.

    No persistence across restarts, deliberately — same exclusion as
    the rest of this module. A production version would write this to
    an append-only store, not a Python list that vanishes on exit.
    """
    _ensure_initialized()
    with _DB_LOCK:
        _AUDIT_LOG.append(dict(event))


def list_audit_events() -> list[dict[str, Any]]:
    """Read-only snapshot of the audit log — for the smoke test / demo UI."""
    _ensure_initialized()
    with _DB_LOCK:
        return list(_AUDIT_LOG)