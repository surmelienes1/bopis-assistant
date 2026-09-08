"""
backend/models.py

Shared Pydantic schemas for the BOPIS prototype.

These are the ONLY definitions of shape for Product, Inventory, Order,
OrderItem, and AIRecommendation. Both the data layer (db.py, candidates.py)
and the API layer (api.py) import from here so a single source of truth
exists for what a valid record looks like.

This file does schema validation only — no retrieval, no filtering, no
ranking, no persistence. If you're tempted to add a method here that
*decides* something (e.g. "is this item resolved?"), it belongs in db.py
or candidates.py instead; a model class here should only ever be able to
answer "am I shaped correctly," never "what should happen next."

Thread-safety is not a concern in this file: these are immutable-in-spirit
value objects with no shared module-level state. Shared, mutable state
(and the locking that guards it) starts in db.py.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------


class OrderStatus(str, Enum):
    """Order-level lifecycle, per the design doc's FR-2."""

    CREATED = "CREATED"
    ASSIGNED = "ASSIGNED"
    PICKING = "PICKING"
    PARTIALLY_PICKED = "PARTIALLY_PICKED"
    EXCEPTION = "EXCEPTION"
    READY = "READY"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class OrderItemStatus(str, Enum):
    """Item-level picking status.

    Not enumerated in the design doc's §8 data model (only Order-level
    statuses were spelled out) — this is a resolved ambiguity, not an
    oversight. PICKED and SUBSTITUTED both count as "resolved" for the
    order-completion gate in §FR-7; UNAVAILABLE and SKIPPED do not.
    """

    PENDING = "PENDING"
    PICKED = "PICKED"
    SUBSTITUTED = "SUBSTITUTED"
    UNAVAILABLE = "UNAVAILABLE"
    SKIPPED = "SKIPPED"


RESOLVED_ITEM_STATUSES = frozenset({OrderItemStatus.PICKED, OrderItemStatus.SUBSTITUTED})


# ----------------------------------------------------------------------
# Catalog / inventory
# ----------------------------------------------------------------------


class Product(BaseModel):
    """Catalog entry. Static product metadata only — no store, no stock."""

    id: str
    name: str
    brand: str
    category: str
    size: float
    unit: str
    price: float
    attributes: dict[str, Any] = Field(default_factory=dict)
    allergens: list[str] = Field(default_factory=list)


class Inventory(BaseModel):
    """Per-(product, store) stock row.

    `version` is load-bearing: it's the optimistic-locking column used by
    the concurrency-safe decrement in db.py (§7.2 of the design doc).

    `reserved_quantity` is carried for schema-fidelity with §8 but is
    deliberately NOT implemented in this prototype — nothing reads or
    writes it. A production BOPIS system would reserve stock at
    order-assignment time; this prototype decrements `quantity` directly
    at substitution-accept time instead. That's a stated scope
    exclusion, not a silently missing feature.
    """

    product_id: str
    store_id: str
    quantity: int
    reserved_quantity: int = 0
    version: int
    updated_at: datetime


# ----------------------------------------------------------------------
# Orders
# ----------------------------------------------------------------------


class OrderItem(BaseModel):
    """One line item within an order."""

    id: str
    order_id: str
    product_id: str
    requested_quantity: float
    picked_quantity: float = 0
    status: OrderItemStatus = OrderItemStatus.PENDING
    substituted_product_id: str | None = None
    exception_reason: str | None = None


class Order(BaseModel):
    """An order assigned to a store, with its items embedded.

    The design doc's §8 lists Order and OrderItem as separate entities
    linked by order_id (a relational split). For this prototype, Order
    embeds its `items` list directly — it matches the API response shape
    the associate's UI actually needs (§FR-3's worked example shows items
    nested under the order), and there's no second consumer of OrderItem
    on its own that would justify keeping them split in the in-memory
    store. `candidates.py` and `transactions.py` still address items by
    (order_id, item_id) rather than reaching into this model's list
    directly, so the split could be reintroduced later without touching
    their call signatures.

    `version` is carried for schema-fidelity with §8 but is NOT enforced
    for concurrency — only `Inventory.version` is (§7.2). Two associates
    editing the same order concurrently isn't a scenario the brief asks
    for; this is a known, stated gap, not a claim that it's handled.
    """

    id: str
    customer_id: str
    store_id: str
    status: OrderStatus = OrderStatus.ASSIGNED
    pickup_deadline: datetime
    created_at: datetime
    updated_at: datetime
    version: int = 1
    items: list[OrderItem] = Field(default_factory=list)


# ----------------------------------------------------------------------
# AI recommendation audit trail
# ----------------------------------------------------------------------


class AIRecommendation(BaseModel):
    """One ranked candidate, as shown to an associate, for the audit trail.

    This answers "why did the system suggest this" — see §38 of the
    original brainstorm and §8 of the final design doc. One row per
    candidate that survived validation and was actually displayed
    (`ai_ranking.py` + `ranking_validation.py`, built in a later prompt,
    are what populate these).

    `accepted` is set once, at creation time, and is NOT mutated later:
    it reflects whether this specific candidate passed validation and was
    shown as a trusted recommendation — not whether the associate
    subsequently clicked Accept on it. The associate's actual decision
    (which product they picked, if any) lives in the audit event written
    by `transactions.py`, cross-referenced by this row's `id`. Keeping
    these separate means the recommendation log always answers "what was
    shown and why," and the transaction audit log always answers "what
    actually happened" — the two are related by ID, never merged into one
    mutable record.
    """

    id: str
    order_id: str
    item_id: str
    model: str
    model_version: str
    candidate_ids: list[str]
    recommended_id: str
    score: float
    reason: str
    prompt_version: str
    accepted: bool
    created_at: datetime