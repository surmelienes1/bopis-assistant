"""
backend/api.py

The FastAPI API layer.

Responsibility, and ONLY responsibility: expose the existing domain
modules over HTTP. This file contains ZERO candidate-filtering,
ranking, validation, inventory-mutation, or transaction logic of its
own — it calls candidates.py, ai_ranking.py, ranking_validation.py,
transactions.py, and db.py, and translates their results into HTTP
responses. Every decision about WHICH product is a good substitute,
whether a substitution is currently valid, or whether an order can be
completed is made by those modules, not here.

Pipeline this file wires up for POST .../unavailable (design doc §6.1,
§9), in the exact order the modules already assume:

    verify order/item exist and are in a substitutable state
        -> candidates.get_substitution_candidates()   (trusted universe)
        -> ai_ranking.rank_candidates()                (untrusted output)
        -> ranking_validation.validate_recommendations() (enforcement)
        -> return validated recommendations

The one exception to "let exceptions propagate" this file makes is
around the ai_ranking.rank_candidates() call specifically: an
unexpected failure there (anything ai_ranking.py itself doesn't
already convert to None — see that module's docstring) is caught and
turned into `raw = None` so that ranking_validation.py's deterministic
fallback path still runs, instead of the AI layer's failure becoming
an HTTP 500 for what should be a resilient, self-healing pipeline
(design doc §7). Candidate retrieval, database access, and validation
exceptions are NOT caught here — those signal real bugs or data
corruption (see candidates.py's own docstring on this) and should
surface loudly.

POST .../resolve-unavailable is a deliberately SEPARATE, short-circuit
path for the case that pipeline can legitimately produce: zero
candidates survived candidates.py's hard filters, so there is nothing
for ai_ranking.py or ranking_validation.py to do. It skips straight to
"associate confirms removal -> item transitions to UNAVAILABLE", never
touching the AI layer or inventory. See that route's own docstring for
why this is a distinct endpoint rather than a special case bolted onto
POST .../unavailable.

Session storage, auth, and persistence across restarts are all out of
scope, matching db.py's own stated exclusions — this is a local,
single-user prototype.

Prompt 10 (design doc §6.3, §6.4) adds two small, deterministic,
LLM-free capabilities alongside the substitution pipeline above:

    GET  /inventory/nearby                         -- §6.3
    POST /inventory/{store_id}/{product_id}/shelf-report -- §6.4

Neither touches ai_ranking.py, candidates.py, ranking_validation.py,
or transactions.py, and neither calls llm_client.py. See each route's
own docstring for why: nearby-store ranking is a deterministic scoring
function over seed metadata + live inventory, and shelf reporting is a
plain audit-log write. Both are search/logging problems, not
generative ones — see this file's own docstring above and the design
doc's §6.1 for the pattern this project uses to decide where the model
belongs at all.

This round adds a small set of read-only catalog/directory endpoints
so the frontend can be a real associate app instead of a single
hardcoded order id: GET /products, GET /orders, GET /stores, and
GET /audit-events. Every one of these is a plain, unfiltered (or
trivially filtered) pass-through to a db.py read function that already
existed or was added alongside it — no new business logic, no new
mutation path, nothing that touches order/inventory state. They exist
because two of this file's tools (nearby-stock lookup, shelf
reporting) are explicitly NOT scoped to a specific order's item list —
an associate can look up any product's nearby availability or report a
shelf issue on any product, independent of what they're currently
picking — so the frontend needs a catalog and a store directory to
build those pickers from, the same way a real "Product Catalog Svc"
and "Inventory Svc" would supply them (design doc §5).

report_item_unavailable() below also now enriches each validated
recommendation with display fields (name/brand/price) pulled from the
SAME trusted candidate_list already used for ranking/validation — see
that route's docstring for exactly why this is safe and doesn't touch
ranking_validation.py's own {product_id, score, reason} contract.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend import ai_ranking, candidates, db, ranking_validation, transactions
from backend.models import (
    Order,
    OrderItemStatus,
    OrderStatus,
    Product,
    RESOLVED_ITEM_STATUSES,
)

load_dotenv()

logger = logging.getLogger(__name__)

_DATA_DIR = os.environ.get(
    "BOPIS_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data"),
)

_FRONTEND_INDEX = os.path.join(
    os.path.dirname(__file__), "..", "frontend", "index.html"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # db.py owns reading the seed JSON files -- api.py never touches
    # data/*.json directly.
    db.init_db(_DATA_DIR)
    yield


app = FastAPI(title="BOPIS Store Associate Assistant", lifespan=lifespan)

# Permissive CORS is intentional: this is a local, unauthenticated
# single-user prototype with a separate frontend and no deployment
# target, same rationale as the Watchtower reference.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# HTTP-only request shapes. These describe wire format, nothing more --
# transactions.accept_substitution() re-validates every value against
# live order/inventory state regardless of what this model accepts.
# ----------------------------------------------------------------------


class SubstitutionRequest(BaseModel):
    """Request body for POST /orders/{order_id}/items/{item_id}/substitution.

    `ai_involved` is optional (default None, "unknown") and is passed
    straight through to transactions.accept_substitution() for the
    audit event only -- see that function's docstring. It records
    whether this acceptance came from the .../unavailable
    recommendation flow, not whether the accepted product actually
    differs from the one originally ordered; transactions.py decides
    PICKED vs SUBSTITUTED itself, independently, by comparing
    product_id against the order line's own product_id.
    """

    product_id: str
    idempotency_key: str
    ai_involved: bool | None = None


class ResolveUnavailableRequest(BaseModel):
    """Request body for POST .../resolve-unavailable.

    `resolution` is a plain str, not a stricter Literal/enum: an
    unsupported value is rejected below as an ordinary 400 business
    error via _SUPPORTED_RESOLUTIONS, the same way SubstitutionRequest's
    `product_id` is validated downstream rather than at the schema
    layer — consistent error semantics (400, not FastAPI's generic 422
    for a failed Pydantic literal) beats a marginally stricter type for
    a single-value enum that has exactly one supported case today.
    """

    resolution: str


# The only resolution this prototype supports (see module-level design
# note on the new endpoint below). A set, not a single constant, so
# adding a second resolution later is a one-line change here rather
# than a restructure of the validation branch that checks it.
_SUPPORTED_RESOLUTIONS = {"REMOVE"}


class ShelfReportRequest(BaseModel):
    """Request body for POST /inventory/{store_id}/{product_id}/shelf-report.

    `status` is a plain str, validated below against
    _SUPPORTED_SHELF_STATUSES the same way SubstitutionRequest's
    resolution field is validated -- an unsupported value is an
    ordinary 400 business error, not a stricter Literal/enum at the
    schema layer, for the same consistency reason ResolveUnavailableRequest
    gives above.
    """

    status: str


# The four shelf-observation statuses design doc §6.4 names. A set, in
# the same spirit as _SUPPORTED_RESOLUTIONS above -- adding a fifth
# status later is a one-line change here, not a restructure of the
# validation branch that checks it.
_SUPPORTED_SHELF_STATUSES = {"empty", "low", "damaged", "misplaced"}

# Availability normalization for GET /inventory/nearby's scoring
# formula (design doc §6.3) -- see that route's docstring for the full
# formula. A raw available_quantity is divided by this constant and
# capped at 1.0, so a nearby store's availability score saturates at
# "fully available" once it's carrying at least this many units,
# rather than letting an arbitrarily large quantity dominate the
# score. Chosen to sit comfortably above every requested_quantity in
# this prototype's seed orders (all 1-2 units) and near the upper end
# of this catalog's typical per-store stock levels (see
# data/inventory.json), so it meaningfully distinguishes "well
# stocked" from "thin" without needing real sales-velocity data this
# prototype doesn't have.
_AVAILABILITY_REFERENCE_QTY = 20.0


# ----------------------------------------------------------------------
# Small local helper -- shaping only, no decisions.
# ----------------------------------------------------------------------


def _product_to_candidate_dict(product: Product) -> dict:
    """Convert a catalog Product into the same {product_id, name, brand,
    size, unit, price} shape candidates.get_substitution_candidates()
    already produces for every candidate, so the original (out-of-stock)
    product can be compared apples-to-apples against candidates by both
    ai_ranking.py's prompt and ranking_validation.py's fallback ranking
    -- both of which expect `original` in exactly this shape (see
    ranking_validation.validate_recommendations()'s docstring).

    Duplicated here rather than imported from candidates.py because
    this shaping is an API-layer concern (assembling the payload for a
    downstream call), not a candidate-retrieval one -- candidates.py's
    own dict construction stays private to that module's filter chain.
    """
    return {
        "product_id": product.id,
        "name": product.name,
        "brand": product.brand,
        "size": product.size,
        "unit": product.unit,
        "price": product.price,
    }


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def read_root() -> FileResponse:
    return FileResponse(_FRONTEND_INDEX)


@app.get("/orders", response_model=list[Order])
def list_orders(store_id: str | None = None, status: OrderStatus | None = None) -> list[Order]:
    """Assigned orders, optionally filtered by store_id/status (design
    doc §9's `GET /orders?store_id&status`). Powers the associate's
    order list/picker screen -- the frontend no longer hardcodes a
    single order id. A thin pass-through to db.list_orders(); no
    filtering logic lives here.
    """
    return db.list_orders(store_id=store_id, status=status)


@app.get("/orders/{order_id}", response_model=Order)
def get_order(order_id: str) -> Order:
    order = db.get_order(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id!r} not found.")
    return order


@app.get("/products", response_model=list[Product])
def list_products() -> list[Product]:
    """The full product catalog. Read-only, unfiltered -- this
    prototype's whole seed catalog is 18 items, small enough for the
    frontend to fetch once and use as a local lookup (product_id ->
    name/brand/price) wherever a bare product_id shows up, and as the
    picker source for the nearby-stock and shelf-report tools, neither
    of which is scoped to a specific order's item list.
    """
    return db.list_products()


@app.get("/stores")
def list_stores() -> list[dict[str, str]]:
    """The known store directory (id + display name only -- never
    distances or inventory_confidence, which are GET /inventory/
    nearby's scoring internals, not directory data). Powers the store
    picker the frontend's order-independent tools use to answer
    "which store am I at" -- a real deployment would get this from
    the associate's authenticated session (see design doc's FR-1 /
    Assumptions; auth is explicitly out of scope for this prototype).
    """
    return db.list_stores()


@app.get("/audit-events")
def list_audit_events(event_type: str | None = None) -> list[dict]:
    """Read-only snapshot of the in-memory audit log (design doc §10),
    optionally filtered by event_type. Powers a small "recent
    activity" feed in the frontend (e.g. shelf reports just
    submitted this session) -- nothing returned here is authoritative
    order/inventory state, it's the same append-only log
    transactions.py and this file's other routes already write to via
    db.append_audit_event().
    """
    events = db.list_audit_events()
    if event_type is not None:
        events = [e for e in events if e.get("event_type") == event_type]
    return events


@app.post("/orders/{order_id}/items/{item_id}/unavailable")
def report_item_unavailable(order_id: str, item_id: str) -> list[dict]:
    """An associate reports the requested item is unavailable and asks
    for substitution recommendations. Runs the full retrieval -> rank
    -> validate pipeline and returns the validated recommendation list
    -- see the module docstring for the exact call order and the
    failure-handling boundary around ai_ranking.rank_candidates().

    Each returned dict is ranking_validation.py's own {product_id,
    score, reason} PLUS display fields (name, brand, price) merged in
    below, after validation, from the SAME trusted `candidate_list`
    already used for ranking. This is safe precisely because it's
    read-only decoration, not a new trust boundary: every product_id
    in the validated result is already guaranteed (by
    ranking_validation.py's own invariant) to be a member of
    candidate_list, so the merge is a plain dict lookup that can never
    miss and never introduces a product the pipeline didn't already
    approve. ranking_validation.py's own return contract is untouched
    -- this enrichment happens here, at the HTTP edge, purely so the
    frontend can show "Sprite 1.5L -- Coca-Cola -- $2.39" instead of a
    bare SKU.
    """
    order = db.get_order(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id!r} not found.")

    item = db.get_order_item(order_id, item_id)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail=f"Order item {item_id!r} not found on order {order_id!r}.",
        )

    if item.status != OrderItemStatus.PENDING:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Order item {item_id!r} is not in a substitutable state "
                f"(current status: {item.status.value})."
            ),
        )

    # candidates.py re-derives the original product internally and
    # raises ValueError if catalog/order data have drifted apart -- a
    # data-corruption signal, not an expected client-recoverable
    # failure (see that module's _get_original() docstring), so it is
    # deliberately left uncaught here and surfaces as a 500.
    candidate_list = candidates.get_substitution_candidates(
        item.product_id, order.store_id
    )

    # get_substitution_candidates() having succeeded means item.product_id
    # is a real catalog entry, so this lookup cannot return None.
    original_product = db.get_product(item.product_id)
    original_dict = _product_to_candidate_dict(original_product)

    # Last-resort safety net around the AI call only -- see module
    # docstring. ai_ranking.rank_candidates() already converts its own
    # known failure modes (LLMUnavailableError, unparseable JSON) to
    # None; this catches anything it doesn't (e.g. a genuine
    # LLMClientError from missing/bad configuration) so a broken AI
    # layer degrades to deterministic fallback instead of a 500.
    try:
        raw = ai_ranking.rank_candidates(original_dict, candidate_list)
    except Exception:
        logger.exception(
            "ai_ranking.rank_candidates failed unexpectedly for "
            "order_id=%r, item_id=%r -- falling back to deterministic ranking.",
            order_id,
            item_id,
        )
        raw = None

    validated = ranking_validation.validate_recommendations(
        raw, candidate_list, original_dict
    )

    candidates_by_id = {c["product_id"]: c for c in candidate_list}
    for rec in validated:
        display = candidates_by_id.get(rec["product_id"], {})
        rec["name"] = display.get("name")
        rec["brand"] = display.get("brand")
        rec["price"] = display.get("price")

    return validated


@app.post("/orders/{order_id}/items/{item_id}/resolve-unavailable")
def resolve_unavailable_item(
    order_id: str, item_id: str, body: ResolveUnavailableRequest
) -> dict:
    """An associate explicitly gives up on a line after
    .../unavailable produced zero valid substitutes, removing it from
    the picking flow so the order can still complete.

    This is deliberately NOT a continuation of the candidate/ranking
    pipeline above: it never calls candidates.py, ai_ranking.py, or
    ranking_validation.py, and it never touches inventory. A zero-
    candidate response from .../unavailable is already a valid,
    terminal outcome of that pipeline (see candidates.py's own
    docstring — a candidate either survives every hard filter or it's
    gone); this endpoint is where the associate ACTS on that outcome
    via one explicit, human-initiated request, not a retry of ranking
    and not something the system infers on the associate's behalf.

    `resolution` is checked against _SUPPORTED_RESOLUTIONS before
    either db.py lookup runs, and REMOVE is the only value this
    prototype defines: it transitions the item straight to
    UNAVAILABLE via the same db.update_order_item_status() write path
    every other item-status change already goes through -- no new
    mutation primitive, no direct dict access.
    """
    if body.resolution not in _SUPPORTED_RESOLUTIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported resolution {body.resolution!r}; supported "
                f"values: {sorted(_SUPPORTED_RESOLUTIONS)}."
            ),
        )

    order = db.get_order(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id!r} not found.")

    item = db.get_order_item(order_id, item_id)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail=f"Order item {item_id!r} not found on order {order_id!r}.",
        )

    if item.status != OrderItemStatus.PENDING:
        # Same convention as report_item_unavailable()'s identical check
        # above: an item not currently PENDING is a state conflict
        # (already resolved, or reported unavailable twice), not a
        # malformed request -- 409, not 400, matching the existing
        # substitution endpoint's use of 409 for "no longer valid to
        # act on." This is what stops a second, redundant REMOVE from
        # silently re-mutating an already-resolved item.
        raise HTTPException(
            status_code=409,
            detail=(
                f"Order item {item_id!r} is not in a resolvable state "
                f"(current status: {item.status.value})."
            ),
        )

    updated = db.update_order_item_status(
        order_id,
        item_id,
        status=OrderItemStatus.UNAVAILABLE,
        exception_reason=(
            "No valid substitution candidates were available; "
            "associate removed the item."
        ),
    )
    if not updated:
        # We just confirmed this exact order/item exists moments ago --
        # same "should be impossible" situation transactions.py raises
        # loudly on after its own post-decrement update. Not folded
        # into a polite conflict response; see that module's docstring
        # for why an invariant violation here is a bug to investigate,
        # not a business outcome to paper over.
        raise RuntimeError(
            "update_order_item_status failed unexpectedly for "
            f"order_id={order_id!r}, item_id={item_id!r} immediately "
            "after a successful get_order_item -- this should be impossible."
        )

    db.append_audit_event(
        {
            "event_type": "item_marked_unavailable",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order_id": order_id,
            "item_id": item_id,
            # Same explicit placeholder as transactions.py's audit
            # events -- no authenticated associate identity exists
            # anywhere in this prototype yet.
            "associate_id": "prototype",
            "resolution": body.resolution,
            "original_product_id": item.product_id,
            "reason": "No acceptable substitution was available for this item.",
        }
    )

    return {
        "status": "ok",
        "item_id": item_id,
        "item_status": OrderItemStatus.UNAVAILABLE.value,
    }


@app.post("/orders/{order_id}/items/{item_id}/substitution")
def accept_substitution(
    order_id: str, item_id: str, body: SubstitutionRequest
) -> JSONResponse:
    """Accept a substitution. All real validation (order/item state,
    replacement availability, idempotency, optimistic locking) happens
    inside transactions.accept_substitution() -- this endpoint only
    maps its result onto an HTTP status code and preserves its JSON
    body unchanged.
    """
    result = transactions.accept_substitution(
        order_id=order_id,
        item_id=item_id,
        product_id=body.product_id,
        idempotency_key=body.idempotency_key,
        ai_involved=body.ai_involved,
    )
    status_code = 200 if result.get("status") == "ok" else 409
    return JSONResponse(status_code=status_code, content=result)


@app.post("/orders/{order_id}/complete")
def complete_order(order_id: str) -> JSONResponse:
    """Transition an order to READY once every item is resolved.

    "Resolved" is defined once, in models.py's RESOLVED_ITEM_STATUSES
    (PICKED and SUBSTITUTED) -- this endpoint reads that definition
    rather than re-stating the state-machine rule itself.
    """
    order = db.get_order(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id!r} not found.")

    open_items = [
        {"item_id": item.id, "status": item.status.value}
        for item in order.items
        if item.status not in RESOLVED_ITEM_STATUSES
    ]
    if open_items:
        return JSONResponse(
            status_code=400,
            content={
                "status": "conflict",
                "reason": "Order cannot be completed while items remain unresolved.",
                "open_items": open_items,
            },
        )

    updated = db.update_order_status(order_id, OrderStatus.READY)
    if not updated:
        # We just fetched this exact order above. A False here means it
        # vanished between the two calls -- not a scenario this
        # in-memory, no-delete prototype can actually trigger, so this
        # is treated as the same kind of invariant violation
        # transactions.py raises loudly on, not folded into a polite
        # conflict response.
        raise RuntimeError(
            f"update_order_status failed unexpectedly for order_id={order_id!r} "
            "immediately after a successful get_order -- this should be impossible."
        )

    return JSONResponse(
        status_code=200,
        content={"status": "ok", "order_status": OrderStatus.READY.value},
    )


# ----------------------------------------------------------------------
# Prompt 10 -- deterministic, LLM-free inventory capabilities (design
# doc §6.3 nearby-store availability, §6.4 shelf-depletion reporting).
#
# Both inventory endpoints are independent of the substitution
# pipeline. They read trusted inventory/store metadata or append an
# audit event; neither calls the LLM.
# ----------------------------------------------------------------------


@app.get("/inventory/current")
def get_current_inventory(store_id: str, product_id: str | None = None) -> dict:
    """Return authoritative current-store inventory for the associate UI.

    This is a deterministic, LLM-free lookup over db.py's trusted
    inventory state. It exists so the frontend can answer the most
    basic operational question first: "How much stock do we currently
    have at the store I am working in?"

    If product_id is supplied, return the quantity for that product.
    If product_id is omitted, return the current inventory quantities
    for every catalog product carried by the selected store.

    A missing inventory row is treated as zero available stock. This
    deliberately does not create an inventory row or mutate state:
    absence of a row means the store does not currently have a
    stock-bearing inventory record for that product.

    This endpoint exposes raw available quantity only. It does not
    expose inventory versioning or confidence metadata because those
    are backend consistency/scoring concerns rather than associate UI
    concerns.
    """
    if not db.store_exists(store_id):
        raise HTTPException(
            status_code=404, detail=f"Store {store_id!r} not found."
        )

    if product_id is not None:
        if db.get_product(product_id) is None:
            raise HTTPException(
                status_code=404, detail=f"Product {product_id!r} not found."
            )

        inventory = db.get_inventory(product_id, store_id)

        return {
            "store_id": store_id,
            "product_id": product_id,
            "available_quantity": inventory.quantity if inventory is not None else 0,
        }

    inventory_by_product = {}
    for product in db.list_products():
        inventory = db.get_inventory(product.id, store_id)
        inventory_by_product[product.id] = (
            inventory.quantity if inventory is not None else 0
        )

    return {
        "store_id": store_id,
        "inventory": inventory_by_product,
    }


@app.get("/inventory/nearby")
def get_nearby_inventory(product_id: str, store_id: str) -> dict:
    """Answer "this product is unavailable at the current store -- is it
    available at nearby stores?" (design doc §6.3).

    This is intentionally a deterministic, LLM-free lookup. It never
    calls llm_client.py, ai_ranking.py, candidates.py, or
    ranking_validation.py -- nearby-store discovery is a search/ranking
    problem over known, structured data (seed store distances + live
    inventory), not a generative one. The model is useful for
    substitution REASONING (why THIS product is a good stand-in for
    THAT one -- see ai_ranking.py), never for a factual lookup like
    "how much stock does store X have," which this prototype already
    knows exactly, with no ambiguity for a model to resolve.

    Scoring formula, applied per nearby store and documented here for
    reproducibility:

        score = availability * (1 / distance_km) * inventory_confidence

    where:
      - availability = min(available_quantity / _AVAILABILITY_REFERENCE_QTY, 1.0)
        A deterministic, capped linear normalization of the nearby
        store's raw on-hand quantity for this product. See
        _AVAILABILITY_REFERENCE_QTY's own comment for why that
        constant was chosen.
      - 1 / distance_km rewards closer stores. `distance_km` comes
        from the deterministic seed metadata in data/stores.json
        (via db.get_store_distances) -- never computed, geocoded, or
        routed.
      - inventory_confidence is a fixed, deterministic per-store trust
        value, also from data/stores.json (via
        db.get_store_inventory_confidence) -- a stand-in for a real
        signal a production system might have (e.g. RFID-tracked vs.
        manual-count stockrooms), not a computed one.

    A nearby store is excluded entirely (not scored at 0) if it has no
    inventory row for `product_id`, or an available_quantity of zero
    or less -- "nearby but has none" isn't useful information for an
    associate deciding where to send a customer. The requesting
    (current) store itself is never included: db.get_store_distances()
    only ever returns OTHER known stores by construction.

    Returns nearby stores sorted by descending score. Raises 404 if
    `product_id` isn't in the catalog or `store_id` isn't a known
    store.
    """
    product = db.get_product(product_id)
    if product is None:
        raise HTTPException(
            status_code=404, detail=f"Product {product_id!r} not found."
        )

    if not db.store_exists(store_id):
        raise HTTPException(status_code=404, detail=f"Store {store_id!r} not found.")

    # By construction (see db.get_store_distances()'s own docstring),
    # this never includes store_id itself.
    distances = db.get_store_distances(store_id) or {}

    nearby_stores: list[dict] = []
    for other_store_id, distance_km in distances.items():
        inv = db.get_inventory(product_id, other_store_id)
        if inv is None or inv.quantity <= 0:
            continue

        confidence = db.get_store_inventory_confidence(other_store_id)
        if confidence is None:
            # Defensive only: a store_id appearing in another store's
            # distances_km map but missing its own top-level entry in
            # stores.json would be a seed-data bug, not a runtime
            # condition to paper over with a fabricated confidence
            # value -- skip it rather than guess.
            continue

        availability = min(inv.quantity / _AVAILABILITY_REFERENCE_QTY, 1.0)
        score = availability * (1.0 / distance_km) * confidence

        nearby_stores.append(
            {
                "store_id": other_store_id,
                "distance_km": distance_km,
                "available_quantity": inv.quantity,
                "inventory_confidence": confidence,
                "score": round(score, 3),
            }
        )

    nearby_stores.sort(key=lambda s: s["score"], reverse=True)

    return {
        "product_id": product_id,
        "source_store_id": store_id,
        "nearby_stores": nearby_stores,
    }


@app.post("/inventory/{store_id}/{product_id}/shelf-report")
def report_shelf_status(
    store_id: str, product_id: str, body: ShelfReportRequest
) -> dict:
    """An associate logs a physical shelf-stock observation (design doc
    §6.4) -- "empty," "low," "damaged," or "misplaced."

    This is a logging endpoint only. It appends one audit event via
    the existing db.append_audit_event() mechanism and returns a
    confirmation; it never touches Inventory.quantity. A shelf report
    is an associate's unverified, in-the-moment observation, not a
    reconciled stock count -- letting it silently adjust the
    authoritative quantity that decrement_inventory()/the substitution
    pipeline depend on would conflate two different kinds of data
    (what accounting believes is in stock vs. what an associate just
    glanced at) with two very different trust levels.

    No LLM call. §6.4's actual AI opportunity is a small forecasting/
    anomaly model trained over historical shelf reports plus inventory
    movement -- e.g. predicting stockout probability, or surfacing a
    pattern like "this SKU repeatedly depletes ~17:00 Fridays." That
    model is deliberately out of scope for this prototype, which only
    needs the event captured for a future training set, not analyzed
    now.

    `status` is checked against _SUPPORTED_SHELF_STATUSES before
    either db.py lookup runs (same ordering convention as
    resolve_unavailable_item()'s resolution check above) -- a
    malformed request shouldn't cost a lookup before it's rejected.
    Raises 404 if `product_id` isn't in the catalog or `store_id`
    isn't a known store.
    """
    if body.status not in _SUPPORTED_SHELF_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported shelf status {body.status!r}; supported "
                f"values: {sorted(_SUPPORTED_SHELF_STATUSES)}."
            ),
        )

    if db.get_product(product_id) is None:
        raise HTTPException(
            status_code=404, detail=f"Product {product_id!r} not found."
        )

    if not db.store_exists(store_id):
        raise HTTPException(status_code=404, detail=f"Store {store_id!r} not found.")

    db.append_audit_event(
        {
            "event_type": "shelf_report",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "store_id": store_id,
            "product_id": product_id,
            "status": body.status,
            # Same explicit placeholder as every other audit event in
            # this prototype -- no authenticated associate identity
            # exists anywhere yet.
            "associate_id": "prototype",
        }
    )

    return {
        "status": "ok",
        "store_id": store_id,
        "product_id": product_id,
        "shelf_status": body.status,
    }