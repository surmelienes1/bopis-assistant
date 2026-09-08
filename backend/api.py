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
    """Request body for POST /orders/{order_id}/items/{item_id}/substitution."""

    product_id: str
    idempotency_key: str


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


@app.get("/orders/{order_id}", response_model=Order)
def get_order(order_id: str) -> Order:
    order = db.get_order(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id!r} not found.")
    return order


@app.post("/orders/{order_id}/items/{item_id}/unavailable")
def report_item_unavailable(order_id: str, item_id: str) -> list[dict]:
    """An associate reports the requested item is unavailable and asks
    for substitution recommendations. Runs the full retrieval -> rank
    -> validate pipeline and returns the validated recommendation list
    -- see the module docstring for the exact call order and the
    failure-handling boundary around ai_ranking.rank_candidates().
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

    return ranking_validation.validate_recommendations(
        raw, candidate_list, original_dict
    )


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