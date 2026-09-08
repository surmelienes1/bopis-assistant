"""
backend/transactions.py

The final, authoritative commit boundary for accepting a substitution.

This is the ONLY module allowed to mutate order state or inventory as
a result of a substitution being accepted. candidates.py produces a
trusted candidate universe; ai_ranking.py and ranking_validation.py
turn that into a recommendation the associate sees. None of that is
trusted here — this module re-fetches everything from db.py and
re-validates from scratch, exactly as if the caller had supplied a
bare product_id with no ranking pedigree at all, because that's
functionally what it is: an associate's decision, not an AI's.

Transaction boundary (design doc §7.1), stated explicitly since this
in-memory prototype can't actually provide it:

    BEGIN
      validate current state (order, item, replacement product,
                               current inventory)
      decrement inventory with optimistic version check
      update order item
      append audit event
    COMMIT

    -- if any step fails, production ROLLS BACK every prior step in
    -- this same transaction.

    The block below (see _execute_substitution) performs exactly these
    steps in this order, but NOT inside a real database transaction --
    it's a sequence of independently-locked db.py calls (see db.py's
    own docstring on this same gap). If decrement_inventory succeeds
    and update_order_item_status then somehow fails, this prototype
    CANNOT roll the decrement back; that scenario raises loudly (see
    _execute_substitution) instead of pretending to be atomic. A real
    database transaction wrapping all three statements in one COMMIT
    is what would actually close this gap -- not a Python lock held
    across the calls, which db.py's own docstring already explains is
    not equivalent to crash atomicity.

Idempotency (design doc §7.3): every call carries an idempotency_key.
A single module-level lock (_IDEMPOTENCY_LOCK) guards both the
"has this key already been processed" check AND the entire execution
that follows for a new key -- the lock is held continuously from the
check until the result is stored, so there is no gap in which a
second concurrent call with the SAME key could see "not yet processed"
and also start executing. This is a genuinely new piece of state that
db.py has no concept of (an idempotency-key -> result map), so it gets
its own lock rather than being shoehorned into db.py's _DB_LOCK or
module dicts -- but every actual data mutation still goes exclusively
through db.py's own functions, which keep their own independent
locking. Two different lock objects, no shared state between them, so
there's no deadlock risk in holding ours across calls that acquire
theirs internally.

Trade-off, stated plainly: holding one global lock across the whole
execution serializes ALL accept_substitution calls process-wide, not
just calls that happen to share an idempotency key -- coarser than a
production system would want (a per-key lock, or more realistically,
delegating this to a real database's unique constraint on the
idempotency key plus a row-level transaction). For a small, single-
process, in-memory prototype this trade-off is deliberate and mirrors
the exact reasoning db.py's own module docstring already gives for its
single process-wide _DB_LOCK: correctness and simplicity over
minimizing lock scope, because contention on an in-memory demo dataset
isn't a real problem worth solving tonight.

Scope restrictions this module observes: no LLM calls, no candidate
retrieval or ranking logic, no FastAPI routes, no direct JSON-file
access, no business logic that belongs in candidates.py or
ranking_validation.py. Every read and write of order/inventory state
goes through db.py; this file decides nothing about WHICH product is
a good substitute, only whether accepting a GIVEN one is currently
valid.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from backend import db
from backend.models import OrderItemStatus

# Guards the idempotency-key -> result map below. See the module
# docstring for why this is a new, transactions.py-owned lock rather
# than a reuse of db.py's _DB_LOCK.
_IDEMPOTENCY_LOCK = threading.Lock()
_IDEMPOTENCY_RESULTS: dict[str, dict[str, Any]] = {}


def _conflict(reason: str) -> dict[str, str]:
    return {"status": "conflict", "reason": reason}


def accept_substitution(
    order_id: str,
    item_id: str,
    product_id: str,
    idempotency_key: str,
    ai_involved: bool | None = None,
) -> dict:
    """Accept a substitution: re-validate everything, commit the state
    change, and record it — or return a business conflict.

    Returns {"status": "ok"} on success, or
    {"status": "conflict", "reason": "..."} for any EXPECTED business
    conflict (bad idempotency key, stale/missing order or item, item
    already resolved, replacement unavailable, a concurrent decrement
    winning the race). None of those raise. An UNEXPECTED failure —
    a real bug, an invariant violation, an infrastructure error — DOES
    raise; this function does not convert every possible failure into
    a polite conflict dict, only the ones an associate can meaningfully
    act on (retry, pick a different candidate, refresh the order).

    `ai_involved` is optional and defaults to None (unknown) because
    this function's four required parameters are exactly what the
    build spec calls for, and nothing in this prototype's current call
    chain tells transactions.py whether the accepted product_id came
    from an AI recommendation or fallback ranking — that information
    lives one layer up, in whatever calls ranking_validation.py. This
    keyword-only addition lets a future API layer (which DOES know)
    populate the audit event's `ai_involved` field honestly, without
    this module fabricating a value it has no way of knowing today.
    This is a different situation from `associate_id` below: an
    associate identity requires real authentication infrastructure this
    prototype doesn't have at all, whereas AI-involvement is ordinary
    per-call information a caller can simply pass once it exists.

    Idempotency: if `idempotency_key` was already processed, the exact
    stored result is returned unchanged and NOTHING below this check
    executes again — no re-decrement, no re-update, no duplicate audit
    event.
    """
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        # Invalid/missing key is a client-caller bug, but an EXPECTED
        # one, not a database or programming error -- reject it the
        # same way every other business conflict is rejected, and do
        # so before ever touching the idempotency map (a None or
        # non-string key isn't safe to use as a dict key here anyway).
        return _conflict("Missing or invalid idempotency key.")

    with _IDEMPOTENCY_LOCK:
        stored = _IDEMPOTENCY_RESULTS.get(idempotency_key)
        if stored is not None:
            # Already processed -- return the exact prior result and
            # apply nothing again. Copied out so the caller can't
            # mutate our stored record by mutating the dict it got back.
            return dict(stored)

        # New key: the lock stays held for the ENTIRE execution below.
        # Holding it IS the "reservation" the build spec asks for --
        # there is no separate in-progress sentinel value, because a
        # concurrent same-key call simply can't reach the `.get()`
        # above until this whole block has finished and stored a
        # result, at which point it takes the "already processed"
        # branch instead.
        result = _execute_substitution(
            order_id, item_id, product_id, ai_involved=ai_involved
        )
        _IDEMPOTENCY_RESULTS[idempotency_key] = dict(result)
        return result


def _execute_substitution(
    order_id: str,
    item_id: str,
    product_id: str,
    *,
    ai_involved: bool | None,
) -> dict:
    """Re-validate current state from db.py and, if everything checks
    out, commit the substitution. Never trusts anything about the
    order, item, product, or inventory beyond what db.py returns RIGHT
    NOW — candidate-generation-time or ranking-time state is not
    consulted anywhere in this function.
    """
    # --- Re-fetch authoritative state -----------------------------------

    order = db.get_order(order_id)
    if order is None:
        return _conflict(f"Order {order_id!r} not found.")

    item = db.get_order_item(order_id, item_id)
    if item is None:
        # db.get_order_item is already scoped to (order_id, item_id)
        # together, so this single check covers both "item exists" and
        # "item belongs to this order" -- there's no item_id that
        # exists on a DIFFERENT order that could pass this check.
        return _conflict(
            f"Order item {item_id!r} not found on order {order_id!r}."
        )

    if item.status != OrderItemStatus.PENDING:
        # PICKED/SUBSTITUTED are already resolved; UNAVAILABLE/SKIPPED
        # are a different associate decision entirely. PENDING is the
        # only state this prototype's flow ever offers a substitution
        # recommendation for, so it's the only state that can accept one.
        return _conflict(
            f"Order item {item_id!r} is not in a substitutable state "
            f"(current status: {item.status.value})."
        )

    replacement = db.get_product(product_id)
    if replacement is None:
        return _conflict(f"Replacement product {product_id!r} not found in catalog.")

    inventory = db.get_inventory(product_id, order.store_id)
    if inventory is None or inventory.quantity <= 0:
        # Deliberately a DIFFERENT, more specific message than the
        # decrement-failure conflict below: this is "there's no stock
        # here at all, right now," checked before we even attempt the
        # optimistic decrement, not "someone else won a race."
        return _conflict(
            f"Replacement product {product_id!r} is not currently "
            f"available at store {order.store_id!r}."
        )

    requested_quantity = item.requested_quantity
    if (
        not isinstance(requested_quantity, (int, float))
        or requested_quantity <= 0
        or requested_quantity != int(requested_quantity)
    ):
        # db.decrement_inventory's `qty` is an int (matching
        # Inventory.quantity); OrderItem.requested_quantity is a float
        # to accommodate weight-based items elsewhere in the schema
        # (see db.py's own docstring on this). A non-whole-number
        # requested_quantity here means this item isn't one of the
        # whole-unit products this prototype's substitution flow
        # supports -- reject it explicitly rather than silently
        # truncating a fractional quantity into a wrong integer.
        return _conflict(
            f"Order item {item_id!r} has a requested_quantity "
            f"({requested_quantity!r}) this substitution flow can't "
            "commit -- only whole-unit quantities are supported."
        )
    quantity = int(requested_quantity)

    # --- Commit boundary --------------------------------------------
    # Everything above this line is validate-only (no mutation).
    # Everything from here down is the "BEGIN ... COMMIT" block
    # described in the module docstring — see that docstring for why
    # this in-memory prototype can't make it crash-atomic the way a
    # real database transaction would.

    decremented = db.decrement_inventory(
        product_id, order.store_id, quantity, inventory.version
    )
    if not decremented:
        # Someone else's request (or a stale expected_version) beat us
        # to it. Per the build spec, this exact reason string and NO
        # retry: the associate needs a fresh exception state and a new
        # recommendation, not a hidden second attempt on their behalf.
        return _conflict("Replacement inventory changed or is no longer available.")

    applied = db.update_order_item_status(
        order_id, item_id, status=OrderItemStatus.PICKED, substituted_product_id=product_id
    )
    if not applied:
        # Should be impossible: we just confirmed this exact order/item
        # exists moments ago, and _IDEMPOTENCY_LOCK has serialized out
        # every other transactions.py mutation for the whole duration
        # of this call, so nothing else could have removed it in
        # between. Inventory has ALREADY been decremented at this
        # point with no matching order-item update -- exactly the
        # inconsistency a real database transaction's COMMIT/ROLLBACK
        # would make physically impossible, and exactly what this
        # in-memory prototype cannot repair. Raised loudly as a bug,
        # not folded into a business conflict, per this function's
        # documented raise-on-the-unexpected contract.
        raise RuntimeError(
            "update_order_item_status failed immediately after "
            f"decrement_inventory succeeded for order_id={order_id!r}, "
            f"item_id={item_id!r}, product_id={product_id!r}. Inventory "
            "has been decremented with no corresponding order-item "
            "update; this violates an invariant this module assumes "
            "always holds and requires manual investigation/reconciliation."
        )

    db.append_audit_event(
        {
            "event_type": "substitution_accepted",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order_id": order_id,
            "item_id": item_id,
            # No authenticated associate context exists anywhere in
            # this prototype (there's no auth layer at all yet) — an
            # explicit placeholder, never a fabricated identity.
            "associate_id": "prototype",
            "original_product_id": item.product_id,
            "replacement_product_id": product_id,
            # None (unknown) unless a future caller supplies it — see
            # accept_substitution()'s docstring.
            "ai_involved": ai_involved,
        }
    )

    return {"status": "ok"}