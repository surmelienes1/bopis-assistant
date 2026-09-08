"""
smoke_test.py

Small, executable integration smoke test for the BOPIS prototype's
vertical slice, run against a REAL, already-running FastAPI server
(`uvicorn backend.api:app`) -- not against the Python modules directly.
This is a demo/integration check, not a replacement for a real test
suite: it exercises exactly one path through the system and stops.

The path, end to end, over real HTTP:

    GET order
      -> POST item unavailable
      -> receive VALIDATED recommendations (AI or deterministic fallback
         -- this script never assumes which; ranking_validation.py
         guarantees both shapes are identical, so treating whatever
         /unavailable returned as ground truth is correct either way)
      -> select a returned candidate (never a hardcoded product_id)
      -> POST substitution with a fresh idempotency key
      -> GET order again, verify the item is PICKED with the right
         substituted_product_id
      -> POST complete, verify the order reaches READY

A note on WHY this script touches two seeded orders, not one
-------------------------------------------------------------
The brief for this script named ORD-1001/STORE-1 as "the" scenario, and
that order IS used below for the headline substitution walk-through --
it's the exact worked example in the design doc (Coca-Cola Zero 1.5L,
§6.2). But running the real filter chain in candidates.py against the
real seed data (not by assumption -- this was actually executed)
surfaces a genuine, documented interaction of two independently correct
policies, not a bug in this script or in candidates.py:

    ITEM-1001-3 is SKU-BANANA-1KG (produce, EUR 1.49). Its only
    same-category neighbors are apples (EUR 2.19) and tomatoes
    (EUR 2.49) -- both outside candidates.py's +/-30% price-substitution
    band. get_substitution_candidates() therefore returns an EMPTY list
    for this item, which is the hard-filter chain working exactly as
    designed (see candidates.py's own docstring: "a candidate either
    qualifies or it's gone, no LLM judgment anywhere in this file").

Separately, the current backend/api.py exposes only two item-mutation
endpoints -- .../unavailable+.../substitution, and no plain "pick this
in-stock item" endpoint (the design doc's §9 lists one; it wasn't part
of what got built -- see README's "Deliberate scope exclusions"). So
the ONLY way any item on an order can leave PENDING today is through
the substitution path.

Put those two true facts together: ITEM-1001-3 has no path out of
PENDING via the current API, so ORD-1001 as seeded cannot reach READY
today -- not because anything is broken, but because a real produce SKU
has no in-policy substitute in this catalog. Forcing this script to
reach READY on ORD-1001 would mean silently loosening the price band or
inventing a pick endpoint -- exactly the kind of scope-widening this
prototype's brief says not to do.

So this script demonstrates BOTH real behaviors instead of hiding
either one:
  1. On ORD-1001: resolve every item that DOES have a valid substitute
     (3 of 4), then call /complete and verify it correctly REFUSES with
     a 400 naming the one item that can't be resolved -- proving the
     FR-7 completion gate works, not just the substitution path.
  2. On ORD-1002 (every item's price is within band of a same-category,
     in-stock neighbor -- also verified against the real filter chain,
     not assumed): resolve every item and verify the order actually
     reaches READY, so the full success path is exercised somewhere.

Nothing above required touching api.py, candidates.py, or the seed
data -- this script only calls the API as a client would.
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any

import requests

BASE_URL = os.environ.get("BOPIS_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 10.0

# Resolved item statuses, mirrored from backend/models.py's
# RESOLVED_ITEM_STATUSES -- duplicated as plain strings here rather than
# imported, since this script talks to the API over HTTP like any other
# client, not by reaching into backend/ internals.
_RESOLVED_STATUSES = {"PICKED", "SUBSTITUTED"}


def fail(message: str) -> None:
    """Print a clear failure reason and exit non-zero. Every failure in
    this script goes through here so a broken run always explains
    itself instead of raising a bare traceback."""
    print(f"\nFAIL: {message}", file=sys.stderr)
    sys.exit(1)


def request(method: str, path: str, **kwargs: Any) -> requests.Response:
    """Thin wrapper around requests.request() that turns a connection
    failure (server not running, wrong port) into the same loud,
    explanatory failure as an HTTP error, instead of a raw traceback
    a reader has to decode.
    """
    url = f"{BASE_URL}{path}"
    try:
        return requests.request(method, url, timeout=TIMEOUT_SECONDS, **kwargs)
    except requests.RequestException as exc:
        fail(
            f"{method} {path} could not reach the server at {BASE_URL} ({exc}). "
            "Is `uvicorn backend.api:app --reload` running?"
        )
        raise  # unreachable -- fail() exits; keeps type checkers happy


def get_order(order_id: str) -> dict:
    resp = request("GET", f"/orders/{order_id}")
    if resp.status_code != 200:
        fail(f"GET /orders/{order_id} returned {resp.status_code}: {resp.text}")
    return resp.json()


def resolve_item_via_substitution(order_id: str, item: dict) -> str | None:
    """Run ONE item through .../unavailable -> select -> .../substitution.

    Returns the accepted product_id on success. Returns None if
    .../unavailable legitimately produced zero recommendations (an
    empty-but-valid candidate universe -- see the module docstring's
    banana example) -- that is reported to the caller as data, not
    raised as a script error, since it's a correct outcome of the hard
    price-band filter, not a malfunction.

    Any OTHER unexpected shape (non-200, non-list body, a
    recommendation with no usable product_id) fails loudly: those would
    mean either the server is broken or ranking_validation.py's
    trust-boundary guarantee was violated, and this script should never
    paper over that.
    """
    item_id = item["id"]
    product_id = item["product_id"]

    resp = request("POST", f"/orders/{order_id}/items/{item_id}/unavailable")
    if resp.status_code != 200:
        fail(
            f"POST /orders/{order_id}/items/{item_id}/unavailable "
            f"returned {resp.status_code}: {resp.text}"
        )

    recommendations = resp.json()
    if not isinstance(recommendations, list):
        fail(
            f"POST .../unavailable for {item_id} returned a non-list body: "
            f"{recommendations!r}"
        )

    if not recommendations:
        print(
            f"    {item_id} ({product_id}): 0 recommendations -- no "
            "same-category, in-stock, in-price-band substitute exists "
            "for this SKU. Leaving it PENDING; this is candidates.py's "
            "hard filter working as designed, not a failure."
        )
        return None

    chosen = recommendations[0]
    chosen_product_id = chosen.get("product_id") if isinstance(chosen, dict) else None
    if not isinstance(chosen_product_id, str) or not chosen_product_id:
        fail(
            f"First recommendation for {item_id} has no usable product_id: "
            f"{chosen!r}"
        )

    idempotency_key = str(uuid.uuid4())
    resp = request(
        "POST",
        f"/orders/{order_id}/items/{item_id}/substitution",
        json={"product_id": chosen_product_id, "idempotency_key": idempotency_key},
    )
    if resp.status_code != 200:
        fail(
            f"POST .../substitution for {item_id} -> {chosen_product_id} "
            f"returned {resp.status_code}: {resp.text}"
        )

    print(
        f"    {item_id} ({product_id}): {len(recommendations)} recommendation(s) "
        f"-> accepted {chosen_product_id} (score={chosen.get('score')})"
    )
    return chosen_product_id


def verify_item_picked(order_id: str, item_id: str, expected_product_id: str) -> None:
    order = get_order(order_id)
    updated = next((i for i in order["items"] if i["id"] == item_id), None)
    if updated is None:
        fail(f"Item {item_id} vanished from order {order_id} after substitution.")
    if updated["status"] != "PICKED":
        fail(
            f"Expected {item_id} status PICKED after substitution, got "
            f"{updated['status']!r}."
        )
    if updated.get("substituted_product_id") != expected_product_id:
        fail(
            f"Expected {item_id}.substituted_product_id == "
            f"{expected_product_id!r}, got "
            f"{updated.get('substituted_product_id')!r}."
        )


def resolve_all_resolvable_items(order_id: str) -> list[str]:
    """Walk every currently-PENDING item on `order_id` through
    resolve_item_via_substitution(), verifying each acceptance against
    a fresh GET before moving on. Returns the ids of items that could
    NOT be resolved (zero recommendations) so the caller can decide
    what to expect from /complete afterward.
    """
    order = get_order(order_id)
    pending = [i for i in order["items"] if i["status"] == "PENDING"]
    if not pending:
        print(f"  (no PENDING items on {order_id} -- already resolved)")

    unresolved: list[str] = []
    for item in pending:
        product_id = resolve_item_via_substitution(order_id, item)
        if product_id is None:
            unresolved.append(item["id"])
            continue
        verify_item_picked(order_id, item["id"], product_id)
    return unresolved


def main() -> None:
    print(f"BOPIS smoke test against {BASE_URL}\n")

    # --- ORD-1001: the design doc's worked example (Coca-Cola Zero 1.5L) ---
    # Resolves every item that has a valid substitute, then proves the
    # FR-7 completion gate correctly refuses to finish the order while
    # ITEM-1001-3 (no in-band substitute -- see module docstring) is
    # still PENDING.
    print("ORD-1001 / STORE-1")
    still_pending = resolve_all_resolvable_items("ORD-1001")

    resp = request("POST", "/orders/ORD-1001/complete")
    if not still_pending:
        # Only true if a future data/policy change gives every item a
        # substitute -- handle it as a real success rather than assuming
        # today's gap forever.
        if resp.status_code != 200 or resp.json().get("order_status") != "READY":
            fail(
                "All ORD-1001 items resolved but /complete did not return "
                f"200/READY: {resp.status_code} {resp.text}"
            )
        print("  /complete -> 200 READY (every item ended up resolvable)")
    else:
        if resp.status_code != 400:
            fail(
                "Expected /complete to refuse with 400 while "
                f"{still_pending} remain PENDING, got {resp.status_code}: "
                f"{resp.text}"
            )
        body = resp.json()
        open_item_ids = {oi.get("item_id") for oi in body.get("open_items", [])}
        if not set(still_pending) <= open_item_ids:
            fail(
                f"/complete's open_items {open_item_ids} did not include "
                f"the still-unresolved item(s) {still_pending}."
            )
        print(
            f"  /complete correctly refused (400) -- open_items includes "
            f"{sorted(open_item_ids)}, confirming the FR-7 completion gate."
        )

    # --- ORD-1002: every item has an in-band substitute (verified against
    # the real filter chain, not assumed) -- this is where the full
    # success path to READY gets exercised. ---
    print("\nORD-1002 / STORE-1")
    still_pending_1002 = resolve_all_resolvable_items("ORD-1002")
    if still_pending_1002:
        fail(
            f"Expected every ORD-1002 item to have a substitute, but "
            f"{still_pending_1002} produced none. Seed data or policy "
            "changed since this script was written."
        )

    resp = request("POST", "/orders/ORD-1002/complete")
    if resp.status_code != 200:
        fail(f"POST /orders/ORD-1002/complete returned {resp.status_code}: {resp.text}")
    order_status = resp.json().get("order_status")
    if order_status != "READY":
        fail(f"Expected order_status READY after complete, got {order_status!r}.")
    print(f"  /complete -> 200, order_status={order_status}")

    print(
        "\nPASS: substitution pipeline (recommend -> validate -> accept -> "
        "commit), the FR-7 completion gate, and a full order -> READY "
        "transition are all verified end-to-end."
    )


if __name__ == "__main__":
    main()