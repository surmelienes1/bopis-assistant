"""
smoke_test.py

Small, executable integration smoke test for the BOPIS prototype's
vertical slice, run against a REAL, already-running FastAPI server
(`uvicorn backend.api:app`) -- not against the Python modules directly.

The script exercises the current end-to-end workflow over real HTTP:

    GET order
      -> POST item unavailable
      -> receive validated recommendations
      -> select a returned candidate
      -> POST substitution with a fresh idempotency key
      -> GET order and verify the item is PICKED
      -> for zero-candidate items, explicitly REMOVE the item
      -> POST complete
      -> verify the order reaches READY

The script deliberately uses two seeded orders.

ORD-1001 is the main worked example from the design: Coca-Cola Zero 1.5L
is unavailable and receives substitution recommendations. Other items
exercise the same substitution path. The banana has no valid candidate
after the deterministic same-category, stock, and +/-30% price filters,
so the script verifies that /complete initially refuses while the item is
PENDING, then exercises the explicit /resolve-unavailable endpoint and
verifies that the order can subsequently reach READY.

ORD-1002 provides a second full success path in which every seeded item
has an in-policy substitution candidate, allowing the smoke test to
verify a complete order -> READY transition independently.

Important boundary:

  - candidates.py determines the trusted candidate universe.
  - ai_ranking.py may rank only those candidates.
  - ranking_validation.py rejects hallucinated or malformed model output.
  - transactions.py performs the authoritative substitution commit.
  - /resolve-unavailable is deterministic and does not call the LLM.
  - /complete remains the authoritative completion gate.

This is an integration smoke test, not a replacement for a full unit or
concurrency test suite.
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
_RESOLVED_STATUSES = {"PICKED", "SUBSTITUTED", "UNAVAILABLE"}


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


def resolve_unavailable_item(order_id: str, item_id: str) -> None:
    """Explicitly remove an item that has no suitable substitute.

    This exercises the API's deterministic zero-candidate resolution
    path: no inventory is modified, no LLM is involved, and the item
    becomes UNAVAILABLE so the order can still reach READY.
    """
    resp = request(
        "POST",
        f"/orders/{order_id}/items/{item_id}/resolve-unavailable",
        json={"resolution": "REMOVE"},
    )
    if resp.status_code != 200:
        fail(
            f"POST /orders/{order_id}/items/{item_id}/resolve-unavailable "
            f"returned {resp.status_code}: {resp.text}"
        )

    order = get_order(order_id)
    updated = next((i for i in order["items"] if i["id"] == item_id), None)
    if updated is None:
        fail(f"Item {item_id} vanished from order {order_id} after removal.")

    if updated["status"] != "UNAVAILABLE":
        fail(
            f"Expected {item_id} status UNAVAILABLE after removal, got "
            f"{updated['status']!r}."
        )

    if updated.get("picked_quantity") != 0:
        fail(
            f"Expected {item_id}.picked_quantity == 0 after removal, got "
            f"{updated.get('picked_quantity')!r}."
        )

    print(
        f"    {item_id}: no suitable substitute -> explicitly removed "
        f"(status=UNAVAILABLE)"
    )


def resolve_all_resolvable_items(order_id: str) -> list[str]:
    """Walk every currently-PENDING item through the substitution path.

    Returns item IDs for which the real candidate pipeline produced zero
    recommendations. Those items are not silently removed: the caller
    must explicitly resolve them through the unavailable-item API.
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
    # Resolve every item that has a valid substitute. The banana has no
    # in-policy substitute, so first prove that /complete correctly refuses
    # while it remains PENDING. Then explicitly remove that item through
    # the deterministic unavailable-item path and prove the full order
    # can reach READY.
    print("ORD-1001 / STORE-1")
    still_pending = resolve_all_resolvable_items("ORD-1001")

    if still_pending:
        resp = request("POST", "/orders/ORD-1001/complete")
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

        for item_id in still_pending:
            resolve_unavailable_item("ORD-1001", item_id)

    resp = request("POST", "/orders/ORD-1001/complete")
    if resp.status_code != 200:
        fail(
            f"POST /orders/ORD-1001/complete returned "
            f"{resp.status_code}: {resp.text}"
        )

    order_status = resp.json().get("order_status")
    if order_status != "READY":
        fail(
            f"Expected ORD-1001 to reach READY after resolving all "
            f"exceptions, got {order_status!r}."
        )

    print(f"  /complete -> 200, order_status={order_status}")

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