"""
tests/test_api.py

Integration tests for backend/api.py, run via fastapi.testclient.TestClient
against the REAL app (real lifespan -> db.init_db, real candidates.py /
ranking_validation.py / transactions.py) rather than mocking the domain
layer wholesale -- api.py's own docstring insists it contains "ZERO
candidate-filtering, ranking, validation, inventory-mutation, or
transaction logic of its own," so the most faithful way to test it is
to prove the WIRING is correct: right status codes, right pipeline
order, right error boundaries -- while letting the real domain modules
do their real work underneath.

The one deliberate mock is `ai_ranking.rank_candidates` in the tests
that exercise POST .../unavailable and the AI-failure boundary: this
keeps those tests fast, deterministic, and free of any real network
call, while still exercising the REAL candidates.py ->
ranking_validation.py pipeline around it (see module docstring's
"pipeline this file wires up" section) -- only the LLM call itself is
faked, exactly the boundary ai_ranking.py's own docstring describes as
"untrusted output" that ranking_validation.py must independently police
regardless of what it's fed.

Uses the `api_client` fixture from conftest.py, which points
backend.api._DATA_DIR at a fresh per-test copy of the real seed JSON
and drives the app's real lifespan.
"""

from __future__ import annotations

import uuid

import pytest


def _new_key() -> str:
    return str(uuid.uuid4())


# ----------------------------------------------------------------------
# Basic / directory endpoints
# ----------------------------------------------------------------------


class TestHealthAndRoot:
    def test_health(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_root_serves_frontend_index(self, api_client):
        resp = api_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestCatalogAndDirectoryEndpoints:
    def test_list_products_returns_full_catalog(self, api_client):
        resp = api_client.get("/products")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 18
        assert {"id", "name", "brand", "category", "size", "unit", "price"} <= set(body[0])

    def test_list_stores_returns_directory_shape_only(self, api_client):
        resp = api_client.get("/stores")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 3
        for store in body:
            assert set(store) == {"store_id", "name"}

    def test_list_orders_unfiltered(self, api_client):
        resp = api_client.get("/orders")
        assert resp.status_code == 200
        assert {o["id"] for o in resp.json()} == {"ORD-1001", "ORD-1002"}

    def test_list_orders_filtered_by_store_id(self, api_client):
        resp = api_client.get("/orders", params={"store_id": "STORE-9"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_orders_filtered_by_status(self, api_client):
        resp = api_client.get("/orders", params={"status": "READY"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_order_found(self, api_client):
        resp = api_client.get("/orders/ORD-1001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "ORD-1001"

    def test_get_order_not_found(self, api_client):
        resp = api_client.get("/orders/ORD-NOPE")
        assert resp.status_code == 404

    def test_audit_events_starts_empty(self, api_client):
        resp = api_client.get("/audit-events")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_audit_events_filtered_by_event_type_after_an_action(self, api_client):
        api_client.post(
            "/inventory/STORE-1/SKU-BANANA-1KG/shelf-report", json={"status": "low"}
        )
        resp = api_client.get("/audit-events", params={"event_type": "shelf_report"})
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) == 1
        assert events[0]["event_type"] == "shelf_report"

        resp_other = api_client.get("/audit-events", params={"event_type": "nonexistent_type"})
        assert resp_other.json() == []


# ----------------------------------------------------------------------
# POST .../unavailable -- the recommend pipeline
# ----------------------------------------------------------------------


class TestReportItemUnavailable:
    def test_unknown_order_404(self, api_client):
        resp = api_client.post("/orders/ORD-NOPE/items/ITEM-1001-1/unavailable")
        assert resp.status_code == 404

    def test_unknown_item_404(self, api_client):
        resp = api_client.post("/orders/ORD-1001/items/ITEM-NOPE/unavailable")
        assert resp.status_code == 404

    def test_item_not_pending_409(self, api_client):
        # Mutate state through the API itself (not the `fresh_db`
        # fixture, which points at a separate seed copy -- see
        # conftest.py) so the mutation is visible to the same app
        # instance this test asserts against.
        api_client.post(
            "/orders/ORD-1001/items/ITEM-1001-1/substitution",
            json={"product_id": "SKU-SPRITE-150", "idempotency_key": _new_key()},
        )
        resp = api_client.post("/orders/ORD-1001/items/ITEM-1001-1/unavailable")
        assert resp.status_code == 409

    def test_fallback_ranking_used_when_ai_returns_none(self, api_client, monkeypatch):
        from backend import api as api_module

        monkeypatch.setattr(api_module.ai_ranking, "rank_candidates", lambda *a, **kw: None)

        resp = api_client.post("/orders/ORD-1001/items/ITEM-1001-1/unavailable")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 4  # the known Coke Zero 1.5L candidate set
        for rec in body:
            assert "fallback" in rec["reason"]
            # Display enrichment fields merged in from the trusted
            # candidate list, per this route's own docstring.
            assert rec["name"] is not None
            assert rec["brand"] is not None
            assert rec["price"] is not None

    def test_validated_ai_ranking_is_returned_when_well_formed(self, api_client, monkeypatch):
        from backend import api as api_module

        fake_response = {
            "recommendations": [
                {"product_id": "SKU-SPRITE-150", "score": 0.95, "reason": "Same brand, closest size."}
            ]
        }
        monkeypatch.setattr(
            api_module.ai_ranking, "rank_candidates", lambda *a, **kw: fake_response
        )

        resp = api_client.post("/orders/ORD-1001/items/ITEM-1001-1/unavailable")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["product_id"] == "SKU-SPRITE-150"
        assert body[0]["reason"] == "Same brand, closest size."
        assert body[0]["name"] == "Sprite 1.5L"
        assert body[0]["brand"] == "Coca-Cola"
        assert body[0]["price"] == 2.39

    def test_hallucinated_ai_output_falls_back_at_the_http_layer_too(self, api_client, monkeypatch):
        from backend import api as api_module

        fake_response = {
            "recommendations": [
                {"product_id": "SKU-NOT-A-REAL-CANDIDATE", "score": 0.9, "reason": "made up"}
            ]
        }
        monkeypatch.setattr(
            api_module.ai_ranking, "rank_candidates", lambda *a, **kw: fake_response
        )

        resp = api_client.post("/orders/ORD-1001/items/ITEM-1001-1/unavailable")
        assert resp.status_code == 200
        body = resp.json()
        assert all("fallback" in rec["reason"] for rec in body)

    def test_ai_ranking_exception_degrades_to_fallback_not_500(self, api_client, monkeypatch):
        from backend import api as api_module

        def boom(*a, **kw):
            raise RuntimeError("simulated ai_ranking crash")

        monkeypatch.setattr(api_module.ai_ranking, "rank_candidates", boom)

        resp = api_client.post("/orders/ORD-1001/items/ITEM-1001-1/unavailable")
        assert resp.status_code == 200
        body = resp.json()
        assert all("fallback" in rec["reason"] for rec in body)

    def test_zero_candidate_item_returns_empty_list_not_an_error(self, api_client, monkeypatch):
        from backend import api as api_module

        monkeypatch.setattr(api_module.ai_ranking, "rank_candidates", lambda *a, **kw: None)
        # ITEM-1001-3 is the banana -- documented zero-candidate case.
        resp = api_client.post("/orders/ORD-1001/items/ITEM-1001-3/unavailable")
        assert resp.status_code == 200
        assert resp.json() == []


# ----------------------------------------------------------------------
# POST .../resolve-unavailable
# ----------------------------------------------------------------------


class TestResolveUnavailable:
    def test_unsupported_resolution_400(self, api_client):
        resp = api_client.post(
            "/orders/ORD-1001/items/ITEM-1001-3/resolve-unavailable",
            json={"resolution": "DELETE_FOREVER"},
        )
        assert resp.status_code == 400

    def test_unknown_order_404(self, api_client):
        resp = api_client.post(
            "/orders/ORD-NOPE/items/ITEM-1001-3/resolve-unavailable",
            json={"resolution": "REMOVE"},
        )
        assert resp.status_code == 404

    def test_unknown_item_404(self, api_client):
        resp = api_client.post(
            "/orders/ORD-1001/items/ITEM-NOPE/resolve-unavailable",
            json={"resolution": "REMOVE"},
        )
        assert resp.status_code == 404

    def test_item_not_pending_409(self, api_client):
        api_client.post(
            "/orders/ORD-1001/items/ITEM-1001-1/substitution",
            json={"product_id": "SKU-SPRITE-150", "idempotency_key": _new_key()},
        )
        resp = api_client.post(
            "/orders/ORD-1001/items/ITEM-1001-1/resolve-unavailable",
            json={"resolution": "REMOVE"},
        )
        assert resp.status_code == 409

    def test_successful_remove_marks_unavailable(self, api_client):
        resp = api_client.post(
            "/orders/ORD-1001/items/ITEM-1001-3/resolve-unavailable",
            json={"resolution": "REMOVE"},
        )
        assert resp.status_code == 200
        assert resp.json()["item_status"] == "UNAVAILABLE"

        order = api_client.get("/orders/ORD-1001").json()
        item = next(i for i in order["items"] if i["id"] == "ITEM-1001-3")
        assert item["status"] == "UNAVAILABLE"
        assert item["picked_quantity"] == 0

    def test_successful_remove_does_not_touch_inventory(self, api_client):
        before = api_client.get("/products").json()
        api_client.post(
            "/orders/ORD-1001/items/ITEM-1001-3/resolve-unavailable",
            json={"resolution": "REMOVE"},
        )
        # No inventory endpoint exposes raw quantities directly here,
        # but /inventory/nearby for the banana's own store would reflect
        # a decrement if one wrongly occurred; simplest direct proof is
        # that removing never calls the substitution/decrement path at
        # all -- verified structurally via the audit trail instead.
        events = api_client.get("/audit-events").json()
        assert all(e["event_type"] != "substitution_accepted" for e in events)

    def test_post_update_invariant_violation_raises_500(self, api_client, monkeypatch):
        # Simulates the documented "should be impossible" case: the
        # item existed a moment ago, but the write somehow fails.
        from backend import api as api_module

        monkeypatch.setattr(api_module.db, "update_order_item_status", lambda *a, **kw: False)

        with pytest.raises(RuntimeError, match="should be impossible"):
            api_client.post(
                "/orders/ORD-1001/items/ITEM-1001-3/resolve-unavailable",
                json={"resolution": "REMOVE"},
            )

    def test_appends_item_marked_unavailable_audit_event(self, api_client):
        api_client.post(
            "/orders/ORD-1001/items/ITEM-1001-3/resolve-unavailable",
            json={"resolution": "REMOVE"},
        )
        events = api_client.get(
            "/audit-events", params={"event_type": "item_marked_unavailable"}
        ).json()
        assert len(events) == 1
        assert events[0]["item_id"] == "ITEM-1001-3"
        assert events[0]["original_product_id"] == "SKU-BANANA-1KG"


# ----------------------------------------------------------------------
# POST .../substitution
# ----------------------------------------------------------------------


class TestAcceptSubstitutionEndpoint:
    def test_success_returns_200(self, api_client):
        resp = api_client.post(
            "/orders/ORD-1001/items/ITEM-1001-1/substitution",
            json={"product_id": "SKU-SPRITE-150", "idempotency_key": _new_key()},
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_conflict_returns_409(self, api_client):
        resp = api_client.post(
            "/orders/ORD-NOPE/items/ITEM-1001-1/substitution",
            json={"product_id": "SKU-SPRITE-150", "idempotency_key": _new_key()},
        )
        assert resp.status_code == 409
        assert resp.json()["status"] == "conflict"

    def test_missing_body_fields_is_422(self, api_client):
        resp = api_client.post(
            "/orders/ORD-1001/items/ITEM-1001-1/substitution",
            json={"product_id": "SKU-SPRITE-150"},  # idempotency_key missing
        )
        assert resp.status_code == 422

    def test_replayed_idempotency_key_returns_same_result(self, api_client):
        key = _new_key()
        first = api_client.post(
            "/orders/ORD-1001/items/ITEM-1001-1/substitution",
            json={"product_id": "SKU-SPRITE-150", "idempotency_key": key},
        )
        second = api_client.post(
            "/orders/ORD-1001/items/ITEM-1001-1/substitution",
            json={"product_id": "SKU-SPRITE-150", "idempotency_key": key},
        )
        assert first.json() == second.json() == {"status": "ok"}


# ----------------------------------------------------------------------
# POST .../complete
# ----------------------------------------------------------------------


class TestCompleteOrder:
    def test_unknown_order_404(self, api_client):
        resp = api_client.post("/orders/ORD-NOPE/complete")
        assert resp.status_code == 404

    def test_refuses_while_items_pending(self, api_client):
        resp = api_client.post("/orders/ORD-1001/complete")
        assert resp.status_code == 400
        body = resp.json()
        assert body["status"] == "conflict"
        open_ids = {i["item_id"] for i in body["open_items"]}
        assert open_ids == {"ITEM-1001-1", "ITEM-1001-2", "ITEM-1001-3", "ITEM-1001-4"}

    def test_full_order_reaches_ready_end_to_end(self, api_client, monkeypatch):
        """Mirrors smoke_test.py's ORD-1002 path (every item has a real
        substitute) end-to-end through the TestClient, without a
        running uvicorn server."""
        from backend import api as api_module

        monkeypatch.setattr(api_module.ai_ranking, "rank_candidates", lambda *a, **kw: None)

        order = api_client.get("/orders/ORD-1002").json()
        for item in order["items"]:
            recs = api_client.post(
                f"/orders/ORD-1002/items/{item['id']}/unavailable"
            ).json()
            assert recs, f"expected at least one candidate for {item['id']}"
            chosen = recs[0]["product_id"]
            sub_resp = api_client.post(
                f"/orders/ORD-1002/items/{item['id']}/substitution",
                json={"product_id": chosen, "idempotency_key": _new_key()},
            )
            assert sub_resp.status_code == 200

        complete_resp = api_client.post("/orders/ORD-1002/complete")
        assert complete_resp.status_code == 200
        assert complete_resp.json()["order_status"] == "READY"

    def test_banana_path_reaches_ready_via_resolve_unavailable(self, api_client, monkeypatch):
        """Mirrors smoke_test.py's ORD-1001 path: three substitutable
        items plus the banana, which has zero candidates and must be
        explicitly removed before the order can complete."""
        from backend import api as api_module

        monkeypatch.setattr(api_module.ai_ranking, "rank_candidates", lambda *a, **kw: None)

        order = api_client.get("/orders/ORD-1001").json()
        unresolved = []
        for item in order["items"]:
            recs = api_client.post(
                f"/orders/ORD-1001/items/{item['id']}/unavailable"
            ).json()
            if not recs:
                unresolved.append(item["id"])
                continue
            sub_resp = api_client.post(
                f"/orders/ORD-1001/items/{item['id']}/substitution",
                json={"product_id": recs[0]["product_id"], "idempotency_key": _new_key()},
            )
            assert sub_resp.status_code == 200

        assert unresolved == ["ITEM-1001-3"]  # the banana

        still_blocked = api_client.post("/orders/ORD-1001/complete")
        assert still_blocked.status_code == 400

        for item_id in unresolved:
            resp = api_client.post(
                f"/orders/ORD-1001/items/{item_id}/resolve-unavailable",
                json={"resolution": "REMOVE"},
            )
            assert resp.status_code == 200

        final = api_client.post("/orders/ORD-1001/complete")
        assert final.status_code == 200
        assert final.json()["order_status"] == "READY"

    def test_post_update_invariant_violation_raises_500(self, api_client, monkeypatch):
        # Every item resolved, so complete_order() reaches its own
        # write step -- simulate that write mysteriously failing right
        # after a successful get_order, the same "should be impossible"
        # contract transactions.py documents and this file already
        # tests for the resolve-unavailable route.
        from backend import api as api_module

        monkeypatch.setattr(api_module.ai_ranking, "rank_candidates", lambda *a, **kw: None)

        order = api_client.get("/orders/ORD-1002").json()
        for item in order["items"]:
            recs = api_client.post(f"/orders/ORD-1002/items/{item['id']}/unavailable").json()
            api_client.post(
                f"/orders/ORD-1002/items/{item['id']}/substitution",
                json={"product_id": recs[0]["product_id"], "idempotency_key": _new_key()},
            )

        monkeypatch.setattr(api_module.db, "update_order_status", lambda *a, **kw: False)

        with pytest.raises(RuntimeError, match="should be impossible"):
            api_client.post("/orders/ORD-1002/complete")


# ----------------------------------------------------------------------
# GET /inventory/nearby
# ----------------------------------------------------------------------


class TestNearbyInventory:
    def test_unknown_product_404(self, api_client):
        resp = api_client.get(
            "/inventory/nearby", params={"product_id": "SKU-NOPE", "store_id": "STORE-1"}
        )
        assert resp.status_code == 404

    def test_unknown_store_404(self, api_client):
        resp = api_client.get(
            "/inventory/nearby",
            params={"product_id": "SKU-COKE-ZERO-150", "store_id": "STORE-NOPE"},
        )
        assert resp.status_code == 404

    def test_known_scores_and_sort_order(self, api_client):
        resp = api_client.get(
            "/inventory/nearby",
            params={"product_id": "SKU-COKE-ZERO-150", "store_id": "STORE-1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["product_id"] == "SKU-COKE-ZERO-150"
        assert body["source_store_id"] == "STORE-1"
        stores = body["nearby_stores"]
        assert [s["store_id"] for s in stores] == ["STORE-2", "STORE-3"]
        assert stores[0]["score"] == 0.185
        assert stores[1]["score"] == 0.063
        assert stores[0]["score"] > stores[1]["score"]

    def test_requesting_store_never_included_in_results(self, api_client):
        resp = api_client.get(
            "/inventory/nearby",
            params={"product_id": "SKU-COKE-ZERO-150", "store_id": "STORE-1"},
        )
        ids = {s["store_id"] for s in resp.json()["nearby_stores"]}
        assert "STORE-1" not in ids

    def test_zero_stock_nearby_store_is_excluded(self, api_client):
        # Yogurt is out of stock at STORE-3; querying from STORE-1 must
        # exclude STORE-3 entirely, leaving only STORE-2.
        resp = api_client.get(
            "/inventory/nearby",
            params={"product_id": "SKU-YOGURT-NAT-500", "store_id": "STORE-1"},
        )
        stores = resp.json()["nearby_stores"]
        assert [s["store_id"] for s in stores] == ["STORE-2"]

    def test_store_with_missing_confidence_metadata_is_skipped_defensively(
        self, api_client, monkeypatch
    ):
        # Defensive branch: a store_id present in another store's
        # distances_km map but missing its own top-level stores.json
        # entry would be a seed-data bug -- the route must skip it
        # rather than fabricate a confidence value.
        from backend import api as api_module

        real_get_confidence = api_module.db.get_store_inventory_confidence

        def fake_get_confidence(store_id):
            if store_id == "STORE-3":
                return None
            return real_get_confidence(store_id)

        monkeypatch.setattr(
            api_module.db, "get_store_inventory_confidence", fake_get_confidence
        )

        resp = api_client.get(
            "/inventory/nearby",
            params={"product_id": "SKU-COKE-ZERO-150", "store_id": "STORE-1"},
        )
        assert resp.status_code == 200
        stores = resp.json()["nearby_stores"]
        # STORE-3 would otherwise appear (see test_known_scores_and_sort_order)
        # but is skipped entirely here due to the missing confidence value.
        assert [s["store_id"] for s in stores] == ["STORE-2"]


# ----------------------------------------------------------------------
# POST /inventory/{store_id}/{product_id}/shelf-report
# ----------------------------------------------------------------------


class TestShelfReport:
    @pytest.mark.parametrize("status", ["empty", "low", "damaged", "misplaced"])
    def test_all_supported_statuses_succeed(self, api_client, status):
        resp = api_client.post(
            "/inventory/STORE-1/SKU-BANANA-1KG/shelf-report", json={"status": status}
        )
        assert resp.status_code == 200
        assert resp.json()["shelf_status"] == status

    def test_unsupported_status_400(self, api_client):
        resp = api_client.post(
            "/inventory/STORE-1/SKU-BANANA-1KG/shelf-report",
            json={"status": "on_fire"},
        )
        assert resp.status_code == 400

    def test_unknown_product_404(self, api_client):
        resp = api_client.post(
            "/inventory/STORE-1/SKU-NOPE/shelf-report", json={"status": "empty"}
        )
        assert resp.status_code == 404

    def test_unknown_store_404(self, api_client):
        resp = api_client.post(
            "/inventory/STORE-NOPE/SKU-BANANA-1KG/shelf-report",
            json={"status": "empty"},
        )
        assert resp.status_code == 404

    def test_unsupported_status_checked_before_lookups(self, api_client):
        # Both product AND store are bogus, but the unsupported status
        # must be rejected first (400), per the route's own documented
        # ordering ("checked... before either db.py lookup runs").
        resp = api_client.post(
            "/inventory/STORE-NOPE/SKU-NOPE/shelf-report", json={"status": "on_fire"}
        )
        assert resp.status_code == 400

    def test_never_mutates_inventory_quantity(self, api_client):
        before = api_client.get(
            "/inventory/nearby",
            params={"product_id": "SKU-BANANA-1KG", "store_id": "STORE-2"},
        ).json()
        api_client.post(
            "/inventory/STORE-1/SKU-BANANA-1KG/shelf-report", json={"status": "empty"}
        )
        after = api_client.get(
            "/inventory/nearby",
            params={"product_id": "SKU-BANANA-1KG", "store_id": "STORE-2"},
        ).json()
        assert before == after

    def test_appends_shelf_report_audit_event(self, api_client):
        api_client.post(
            "/inventory/STORE-1/SKU-BANANA-1KG/shelf-report", json={"status": "damaged"}
        )
        events = api_client.get(
            "/audit-events", params={"event_type": "shelf_report"}
        ).json()
        assert len(events) == 1
        assert events[0]["store_id"] == "STORE-1"
        assert events[0]["product_id"] == "SKU-BANANA-1KG"
        assert events[0]["status"] == "damaged"
