# BOPIS Store Associate Assistant

A vertical slice of the substitution-recommendation exception flow from the design
doc's §13: an associate reports an item unavailable, a trusted candidate set is
retrieved and hard-filtered deterministically, an LLM ranks and explains that set,
the ranking is validated before the associate ever sees it, and only a re-validated,
associate-approved acceptance is allowed to touch inventory or order state.

## Architecture

![BOPIS Store Associate Assistant Architecture](docs/images/architecture.png)

**The LLM ranks and explains a trusted, deterministically filtered candidate set — it never generates candidates, and it never touches inventory or order state.**

`candidates.py` builds the only universe a recommendation may reference (same category, in-stock at this store, within a ±30% price band); `ai_ranking.py` asks the model to order and justify that set; `ranking_validation.py` checks every `product_id` in the model's response against the trusted set before anything reaches the UI, discarding the whole response for a deterministic fallback ranking on any single failure; `transactions.py` re-fetches order and inventory state from scratch and commits with an optimistic-locking check, treating an accepted recommendation as nothing more than "an associate's chosen product_id" with no special trust attached to its AI origin.

This boundary exists for two independent reasons, not one: model output is **untrusted** (it can hallucinate a product, or simply be temporarily unavailable), and the world is **stale by the time it matters** (inventory can change in the seconds between a recommendation being shown and the associate tapping Accept). A recommendation answers "what looks like a good substitute"; only a fresh backend check answers "is this actually still true right now" — conflating the two would mean either trusting the model with a decision it can't verify, or trusting a snapshot that's already out of date.

Every arrow after "LLM ranking" is deterministic Python. The LLM sits at exactly one point in this pipeline, and its output is disposable — a bad or missing response never blocks the associate, it just falls back to a ranking by brand, size, and price distance instead (`ranking_validation._fallback_rank`).

## How to run

```bash
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

uvicorn backend.api:app --reload
```

Open **http://localhost:8000/** for the associate UI (`api.py`'s `GET /` serves
`frontend/index.html` directly — there's no separate frontend server).

**No `.env` is required to run this.** With no LLM provider configured,
`llm_client.py` raises `LLMClientError` on the first ranking call, `api.py` catches
it and falls back to deterministic ranking, and the UI is indistinguishable to the
associate except for the reason text ("...fallback ranking)" appended by
`ranking_validation._fallback_rank`) — this is the intended, demoable behavior for
the design doc's §12 tradeoff ("AI is optional infrastructure, not critical
path"), not a degraded mode you need to avoid.

To exercise the real LLM path instead, set either:

```bash
# Plain OpenAI-compatible provider
export OPENAI_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini          # or your provider's model name
# export OPENAI_BASE_URL=...          # optional, for an OpenAI-compatible non-OpenAI endpoint

# --- OR: SAP AI Core / GenAI Hub ---
export AICORE_CLIENT_ID=...
export AICORE_CLIENT_SECRET=...
export AICORE_AUTH_URL=...
export AICORE_BASE_URL=...
export AICORE_RESOURCE_GROUP=...
export LLM_MODEL=...                  # your GenAI Hub deployment's model name
```

Then run the smoke test against the running server:

```bash
python smoke_test.py
```

## Manual API smoke test

The sequence below shows the shape of each call. `smoke_test.py` is the preferred
way to actually exercise this end to end — the substitution `product_id` and every
`idempotency_key` must come from a real response, not be typed in ahead of time
(that's the whole point of the validation boundary this app is built around), so
the curl version below is illustrative, not copy-paste-to-completion:

```bash
# 1. Look at the order
curl -s http://localhost:8000/orders/ORD-1001 | python3 -m json.tool

# 2. Report the Coca-Cola Zero line unavailable -> get validated recommendations
curl -s -X POST http://localhost:8000/orders/ORD-1001/items/ITEM-1001-1/unavailable \
  | python3 -m json.tool
# Copy a "product_id" from the response above into PRODUCT_ID below.

# 3. Accept it, with a fresh idempotency key
curl -s -X POST http://localhost:8000/orders/ORD-1001/items/ITEM-1001-1/substitution \
  -H "Content-Type: application/json" \
  -d "{\"product_id\": \"PRODUCT_ID\", \"idempotency_key\": \"$(uuidgen)\"}"

# 4. Confirm the item is PICKED with that substituted_product_id
curl -s http://localhost:8000/orders/ORD-1001 | python3 -m json.tool

# 5. Try to complete the order
curl -s -X POST http://localhost:8000/orders/ORD-1001/complete
```

Step 5 will return **400**, not 200, after only step 3 — ORD-1001 has 4 items and
this walk-through resolved 1. That's the FR-7 completion gate working correctly,
not a bug in the sequence; see "Demo scenario" below for why one of the remaining
3 items (bananas) can never be resolved through this API at all, and run
`smoke_test.py` to see the gate verified properly alongside a real order that does
reach `READY`.

## Deliberate scope exclusions

This is the "one vertical slice" build the design doc's §13 calls for, not the
full §5 architecture. Explicitly out of scope, not silently missing:

- **No persistent database** — `db.py` is in-memory, seeded from the three JSON
  files at startup; a restart resets to that seed state.
- **No authentication/authorization** — no associate identity exists anywhere;
  audit events record `"associate_id": "prototype"` as an explicit placeholder.
- **No offline synchronization** — the design doc's §7.4 local-queue/sync model
  isn't built; the frontend talks to a reachable server or shows a connection
  error.
- **No nearby-store fulfillment** — use case B (§6.3) is a different, purely
  deterministic ranking problem and isn't part of this slice.
- **No production reservation system** — `Inventory.reserved_quantity` exists in
  the schema for fidelity with §8 but is never read or written; this prototype
  decrements `quantity` directly at accept-time instead of reserving at
  order-assignment time.
- **No production-grade distributed transaction infrastructure** — `db.py` and
  `transactions.py` are both explicit about this: each mutation is independently
  locked, not wrapped in one atomic multi-step transaction, so a failure between
  the inventory decrement and the order-item update raises loudly for manual
  reconciliation instead of silently rolling back (impossible without a real
  database transaction).

The production version of this would replace the in-memory store and the
sequenced-lock "transaction" in `transactions.py` with a real transactional
database and a proper reservation mechanism, and would add nearby-store
fulfillment as a second, independently-scored exception path alongside
substitution.

## Demo scenario

Seeded in `data/orders.json` / `data/inventory.json`:

- **ORD-1001**, assigned to **STORE-1**, 4 items — the design doc's own worked
  example (§6.2) is item 1: **Coca-Cola Zero 1.5L, quantity 2**, with **zero units**
  in stock at STORE-1. Its same-category, in-stock, in-price-band neighbors (1L
  Coke Zero, 1.5L Coke Original, Pepsi Max, Sprite) give the AI/fallback ranker a
  real 4-candidate set to work with — the recommendation is validated against
  exactly that set before it ever reaches the associate.
- Items 2 and 4 on the same order (spaghetti, whole milk) each have their own
  smaller in-band candidate set and resolve the same way.
- Item 3 (**bananas**) has no same-category neighbor inside the ±30% price band —
  apples and tomatoes are both priced too far above bananas for this catalog's
  substitution policy. `get_substitution_candidates()` correctly returns an empty
  list for it, and since this build has no plain "pick an in-stock item" endpoint
  (only unavailable/substitution), that item has no path out of `PENDING` today.
  `smoke_test.py` verifies this produces a 400 from `/complete`, naming the open
  item — the completion gate refusing correctly, not a defect.
- **ORD-1002** (also STORE-1) is the order where every item's price band does
  contain an in-stock same-category neighbor, so it's the one `smoke_test.py` uses
  to demonstrate the full path through to `order_status == "READY"`.

## Repository structure

```
backend/
├── models.py              Pydantic schemas — the only source of shape
├── db.py                  in-memory data access, seed load, locking
├── candidates.py          deterministic candidate retrieval + hard filters
├── llm_client.py          OpenAI-compatible provider wrapper (GenAI Hub or plain)
├── ai_ranking.py          untrusted LLM ranking call
├── ranking_validation.py  the enforcement point: validate-or-fallback
├── transactions.py        the only module allowed to mutate order/inventory
└── api.py                 FastAPI routes — wiring only, no business logic
frontend/
└── index.html             associate UI, no build step
data/
└── products.json, inventory.json, orders.json
smoke_test.py               end-to-end integration check against a live server
requirements.txt
```