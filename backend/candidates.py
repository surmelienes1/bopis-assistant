"""
backend/candidates.py

Deterministic substitution-candidate retrieval for the BOPIS prototype.

Exposes exactly one public function, get_substitution_candidates(), which
answers "what could an associate offer as a substitute for this item, at
this store, right now" using only catalog and inventory data already
exposed by db.py. This is the trusted candidate universe for everything
downstream:

    ai_ranking.py may re-order and explain these candidates. It may
    NEVER invent a product_id that didn't come out of this function.
    ranking_validation.py's entire job is checking that every
    product_id an LLM recommendation returns is a member of the list
    this function produced for that (product_id, store_id) call — if
    it isn't, the whole recommendation is discarded and the system
    falls back deterministically. That check is only meaningful if
    this function's output is itself trustworthy, so every filter
    below is a hard filter: a candidate either qualifies or it's gone,
    no LLM judgment anywhere in this file.

Filter chain, each step a private function so it's independently
testable and the pipeline reads top-to-bottom as the actual policy:

    1. _get_original(product_id)                        -> Product
    2. _same_category_candidates(original, store_id)     -> list[Product]
    3. _filter_in_stock(candidates, store_id)             -> list[Product]
    4. _filter_price_band(candidates, original.price)     -> list[Product]
    5. _filter_customer_allergens(candidates, allergies)  -> list[Product]

Step 5 — allergen safety — deliberately does NOT use the heuristic an
earlier draft of this spec proposed: exclude a candidate if it carries
an allergen the *original product* doesn't list. That heuristic
conflates two different things — what allergens a product happens to
contain, and what a specific customer is actually allergic to. A
customer who ordered a chocolate bar containing milk and soy is not
thereby known to tolerate milk and soy in a substitute, and a customer
who ordered a product with zero listed allergens is not thereby known
to be allergic to everything else. Filtering on the original product's
own allergen list as a proxy for customer allergy data produces both
false exclusions (safe substitutes dropped because the original
happened not to list some allergen) and false confidence (unsafe
substitutes admitted because the original happened to share one) —
worse than doing nothing, because it looks like a safety check without
being one.

This prototype's data model has no customer-allergy source to filter
against: Order carries only a customer_id string, and there is no
Customer/CustomerProfile record anywhere in data/ or models.py. So
_filter_customer_allergens() takes an explicit `customer_allergies`
parameter that defaults to None, and is a documented no-op when it's
None — candidates are neither included nor excluded on the strength of
allergen data that doesn't exist. The parameter exists so a later
prompt can wire in a real customer-allergy source without changing this
function's signature, only its call site. Until that source exists,
"no data" means "don't filter" — not "filter using the closest-looking
field as a stand-in." Stated scope gap, not a silently missing feature.
"""

from __future__ import annotations

from backend import db
from backend.models import Product

# Substitution price-band policy. A stand-in for a real store/merchant
# substitution policy, which in production would come from a policy
# service (possibly varying by category, store, or brand tier) rather
# than one constant applied uniformly across the whole catalog.
_PRICE_BAND_RATIO = 0.30


def get_substitution_candidates(
    product_id: str,
    store_id: str,
    customer_allergies: list[str] | None = None,
) -> list[dict]:
    """Return the trusted substitution-candidate universe for one item.

    This is the ONLY function downstream code may call to obtain
    candidate products for substitution ranking. Every filter here is
    hard and deterministic — a candidate either survives every step or
    it's gone. Nothing in this file ranks, scores, or explains
    candidates; that's ai_ranking.py's job, operating strictly on the
    list this function returns.

    `customer_allergies`, when provided, is a list of allergen strings
    using the same vocabulary as Product.allergens (e.g. "milk",
    "gluten", "soy") — an explicit, customer-sourced constraint. When
    None (the only mode this prototype's data ever exercises — see
    module docstring), no allergen filtering happens at all.

    Returns plain dicts, not Product objects, with only the fields the
    ranking step needs: product_id, name, brand, size, unit, price.
    Trimming the shape here, rather than downstream, keeps the LLM
    prompt small and keeps ai_ranking.py from being tempted to reach
    into Product fields (allergens, attributes) it has no business
    making substitution judgments about.
    """
    original = _get_original(product_id)

    candidates = _same_category_candidates(original, store_id)
    candidates = _filter_in_stock(candidates, store_id)
    candidates = _filter_price_band(candidates, original.price)
    candidates = _filter_customer_allergens(candidates, customer_allergies)

    return [
        {
            "product_id": c.id,
            "name": c.name,
            "brand": c.brand,
            "size": c.size,
            "unit": c.unit,
            "price": c.price,
        }
        for c in candidates
    ]


# ----------------------------------------------------------------------
# Filter chain — each step named and isolated so it's testable on its
# own and the pipeline above reads as the actual policy, not a black box.
# ----------------------------------------------------------------------


def _get_original(product_id: str) -> Product:
    """Step 1: look up the product being substituted.

    Raises rather than returning None: by the time
    get_substitution_candidates() is called, `product_id` has already
    come from a real OrderItem in a real order (see transactions.py's
    call site), so a missing product here means catalog and order data
    have drifted apart — a bug or data-corruption signal, not an
    expected, caller-recoverable outcome. That's the opposite
    convention from db.py's write functions, which return False/None
    for expected failures (stale version, an id an external caller
    supplied) — here the id is internally sourced and trusted, so
    failing silently would be more dangerous than failing loudly.
    """
    original = db.get_product(product_id)
    if original is None:
        raise ValueError(
            f"get_substitution_candidates: product_id {product_id!r} not "
            "found in catalog — caller passed an id that doesn't exist, "
            "or catalog and order data have drifted apart."
        )
    return original


def _same_category_candidates(original: Product, store_id: str) -> list[Product]:
    """Step 2: same-category products, excluding the original itself.

    `store_id` isn't used for the category lookup — db.list_products_
    by_category is store-agnostic, a catalog query rather than a stock
    query — but it's threaded through so this step's signature stays
    uniform with the rest of the chain, and a reader of the pipeline
    doesn't have to wonder why one step alone drops a parameter every
    neighboring step takes.
    """
    return db.list_products_by_category(
        original.category, exclude_product_id=original.id
    )


def _filter_in_stock(candidates: list[Product], store_id: str) -> list[Product]:
    """Step 3: hard filter — exclude anything with zero stock at this store.

    A candidate with no Inventory row at all is treated the same as
    zero stock (excluded), not as "unknown, so include it": an
    associate acting on a recommendation needs "this is actually on
    the shelf," and a missing inventory row is not evidence of that.
    """
    in_stock: list[Product] = []
    for candidate in candidates:
        inv = db.get_inventory(candidate.id, store_id)
        if inv is not None and inv.quantity > 0:
            in_stock.append(candidate)
    return in_stock


def _filter_price_band(
    candidates: list[Product], original_price: float
) -> list[Product]:
    """Step 4: hard filter — exclude anything outside ±30% of original price.

    _PRICE_BAND_RATIO stands in for a real store substitution policy —
    see the module-level comment above the constant.
    """
    low = original_price * (1 - _PRICE_BAND_RATIO)
    high = original_price * (1 + _PRICE_BAND_RATIO)
    return [c for c in candidates if low <= c.price <= high]


def _filter_customer_allergens(
    candidates: list[Product], customer_allergies: list[str] | None
) -> list[Product]:
    """Step 5: hard filter against EXPLICIT customer allergy data, when
    such data exists — never against the original product's own
    allergen list. See the module docstring for why that proxy was
    rejected.

    When `customer_allergies` is None or empty, this is a deliberate
    no-op: `candidates` is returned unchanged. "No customer allergy
    data" must mean "don't filter," not "filter on the nearest
    available field."

    When `customer_allergies` IS provided, a candidate is excluded if
    it contains ANY of the listed allergens — strict and symmetric,
    with no partial-match exceptions, since getting this wrong in the
    permissive direction is a customer-safety issue, not a UX
    tradeoff.
    """
    if not customer_allergies:
        return candidates
    flagged = set(customer_allergies)
    return [c for c in candidates if not (flagged & set(c.allergens))]