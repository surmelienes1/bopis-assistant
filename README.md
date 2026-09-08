# BOPIS Assistant

AI-assisted mobile application prototype for supermarket store associates
picking online BOPIS (Buy Online, Pick Up In Store) orders.

## Core Engineering Thesis

Use AI for reasoning, ranking, and explanation at points of ambiguity.

Keep state-changing operations deterministic, validated, auditable, and
outside the control of the language model.

The LLM should never be the authority for:

- inventory truth
- authorization
- business-rule validation
- order-state transitions
- transactional writes

## Core AI Use Case

Product substitution recommendation:

1. Retrieve candidates from trusted catalog/inventory systems.
2. Apply deterministic hard constraints.
3. Ask the AI to rank the remaining candidates.
4. Validate the structured AI response.
5. Show the recommendation to the associate.
6. Require associate approval.
7. Revalidate inventory on the backend.
8. Commit the state change transactionally.

## Core Prototype Flow

Order
→ Pick
→ Product unavailable
→ Retrieve candidates
→ Apply hard filters
→ AI ranking
→ Show recommendation
→ Associate accepts
→ Backend validates
→ Update order

## Architecture

Store Associate Mobile
        |
       API
        |
  +-----+----------------+
  |                      |
Order/Picking      Inventory/Catalog
  |                      |
  +----------+-----------+
             |
       AI Orchestrator
             |
            LLM

## Technology

- Python
- FastAPI
- Pydantic
- HTML/CSS/JavaScript
- LLM provider abstraction
- SAP AI Core / Gen AI Hub
- pytest

## Reliability Principles

- Backend is authoritative.
- Inventory may be stale on the mobile client.
- Inventory is revalidated before mutation.
- Optimistic concurrency protects inventory updates.
- Mobile retries should use idempotency keys.
- Offline clients may cache and queue observations.
- AI failure must not prevent core picking.
- Deterministic fallback ranking exists when AI is unavailable.

## Security

- Never commit credentials.
- AI receives only necessary information.
- Retrieved product data is treated as untrusted content.
- AI output is schema validated.
- Business rules remain outside the model.

## Deliberate Prototype Exclusions

- Production authentication
- Production event bus
- Full offline synchronization
- Advanced replenishment forecasting
- Production notification platform
- Customer application
- Microservice deployment

These are production extensions to discuss rather than initial implementation
requirements.

## Development

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Run tests:

    pytest -q

Run API:

    uvicorn backend.api:app --reload
