"""
tests/conftest.py

Shared fixtures for the whole suite.

Design choices worth stating up front, since they shape every test file
in this directory:

- `db` module state is a set of module-level dicts guarded by one lock
  (see backend/db.py's own docstring). That means tests CANNOT run
  safely in parallel against the shared, real `data/` seed files without
  stepping on each other's mutations (e.g. one test decrementing
  inventory that another test's assertions depend on). The `fresh_db`
  fixture below re-initializes `backend.db` from a *per-test temporary
  copy* of the seed JSON before every test, so:

    1. every test starts from a known, pristine state, and
    2. a test that mutates inventory/orders (decrement, status changes)
       can never leak that mutation into another test, including ones
       that run in the same process/session.

- We copy the real seed JSON (data/products.json etc.) rather than
  hand-rolling synthetic fixtures, because several tests are written
  against the *actual* product/inventory/order graph shipped with this
  project (e.g. "Coca-Cola Zero 1.5L out of stock at STORE-1" is a real,
  documented scenario from the design doc and smoke_test.py). Copying
  keeps those tests honest against the real data instead of a
  hand-maintained shadow copy that can drift from it.

- `api_client` builds a fastapi.testclient.TestClient over the real
  `backend.api.app`, but repoints `backend.api._DATA_DIR` at the same
  temp seed copy `fresh_db` just initialized, and drives the app's own
  `lifespan` (via the `with TestClient(...) as client` context) so
  `db.init_db()` runs exactly the way it does in production, not via a
  test-only shortcut.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_DATA_DIR = REPO_ROOT / "data"


@pytest.fixture()
def seed_data_dir(tmp_path: Path) -> Path:
    """A per-test temp copy of the real data/*.json seed files."""
    dest = tmp_path / "data"
    shutil.copytree(SEED_DATA_DIR, dest)
    return dest


@pytest.fixture()
def fresh_db(seed_data_dir: Path):
    """Reset backend.db's module-level state from a pristine seed copy
    before the test runs, and again after, so no test can leak mutated
    inventory/order state into a test that runs after it.
    """
    from backend import db

    db.init_db(seed_data_dir)
    yield db
    # Reset again afterwards purely for hygiene; the next test's own
    # fresh_db call will re-init anyway, but this keeps any code that
    # imports db directly at module scope (rather than through this
    # fixture) from ever observing a mutated tail state.
    db.init_db(seed_data_dir)


@pytest.fixture()
def seed_json(seed_data_dir: Path):
    """Load the raw seed JSON as plain dicts/lists, for tests that want
    to assert against the source-of-truth numbers directly (e.g. "STORE-1's
    Coca-Cola Zero 1.5L quantity is exactly 0") without depending on
    backend.db's parsed Pydantic view of the same data.
    """

    def _load(name: str):
        return json.loads((seed_data_dir / f"{name}.json").read_text())

    return _load


@pytest.fixture()
def api_client(seed_data_dir: Path, monkeypatch):
    """A TestClient over the real FastAPI app, pointed at an isolated
    per-test seed copy, with the app's real lifespan (db.init_db) run.
    """
    from fastapi.testclient import TestClient

    from backend import api

    monkeypatch.setattr(api, "_DATA_DIR", str(seed_data_dir))
    with TestClient(api.app) as client:
        yield client
