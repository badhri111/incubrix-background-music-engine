# tests/conftest.py
"""
Shared pytest fixtures for the IncuBrix Music Engine test suite.
Overrides MUSIC_PROVIDER=mock so tests run without network or paid APIs.

Direct function-call approach: We call the FastAPI route functions
directly as normal Python callables, bypassing HTTP entirely.
This avoids the Python 3.14 + anyio TestClient deadlock and the
sandbox network restrictions.
"""

import os
import shutil
import pytest

# Force mock provider and a scratch output dir for tests
os.environ["MUSIC_PROVIDER"] = "mock"
os.environ["OUTPUT_DIR"] = "outputs_test"


@pytest.fixture(scope="session")
def test_output_dir():
    path = "outputs_test"
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def ensure_output_dir():
    """Make sure outputs_test exists before all tests."""
    os.makedirs("outputs_test", exist_ok=True)
    yield
    shutil.rmtree("outputs_test", ignore_errors=True)
