"""Pytest configuration for FairSense-AgentiX tests.

Design Choices
--------------
1. **Auto-reset registry**: Reset global tool registry before each test
2. **Test-safe defaults**: Override settings to use fake tools by default
3. **Test isolation**: Prevent state pollution between tests
4. **Autouse fixtures**: Automatically applied, no manual decoration needed

What This Enables
-----------------
- Tests can run in any order without failures
- Each test gets a fresh registry with correct configuration
- Tests use fake tools by default (fast, no GPU/model requirements)
- Prevents cached tools from one test affecting another
- Fixes the ~78 test failures caused by singleton registry pollution

Why This Way
------------
- Global registry singleton pattern is great for production (no model reloading)
- But tests MUST have isolation - each test should be independent
- Default settings ("auto") try to load real models → slow, GPU-dependent
- Tests should be fast and work on CPU-only machines
- Autouse fixtures ensure developers don't forget to reset

Examples
--------
Without this fixture:
    Test A: ocr_tool="fake" → registry cached
    Test B: Expects ocr_tool="tesseract" → gets cached "fake" → FAILS

With this fixture:
    Test A: ocr_tool="fake" → registry cached
    [reset_tool_registry called, fake defaults set]
    Test B: Expects fresh registry → gets fake tools → PASSES
"""

import pytest

from fairsense_agentix.tools.registry import reset_tool_registry


@pytest.fixture(autouse=True, scope="function")
def configure_test_environment(monkeypatch):
    """Configure test-safe environment before each test.

    Design Choice: Monkeypatch global settings to force fake tools
    Why: Global settings singleton created at import time with original env
    What This Enables: Tests use fake tools without modifying real settings

    This runs BEFORE each test to:
    1. Monkeypatch global settings to use fake tools
    2. Reset global registry
    3. Tests run with fake tools (fast, no GPU)
    4. Restore original settings after test

    Prevents issues like:
    - ocr_tool="auto" tries to load PaddleOCR → CUDA errors on CPU machines
    - caption_model="auto" tries to load BLIP-2 → 5GB model download in tests
    - embedding_model="all-MiniLM-L6-v2" → sentence-transformers GPU loading

    Technical Note:
    - Can't just set env vars because settings already loaded at import time
    - Must directly monkeypatch the global settings attributes
    - monkeypatch from pytest automatically reverts after test
    """
    # Design Choice: Import inside fixture to ensure fresh module state
    # Why: Settings module imported at test collection time
    # What This Enables: Monkeypatch works correctly on already-loaded module
    from fairsense_agentix.configs import settings  # noqa: PLC0415

    # Monkeypatch global settings to use fake tools
    # Design Choice: Directly patch attributes instead of recreating Settings
    # Why: Settings has many fields, easier to patch just what tests need
    # What This Enables: Fast test execution without model loading
    monkeypatch.setattr(settings, "ocr_tool", "fake")
    monkeypatch.setattr(settings, "caption_model", "fake")
    monkeypatch.setattr(settings, "embedding_model", "fake")
    monkeypatch.setattr(settings, "llm_provider", "fake")

    # Reset registry to force recreation with patched settings
    reset_tool_registry()

    # Run test (monkeypatch auto-reverts after yield)
    yield

    # Reset registry again to clear test state
    reset_tool_registry()
