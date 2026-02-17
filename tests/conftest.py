"""Pytest configuration for FairSense-AgentiX.

This file is responsible for configuring deterministic test behavior. We force the
project to run with the lightweight "fake" toolchain regardless of a developer's
`.env` so that CI never attempts to hit real APIs or load heavy OCR/caption
models. If someone truly wants to run the suite against real providers they can
opt-in by exporting `FAIRSENSE_TEST_USE_REAL=1` before running pytest.
"""

from __future__ import annotations

import os

import pytest


def _should_force_fake_stack() -> bool:
    """Return True when tests should patch environment for fake providers."""
    flag = os.environ.get("FAIRSENSE_TEST_USE_REAL", "").lower()
    return flag not in {"1", "true", "yes", "on"}


if _should_force_fake_stack():
    _TEST_ENV_OVERRIDES: dict[str, str] = {
        # LLM + evaluators use fake models to avoid network calls.
        "FAIRSENSE_LLM_PROVIDER": "fake",
        "FAIRSENSE_EVALUATOR_ENABLED": "true",
        # OCR/caption tools remain fake to avoid heavy optional dependencies.
        "FAIRSENSE_OCR_TOOL": "fake",
        "FAIRSENSE_CAPTION_MODEL": "fake",
        "FAIRSENSE_CAPTION_PRELOAD": "false",
        # Disable caching to stop filesystem churn inside CI sandboxes.
        "FAIRSENSE_LLM_CACHE_ENABLED": "false",
        # Keep telemetry off to reduce noisy logs in CI.
        "FAIRSENSE_TELEMETRY_ENABLED": "false",
        # Disable eager loading to speed up test collection and avoid
        # loading heavy models at import time (60s -> instant).
        "FAIRSENSE_DISABLE_EAGER_LOADING": "true",
        # Test environment defaults
        "FAIRSENSE_CAPTION_MAX_LENGTH": "50",
        "FAIRSENSE_CAPTION_FORCE_CPU": "true",
        "FAIRSENSE_LLM_MODEL_NAME": "gpt-4-turbo",
    }

    for key, value in _TEST_ENV_OVERRIDES.items():
        os.environ[key] = value


@pytest.fixture
def my_test_number() -> int:
    """Keep backwards-compatible sample fixture."""
    return 42
