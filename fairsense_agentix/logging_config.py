"""Logging configuration for FairSense-AgentiX.

This module configures logging levels to suppress verbose HTTP logs from
third-party libraries while keeping our application logs visible. Import
this module at application startup for side effects.

Usage
-----
Import at the top of main application files (api.py, server.py):

    from fairsense_agentix import logging_config  # noqa: F401

This will suppress verbose HTTP request/response logs from httpx, urllib3,
openai, and anthropic clients, making console output much more readable.
"""

import logging


def ensure_root_logging(level: int = logging.INFO) -> None:
    """Attach a stderr handler if the root logger has no handlers yet.

    ``configure_logging()`` only adjusts named logger levels; it does not add
    handlers. Call this from CLI entrypoints (``run_server.py``,
    ``server.start()``) so INFO/DEBUG lines are visible when the host process
    never called ``logging.basicConfig``. No-op if handlers already exist.
    """
    if not logging.root.handlers:
        logging.basicConfig(
            level=level,
            format="%(levelname)s:%(name)s:%(message)s",
        )


def configure_logging() -> None:
    """Configure logging levels for clean console output.

    Strategy:
    - Suppress VERBOSE HTTP logs (request/response details)
    - Suppress VERBOSE model internals (layer warnings, tensor shapes)
    - KEEP meaningful progress logs (model loading, initialization)

    This creates a clean output that shows progress without spam.
    """
    # Suppress HTTP request/response logs (major noise source)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

    # Suppress LLM provider API call logs (not the actual responses)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    # Transformers: Show loading progress, hide verbose warnings
    logging.getLogger("transformers").setLevel(logging.INFO)  # ← Changed: show loading
    logging.getLogger("transformers.modeling_utils").setLevel(
        logging.WARNING,
    )  # ← Hide tensor warnings
    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
    logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.WARNING)

    # Sentence Transformers: Show loading progress
    logging.getLogger("sentence_transformers").setLevel(
        logging.INFO,
    )  # ← Keep progress visible

    # Keep our application logs visible
    logging.getLogger("fairsense_agentix").setLevel(logging.INFO)

    # Keep LangGraph logs visible (workflow execution)
    logging.getLogger("langgraph").setLevel(logging.INFO)


# Auto-configure on import (side effect)
configure_logging()
