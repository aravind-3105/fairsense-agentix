"""FairSense-AgentiX: Agentic Multimodal Bias & AI-Risk Platform.

Quick Start
-----------
Simple usage:

    >>> from fairsense_agentix import analyze_text
    >>> result = analyze_text("Job posting text")
    >>> print(result.bias_detected, result.risk_level)

Class-based (reusable):

    >>> from fairsense_agentix import FairSense
    >>> fs = FairSense()
    >>> result = fs.analyze_text("Text")

Advanced (full control):

    >>> from fairsense_agentix import create_orchestrator_graph
    >>> graph = create_orchestrator_graph()
    >>> result = graph.invoke({"input_type": "text", "content": "...", "options": {}})
"""

# High-level API (recommended)
from fairsense_agentix.api import (
    BiasResult,
    FairSense,
    ResultMetadata,
    RiskResult,
    analyze_image,
    analyze_text,
    assess_risk,
)

# Configuration
from fairsense_agentix.configs.settings import Settings

# Advanced API (for power users)
from fairsense_agentix.graphs.orchestrator_graph import create_orchestrator_graph


__version__ = "0.1.0"

__all__ = [
    # Main API
    "FairSense",
    # Result models
    "BiasResult",
    "RiskResult",
    "ResultMetadata",
    # Convenience functions
    "analyze_text",
    "analyze_image",
    "assess_risk",
    # Configuration
    "Settings",
    # Advanced
    "create_orchestrator_graph",
]

# ==============================================================================
# Eager Loading: Preload models at import time for fast first use
# ==============================================================================
# This loads all models (LLM, OCR, embeddings, FAISS) when the module is
# imported, making all subsequent function calls instant (~0.03s instead of ~30s).
#
# Trade-off: Import takes 30-60s, but every analyze_text() call is instant.
# Perfect for: Production servers, demos, notebooks (import once, use many times)
#
# To disable: Set FAIRSENSE_DISABLE_EAGER_LOADING=true in environment
import os as _os


if _os.getenv("FAIRSENSE_DISABLE_EAGER_LOADING", "").lower() not in (
    "true",
    "1",
    "yes",
):
    from fairsense_agentix.tools.registry import get_tool_registry as _get_registry

    # Force singleton creation (preloads all models)
    _ = _get_registry()
    del _get_registry  # Clean up namespace
