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
