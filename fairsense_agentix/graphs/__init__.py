"""Graph orchestration modules for FairSense-AgentiX.

This package contains state models and graph definitions for the agentic
bias detection and risk assessment system.

Key Components:
    - state.py: Pydantic models representing state at each stage
    - orchestrator_graph.py: Supergraph coordinating all workflows
    - bias_text_graph.py: Text bias analysis subgraph
    - bias_image_graph.py: Image bias analysis subgraph (OCR + Caption)
    - risk_graph.py: AI risk assessment subgraph

Architecture:
    The system uses a supergraph/subgraph pattern where the orchestrator
    acts as an agentic coordinator, delegating work to specialized workflows
    and managing quality refinement loops.
"""

from fairsense_agentix.graphs.state import (
    BiasImageState,
    BiasTextState,
    EvaluationResult,
    OrchestratorState,
    RiskState,
    SelectionPlan,
)


__all__ = [
    "SelectionPlan",
    "EvaluationResult",
    "OrchestratorState",
    "BiasTextState",
    "BiasImageState",
    "RiskState",
]

# Note: create_orchestrator_graph is not exported here to avoid circular imports.
# Import directly from the module:
#   from fairsense_agentix.graphs.orchestrator_graph import create_orchestrator_graph
