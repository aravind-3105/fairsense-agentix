"""State models for FairSense-AgentiX graph orchestration.

This package defines Pydantic models that represent state at each stage of
the agentic workflow system. These models ensure type safety and validation
as data flows through the orchestrator and subgraphs.

Key Design Principles:
    - Type safety: All fields strongly typed with Pydantic validation
    - Immutability: State updates return new instances (functional pattern)
    - Observability: run_id propagates through entire execution
    - Refinement support: Fields for tracking iterative improvement loops

Implementation is split across focused modules:
    - :mod:`.orchestrator` — SelectionPlan, EvaluationResult, OrchestratorState
    - :mod:`.bias_text`    — BiasTextState
    - :mod:`.bias_image`   — BiasImageState
    - :mod:`.bias_image_vlm` — BiasImageVLMState
    - :mod:`.risk`         — RiskState

All public names are re-exported here so existing imports of the form
``from fairsense_agentix.graphs.state import ...`` continue to work.

Examples
--------
    >>> plan = SelectionPlan(
    ...     workflow_id="bias_text",
    ...     reasoning="Text input detected",
    ...     confidence=0.95,
    ...     tool_preferences={"llm": "gpt-4"},
    ...     node_params={},
    ...     fallbacks=[],
    ... )
    >>> state = OrchestratorState(input_type="text", content="Sample text", plan=plan)
"""

from fairsense_agentix.graphs.state.bias_image import BiasImageState
from fairsense_agentix.graphs.state.bias_image_vlm import BiasImageVLMState
from fairsense_agentix.graphs.state.bias_text import BiasTextState
from fairsense_agentix.graphs.state.orchestrator import (
    EvaluationResult,
    OrchestratorState,
    SelectionPlan,
)
from fairsense_agentix.graphs.state.risk import RiskState


__all__ = [
    "SelectionPlan",
    "EvaluationResult",
    "OrchestratorState",
    "BiasTextState",
    "BiasImageState",
    "BiasImageVLMState",
    "RiskState",
]
