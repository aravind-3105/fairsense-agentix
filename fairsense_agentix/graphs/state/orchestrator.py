"""State models for the orchestrator supergraph."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class SelectionPlan(BaseModel):
    """Plan created by router for workflow execution.

    The router analyzes input and creates a structured plan that tells the
    orchestrator which workflow to run and how to configure it. This enables
    dynamic model selection, parameter tuning, and fallback strategies.

    Attributes
    ----------
    workflow_id : Literal["bias_text", "bias_image", "risk"]
        Which workflow to execute
    reasoning : str
        Human-readable explanation of why this workflow was chosen
    confidence : float
        Router's confidence in this decision (0.0-1.0)
    tool_preferences : dict[str, Any]
        Which tools to use (e.g., {"llm": "gpt-4", "temperature": 0.3})
    node_params : dict[str, dict[str, Any]]
        Per-node configuration (e.g., {"bias_llm": {"prompt_version": "v2"}})
    fallbacks : list[str]
        Alternative workflow_ids if this plan fails

    Examples
    --------
    >>> plan = SelectionPlan(
    ...     workflow_id="bias_text",
    ...     reasoning="Text input with moderate length",
    ...     confidence=0.95,
    ...     tool_preferences={"llm": "gpt-4", "temperature": 0.3},
    ...     node_params={"bias_llm": {"prompt_version": "v1"}},
    ...     fallbacks=[],
    ... )
    """

    workflow_id: Literal["bias_text", "bias_image", "bias_image_vlm", "risk"] = Field(
        description="Which workflow to execute",
    )

    reasoning: str = Field(description="Human-readable explanation of router decision")

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Router confidence in this decision",
    )

    tool_preferences: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool selection preferences (e.g., which LLM to use)",
    )

    node_params: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-node configuration parameters",
    )

    fallbacks: list[str] = Field(
        default_factory=list,
        description="Alternative workflow IDs if this plan fails",
    )


class EvaluationResult(BaseModel):
    """Result of evaluating workflow output quality.

    Evaluators assess whether workflow output meets quality standards and
    provide structured feedback for potential refinement. This enables the
    orchestrator to make informed decisions about accepting, refining, or
    failing a workflow execution.

    Attributes
    ----------
    passed : bool
        Whether output meets quality threshold
    score : float
        Quality score (0.0-1.0), higher is better
    issues : list[str]
        Specific problems found (empty if passed=True)
    refinement_hints : dict[str, Any]
        Suggested changes to improve quality (e.g., {"temperature": 0.2})
    explanation : str
        Human-readable reasoning for the evaluation

    Examples
    --------
    >>> eval_result = EvaluationResult(
    ...     passed=False,
    ...     score=0.6,
    ...     issues=["Missing mitigation suggestions"],
    ...     refinement_hints={"temperature": 0.2, "prompt_suffix": "Add mitigations"},
    ...     explanation="Output lacks actionable mitigation strategies",
    ... )
    """

    passed: bool = Field(description="Whether output meets quality threshold")

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality score, higher is better",
    )

    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems found in output",
    )

    refinement_hints: dict[str, Any] = Field(
        default_factory=dict,
        description="Suggested changes to improve output quality",
    )

    explanation: str = Field(
        default="",
        description="Human-readable reasoning for evaluation",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluator-specific metadata (e.g., raw critique payloads)",
    )


class OrchestratorState(BaseModel):
    """State for orchestrator supergraph.

    The orchestrator manages the entire workflow execution, including routing,
    evaluation, and refinement loops. This state model tracks all data needed
    for the agentic coordination process.

    Attributes
    ----------
    input_type : Literal["text", "image", "csv"]
        Type of input being processed
    content : str | bytes
        The actual input content
    options : dict[str, Any]
        User-provided configuration options
    plan : SelectionPlan | None
        Execution plan from router
    preflight_eval : EvaluationResult | None
        Pre-execution validation result
    posthoc_eval : EvaluationResult | None
        Post-execution quality assessment
    workflow_result : dict[str, Any] | None
        Result from executed workflow
    decision : Literal["accept", "refine", "fail"] | None
        Orchestrator decision on how to proceed
    refinement_count : int
        Number of refinement iterations performed (for bounded loop)
    final_result : dict[str, Any] | None
        Final packaged result for client
    errors : list[str]
        Error messages accumulated during execution
    warnings : list[str]
        Warning messages (non-fatal issues)
    run_id : str
        Unique identifier for this execution (for tracing)

    Examples
    --------
    >>> state = OrchestratorState(
    ...     input_type="text",
    ...     content="Sample job posting text",
    ...     options={"enable_refinement": False},
    ... )
    >>> state.run_id  # Auto-generated UUID
    '550e8400-e29b-41d4-a716-446655440000'
    """

    # Input
    input_type: Literal["text", "image", "csv"] = Field(
        description="Type of input being processed",
    )

    content: str | bytes = Field(description="The actual input content")

    options: dict[str, Any] = Field(
        default_factory=dict,
        description="User-provided configuration options",
    )

    # Router output
    plan: SelectionPlan | None = Field(
        default=None,
        description="Execution plan from router",
    )

    # Evaluation results
    preflight_eval: EvaluationResult | None = Field(
        default=None,
        description="Pre-execution validation result",
    )

    posthoc_eval: EvaluationResult | None = Field(
        default=None,
        description="Post-execution quality assessment",
    )

    # Workflow execution
    workflow_result: dict[str, Any] | None = Field(
        default=None,
        description="Result from executed workflow",
    )

    # Refinement loop state (Phase 7+)
    decision: Literal["accept", "refine", "fail"] | None = Field(
        default=None,
        description="Orchestrator decision on how to proceed",
    )

    refinement_count: int = Field(
        default=0,
        ge=0,
        description="Number of refinement iterations performed",
    )

    # Output
    final_result: dict[str, Any] | None = Field(
        default=None,
        description="Final packaged result for client",
    )

    errors: list[str] = Field(
        default_factory=list,
        description="Error messages accumulated during execution",
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Warning messages (non-fatal issues)",
    )

    # Observability
    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for execution tracing",
    )
