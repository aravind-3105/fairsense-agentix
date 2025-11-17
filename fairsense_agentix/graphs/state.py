"""State models for FairSense-AgentiX graph orchestration.

This module defines Pydantic models that represent state at each stage of
the agentic workflow system. These models ensure type safety and validation
as data flows through the orchestrator and subgraphs.

Key Design Principles:
    - Type safety: All fields strongly typed with Pydantic validation
    - Immutability: State updates return new instances (functional pattern)
    - Observability: run_id propagates through entire execution
    - Refinement support: Fields for tracking iterative improvement loops

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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    pass


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

    workflow_id: Literal["bias_text", "bias_image", "risk"] = Field(
        description="Which workflow to execute"
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
        description="Type of input being processed"
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


class BiasTextState(BaseModel):
    """State for text bias analysis workflow.

    This workflow analyzes text for various forms of bias (gender, age, race,
    ability, etc.), provides a detailed analysis, optional summary, and
    highlighted HTML output.

    Note: Quality assessment is handled by orchestrator's posthoc_eval node,
    not by the subgraph itself.

    Attributes
    ----------
    text : str
        Input text to analyze for bias
    options : dict[str, Any]
        Configuration options (temperature, prompt_version, etc.)
    bias_analysis : str | None
        Detailed bias analysis from LLM
    summary : str | None
        Optional condensed summary of bias findings
    highlighted_html : str | None
        HTML with bias spans highlighted
    run_id : str
        Unique identifier inherited from orchestrator

    Examples
    --------
    >>> state = BiasTextState(
    ...     text="Sample job posting text", options={"temperature": 0.3}
    ... )
    """

    text: str = Field(description="Input text to analyze for bias")

    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration options for workflow",
    )

    bias_analysis: Any = Field(
        default=None,
        description="Structured bias analysis from LLM (BiasAnalysisOutput object)",
    )

    summary: str | None = Field(
        default=None,
        description="Optional condensed summary of findings",
    )

    highlighted_html: str | None = Field(
        default=None,
        description="HTML with bias spans highlighted",
    )

    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for tracing",
    )


class BiasImageState(BaseModel):
    """State for image bias analysis workflow.

    This workflow extracts text from images (OCR) and generates captions
    (captioning), then analyzes the combined text for bias. Parallel execution
    of OCR and captioning enables faster processing.

    Note: Quality assessment is handled by orchestrator's posthoc_eval node,
    not by the subgraph itself.

    Attributes
    ----------
    image_bytes : bytes
        Raw image data to analyze
    options : dict[str, Any]
        Configuration options
    ocr_text : str | None
        Text extracted via OCR
    caption_text : str | None
        Generated image caption
    merged_text : str | None
        Combined OCR + caption text (deduplicated)
    bias_analysis : str | None
        Detailed bias analysis from LLM
    summary : str | None
        Optional condensed summary
    highlighted_html : str | None
        HTML with bias spans highlighted
    run_id : str
        Unique identifier for tracing

    Examples
    --------
    >>> state = BiasImageState(image_bytes=b"...", options={"ocr_tool": "tesseract"})
    """

    image_bytes: bytes = Field(description="Raw image data to analyze")

    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration options for workflow",
    )

    # Parallel extraction
    ocr_text: str | None = Field(
        default=None,
        description="Text extracted via OCR",
    )

    caption_text: str | None = Field(
        default=None,
        description="Generated image caption",
    )

    merged_text: str | None = Field(
        default=None,
        description="Combined OCR + caption text",
    )

    # Analysis (same as BiasTextState after merge)
    bias_analysis: Any = Field(
        default=None,
        description="Structured bias analysis from LLM (BiasAnalysisOutput object)",
    )

    summary: str | None = Field(
        default=None,
        description="Optional condensed summary",
    )

    highlighted_html: str | None = Field(
        default=None,
        description="HTML with bias spans highlighted",
    )

    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for tracing",
    )


class RiskState(BaseModel):
    """State for AI risk assessment workflow.

    This workflow embeds scenario text, searches for relevant AI risks via
    FAISS, retrieves corresponding AI-RMF (Risk Management Framework)
    recommendations, and outputs results as an HTML table and CSV.

    Note: Quality assessment is handled by orchestrator's posthoc_eval node,
    not by the subgraph itself.

    Attributes
    ----------
    scenario_text : str
        Input scenario to assess for AI risks
    options : dict[str, Any]
        Configuration options (top_k, etc.)
    embedding : list[float] | None
        Vector embedding of scenario text
    risks : list[dict[str, Any]] | None
        Top-k AI risks from FAISS search
    rmf_recommendations : dict[str, list[dict[str, Any]]] | None
        RMF recommendations per risk (keyed by risk ID)
    joined_table : list[dict[str, Any]] | None
        Risks + RMF combined into table rows
    html_table : str | None
        Rendered HTML table
    csv_path : str | None
        Path to generated CSV file
    run_id : str
        Unique identifier for tracing

    Examples
    --------
    >>> state = RiskState(
    ...     scenario_text="Deploying facial recognition in public spaces",
    ...     options={"top_k": 5},
    ... )
    """

    scenario_text: str = Field(description="Input scenario to assess for AI risks")

    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration options for workflow",
    )

    # Embedding
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding of scenario text",
    )

    # FAISS search results
    risks: list[dict[str, Any]] | None = Field(
        default=None,
        description="Top-k AI risks from FAISS search",
    )

    rmf_recommendations: dict[str, list[dict[str, Any]]] | None = Field(
        default=None,
        description="RMF recommendations per risk (keyed by risk ID)",
    )

    # Join and formatting
    joined_table: list[dict[str, Any]] | None = Field(
        default=None,
        description="Risks + RMF combined into table rows",
    )

    html_table: str | None = Field(
        default=None,
        description="Rendered HTML table",
    )

    csv_path: str | None = Field(
        default=None,
        description="Path to generated CSV file",
    )

    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for tracing",
    )
