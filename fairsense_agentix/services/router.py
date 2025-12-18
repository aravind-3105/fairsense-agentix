"""Router module for workflow selection and plan creation.

The router analyzes input and creates a structured SelectionPlan that tells
the orchestrator which workflow to run and how to configure it. This enables
intelligent workflow selection, dynamic model selection, and parameter tuning.

In Phase 2-6, the router uses deterministic logic based on input_type.
In Phase 7+, this can be enhanced with LLM-based routing for complex inputs.

Example:
    >>> from fairsense_agentix.services.router import create_selection_plan
    >>> plan = create_selection_plan(
    ...     input_type="text", content="Sample job posting", options={}
    ... )
    >>> plan.workflow_id
    'bias_text'
    >>> plan.reasoning
    'Text input detected, using text bias analysis workflow'
"""

from typing import Any, Literal

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.state import SelectionPlan


def create_selection_plan(
    input_type: Literal["text", "image", "csv"],
    content: str | bytes,
    options: dict[str, Any] | None = None,
) -> SelectionPlan:
    """Create execution plan by analyzing input.

    The router determines which workflow to execute based on input type
    and creates a structured plan with tool preferences and configuration
    parameters.

    Parameters
    ----------
    input_type : Literal["text", "image", "csv"]
        Type of input being processed
    content : str | bytes
        The actual input content (used for future heuristics)
    options : dict[str, Any] | None, optional
        User-provided options that override defaults, by default None

    Returns
    -------
    SelectionPlan
        Structured plan for orchestrator execution

    Examples
    --------
    >>> # Text input routing
    >>> plan = create_selection_plan("text", "Job posting text", {})
    >>> plan.workflow_id
    'bias_text'

    >>> # Image input routing
    >>> # (returns bias_image_vlm if VLM mode enabled, else bias_image)
    >>> plan = create_selection_plan("image", b"image_bytes", {})
    >>> plan.workflow_id  # 'bias_image_vlm' or 'bias_image' depending on settings
    'bias_image_vlm'

    >>> # CSV/risk input routing
    >>> plan = create_selection_plan("csv", "AI scenario", {})
    >>> plan.workflow_id
    'risk'
    """
    options = options or {}

    # Phase 2-6: Deterministic routing based on input_type
    # Phase 7+: Can enhance with LLM-based routing for ambiguous cases

    if input_type == "text":
        return _create_text_bias_plan(content, options)
    if input_type == "image":
        return _create_image_bias_plan(content, options)
    if input_type == "csv":
        return _create_risk_plan(content, options)
    # Fallback (should never happen due to Literal type)
    return SelectionPlan(
        workflow_id="bias_text",
        reasoning=f"Unknown input_type '{input_type}', defaulting to text workflow",
        confidence=0.5,
        tool_preferences={},
        node_params={},
        fallbacks=["bias_image", "risk"],
    )


def _create_text_bias_plan(
    content: str | bytes, options: dict[str, Any]
) -> SelectionPlan:
    """Create plan for text bias analysis workflow.

    Parameters
    ----------
    content : str | bytes
        Text content to analyze
    options : dict[str, Any]
        User options that override defaults

    Returns
    -------
    SelectionPlan
        Plan configured for text bias workflow
    """
    # Extract text length for future heuristics
    text = (
        content
        if isinstance(content, str)
        else content.decode("utf-8", errors="ignore")
    )
    text_length = len(text)

    # Determine tool preferences from settings and options
    llm_model = options.get("llm_model", settings.llm_model_name)
    temperature = options.get("temperature", settings.llm_temperature)

    # Node-specific parameters
    node_params = {
        "bias_llm": {
            "model": llm_model,
            "temperature": temperature,
            "prompt_version": "v1",
        },
        "summarize": {
            "enabled": text_length > 500,  # Summarize long texts
            "max_length": options.get("summary_max_length", 200),
        },
    }

    return SelectionPlan(
        workflow_id="bias_text",
        reasoning=(
            f"Text input detected ({text_length} chars), "
            "using text bias analysis workflow"
        ),
        confidence=1.0,  # Deterministic in Phase 2
        tool_preferences={
            "llm": llm_model,
            "temperature": temperature,
        },
        node_params=node_params,
        fallbacks=[],  # No fallbacks for deterministic routing
    )


def _create_image_bias_plan(
    content: str | bytes, options: dict[str, Any]
) -> SelectionPlan:
    """Create plan for image bias analysis workflow.

    Parameters
    ----------
    content : str | bytes
        Image bytes to analyze
    options : dict[str, Any]
        User options that override defaults

    Returns
    -------
    SelectionPlan
        Plan configured for image bias workflow (VLM or traditional OCR+Caption)
    """
    # Check if VLM mode is enabled (default as of Phase 6)
    if settings.image_analysis_mode == "vlm":
        return _create_image_bias_vlm_plan(content, options)

    # Traditional mode: OCR + Caption
    # Determine tool preferences
    ocr_tool = options.get("ocr_tool", settings.ocr_tool)
    caption_model = options.get("caption_model", settings.caption_model)
    llm_model = options.get("llm_model", settings.llm_model_name)

    # Node-specific parameters
    node_params = {
        "extract_ocr": {
            "tool": ocr_tool,
            "language": options.get("ocr_language", settings.ocr_language),
            "confidence_threshold": settings.ocr_confidence_threshold,
        },
        "generate_caption": {
            "model": caption_model,
            "max_length": settings.caption_max_length,
        },
        "bias_llm": {
            "model": llm_model,
            "temperature": options.get("temperature", settings.llm_temperature),
        },
    }

    return SelectionPlan(
        workflow_id="bias_image",
        reasoning=(
            "Image input detected, using traditional image bias analysis workflow "
            "(OCR + Caption → LLM analysis)"
        ),
        confidence=1.0,  # Deterministic in Phase 2
        tool_preferences={
            "ocr": ocr_tool,
            "caption": caption_model,
            "llm": llm_model,
        },
        node_params=node_params,
        fallbacks=[
            "bias_text"
        ],  # Fallback: extract text manually and use text workflow
    )


def _create_image_bias_vlm_plan(
    content: str | bytes, options: dict[str, Any]
) -> SelectionPlan:
    """Create plan for VLM-based image bias analysis workflow.

    Uses Vision-Language Models (GPT-4o Vision or Claude Sonnet Vision) with
    Chain-of-Thought reasoning for direct image analysis. Much faster than
    traditional OCR + Caption approach (2-5s vs 30-180s).

    Parameters
    ----------
    content : str | bytes
        Image bytes to analyze
    options : dict[str, Any]
        User options that override defaults

    Returns
    -------
    SelectionPlan
        Plan configured for VLM-based image bias workflow
    """
    # Determine tool preferences
    llm_model = options.get("llm_model", settings.llm_model_name)
    temperature = options.get("temperature", settings.llm_temperature)

    # Node-specific parameters for VLM workflow
    node_params = {
        "visual_analyze": {
            "model": llm_model,
            "temperature": temperature,
            "max_tokens": options.get("vlm_max_tokens", 2000),
        },
        "summarize": {
            "enabled": True,
            "max_length": options.get("summary_max_length", 200),
        },
    }

    return SelectionPlan(
        workflow_id="bias_image_vlm",
        reasoning=(
            "Image input detected, using VLM-based image bias analysis workflow "
            "(Direct visual analysis with Chain-of-Thought reasoning)"
        ),
        confidence=1.0,  # Deterministic in Phase 6+
        tool_preferences={
            "llm": llm_model,
            "temperature": temperature,
        },
        node_params=node_params,
        fallbacks=[
            "bias_image",  # Fallback to traditional OCR+Caption if VLM fails
            "bias_text",
        ],
    )


def _create_risk_plan(content: str | bytes, options: dict[str, Any]) -> SelectionPlan:
    """Create plan for AI risk assessment workflow.

    Parameters
    ----------
    content : str | bytes
        Scenario text to assess
    options : dict[str, Any]
        User options that override defaults

    Returns
    -------
    SelectionPlan
        Plan configured for risk assessment workflow (FAISS-based)
    """
    # Extract scenario length for future heuristics
    scenario = (
        content
        if isinstance(content, str)
        else content.decode("utf-8", errors="ignore")
    )
    scenario_length = len(scenario)

    # Determine tool preferences
    embedding_model = options.get("embedding_model", settings.embedding_model)
    top_k = options.get("top_k", settings.faiss_top_k)

    # Node-specific parameters
    node_params = {
        "embed_scenario": {
            "model": embedding_model,
        },
        "search_risks": {
            "index_path": str(settings.faiss_risks_index_path),
            "top_k": top_k,
        },
        "search_rmf_per_risk": {
            "index_path": str(settings.faiss_rmf_index_path),
            "top_k": 3,  # Top 3 RMF recommendations per risk
        },
    }

    return SelectionPlan(
        workflow_id="risk",
        reasoning=(
            f"CSV/scenario input detected ({scenario_length} chars), "
            "using AI risk assessment workflow (FAISS-based)"
        ),
        confidence=1.0,  # Deterministic in Phase 2
        tool_preferences={
            "embedding": embedding_model,
            "top_k": top_k,
        },
        node_params=node_params,
        fallbacks=[],  # No fallbacks for risk workflow
    )
