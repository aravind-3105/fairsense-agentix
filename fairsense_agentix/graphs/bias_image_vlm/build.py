"""Graph construction for the VLM image bias workflow.

Public entry point
------------------
Use :func:`create_bias_image_vlm_graph` via the stable re-export in
:mod:`fairsense_agentix.graphs.bias_image_vlm_graph`.
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.state import BiasImageVLMState

from .nodes import highlight, summarize, visual_analyze


logger = logging.getLogger(__name__)


def create_bias_image_vlm_graph() -> CompiledStateGraph:
    """Create and compile the VLM-based image bias analysis graph.

    Simple 3-node workflow:
        visual_analyze (VLM CoT) → summarize → highlight

    This graph is 10-60x faster than the traditional OCR+Caption approach
    while providing higher quality bias detection through systematic
    Chain-of-Thought reasoning.

    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready for execution

    Notes
    -----
    Provider routing is automatic:
        - llm_provider="openai" → Uses GPT-4o Vision
        - llm_provider="anthropic" → Uses Claude Sonnet Vision
        - llm_provider="fake" → Uses FakeVLMTool for testing

    The graph reuses existing summarizer and formatter tools,
    demonstrating code reuse across the platform.

    Examples
    --------
    >>> graph = create_bias_image_vlm_graph()
    >>> result = graph.invoke(
    ...     {
    ...         "image_bytes": image_data,
    ...         "options": {"temperature": 0.3},
    ...         "run_id": "test-123",
    ...     }
    ... )
    >>> print(result.vlm_analysis.bias_analysis.overall_assessment)
    """
    workflow = StateGraph(BiasImageVLMState)

    workflow.add_node("visual_analyze", visual_analyze)
    workflow.add_node("summarize", summarize)
    workflow.add_node("highlight", highlight)

    workflow.add_edge(START, "visual_analyze")
    workflow.add_edge("visual_analyze", "summarize")
    workflow.add_edge("summarize", "highlight")
    workflow.add_edge("highlight", END)

    logger.info(
        "Compiled bias_image_vlm_graph: 3 nodes, "
        f"provider={settings.llm_provider}, model={settings.llm_model_name}",
    )

    return workflow.compile()
