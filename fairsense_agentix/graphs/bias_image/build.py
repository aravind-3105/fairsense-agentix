"""Compile the traditional bias-image LangGraph (OCR + caption)."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fairsense_agentix.graphs.bias_image.nodes_analysis import (
    analyze_bias,
    highlight,
    summarize,
)
from fairsense_agentix.graphs.bias_image.nodes_extraction import (
    extract_ocr,
    generate_caption,
    merge_text,
)
from fairsense_agentix.graphs.state import BiasImageState


def create_bias_image_graph() -> CompiledStateGraph:
    """Create and compile the BiasImageGraph subgraph.

    Builds the image bias analysis workflow with parallel OCR and captioning.

    Graph Structure:
        START → extract_ocr (parallel with generate_caption)
        START → generate_caption (parallel with extract_ocr)
        Both → merge_text → analyze_bias → summarize → highlight → END

    Note: Quality assessment is handled by orchestrator's posthoc_eval node.

    Returns
    -------
    StateGraph
        Compiled bias image graph ready for invocation

    Examples
    --------
    >>> graph = create_bias_image_graph()
    >>> result = graph.invoke({"image_bytes": b"fake_image_data", "options": {}})
    >>> result["bias_analysis"]
    '...'
    """
    # Create graph with BiasImageState
    workflow = StateGraph(BiasImageState)

    # Add nodes
    workflow.add_node("extract_ocr", extract_ocr)
    workflow.add_node("generate_caption", generate_caption)
    workflow.add_node("merge_text", merge_text)
    workflow.add_node("analyze_bias", analyze_bias)
    workflow.add_node("summarize", summarize)
    workflow.add_node("highlight", highlight)

    # Add edges for parallel execution
    # START → both OCR and caption (parallel)
    workflow.add_edge(START, "extract_ocr")
    workflow.add_edge(START, "generate_caption")

    # Both parallel nodes → merge (fan-in)
    workflow.add_edge("extract_ocr", "merge_text")
    workflow.add_edge("generate_caption", "merge_text")

    # Sequential after merge
    workflow.add_edge("merge_text", "analyze_bias")
    workflow.add_edge("analyze_bias", "summarize")
    workflow.add_edge("summarize", "highlight")
    workflow.add_edge("highlight", END)

    # Compile graph
    return workflow.compile()
