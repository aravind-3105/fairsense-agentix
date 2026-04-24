"""Compile the BiasTextGraph subgraph."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fairsense_agentix.graphs.bias_text.nodes import analyze_bias, highlight, summarize
from fairsense_agentix.graphs.bias_text.routing import should_summarize
from fairsense_agentix.graphs.state import BiasTextState


def create_bias_text_graph() -> CompiledStateGraph:
    """Create and compile the BiasTextGraph subgraph.

    Builds the text bias analysis workflow with conditional summarization.

    Graph Structure:
        START → analyze_bias → [summarize (conditional)] → highlight → END

        Conditional: If should_summarize() is True, route to summarize node.
                     Otherwise, skip directly to highlight node.

    Note: Quality assessment is handled by orchestrator's posthoc_eval node.

    Returns
    -------
    StateGraph
        Compiled bias text graph ready for invocation

    Examples
    --------
    >>> graph = create_bias_text_graph()
    >>> result = graph.invoke({"text": "Sample job posting", "options": {}})
    >>> result["bias_analysis"]
    '...'
    """
    # Create graph with BiasTextState
    workflow = StateGraph(BiasTextState)

    # Add nodes
    workflow.add_node("analyze_bias", analyze_bias)
    workflow.add_node("summarize", summarize)
    workflow.add_node("highlight", highlight)

    # Add edges
    # START → analyze_bias
    workflow.add_edge(START, "analyze_bias")

    # analyze_bias → [conditional routing]
    # If should_summarize: analyze_bias → summarize → highlight
    # Else: analyze_bias → highlight (skip summarize)
    workflow.add_conditional_edges(
        "analyze_bias",
        should_summarize,
        {
            True: "summarize",
            False: "highlight",
        },
    )

    # summarize → highlight (when summarize runs)
    workflow.add_edge("summarize", "highlight")

    # highlight → END (always)
    workflow.add_edge("highlight", END)

    # Compile graph
    return workflow.compile()
