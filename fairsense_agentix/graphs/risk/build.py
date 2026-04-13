"""Compile the RiskGraph subgraph."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fairsense_agentix.graphs.risk.nodes_output import (
    export_csv,
    format_html,
    join_data,
)
from fairsense_agentix.graphs.risk.nodes_retrieval import (
    embed_scenario,
    search_risks,
    search_rmf_per_risk,
)
from fairsense_agentix.graphs.state import RiskState


def create_risk_graph() -> CompiledStateGraph:
    """Create and compile the RiskGraph subgraph.

    Builds the AI risk assessment workflow with FAISS-based retrieval.

    Graph Structure:
        START → embed_scenario → search_risks → search_rmf_per_risk
             → join_data → format_html → export_csv → END

    Note: Quality assessment is handled by orchestrator's posthoc_eval node.

    Returns
    -------
    StateGraph
        Compiled risk assessment graph ready for invocation

    Examples
    --------
    >>> graph = create_risk_graph()
    >>> result = graph.invoke(
    ...     {"scenario_text": "Deploying facial recognition", "options": {"top_k": 3}}
    ... )
    >>> result["html_table"]
    '<table>...</table>'
    """
    # Create graph with RiskState
    workflow = StateGraph(RiskState)

    # Add nodes
    workflow.add_node("embed_scenario", embed_scenario)
    workflow.add_node("search_risks", search_risks)
    workflow.add_node("search_rmf_per_risk", search_rmf_per_risk)
    workflow.add_node("join_data", join_data)
    workflow.add_node("format_html", format_html)
    workflow.add_node("export_csv", export_csv)

    # Add edges (sequential workflow)
    workflow.add_edge(START, "embed_scenario")
    workflow.add_edge("embed_scenario", "search_risks")
    workflow.add_edge("search_risks", "search_rmf_per_risk")
    workflow.add_edge("search_rmf_per_risk", "join_data")
    workflow.add_edge("join_data", "format_html")
    workflow.add_edge("format_html", "export_csv")
    workflow.add_edge("export_csv", END)

    # Compile graph
    return workflow.compile()
