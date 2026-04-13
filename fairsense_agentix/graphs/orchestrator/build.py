"""Compile the orchestrator LangGraph."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fairsense_agentix.graphs.orchestrator.decision_finalize import (
    apply_refinement,
    decide_action,
    finalize,
)
from fairsense_agentix.graphs.orchestrator.evaluation import posthoc_eval
from fairsense_agentix.graphs.orchestrator.execution import execute_workflow
from fairsense_agentix.graphs.orchestrator.planning import (
    preflight_eval,
    request_plan,
)
from fairsense_agentix.graphs.orchestrator.routing import (
    route_after_decision,
    should_execute_workflow,
)
from fairsense_agentix.graphs.state import OrchestratorState


def create_orchestrator_graph() -> CompiledStateGraph:
    """Create and compile the orchestrator supergraph.

    Builds the complete graph with all nodes, edges, and conditional routing.

    Returns
    -------
    StateGraph
        Compiled orchestrator graph ready for invocation

    Examples
    --------
    >>> graph = create_orchestrator_graph()
    >>> result = graph.invoke(
    ...     {"input_type": "text", "content": "Sample text", "options": {}}
    ... )
    >>> result["final_result"]["status"]
    'success'
    """
    # Create graph with OrchestratorState
    workflow = StateGraph(OrchestratorState)

    # Add nodes
    workflow.add_node("request_plan", request_plan)
    workflow.add_node("preflight_eval", preflight_eval)
    workflow.add_node("execute_workflow", execute_workflow)
    workflow.add_node("posthoc_eval", posthoc_eval)
    workflow.add_node("decide_action", decide_action)
    workflow.add_node("apply_refinement", apply_refinement)
    workflow.add_node("finalize", finalize)

    # Add edges
    # START → request_plan
    workflow.add_edge(START, "request_plan")

    # request_plan → preflight_eval
    workflow.add_edge("request_plan", "preflight_eval")

    # preflight_eval → (conditional) → execute_workflow or finalize
    workflow.add_conditional_edges(
        "preflight_eval",
        should_execute_workflow,
        {
            "execute": "execute_workflow",
            "finalize": "finalize",
        },
    )

    # execute_workflow → posthoc_eval
    workflow.add_edge("execute_workflow", "posthoc_eval")

    # posthoc_eval → decide_action
    workflow.add_edge("posthoc_eval", "decide_action")

    # decide_action → (conditional) → finalize or apply_refinement
    workflow.add_conditional_edges(
        "decide_action",
        route_after_decision,
        {
            "finalize": "finalize",
            "apply_refinement": "apply_refinement",
        },
    )

    # apply_refinement → request_plan (refinement loop)
    workflow.add_edge("apply_refinement", "request_plan")

    # finalize → END
    workflow.add_edge("finalize", END)

    # Compile graph
    return workflow.compile()
