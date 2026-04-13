"""Conditional edges for the orchestrator graph."""

from typing import Literal

from fairsense_agentix.graphs.state import OrchestratorState


def should_execute_workflow(state: OrchestratorState) -> Literal["execute", "finalize"]:
    """Conditional edge: Decide whether to execute workflow after preflight.

    If preflight_eval passes, proceed to execution. Otherwise, skip to finalize.

    Parameters
    ----------
    state : OrchestratorState
        Current state with preflight_eval

    Returns
    -------
    Literal["execute", "finalize"]
        Next node to visit
    """
    if state.preflight_eval and state.preflight_eval.passed:
        return "execute"
    return "finalize"


def route_after_decision(
    state: OrchestratorState,
) -> Literal["finalize", "apply_refinement"]:
    """Conditional edge: Route based on decision.

    Routes to:
        - "finalize" if decision is "accept" or "fail"
        - "apply_refinement" if decision is "refine"

    Parameters
    ----------
    state : OrchestratorState
        Current state with decision

    Returns
    -------
    Literal["finalize", "apply_refinement"]
        Next node to visit
    """
    if state.decision == "refine":
        return "apply_refinement"
    # "accept" or "fail" both go to finalize
    return "finalize"
