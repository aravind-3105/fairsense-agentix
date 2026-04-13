"""Orchestrator supergraph for FairSense-AgentiX.

The orchestrator is an agentic coordinator that implements a ReAct-style reasoning
loop: Reason (Router) → Act (Execute) → Observe (Evaluate) → Reflect (Decide).

It delegates to specialized workflow subgraphs (bias_text, bias_image, risk) and
coordinates evaluators, refinement loops, and error handling.

Graph Structure:
    START → request_plan → preflight_eval → execute_workflow → posthoc_eval
         → decide_action → [accept/refine/fail] → finalize → END

Refinement Loop (Phase 7+):
    When posthoc_eval fails and settings.enable_refinement=True:
        decide_action → apply_refinement → request_plan (loop back)

Implementation modules live under :mod:`fairsense_agentix.graphs.orchestrator`.

Example:
    >>> from fairsense_agentix.graphs.orchestrator_graph import (
    ...     create_orchestrator_graph,
    ... )
    >>> graph = create_orchestrator_graph()
    >>> result = graph.invoke(
    ...     {"input_type": "text", "content": "Sample text", "options": {}}
    ... )
    >>> result["final_result"]
    {...}
"""

from fairsense_agentix.graphs.orchestrator.build import create_orchestrator_graph


__all__ = ["create_orchestrator_graph"]
