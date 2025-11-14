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

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.bias_image_graph import create_bias_image_graph
from fairsense_agentix.graphs.bias_text_graph import create_bias_text_graph
from fairsense_agentix.graphs.risk_graph import create_risk_graph
from fairsense_agentix.graphs.state import EvaluationResult, OrchestratorState
from fairsense_agentix.services.router import create_selection_plan
from fairsense_agentix.services.telemetry import telemetry


# ============================================================================
# Node Functions
# ============================================================================


def request_plan(state: OrchestratorState) -> dict:
    """Router node: Analyze input and create execution plan.

    Calls the external router module to generate a SelectionPlan that tells
    the orchestrator which workflow to run and how to configure it.

    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : OrchestratorState
        Current orchestrator state with input_type, content, options

    Returns
    -------
    dict
        State update with plan field populated

    Examples
    --------
    >>> state = OrchestratorState(input_type="text", content="test", options={})
    >>> update = request_plan(state)
    >>> update["plan"].workflow_id
    'bias_text'
    """
    with telemetry.timer("orchestrator.request_plan", input_type=state.input_type):
        telemetry.log_info(
            "orchestrator_plan_start",
            input_type=state.input_type,
            run_id=state.run_id or "unknown",
            refinement_count=state.refinement_count,
        )

        try:
            with telemetry.timer("orchestrator.router_service"):
                plan = create_selection_plan(
                    input_type=state.input_type,
                    content=state.content,
                    options=state.options,
                )

            telemetry.log_info(
                "orchestrator_plan_complete",
                workflow_id=plan.workflow_id,
                confidence=plan.confidence,
            )
            return {"plan": plan}
        except Exception as e:
            telemetry.log_error("orchestrator_request_plan_failed", error=e)
            return {
                "plan": None,
                "errors": state.errors + [f"Router failed: {e!s}"],
            }


def preflight_eval(state: OrchestratorState) -> dict:
    """Validate plan before execution (pre-flight check).

    In Phase 2-6, this is a stub that always passes. In Phase 7+, this can
    check for invalid configurations, missing resources, etc.

    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : OrchestratorState
        Current state with plan

    Returns
    -------
    dict
        State update with preflight_eval result

    Examples
    --------
    >>> state = OrchestratorState(input_type="text", content="test", plan=...)
    >>> update = preflight_eval(state)
    >>> update["preflight_eval"].passed
    True
    """
    with telemetry.timer("orchestrator.preflight_eval"):
        telemetry.log_info(
            "orchestrator_preflight_start",
            run_id=state.run_id or "unknown",
            has_plan=state.plan is not None,
        )

        try:
            # Phase 2-6: Always pass (stub)
            # Phase 7+: Add validation logic (check plan.confidence, resources, etc.)

            if state.plan is None:
                telemetry.log_warning("preflight_eval_failed", reason="no_plan")
                return {
                    "preflight_eval": EvaluationResult(
                        passed=False,
                        score=0.0,
                        issues=["No plan available"],
                        explanation="Router failed to create a plan",
                    )
                }

            # Stub: Always pass
            telemetry.log_info(
                "orchestrator_preflight_complete",
                passed=True,
                workflow_id=state.plan.workflow_id,
            )
            return {
                "preflight_eval": EvaluationResult(
                    passed=True,
                    score=1.0,
                    issues=[],
                    explanation=f"Plan validation passed for workflow '{state.plan.workflow_id}'",
                )
            }

        except Exception as e:
            telemetry.log_error("orchestrator_preflight_eval_failed", error=e)
            raise


def execute_workflow(state: OrchestratorState) -> dict:
    """Executor node: Delegate to appropriate workflow subgraph.

    Routes to bias_text, bias_image, or risk subgraphs based on plan.workflow_id.

    Checkpoint 2.4: bias_text uses real BiasTextGraph
    Checkpoint 2.5: bias_image will use real BiasImageGraph
    Checkpoint 2.6: risk will use real RiskGraph
    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : OrchestratorState
        Current state with plan

    Returns
    -------
    dict
        State update with workflow_result

    Examples
    --------
    >>> state = OrchestratorState(input_type="text", content="test", plan=...)
    >>> update = execute_workflow(state)
    >>> update["workflow_result"]["workflow_id"]
    'bias_text'
    """
    with telemetry.timer(
        "orchestrator.execute_workflow",
        workflow_id=state.plan.workflow_id if state.plan else "none",
    ):
        telemetry.log_info(
            "orchestrator_execute_start",
            run_id=state.run_id or "unknown",
            workflow_id=state.plan.workflow_id if state.plan else None,
        )

        if state.plan is None:
            telemetry.log_error("execute_workflow_failed", reason="no_plan")
            return {
                "workflow_result": None,
                "errors": state.errors + ["Cannot execute: no plan available"],
            }

        workflow_id = state.plan.workflow_id

        try:
            if workflow_id == "bias_text":
                # Checkpoint 2.4: Real BiasTextGraph
                with telemetry.timer("orchestrator.bias_text_subgraph"):
                    text_graph = create_bias_text_graph()

                    # Extract text content
                    text = (
                        state.content
                        if isinstance(state.content, str)
                        else state.content.decode("utf-8", errors="ignore")
                    )

                    # Invoke subgraph with options from plan
                    subgraph_result = text_graph.invoke(
                        {
                            "text": text,
                            "options": state.options,
                            "run_id": state.run_id,  # Propagate run_id for tracing
                        }
                    )

                # Package result
                # Note: summary is optional (conditional node in BiasTextGraph)
                result = {
                    "workflow_id": "bias_text",
                    "bias_analysis": subgraph_result["bias_analysis"],
                    "summary": subgraph_result.get("summary"),  # Optional field
                    "highlighted_html": subgraph_result["highlighted_html"],
                }

            elif workflow_id == "bias_image":
                # Checkpoint 2.5: Real BiasImageGraph
                with telemetry.timer("orchestrator.bias_image_subgraph"):
                    image_graph = create_bias_image_graph()

                    # Extract image bytes
                    image_bytes = (
                        state.content
                        if isinstance(state.content, bytes)
                        else state.content.encode("utf-8")
                    )

                    # Invoke subgraph with options from plan
                    subgraph_result = image_graph.invoke(
                        {
                            "image_bytes": image_bytes,
                            "options": state.options,
                            "run_id": state.run_id,  # Propagate run_id for tracing
                        }
                    )

                # Package result
                # Note: summary is optional (conditional node in BiasImageGraph)
                result = {
                    "workflow_id": "bias_image",
                    "ocr_text": subgraph_result["ocr_text"],
                    "caption_text": subgraph_result["caption_text"],
                    "merged_text": subgraph_result["merged_text"],
                    "bias_analysis": subgraph_result["bias_analysis"],
                    "summary": subgraph_result.get("summary"),  # Optional field
                    "highlighted_html": subgraph_result["highlighted_html"],
                }
            elif workflow_id == "risk":
                # Checkpoint 2.6: Real RiskGraph
                with telemetry.timer("orchestrator.risk_subgraph"):
                    risk_graph = create_risk_graph()

                    # Extract scenario text
                    scenario_text = (
                        state.content
                        if isinstance(state.content, str)
                        else state.content.decode("utf-8", errors="ignore")
                    )

                    # Invoke subgraph with options from plan
                    subgraph_result = risk_graph.invoke(
                        {
                            "scenario_text": scenario_text,
                            "options": state.options,
                            "run_id": state.run_id,  # Propagate run_id for tracing
                        }
                    )

                # Package result
                result = {
                    "workflow_id": "risk",
                    "embedding": subgraph_result["embedding"],
                    "risks": subgraph_result["risks"],
                    "rmf_recommendations": subgraph_result["rmf_recommendations"],
                    "joined_table": subgraph_result["joined_table"],
                    "html_table": subgraph_result["html_table"],
                    "csv_path": subgraph_result["csv_path"],
                }
            else:
                telemetry.log_error("unknown_workflow_id", workflow_id=workflow_id)
                raise ValueError(f"Unknown workflow_id: {workflow_id}")

            telemetry.log_info("orchestrator_execute_complete", workflow_id=workflow_id)
            return {"workflow_result": result}

        except Exception as e:
            telemetry.log_error(
                "orchestrator_execute_workflow_failed", workflow_id=workflow_id, error=e
            )
            return {
                "workflow_result": None,
                "errors": state.errors + [f"Workflow execution failed: {e!s}"],
            }


def posthoc_eval(state: OrchestratorState) -> dict:
    """Assess output quality after execution.

    The orchestrator owns all quality evaluation. Subgraphs produce output,
    orchestrator evaluates whether it meets quality standards.

    Phase 2-6: Stub that always passes with default score.
    Phase 7+: Real LLM-based quality assessment with refinement hints.
    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : OrchestratorState
        Current state with workflow_result

    Returns
    -------
    dict
        State update with posthoc_eval result

    Examples
    --------
    >>> state = OrchestratorState(..., workflow_result={...})
    >>> update = posthoc_eval(state)
    >>> update["posthoc_eval"].passed
    True
    """
    with telemetry.timer("orchestrator.posthoc_eval"):
        telemetry.log_info(
            "orchestrator_posthoc_start",
            run_id=state.run_id or "unknown",
            has_result=state.workflow_result is not None,
        )

        try:
            # Phase 2-6: Always pass (stub)
            # Phase 7+: Add LLM-based quality assessment:
            #   - Evaluate completeness, accuracy, actionability
            #   - Return refinement hints if quality is low
            #   - Enable refinement loop via settings.enable_refinement

            if state.workflow_result is None:
                telemetry.log_warning(
                    "posthoc_eval_failed", reason="no_workflow_result"
                )
                return {
                    "posthoc_eval": EvaluationResult(
                        passed=False,
                        score=0.0,
                        issues=["No workflow result to evaluate"],
                        explanation="Workflow execution failed",
                    )
                }

            # Stub: Always pass with default score
            telemetry.log_info("orchestrator_posthoc_complete", passed=True, score=0.85)
            return {
                "posthoc_eval": EvaluationResult(
                    passed=True,
                    score=0.85,  # Default acceptable quality
                    issues=[],
                    explanation="Phase 2-6 stub: Output accepted without evaluation",
                )
            }

        except Exception as e:
            telemetry.log_error("orchestrator_posthoc_eval_failed", error=e)
            raise


def decide_action(state: OrchestratorState) -> dict:
    """Decision node: Determine whether to accept, refine, or fail.

    Implements the "Reflect" step of the ReAct loop. Based on posthoc_eval
    results and refinement settings, decides the next action.

    Decision Logic:
        - If refinement disabled (Phase 2-6): Always accept
        - If refinement enabled (Phase 7+):
            - posthoc_eval.passed → accept
            - posthoc_eval.passed=False AND refinement_count < max → refine
            - posthoc_eval.passed=False AND refinement_count >= max → fail

    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : OrchestratorState
        Current state with posthoc_eval

    Returns
    -------
    dict
        State update with decision field

    Examples
    --------
    >>> state = OrchestratorState(..., posthoc_eval=EvaluationResult(passed=True, ...))
    >>> update = decide_action(state)
    >>> update["decision"]
    'accept'
    """
    with telemetry.timer("orchestrator.decide_action"):
        telemetry.log_info(
            "orchestrator_decide_start",
            run_id=state.run_id or "unknown",
            refinement_count=state.refinement_count,
            refinement_enabled=settings.enable_refinement,
        )

        try:
            if state.posthoc_eval is None:
                telemetry.log_error("decide_action_failed", reason="no_posthoc_eval")
                return {
                    "decision": "fail",
                    "errors": state.errors + ["Cannot decide: no posthoc evaluation"],
                }

            # Phase 2-6: Refinement disabled, always accept
            if not settings.enable_refinement:
                telemetry.log_info(
                    "orchestrator_decide_complete",
                    decision="accept",
                    reason="refinement_disabled",
                )
                return {"decision": "accept"}

            # Phase 7+: Refinement enabled, make decision based on evaluation
            if state.posthoc_eval.passed:
                telemetry.log_info(
                    "orchestrator_decide_complete",
                    decision="accept",
                    reason="eval_passed",
                )
                return {"decision": "accept"}

            # Evaluation failed - check if we can refine
            if state.refinement_count < settings.max_refinement_iterations:
                telemetry.log_info(
                    "orchestrator_decide_complete",
                    decision="refine",
                    refinement_count=state.refinement_count + 1,
                )
                return {
                    "decision": "refine",
                    "warnings": state.warnings
                    + [
                        f"Refinement {state.refinement_count + 1}/{settings.max_refinement_iterations} triggered"
                    ],
                }

            telemetry.log_warning(
                "orchestrator_decide_complete",
                decision="fail",
                reason="max_refinements_reached",
            )
            return {
                "decision": "fail",
                "errors": state.errors
                + [
                    f"Max refinements ({settings.max_refinement_iterations}) reached, "
                    "output still does not meet quality threshold"
                ],
            }

        except Exception as e:
            telemetry.log_error("orchestrator_decide_action_failed", error=e)
            raise


def apply_refinement(state: OrchestratorState) -> dict:
    """Refinement node: Update plan based on evaluator feedback.

    Extracts refinement hints from posthoc_eval and updates the plan's
    tool_preferences and node_params. This enables the orchestrator to
    try again with adjusted parameters.

    Phase 2-6: This node is unreachable (refinement disabled).
    Phase 7+: Applies refinement hints and increments refinement_count.
    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : OrchestratorState
        Current state with posthoc_eval and plan

    Returns
    -------
    dict
        State update with modified plan and incremented refinement_count

    Examples
    --------
    >>> state = OrchestratorState(
    ...     ...,
    ...     posthoc_eval=EvaluationResult(
    ...         passed=False, refinement_hints={"temperature": 0.2}
    ...     ),
    ... )
    >>> update = apply_refinement(state)
    >>> update["refinement_count"]
    1
    """
    with telemetry.timer("orchestrator.apply_refinement"):
        telemetry.log_info(
            "orchestrator_refine_start",
            run_id=state.run_id or "unknown",
            current_refinement_count=state.refinement_count,
        )

        try:
            if state.posthoc_eval is None or state.plan is None:
                telemetry.log_error(
                    "apply_refinement_failed", reason="missing_eval_or_plan"
                )
                return {
                    "errors": state.errors
                    + ["Cannot refine: missing evaluation or plan"]
                }

            # Extract refinement hints
            hints = state.posthoc_eval.refinement_hints

            # Update plan with hints (merge into tool_preferences)
            updated_preferences = {**state.plan.tool_preferences, **hints}

            # Create updated plan (Pydantic models are immutable)
            updated_plan = state.plan.model_copy(
                update={"tool_preferences": updated_preferences}
            )

            telemetry.log_info(
                "orchestrator_refine_complete",
                new_refinement_count=state.refinement_count + 1,
                hints_applied=len(hints),
            )
            return {
                "plan": updated_plan,
                "refinement_count": state.refinement_count + 1,
            }

        except Exception as e:
            telemetry.log_error("orchestrator_apply_refinement_failed", error=e)
            raise


def finalize(state: OrchestratorState) -> dict:
    """Package result for client.

    Constructs the final result dictionary with all relevant outputs,
    metadata, and observability information.

    Phase 6.4: Added telemetry for observability and performance tracking.

    Parameters
    ----------
    state : OrchestratorState
        Final state with workflow_result

    Returns
    -------
    dict
        State update with final_result field

    Examples
    --------
    >>> state = OrchestratorState(..., workflow_result={...}, decision="accept")
    >>> update = finalize(state)
    >>> update["final_result"]["status"]
    'success'
    """
    with telemetry.timer("orchestrator.finalize"):
        telemetry.log_info(
            "orchestrator_finalize_start",
            run_id=state.run_id or "unknown",
            decision=state.decision,
            refinement_count=state.refinement_count,
        )

        try:
            # Determine status based on decision
            if state.decision == "accept":
                status = "success"
            elif state.decision == "fail":
                status = "failed"
            else:
                # Should not reach here, but handle gracefully
                status = "unknown"

            # Package final result
            final_result = {
                "status": status,
                "workflow_id": state.plan.workflow_id if state.plan else None,
                "output": state.workflow_result,
                "refinement_count": state.refinement_count,
                "errors": state.errors,
                "warnings": state.warnings,
                "run_id": state.run_id,
                # Metadata for observability
                "metadata": {
                    "plan_reasoning": state.plan.reasoning if state.plan else None,
                    "plan_confidence": state.plan.confidence if state.plan else None,
                    "preflight_score": (
                        state.preflight_eval.score if state.preflight_eval else None
                    ),
                    "posthoc_score": state.posthoc_eval.score
                    if state.posthoc_eval
                    else None,
                },
            }

            telemetry.log_info(
                "orchestrator_finalize_complete",
                status=status,
                workflow_id=final_result["workflow_id"],
                error_count=len(state.errors),
                warning_count=len(state.warnings),
            )
            return {"final_result": final_result}

        except Exception as e:
            telemetry.log_error("orchestrator_finalize_failed", error=e)
            raise


# ============================================================================
# Conditional Edge Functions
# ============================================================================


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


# ============================================================================
# Graph Construction
# ============================================================================


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
