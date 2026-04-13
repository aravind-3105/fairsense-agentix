"""Orchestrator nodes: routing plan and preflight."""

from fairsense_agentix.graphs.state import EvaluationResult, OrchestratorState
from fairsense_agentix.services import telemetry
from fairsense_agentix.services.router import create_selection_plan


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
                    ),
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
                    explanation=(
                        f"Plan validation passed for workflow "
                        f"'{state.plan.workflow_id}'"
                    ),
                ),
            }

        except Exception as e:
            telemetry.log_error("orchestrator_preflight_eval_failed", error=e)
            raise
