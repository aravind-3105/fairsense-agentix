"""Orchestrator nodes: refinement decision and client-facing final result."""

from typing import Any

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.state import OrchestratorState
from fairsense_agentix.services import telemetry


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
                        f"Refinement {state.refinement_count + 1}/"
                        f"{settings.max_refinement_iterations} triggered",
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
                    "output still does not meet quality threshold",
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
                    "apply_refinement_failed",
                    reason="missing_eval_or_plan",
                )
                return {
                    "errors": state.errors
                    + ["Cannot refine: missing evaluation or plan"],
                }

            # Extract refinement hints
            hints = state.posthoc_eval.refinement_hints or {}
            preference_hints = hints.get("tool_preferences")
            option_hints = hints.get("options")

            # Backwards compatibility: legacy hints map directly to preferences
            if preference_hints is None and option_hints is None and hints:
                preference_hints = hints

            updated_preferences = {**state.plan.tool_preferences}
            if preference_hints:
                updated_preferences.update(preference_hints)

            updated_options = {**state.options}
            if option_hints:
                for key, value in option_hints.items():
                    if isinstance(value, list) and isinstance(
                        updated_options.get(key),
                        list,
                    ):
                        updated_options[key] = [
                            *updated_options[key],
                            *value,
                        ]
                    else:
                        updated_options[key] = value

            # Create updated plan (Pydantic models are immutable)
            updated_plan = state.plan.model_copy(
                update={"tool_preferences": updated_preferences},
            )

            telemetry.log_info(
                "orchestrator_refine_complete",
                new_refinement_count=state.refinement_count + 1,
                hints_applied=len(hints),
            )
            return {
                "plan": updated_plan,
                "options": updated_options,
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
            metadata: dict[str, Any] = {
                "plan_reasoning": state.plan.reasoning if state.plan else None,
                "plan_confidence": state.plan.confidence if state.plan else None,
                "preflight_score": (
                    state.preflight_eval.score if state.preflight_eval else None
                ),
                "posthoc_score": state.posthoc_eval.score
                if state.posthoc_eval
                else None,
            }

            if state.posthoc_eval and state.posthoc_eval.metadata:
                metadata["evaluators"] = state.posthoc_eval.metadata

            final_result = {
                "status": status,
                "workflow_id": state.plan.workflow_id if state.plan else None,
                "output": state.workflow_result,
                "refinement_count": state.refinement_count,
                "errors": state.errors,
                "warnings": state.warnings,
                "run_id": state.run_id,
                "metadata": metadata,
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
