"""Orchestrator node: post-workflow quality evaluation."""

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.state import EvaluationResult, OrchestratorState
from fairsense_agentix.services import (
    EvaluationContext,
    evaluate_bias_output,
    telemetry,
)


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
                    "posthoc_eval_failed",
                    reason="no_workflow_result",
                )
                return {
                    "posthoc_eval": EvaluationResult(
                        passed=False,
                        score=0.0,
                        issues=["No workflow result to evaluate"],
                        explanation="Workflow execution failed",
                    ),
                }

            workflow_id = state.workflow_result.get("workflow_id")

            if (
                workflow_id in {"bias_text", "bias_image", "bias_image_vlm"}
                and settings.evaluator_enabled
            ):
                original_text = _extract_bias_source_text(state, workflow_id)
                evaluation = evaluate_bias_output(
                    state.workflow_result,
                    original_text=original_text,
                    options=state.options,
                    context=EvaluationContext(
                        workflow_id=workflow_id,
                        run_id=state.run_id,
                    ),
                )
            elif workflow_id == "risk" and settings.evaluator_enabled:
                # Phase 7.2: Risk evaluator for quality checks
                from fairsense_agentix.services.evaluator import evaluate_risk_output

                evaluation = evaluate_risk_output(
                    state.workflow_result,
                    options=state.options,
                    context=EvaluationContext(
                        workflow_id=workflow_id,
                        run_id=state.run_id,
                    ),
                )
            else:
                evaluation = EvaluationResult(
                    passed=True,
                    score=0.9,
                    issues=[],
                    explanation="Evaluator not required for this workflow",
                )

            telemetry.log_info(
                "orchestrator_posthoc_complete",
                passed=evaluation.passed,
                score=evaluation.score,
                workflow_id=workflow_id,
            )
            return {"posthoc_eval": evaluation}

        except Exception as e:
            telemetry.log_error("orchestrator_posthoc_eval_failed", error=e)
            raise


def _extract_bias_source_text(
    state: OrchestratorState,
    workflow_id: str | None,
) -> str | None:
    """Return best-effort source text for evaluator prompts."""
    if workflow_id == "bias_text" and isinstance(state.content, str):
        return state.content
    if workflow_id == "bias_image":
        merged_text = (
            state.workflow_result.get("merged_text") if state.workflow_result else None
        )
        if isinstance(merged_text, str):
            return merged_text
    if workflow_id == "bias_image_vlm":
        # For VLM mode, use visual_description as source text
        visual_description = (
            state.workflow_result.get("visual_description")
            if state.workflow_result
            else None
        )
        if isinstance(visual_description, str):
            return visual_description
    return None
