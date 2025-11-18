"""Tests for evaluator service helpers."""

from __future__ import annotations

import pytest

from fairsense_agentix.graphs.state import EvaluationResult
from fairsense_agentix.services import EvaluationContext, evaluate_bias_output
from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput, BiasInstance


def _make_bias_workflow_result() -> dict:
    analysis = BiasAnalysisOutput(
        bias_detected=True,
        bias_instances=[
            BiasInstance(
                type="gender",
                severity="high",
                text_span="young rockstar",
                explanation="Gendered language",
                start_char=0,
                end_char=13,
            )
        ],
        overall_assessment="High severity bias",
        risk_level="high",
    )
    return {
        "workflow_id": "bias_text",
        "bias_analysis": analysis,
        "summary": "Bias found",
        "highlighted_html": "<p>...</p>",
    }


def _make_context() -> EvaluationContext:
    return EvaluationContext(workflow_id="bias_text", run_id="test-run")


def test_bias_evaluator_forced_score_failure():
    """Force evaluator to return < threshold and expect refinement hints."""
    workflow_result = _make_bias_workflow_result()
    evaluation = evaluate_bias_output(
        workflow_result,
        original_text="young rockstar developer needed",
        options={"force_bias_eval_score": 60},
        context=_make_context(),
    )

    assert isinstance(evaluation, EvaluationResult)
    assert evaluation.passed is False
    assert evaluation.score == pytest.approx(0.6)
    assert "bias_prompt_feedback" in evaluation.refinement_hints.get("options", {})
    assert "bias" in evaluation.metadata


def test_bias_evaluator_pass_after_feedback():
    """Existing feedback should cause simulated evaluator to pass."""
    workflow_result = _make_bias_workflow_result()
    evaluation = evaluate_bias_output(
        workflow_result,
        original_text="young rockstar developer needed",
        options={"bias_prompt_feedback": ["Already fixed"]},
        context=_make_context(),
    )

    assert evaluation.passed is True
    assert evaluation.score >= 0.75
