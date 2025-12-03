"""Tests for evaluator service helpers."""

from __future__ import annotations

import pytest

from fairsense_agentix.graphs.state import EvaluationResult
from fairsense_agentix.services import EvaluationContext, evaluate_bias_output
from fairsense_agentix.services.evaluator import (
    _check_duplicate_risks,
    _check_risk_coverage,
    _check_rmf_breadth,
    _check_score_sanity,
    _compute_risk_score,
    evaluate_risk_output,
)
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


# ============================================================================
# Phase 7.2: Risk Evaluator Tests
# ============================================================================


class TestRiskEvaluatorHelpers:
    """Tests for risk evaluator helper functions."""

    def test_check_rmf_breadth_pass(self):
        """Test RMF breadth check passes with sufficient function coverage."""
        rmf_recs = {
            "RISK1": [
                {"function": "Govern"},
                {"function": "Measure"},
            ],
            "RISK2": [
                {"function": "Map"},
            ],
        }
        passed, count, reason = _check_rmf_breadth(rmf_recs, min_functions=3)
        assert passed is True
        assert count == 3
        assert "GOVERN" in reason
        assert "MAP" in reason
        assert "MEASURE" in reason

    def test_check_rmf_breadth_fail_narrow(self):
        """Test RMF breadth check fails with narrow function coverage."""
        rmf_recs = {
            "RISK1": [
                {"function": "Govern"},
                {"function": "Govern"},  # Same function
            ],
        }
        passed, count, reason = _check_rmf_breadth(rmf_recs, min_functions=3)
        assert passed is False
        assert count == 1
        assert "Only 1/3" in reason

    def test_check_rmf_breadth_empty(self):
        """Test RMF breadth check handles empty recommendations."""
        passed, count, reason = _check_rmf_breadth({}, min_functions=3)
        assert passed is False
        assert count == 0
        assert "No RMF recommendations" in reason

    def test_check_rmf_breadth_case_normalization(self):
        """Test RMF breadth normalizes function names (Govern vs GOVERN)."""
        rmf_recs = {
            "RISK1": [
                {"function": "govern"},
                {"function": "GOVERN"},
                {"function": "Govern"},
            ],
        }
        passed, count, reason = _check_rmf_breadth(rmf_recs, min_functions=2)
        assert count == 1  # Should deduplicate

    def test_check_duplicate_risks_no_duplicates(self):
        """Test duplicate check passes when no duplicates exist."""
        risks = [
            {"id": "R1", "description": "Privacy risk"},
            {"id": "R2", "description": "Bias risk"},
        ]
        passed, dup_count, pairs = _check_duplicate_risks(risks)
        assert passed is True
        assert dup_count == 0
        assert len(pairs) == 0

    def test_check_duplicate_risks_exact_duplicate(self):
        """Test duplicate check detects exact duplicates."""
        risks = [
            {"id": "R1", "description": "Same risk"},
            {"id": "R2", "description": "Same risk"},
        ]
        passed, dup_count, pairs = _check_duplicate_risks(risks)
        assert passed is False
        assert dup_count == 1
        assert len(pairs) == 1
        assert pairs[0] == ("R1", "R2")

    def test_check_duplicate_risks_single_risk(self):
        """Test duplicate check passes with single risk (can't duplicate)."""
        risks = [{"id": "R1", "description": "Only risk"}]
        passed, dup_count, pairs = _check_duplicate_risks(risks)
        assert passed is True
        assert dup_count == 0

    def test_check_score_sanity_pass(self):
        """Test score sanity check passes with good scores."""
        risks = [
            {"score": 0.8},
            {"score": 0.7},
            {"score": 0.6},
        ]
        passed, avg_score, reason = _check_score_sanity(risks, min_avg_score=0.5)
        assert passed is True
        assert avg_score == pytest.approx(0.7)
        assert "Good FAISS scores" in reason

    def test_check_score_sanity_fail(self):
        """Test score sanity check fails with low scores."""
        risks = [
            {"score": 0.2},
            {"score": 0.1},
            {"score": 0.15},
        ]
        passed, avg_score, reason = _check_score_sanity(risks, min_avg_score=0.3)
        assert passed is False
        assert avg_score == pytest.approx(0.15)
        assert "Low FAISS scores" in reason

    def test_check_score_sanity_percentage_conversion(self):
        """Test score sanity handles percentage scores (0-100)."""
        risks = [
            {"score": 80.0},  # Percentage format
            {"score": 70.0},
        ]
        passed, avg_score, reason = _check_score_sanity(risks, min_avg_score=0.5)
        assert passed is True
        assert avg_score == pytest.approx(0.75)  # Converted to 0-1 scale

    def test_check_risk_coverage_pass(self):
        """Test coverage check passes with sufficient risks."""
        risks = [{"id": "R1"}, {"id": "R2"}, {"id": "R3"}]
        passed, count, reason = _check_risk_coverage(risks, min_risks=2)
        assert passed is True
        assert count == 3
        assert "Sufficient coverage" in reason

    def test_check_risk_coverage_fail(self):
        """Test coverage check fails with insufficient risks."""
        risks = [{"id": "R1"}]
        passed, count, reason = _check_risk_coverage(risks, min_risks=3)
        assert passed is False
        assert count == 1
        assert "Insufficient coverage" in reason

    def test_compute_risk_score_all_pass(self):
        """Test score computation with all checks passing."""
        score = _compute_risk_score(True, True, True, True)
        assert score == 100

    def test_compute_risk_score_partial(self):
        """Test score computation with partial checks passing."""
        score = _compute_risk_score(True, True, False, False)
        assert score == 50  # 2/4 checks = 50%

    def test_compute_risk_score_all_fail(self):
        """Test score computation with all checks failing."""
        score = _compute_risk_score(False, False, False, False)
        assert score == 0


class TestEvaluateRiskOutput:
    """Integration tests for evaluate_risk_output function."""

    def _make_risk_workflow_result(
        self,
        risks: list[dict] | None = None,
        rmf_recs: dict | None = None,
    ) -> dict:
        """Create risk workflow result for testing."""
        if risks is None:
            risks = [
                {"id": "R1", "description": "Privacy risk", "score": 0.8},
                {"id": "R2", "description": "Bias risk", "score": 0.7},
                {"id": "R3", "description": "Safety risk", "score": 0.6},
            ]
        if rmf_recs is None:
            rmf_recs = {
                "R1": [{"function": "Govern"}, {"function": "Measure"}],
                "R2": [{"function": "Map"}],
                "R3": [{"function": "Manage"}],
            }
        return {
            "workflow_id": "risk",
            "risks": risks,
            "rmf_recommendations": rmf_recs,
            "joined_table": [],
        }

    def _make_risk_context(self) -> EvaluationContext:
        """Create evaluation context for testing."""
        return EvaluationContext(workflow_id="risk", run_id="test-run")

    def test_evaluate_risk_output_all_checks_pass(self):
        """Test evaluation passes when all quality checks succeed."""
        workflow_result = self._make_risk_workflow_result()
        evaluation = evaluate_risk_output(
            workflow_result,
            options={},
            context=self._make_risk_context(),
        )

        assert isinstance(evaluation, EvaluationResult)
        assert evaluation.passed is True
        assert evaluation.score == 1.0  # 100% converted to 0-1 scale
        assert len(evaluation.issues) == 0
        assert "risk" in evaluation.metadata

    def test_evaluate_risk_output_rmf_breadth_failure(self):
        """Test evaluation fails with narrow RMF coverage."""
        workflow_result = self._make_risk_workflow_result(
            rmf_recs={
                "R1": [{"function": "Govern"}, {"function": "Govern"}],
            }
        )
        evaluation = evaluate_risk_output(
            workflow_result,
            options={},
            context=self._make_risk_context(),
        )

        assert evaluation.passed is False
        assert any("RMF breadth insufficient" in issue for issue in evaluation.issues)
        assert "top_k" in evaluation.refinement_hints["options"]

    def test_evaluate_risk_output_duplicate_risks(self):
        """Test evaluation fails with duplicate risks."""
        workflow_result = self._make_risk_workflow_result(
            risks=[
                {"id": "R1", "description": "Same risk", "score": 0.8},
                {"id": "R2", "description": "Same risk", "score": 0.7},
            ]
        )
        evaluation = evaluate_risk_output(
            workflow_result,
            options={},
            context=self._make_risk_context(),
        )

        assert evaluation.passed is False
        assert any("Duplicate risks found" in issue for issue in evaluation.issues)

    def test_evaluate_risk_output_low_scores(self):
        """Test evaluation fails with low FAISS scores."""
        workflow_result = self._make_risk_workflow_result(
            risks=[
                {"id": "R1", "description": "Risk 1", "score": 0.1},
                {"id": "R2", "description": "Risk 2", "score": 0.2},
            ]
        )
        evaluation = evaluate_risk_output(
            workflow_result,
            options={},
            context=self._make_risk_context(),
        )

        assert evaluation.passed is False
        assert any("Low FAISS scores" in issue for issue in evaluation.issues)

    def test_evaluate_risk_output_insufficient_coverage(self):
        """Test evaluation fails with too few risks."""
        workflow_result = self._make_risk_workflow_result(
            risks=[{"id": "R1", "description": "Only risk", "score": 0.8}]
        )
        evaluation = evaluate_risk_output(
            workflow_result,
            options={},
            context=self._make_risk_context(),
        )

        assert evaluation.passed is False
        assert any("Insufficient coverage" in issue for issue in evaluation.issues)

    def test_evaluate_risk_output_empty_risks(self):
        """Test evaluation handles empty risk results."""
        workflow_result = self._make_risk_workflow_result(risks=[], rmf_recs={})
        evaluation = evaluate_risk_output(
            workflow_result,
            options={},
            context=self._make_risk_context(),
        )

        assert evaluation.passed is False
        assert evaluation.score == 0.0
        assert "No risks found" in evaluation.issues[0]
        assert evaluation.refinement_hints["options"]["top_k"] == 10  # K-bump

    def test_evaluate_risk_output_k_bump_refinement(self):
        """Test k-bump refinement hint generation."""
        workflow_result = self._make_risk_workflow_result(
            risks=[{"id": "R1", "description": "Risk", "score": 0.1}]
        )
        evaluation = evaluate_risk_output(
            workflow_result,
            options={"top_k": 5},
            context=self._make_risk_context(),
        )

        assert evaluation.passed is False
        assert evaluation.refinement_hints["options"]["top_k"] == 10  # 5 + 5

    def test_evaluate_risk_output_metadata_structure(self):
        """Test evaluation metadata has correct structure."""
        workflow_result = self._make_risk_workflow_result()
        evaluation = evaluate_risk_output(
            workflow_result,
            options={},
            context=self._make_risk_context(),
        )

        assert "risk" in evaluation.metadata
        risk_meta = evaluation.metadata["risk"]
        assert "score" in risk_meta
        assert "function_count" in risk_meta
        assert "duplicate_count" in risk_meta
        assert "avg_faiss_score" in risk_meta
        assert "risk_count" in risk_meta
        assert "checks" in risk_meta
