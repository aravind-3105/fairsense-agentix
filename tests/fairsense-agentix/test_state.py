"""Minimal tests for state models in fairsense_agentix.graphs.state.

These tests verify basic functionality:
- State models instantiate correctly
- Critical validation works (Literal types, ranges)
- Serialization works (for graph state passing)

Comprehensive integration tests will be added after Phase 2 completion.
"""

import pytest
from pydantic import ValidationError

from fairsense_agentix.graphs.state import (
    BiasImageState,
    BiasTextState,
    EvaluationResult,
    OrchestratorState,
    RiskState,
    SelectionPlan,
)


class TestStateModelsBasic:
    """Basic smoke tests for all state models."""

    def test_selection_plan_instantiation(self) -> None:
        """Test SelectionPlan creates with required fields and defaults."""
        plan = SelectionPlan(
            workflow_id="bias_text",
            reasoning="Text input detected",
            confidence=0.95,
        )

        assert plan.workflow_id == "bias_text"
        assert plan.reasoning == "Text input detected"
        assert plan.confidence == 0.95
        assert plan.tool_preferences == {}  # Default
        assert plan.fallbacks == []  # Default

    def test_evaluation_result_instantiation(self) -> None:
        """Test EvaluationResult with pass and fail scenarios."""
        # Passed evaluation
        passed = EvaluationResult(passed=True, score=0.95)
        assert passed.passed is True
        assert passed.score == 0.95
        assert passed.issues == []  # Default

        # Failed evaluation with hints
        failed = EvaluationResult(
            passed=False,
            score=0.6,
            issues=["Missing mitigations"],
            refinement_hints={"temperature": 0.2},
        )
        assert failed.passed is False
        assert len(failed.issues) == 1
        assert "temperature" in failed.refinement_hints

    def test_orchestrator_state_instantiation(self) -> None:
        """Test OrchestratorState creates with minimal data."""
        state = OrchestratorState(
            input_type="text",
            content="Sample text",
        )

        assert state.input_type == "text"
        assert state.content == "Sample text"
        assert state.refinement_count == 0
        assert state.run_id != ""  # Auto-generated UUID
        assert len(state.run_id) == 36  # UUID format

    def test_workflow_states_instantiation(self) -> None:
        """Test BiasTextState, BiasImageState, and RiskState instantiate."""
        # BiasTextState
        text_state = BiasTextState(text="Job posting")
        assert text_state.text == "Job posting"
        assert text_state.bias_analysis is None

        # BiasImageState
        image_state = BiasImageState(image_bytes=b"fake_image")
        assert image_state.image_bytes == b"fake_image"
        assert image_state.ocr_text is None

        # RiskState
        risk_state = RiskState(scenario_text="AI deployment")
        assert risk_state.scenario_text == "AI deployment"
        assert risk_state.embedding is None


class TestCriticalValidation:
    """Test critical field validation that prevents bugs."""

    def test_workflow_id_literal_validation(self) -> None:
        """Test workflow_id only accepts valid values."""
        # Valid
        SelectionPlan(
            workflow_id="bias_text",
            reasoning="test",
            confidence=0.9,
        )

        # Invalid
        with pytest.raises(ValidationError):
            SelectionPlan(
                workflow_id="invalid",  # type: ignore[arg-type]
                reasoning="test",
                confidence=0.9,
            )

    def test_confidence_and_score_range_validation(self) -> None:
        """Test confidence/score must be 0.0-1.0."""
        # Valid
        SelectionPlan(workflow_id="risk", reasoning="test", confidence=0.5)
        EvaluationResult(passed=True, score=0.8)

        # Invalid: out of range
        with pytest.raises(ValidationError):
            SelectionPlan(workflow_id="risk", reasoning="test", confidence=1.5)

        with pytest.raises(ValidationError):
            EvaluationResult(passed=True, score=-0.1)


class TestSerialization:
    """Test state models can be serialized/deserialized."""

    def test_orchestrator_state_json_round_trip(self) -> None:
        """Test OrchestratorState survives JSON serialization."""
        original = OrchestratorState(
            input_type="text",
            content="test",
            refinement_count=2,
        )

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = OrchestratorState.model_validate_json(json_str)

        assert restored.input_type == original.input_type
        assert restored.content == original.content
        assert restored.refinement_count == original.refinement_count
