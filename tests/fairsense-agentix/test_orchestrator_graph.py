"""Minimal tests for orchestrator graph.

These tests verify:
- Graph compiles successfully
- Basic end-to-end execution with stub workflows
- Conditional routing (preflight failure, refinement disabled)
- Final result packaging
- VLM vs traditional image analysis routing

Comprehensive integration tests will be added after Phase 2 completion.
"""

import pytest

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.orchestrator_graph import create_orchestrator_graph
from fairsense_agentix.tools import reset_tool_registry


@pytest.mark.integration_test
class TestOrchestratorGraphBasic:
    """Basic smoke tests for orchestrator graph."""

    def test_graph_compiles(self) -> None:
        """Test orchestrator graph compiles without errors."""
        graph = create_orchestrator_graph()
        assert graph is not None
        # Verify graph has expected structure
        assert hasattr(graph, "invoke")

    def test_text_workflow_end_to_end(self) -> None:
        """Test complete execution flow for text input."""
        graph = create_orchestrator_graph()

        # Invoke with text input
        result = graph.invoke(
            {
                "input_type": "text",
                "content": "Sample job posting text",
                "options": {},
            }
        )

        # Verify execution completed
        assert result["final_result"] is not None
        assert result["final_result"]["status"] == "success"
        assert result["final_result"]["workflow_id"] == "bias_text"

        # Verify plan was created
        assert result["plan"] is not None
        assert result["plan"].workflow_id == "bias_text"

        # Verify workflow executed (stub result)
        assert result["workflow_result"] is not None
        assert result["workflow_result"]["workflow_id"] == "bias_text"
        assert "bias_analysis" in result["workflow_result"]

        # Verify evaluations ran
        assert result["preflight_eval"] is not None
        assert result["preflight_eval"].passed is True
        assert result["posthoc_eval"] is not None
        assert result["posthoc_eval"].passed is True

        # Verify decision made
        assert result["decision"] == "accept"

    def test_image_workflow_end_to_end(self) -> None:
        """Test complete execution flow for image input."""
        graph = create_orchestrator_graph()

        # Invoke with image input
        result = graph.invoke(
            {
                "input_type": "image",
                "content": b"fake_image_bytes",
                "options": {"validate_image_bytes": False},
            }
        )

        # Verify execution completed
        assert result["final_result"]["status"] == "success"

        # VLM is now default, so workflow could be either bias_image or bias_image_vlm
        workflow_id = result["final_result"]["workflow_id"]
        assert workflow_id in ["bias_image", "bias_image_vlm"]

        # Verify plan selected image workflow
        assert result["plan"].workflow_id in ["bias_image", "bias_image_vlm"]

        # Verify result has appropriate fields based on workflow
        if workflow_id == "bias_image_vlm":
            assert "vlm_analysis" in result["workflow_result"]
        else:
            assert "ocr_text" in result["workflow_result"]
            assert "caption_text" in result["workflow_result"]

    def test_csv_risk_workflow_end_to_end(self) -> None:
        """Test complete execution flow for CSV/risk input."""
        graph = create_orchestrator_graph()

        # Invoke with CSV input
        result = graph.invoke(
            {
                "input_type": "csv",
                "content": "AI deployment scenario text",
                "options": {},
            }
        )

        # Verify execution completed
        assert result["final_result"]["status"] == "success"
        assert result["final_result"]["workflow_id"] == "risk"

        # Verify plan selected risk workflow
        assert result["plan"].workflow_id == "risk"

        # Verify stub result has risk-specific fields
        assert "risks" in result["workflow_result"]
        assert "rmf_recommendations" in result["workflow_result"]
        assert "html_table" in result["workflow_result"]

    def test_text_workflow_refines_when_evaluator_fails(self) -> None:
        """Bias evaluator score below threshold should trigger refinement loop."""
        graph = create_orchestrator_graph()

        result = graph.invoke(
            {
                "input_type": "text",
                "content": "Sample job posting text with biased language",
                "options": {"force_bias_eval_score": 60},
            }
        )

        assert result["final_result"]["refinement_count"] == 1
        assert result["decision"] == "accept"
        assert result["final_result"]["status"] == "success"
        evaluators_meta = result["final_result"]["metadata"]["evaluators"]
        assert evaluators_meta["bias"]["score"] >= 75


@pytest.mark.integration_test
class TestOrchestratorConditionalRouting:
    """Test conditional edges and routing logic."""

    def test_options_propagate_to_router(self) -> None:
        """Test user options override defaults in router."""
        graph = create_orchestrator_graph()

        # Invoke with custom LLM options
        result = graph.invoke(
            {
                "input_type": "text",
                "content": "Test text",
                "options": {"llm_model": "gpt-4-turbo", "temperature": 0.7},
            }
        )

        # Verify options propagated to plan
        assert result["plan"].tool_preferences["llm"] == "gpt-4-turbo"
        assert result["plan"].tool_preferences["temperature"] == 0.7

        # Verify options in node params
        assert result["plan"].node_params["bias_llm"]["model"] == "gpt-4-turbo"
        assert result["plan"].node_params["bias_llm"]["temperature"] == 0.7


@pytest.mark.integration_test
class TestOrchestratorVLMRouting:
    """Test VLM vs traditional image analysis routing."""

    def setup_method(self) -> None:
        """Reset tool registry before each test."""
        reset_tool_registry()

    def teardown_method(self) -> None:
        """Reset tool registry and settings after each test."""
        reset_tool_registry()
        # Reset to default
        settings.image_analysis_mode = "vlm"

    def test_vlm_mode_routing(self) -> None:
        """Test orchestrator routes to VLM graph when image_analysis_mode='vlm'.

        Default mode is VLM, so this tests the standard path.
        """
        # Ensure VLM mode
        settings.image_analysis_mode = "vlm"
        reset_tool_registry()  # Reset registry after changing mode

        graph = create_orchestrator_graph()

        result = graph.invoke(
            {
                "input_type": "image",
                "content": b"fake_image_bytes",
                "options": {"validate_image_bytes": False},
            }
        )

        # Verify execution completed
        assert result["final_result"]["status"] == "success"
        assert result["final_result"]["workflow_id"] == "bias_image_vlm"

        # Verify VLM-specific fields in result
        workflow_result = result["workflow_result"]
        assert workflow_result["workflow_id"] == "bias_image_vlm"
        assert "vlm_analysis" in workflow_result
        assert "visual_description" in workflow_result
        assert "reasoning_trace" in workflow_result

        # Should NOT have OCR/caption fields
        assert "ocr_text" not in workflow_result
        assert "caption_text" not in workflow_result
        assert "merged_text" not in workflow_result

    def test_traditional_mode_routing(self) -> None:
        """Test orchestrator routes to traditional graph.

        Routes when image_analysis_mode='traditional'.
        """
        # Switch to traditional mode
        settings.image_analysis_mode = "traditional"
        reset_tool_registry()  # Reset to pick up new mode

        graph = create_orchestrator_graph()

        result = graph.invoke(
            {
                "input_type": "image",
                "content": b"fake_image_bytes",
                "options": {"validate_image_bytes": False},
            }
        )

        # Verify execution completed
        assert result["final_result"]["status"] == "success"
        assert result["final_result"]["workflow_id"] == "bias_image"

        # Verify traditional-specific fields in result
        workflow_result = result["workflow_result"]
        assert workflow_result["workflow_id"] == "bias_image"
        assert "ocr_text" in workflow_result
        assert "caption_text" in workflow_result
        assert "merged_text" in workflow_result

        # Should NOT have VLM-specific fields
        assert "vlm_analysis" not in workflow_result
        assert "visual_description" not in workflow_result
        assert "reasoning_trace" not in workflow_result

    def test_vlm_mode_provider_validation(self) -> None:
        """Test VLM mode validates provider.

        Fails gracefully for unsupported providers.
        """
        # Try using VLM mode with unsupported provider
        settings.image_analysis_mode = "vlm"
        original_provider = settings.llm_provider
        settings.llm_provider = "local"  # Unsupported for VLM
        reset_tool_registry()

        graph = create_orchestrator_graph()

        result = graph.invoke(
            {
                "input_type": "image",
                "content": b"fake_image_bytes",
                "options": {"validate_image_bytes": False},
            }
        )

        # Should fail with clear error
        assert result["final_result"]["status"] == "failed"
        assert len(result["errors"]) > 0

        # Error message should mention provider requirement
        error_text = " ".join(result["errors"]).lower()
        assert "vlm" in error_text or "provider" in error_text

        # Restore original provider
        settings.llm_provider = original_provider

    def test_vlm_mode_with_fake_provider(self) -> None:
        """Test VLM mode works with fake provider (for testing)."""
        settings.image_analysis_mode = "vlm"
        settings.llm_provider = "fake"
        reset_tool_registry()

        graph = create_orchestrator_graph()

        result = graph.invoke(
            {
                "input_type": "image",
                "content": b"fake_image_bytes",
                "options": {"validate_image_bytes": False},
            }
        )

        # Should succeed with fake provider
        assert result["final_result"]["status"] == "success"
        assert result["final_result"]["workflow_id"] == "bias_image_vlm"

        # Verify VLM fields present
        assert "visual_description" in result["workflow_result"]
        assert "reasoning_trace" in result["workflow_result"]

    def test_traditional_mode_backward_compatibility(self) -> None:
        """Test traditional mode preserves exact same behavior as before VLM.

        This ensures backward compatibility for users who want
        the original OCR+Caption approach.
        """
        settings.image_analysis_mode = "traditional"
        reset_tool_registry()

        graph = create_orchestrator_graph()

        result = graph.invoke(
            {
                "input_type": "image",
                "content": b"fake_image_bytes",
                "options": {"validate_image_bytes": False},
            }
        )

        # Should have exact same structure as original bias_image workflow
        workflow_result = result["workflow_result"]
        assert workflow_result["workflow_id"] == "bias_image"

        # Check all traditional fields present
        required_fields = [
            "ocr_text",
            "caption_text",
            "merged_text",
            "bias_analysis",
            "highlighted_html",
        ]
        for field in required_fields:
            assert field in workflow_result, f"Missing field: {field}"

        # Optional summary field
        assert "summary" in workflow_result  # May be None
