"""Minimal tests for orchestrator graph.

These tests verify:
- Graph compiles successfully
- Basic end-to-end execution with stub workflows
- Conditional routing (preflight failure, refinement disabled)
- Final result packaging

Comprehensive integration tests will be added after Phase 2 completion.
"""

from fairsense_agentix.graphs.orchestrator_graph import create_orchestrator_graph


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
                "options": {},
            }
        )

        # Verify execution completed
        assert result["final_result"]["status"] == "success"
        assert result["final_result"]["workflow_id"] == "bias_image"

        # Verify plan selected image workflow
        assert result["plan"].workflow_id == "bias_image"

        # Verify stub result has image-specific fields
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
