"""Minimal tests for router module.

These tests verify:
- Router correctly maps input_type to workflow_id
- SelectionPlan contains required fields (reasoning, confidence, etc.)
- Options override default tool preferences

Comprehensive integration tests will be added after Phase 2 completion.
"""

import pytest

from fairsense_agentix.services.router import create_selection_plan


pytestmark = pytest.mark.unit


class TestRouterBasic:
    """Basic routing tests for each input type."""

    def test_text_input_routes_to_bias_text(self) -> None:
        """Test text input selects bias_text workflow."""
        plan = create_selection_plan(
            input_type="text",
            content="Sample job posting text",
            options={},
        )

        assert plan.workflow_id == "bias_text"
        assert plan.confidence == 1.0  # Deterministic routing
        assert "text" in plan.reasoning.lower()
        assert "llm" in plan.tool_preferences
        assert "bias_llm" in plan.node_params

    def test_image_input_routes_to_bias_image(self) -> None:
        """Test image input selects bias_image_vlm workflow.

        VLM is default as of Phase 6.
        """
        plan = create_selection_plan(
            input_type="image",
            content=b"fake_image_bytes",
            options={},
        )

        # VLM is now the default mode for image analysis
        assert plan.workflow_id == "bias_image_vlm"
        assert plan.confidence == 1.0
        assert "image" in plan.reasoning.lower()
        # VLM mode uses llm instead of ocr+caption
        assert "llm" in plan.tool_preferences
        assert "visual_analyze" in plan.node_params
        assert plan.fallbacks == ["bias_image", "bias_text"]  # VLM can fall back

    def test_csv_input_routes_to_risk(self) -> None:
        """Test CSV input selects risk workflow."""
        plan = create_selection_plan(
            input_type="csv",
            content="AI deployment scenario text",
            options={},
        )

        assert plan.workflow_id == "risk"
        assert plan.confidence == 1.0
        assert "risk" in plan.reasoning.lower() or "scenario" in plan.reasoning.lower()
        assert "embedding" in plan.tool_preferences
        assert "embed_scenario" in plan.node_params
        assert "search_risks" in plan.node_params


class TestRouterOptions:
    """Test that user options override defaults."""

    def test_options_override_llm_model(self) -> None:
        """Test user can override LLM model via options."""
        plan = create_selection_plan(
            input_type="text",
            content="Test text",
            options={"llm_model": "gpt-4-turbo", "temperature": 0.5},
        )

        assert plan.tool_preferences["llm"] == "gpt-4-turbo"
        assert plan.tool_preferences["temperature"] == 0.5
        assert plan.node_params["bias_llm"]["model"] == "gpt-4-turbo"
        assert plan.node_params["bias_llm"]["temperature"] == 0.5
