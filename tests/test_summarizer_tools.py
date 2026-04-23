"""Tests for summarizer tools.

This module tests the summarizer tool implementations, including protocol
compliance, fake tool behavior, registry resolution, and real tool functionality.
"""

from unittest.mock import MagicMock, Mock

import pytest
from langchain_core.messages import AIMessage

from fairsense_agentix.configs.settings import Settings
from fairsense_agentix.prompts.prompt_loader import PromptLoader
from fairsense_agentix.services.telemetry import TelemetryService
from fairsense_agentix.tools.exceptions import LLMError
from fairsense_agentix.tools.fake import FakeSummarizerTool
from fairsense_agentix.tools.interfaces import SummarizerTool
from fairsense_agentix.tools.registry import reset_tool_registry
from fairsense_agentix.tools.summarizer import LLMSummarizer


pytestmark = pytest.mark.unit


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestProtocolCompliance:
    """Test that summarizer implementations satisfy the protocol."""

    def test_fake_summarizer_satisfies_protocol(self):
        """Verify FakeSummarizerTool implements SummarizerTool protocol."""
        fake_summarizer = FakeSummarizerTool()
        assert isinstance(fake_summarizer, SummarizerTool)

    def test_llm_summarizer_satisfies_protocol(self):
        """Verify LLMSummarizer implements SummarizerTool protocol."""
        mock_model = Mock()
        mock_telemetry = Mock(spec=TelemetryService)
        mock_settings = Mock(spec=Settings)
        mock_prompt_loader = Mock(spec=PromptLoader)

        summarizer = LLMSummarizer(
            langchain_model=mock_model,
            telemetry=mock_telemetry,
            settings=mock_settings,
            prompt_loader=mock_prompt_loader,
        )

        assert isinstance(summarizer, SummarizerTool)


# ============================================================================
# Fake Summarizer Tests
# ============================================================================


class TestFakeSummarizer:
    """Test fake summarizer for unit testing and development."""

    def test_default_behavior(self):
        """Test fake summarizer returns deterministic default summary."""
        summarizer = FakeSummarizerTool()

        text = "This is a long detailed analysis that needs summarizing."
        summary = summarizer.summarize(text, max_length=200)

        assert isinstance(summary, str)
        assert len(summary) <= 200
        assert "Summary" in summary or "bias" in summary.lower()

    def test_custom_return_value(self):
        """Test fake summarizer can be configured with custom return value."""
        custom_summary = "Custom summary for testing purposes"
        summarizer = FakeSummarizerTool(return_summary=custom_summary)

        summary = summarizer.summarize("Any text", max_length=100)

        assert summary == custom_summary

    def test_call_tracking(self):
        """Test fake summarizer tracks method calls."""
        summarizer = FakeSummarizerTool()

        assert summarizer.call_count == 0

        summarizer.summarize("Text 1", max_length=100)
        assert summarizer.call_count == 1

        summarizer.summarize("Text 2", max_length=200)
        assert summarizer.call_count == 2

    def test_last_call_args_tracking(self):
        """Test fake summarizer records last call arguments."""
        summarizer = FakeSummarizerTool()

        text = "Sample text to summarize"
        max_length = 150

        summarizer.summarize(text, max_length=max_length)

        assert summarizer.last_call_args["text_len"] == len(text)
        assert summarizer.last_call_args["max_length"] == max_length

    def test_respects_max_length(self):
        """Test fake summarizer respects max_length parameter."""
        long_summary = "A" * 500
        summarizer = FakeSummarizerTool(return_summary=long_summary)

        summary = summarizer.summarize("Any text", max_length=100)

        assert len(summary) == 100


# ============================================================================
# Registry Resolution Tests
# ============================================================================


class TestRegistryResolution:
    """Test summarizer tool resolution from registry."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_tool_registry()

    def test_fake_provider_returns_fake_summarizer(self):
        """Test registry returns FakeSummarizerTool for fake provider."""
        from fairsense_agentix.tools.registry import (  # noqa: PLC0415
            create_tool_registry,
        )

        settings = Settings(llm_provider="fake")
        registry = create_tool_registry(settings)

        assert isinstance(registry.summarizer, FakeSummarizerTool)

    @pytest.mark.requires_api
    def test_openai_provider_returns_llm_summarizer(self):
        """Test registry returns LLMSummarizer for OpenAI provider."""
        from fairsense_agentix.tools.registry import (  # noqa: PLC0415
            create_tool_registry,
        )

        settings = Settings(
            llm_provider="openai",
            llm_api_key="sk-test-key",  # Required for validation
        )

        registry = create_tool_registry(settings)

        assert isinstance(registry.summarizer, LLMSummarizer)

    @pytest.mark.requires_api
    def test_anthropic_provider_returns_llm_summarizer(self):
        """Test registry returns LLMSummarizer for Anthropic provider."""
        from fairsense_agentix.tools.registry import (  # noqa: PLC0415
            create_tool_registry,
        )

        settings = Settings(
            llm_provider="anthropic",
            llm_api_key="sk-ant-test-key",  # Required for validation
        )

        registry = create_tool_registry(settings)

        assert isinstance(registry.summarizer, LLMSummarizer)


# ============================================================================
# LLM Summarizer Tests (Mocked)
# ============================================================================


class TestLLMSummarizer:
    """Test real LLM summarizer with mocked dependencies."""

    def test_initialization(self):
        """Test LLMSummarizer initializes correctly."""
        mock_model = Mock()
        mock_model.model_name = "gpt-4"
        mock_telemetry = Mock(spec=TelemetryService)
        mock_settings = Mock(spec=Settings)
        mock_prompt_loader = Mock(spec=PromptLoader)

        summarizer = LLMSummarizer(
            langchain_model=mock_model,
            telemetry=mock_telemetry,
            settings=mock_settings,
            prompt_loader=mock_prompt_loader,
        )

        assert summarizer.model == mock_model
        assert summarizer.telemetry == mock_telemetry
        assert summarizer.settings == mock_settings
        assert summarizer.prompt_loader == mock_prompt_loader

    def test_summarize_success(self):
        """Test successful summarization."""
        # Setup mocks
        mock_model = Mock()
        mock_model.model_name = "gpt-4"
        mock_model.invoke.return_value = AIMessage(
            content="Brief summary of the analysis highlighting key findings.",
        )

        mock_telemetry = Mock(spec=TelemetryService)
        mock_telemetry.timer.return_value.__enter__ = Mock(return_value={})
        mock_telemetry.timer.return_value.__exit__ = Mock(return_value=False)

        mock_settings = Mock(spec=Settings)

        mock_prompt_loader = Mock(spec=PromptLoader)
        mock_prompt_loader.load.return_value = "Summarize this text: {text}"

        # Create summarizer
        summarizer = LLMSummarizer(
            langchain_model=mock_model,
            telemetry=mock_telemetry,
            settings=mock_settings,
            prompt_loader=mock_prompt_loader,
        )

        # Summarize text
        text = "This is a detailed analysis that needs to be summarized."
        summary = summarizer.summarize(text, max_length=200)

        # Verify
        assert isinstance(summary, str)
        assert len(summary) <= 200
        assert "summary" in summary.lower() or "brief" in summary.lower()

        # Verify mocks called
        mock_prompt_loader.load.assert_called_once()
        mock_model.invoke.assert_called_once()
        mock_telemetry.timer.assert_called_once()

    def test_summarize_truncates_long_output(self):
        """Test summarizer truncates output exceeding max_length."""
        # Setup mocks with very long response
        long_response = "A" * 500  # 500 character response
        mock_model = Mock()
        mock_model.model_name = "gpt-4"
        mock_model.invoke.return_value = AIMessage(content=long_response)

        mock_telemetry = Mock(spec=TelemetryService)
        mock_telemetry.timer.return_value.__enter__ = Mock(return_value={})
        mock_telemetry.timer.return_value.__exit__ = Mock(return_value=False)

        mock_settings = Mock(spec=Settings)
        mock_prompt_loader = Mock(spec=PromptLoader)
        mock_prompt_loader.load.return_value = "Summarize: {text}"

        summarizer = LLMSummarizer(
            langchain_model=mock_model,
            telemetry=mock_telemetry,
            settings=mock_settings,
            prompt_loader=mock_prompt_loader,
        )

        # Summarize with small max_length
        summary = summarizer.summarize("Any text", max_length=100)

        # Verify truncation
        assert len(summary) == 100
        assert summary == "A" * 100

    def test_empty_text_raises_error(self):
        """Test summarizing empty text raises LLMError."""
        mock_model = Mock()
        mock_telemetry = Mock(spec=TelemetryService)
        mock_settings = Mock(spec=Settings)
        mock_prompt_loader = Mock(spec=PromptLoader)

        summarizer = LLMSummarizer(
            langchain_model=mock_model,
            telemetry=mock_telemetry,
            settings=mock_settings,
            prompt_loader=mock_prompt_loader,
        )

        with pytest.raises(LLMError) as exc_info:
            summarizer.summarize("", max_length=200)

        assert "empty" in str(exc_info.value).lower()
        assert "text_length" in exc_info.value.context

    def test_whitespace_only_text_raises_error(self):
        """Test summarizing whitespace-only text raises LLMError."""
        mock_model = Mock()
        mock_telemetry = Mock(spec=TelemetryService)
        mock_settings = Mock(spec=Settings)
        mock_prompt_loader = Mock(spec=PromptLoader)

        summarizer = LLMSummarizer(
            langchain_model=mock_model,
            telemetry=mock_telemetry,
            settings=mock_settings,
            prompt_loader=mock_prompt_loader,
        )

        with pytest.raises(LLMError) as exc_info:
            summarizer.summarize("   \n\t  ", max_length=200)

        assert (
            "empty" in str(exc_info.value).lower()
            or "whitespace" in str(exc_info.value).lower()
        )

    def test_negative_max_length_raises_error(self):
        """Test negative max_length raises LLMError."""
        mock_model = Mock()
        mock_telemetry = Mock(spec=TelemetryService)
        mock_settings = Mock(spec=Settings)
        mock_prompt_loader = Mock(spec=PromptLoader)

        summarizer = LLMSummarizer(
            langchain_model=mock_model,
            telemetry=mock_telemetry,
            settings=mock_settings,
            prompt_loader=mock_prompt_loader,
        )

        with pytest.raises(LLMError) as exc_info:
            summarizer.summarize("Valid text", max_length=-100)

        assert "positive" in str(exc_info.value).lower()
        assert "max_length" in exc_info.value.context

    def test_zero_max_length_raises_error(self):
        """Test zero max_length raises LLMError."""
        mock_model = Mock()
        mock_telemetry = Mock(spec=TelemetryService)
        mock_settings = Mock(spec=Settings)
        mock_prompt_loader = Mock(spec=PromptLoader)

        summarizer = LLMSummarizer(
            langchain_model=mock_model,
            telemetry=mock_telemetry,
            settings=mock_settings,
            prompt_loader=mock_prompt_loader,
        )

        with pytest.raises(LLMError) as exc_info:
            summarizer.summarize("Valid text", max_length=0)

        assert "positive" in str(exc_info.value).lower()

    def test_llm_error_propagates_with_context(self):
        """Test LLM errors include helpful context."""
        mock_model = Mock()
        mock_model.model_name = "gpt-4"
        mock_model.invoke.side_effect = Exception("API rate limit exceeded")

        mock_telemetry = Mock(spec=TelemetryService)
        mock_telemetry.timer.return_value.__enter__ = Mock(return_value={})
        mock_telemetry.timer.return_value.__exit__ = Mock(return_value=False)

        mock_settings = Mock(spec=Settings)
        mock_prompt_loader = Mock(spec=PromptLoader)
        mock_prompt_loader.load.return_value = "Summarize: {text}"

        summarizer = LLMSummarizer(
            langchain_model=mock_model,
            telemetry=mock_telemetry,
            settings=mock_settings,
            prompt_loader=mock_prompt_loader,
        )

        with pytest.raises(LLMError) as exc_info:
            summarizer.summarize("Any text", max_length=200)

        # Verify error context
        error = exc_info.value
        assert "summarization failed" in str(error).lower()
        assert "text_length" in error.context
        assert "max_length" in error.context
        assert "error_type" in error.context
        assert error.context["error_type"] == "Exception"

    def test_prompt_loader_receives_correct_params(self):
        """Test prompt loader receives text and max_length."""
        mock_model = Mock()
        mock_model.invoke.return_value = AIMessage(content="Summary")

        mock_telemetry = Mock(spec=TelemetryService)
        mock_telemetry.timer.return_value.__enter__ = Mock(return_value={})
        mock_telemetry.timer.return_value.__exit__ = Mock(return_value=False)

        mock_settings = Mock(spec=Settings)

        mock_prompt_loader = Mock(spec=PromptLoader)
        mock_prompt_loader.load.return_value = "Prompt"

        summarizer = LLMSummarizer(
            langchain_model=mock_model,
            telemetry=mock_telemetry,
            settings=mock_settings,
            prompt_loader=mock_prompt_loader,
        )

        text = "Text to summarize"
        max_length = 150
        summarizer.summarize(text, max_length=max_length)

        # Verify prompt_loader called with correct template and params
        mock_prompt_loader.load.assert_called_once_with(
            "summarize_v1",
            text=text,
            max_length=max_length,
        )

    def test_telemetry_timer_called_with_metrics(self):
        """Test telemetry timer receives correct metrics."""
        mock_model = Mock()
        mock_model.model_name = "gpt-4"
        mock_model.invoke.return_value = AIMessage(content="Summary")

        mock_timer_context = MagicMock()
        mock_telemetry = Mock(spec=TelemetryService)
        mock_telemetry.timer.return_value = mock_timer_context

        mock_settings = Mock(spec=Settings)
        mock_prompt_loader = Mock(spec=PromptLoader)
        mock_prompt_loader.load.return_value = "Prompt"

        summarizer = LLMSummarizer(
            langchain_model=mock_model,
            telemetry=mock_telemetry,
            settings=mock_settings,
            prompt_loader=mock_prompt_loader,
        )

        text = "Text to summarize"
        max_length = 150
        summarizer.summarize(text, max_length=max_length)

        # Verify timer called with operation name and metrics
        mock_telemetry.timer.assert_called_once()
        call_args = mock_telemetry.timer.call_args
        assert call_args[0][0] == "summarize"  # operation name
        assert call_args[1]["model"] == "gpt-4"
        assert call_args[1]["input_length"] == len(text)
        assert call_args[1]["max_length"] == max_length
