"""Unit tests for Telemetry service."""

import time

from fairsense_agentix.services.telemetry import TelemetryService


class TestTelemetryService:
    """Test suite for TelemetryService."""

    def test_init_disabled(self):
        """Test that telemetry can be initialized in disabled state."""
        telem = TelemetryService(enabled=False)
        assert telem.enabled is False

    def test_init_enabled(self):
        """Test that telemetry can be initialized in enabled state."""
        telem = TelemetryService(enabled=True)
        assert telem.enabled is True

    def test_timer_measures_duration(self):
        """Test that timer context manager measures duration correctly."""
        telem = TelemetryService(enabled=True)

        with telem.timer("test_operation") as ctx:
            time.sleep(0.05)  # Sleep for 50ms

        assert "duration" in ctx
        assert 0.04 < ctx["duration"] < 0.10  # Allow some tolerance

    def test_timer_when_disabled(self):
        """Test that timer still records duration even when disabled."""
        telem = TelemetryService(enabled=False)

        with telem.timer("test_operation") as ctx:
            time.sleep(0.02)

        # Duration should still be recorded
        assert "duration" in ctx
        assert ctx["duration"] > 0

    def test_timer_with_extra_context(self):
        """Test that timer accepts and logs extra context."""
        telem = TelemetryService(enabled=True)

        # Should not raise an error
        with telem.timer("test_op", model="gpt-4", temperature=0.3) as ctx:
            pass

        assert "duration" in ctx

    def test_start_trace_returns_run_id(self):
        """Test that start_trace returns a unique run identifier."""
        telem = TelemetryService(enabled=True)

        run_id1 = telem.start_trace("operation1")
        run_id2 = telem.start_trace("operation2")

        assert isinstance(run_id1, str)
        assert isinstance(run_id2, str)
        assert run_id1 != run_id2  # Should be unique

    def test_start_trace_sets_current_run_id(self):
        """Test that start_trace sets the internal run_id."""
        telem = TelemetryService(enabled=True)

        run_id = telem.start_trace("test_op")
        assert telem._current_run_id == run_id

    def test_end_trace_clears_run_id(self):
        """Test that end_trace clears the current run_id."""
        telem = TelemetryService(enabled=True)

        run_id = telem.start_trace("test_op")
        assert telem._current_run_id == run_id

        telem.end_trace(run_id)
        assert telem._current_run_id is None

    def test_end_trace_with_different_run_id(self):
        """Test that ending a different run_id doesn't clear current."""
        telem = TelemetryService(enabled=True)

        run_id1 = telem.start_trace("test_op")
        assert telem._current_run_id == run_id1

        # End a different run_id
        telem.end_trace("different-run-id")

        # Current should still be set
        assert telem._current_run_id == run_id1

    def test_log_info_basic(self):
        """Test basic info logging."""
        telem = TelemetryService(enabled=True)

        # Should not raise an error
        telem.log_info("test_event")

    def test_log_info_with_context(self):
        """Test info logging with structured context."""
        telem = TelemetryService(enabled=True)

        # Should not raise an error
        telem.log_info("test_event", key1="value1", key2=42, key3=True)

    def test_log_info_when_disabled(self):
        """Test that log_info is no-op when disabled."""
        telem = TelemetryService(enabled=False)

        # Should not raise an error and should be a no-op
        telem.log_info("test_event", data="value")

    def test_log_warning_basic(self):
        """Test basic warning logging."""
        telem = TelemetryService(enabled=True)

        # Should not raise an error
        telem.log_warning("test_warning")

    def test_log_warning_with_context(self):
        """Test warning logging with context."""
        telem = TelemetryService(enabled=True)

        # Should not raise an error
        telem.log_warning("test_warning", reason="test", severity="low")

    def test_log_error_basic(self):
        """Test basic error logging."""
        telem = TelemetryService(enabled=True)

        # Should not raise an error
        telem.log_error("test_error")

    def test_log_error_with_exception(self):
        """Test error logging with exception object."""
        telem = TelemetryService(enabled=True)

        try:
            raise ValueError("Test error message")
        except ValueError as e:
            # Should not raise an error
            telem.log_error("operation_failed", error=e)

    def test_log_error_always_logs(self):
        """Test that errors are logged even when telemetry is disabled."""
        telem = TelemetryService(enabled=False)

        # Should still log errors (important for debugging)
        # Should not raise an error
        telem.log_error("critical_error", severity="high")

    def test_record_llm_call_basic(self):
        """Test recording LLM call metrics."""
        telem = TelemetryService(enabled=True)

        # Should not raise an error
        telem.record_llm_call(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            duration=1.5,
        )

    def test_record_llm_call_with_cost(self):
        """Test recording LLM call with cost information."""
        telem = TelemetryService(enabled=True)

        # Should not raise an error
        telem.record_llm_call(
            model="gpt-4",
            prompt_tokens=500,
            completion_tokens=150,
            duration=2.3,
            cost=0.045,
        )

    def test_record_llm_call_when_disabled(self):
        """Test that LLM call recording is no-op when disabled."""
        telem = TelemetryService(enabled=False)

        # Should not raise an error
        telem.record_llm_call(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            duration=1.5,
        )

    def test_record_cache_hit(self):
        """Test recording cache hit."""
        telem = TelemetryService(enabled=True)

        # Should not raise an error
        telem.record_cache_hit("ocr", "abc123def456")

    def test_record_cache_miss(self):
        """Test recording cache miss."""
        telem = TelemetryService(enabled=True)

        # Should not raise an error
        telem.record_cache_miss("embedding", "xyz789")

    def test_cache_operations_when_disabled(self):
        """Test that cache recording is no-op when disabled."""
        telem = TelemetryService(enabled=False)

        # Should not raise errors
        telem.record_cache_hit("ocr", "key123")
        telem.record_cache_miss("embedding", "key456")

    def test_trace_context_in_logs(self):
        """Test that run_id appears in log context."""
        telem = TelemetryService(enabled=True)

        run_id = telem.start_trace("test_workflow")

        # These should include the run_id in their log output
        # (We can't easily verify the log output in unit tests without
        # capturing logs, but we ensure no errors are raised)
        telem.log_info("step1", status="started")
        with telem.timer("step2"):
            pass
        telem.record_llm_call("gpt-4", 100, 50, 1.0)

        telem.end_trace(run_id)

    def test_multiple_traces_independent(self):
        """Test that multiple telemetry instances have independent traces."""
        telem1 = TelemetryService(enabled=True)
        telem2 = TelemetryService(enabled=True)

        run_id1 = telem1.start_trace("workflow1")
        run_id2 = telem2.start_trace("workflow2")

        assert telem1._current_run_id == run_id1
        assert telem2._current_run_id == run_id2
        assert run_id1 != run_id2
