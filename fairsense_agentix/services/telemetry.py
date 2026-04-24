"""Telemetry and observability service for tracking performance and behavior.

This module provides cross-cutting telemetry capabilities including:
- Timing measurements for operations
- Structured logging with context propagation
- Token counting and cost tracking for LLM calls
- OpenTelemetry integration (when enabled)

Example:
    >>> from fairsense_agentix.services.telemetry import telemetry
    >>> with telemetry.timer("ocr_extraction"):
    ...     result = perform_ocr(image)
    >>> telemetry.log_info("workflow_complete", run_id="abc123", status="success")
"""

import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from fairsense_agentix.configs import settings


# Configure logger for the module
logger = logging.getLogger(__name__)


class TelemetryService:
    """Service for telemetry, logging, and observability.

    This service provides a unified interface for all telemetry needs across
    the application. It can be enabled/disabled via Settings and supports
    both simple logging and OpenTelemetry integration.

    Attributes
    ----------
    enabled : bool
        Whether telemetry is enabled globally
    _current_run_id : str | None
        Current trace/run identifier for context propagation

    """

    def __init__(self, enabled: bool = False, log_level: str = "INFO") -> None:
        """Initialize the telemetry service.

        Parameters
        ----------
        enabled : bool, optional
            Whether to enable telemetry, by default False
        log_level : str, optional
            Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL), by default "INFO"
        """
        self.enabled = enabled
        self._current_run_id: str | None = None
        self._observers: list[Callable[[dict[str, Any]], None]] = []

        # Configure logging level
        log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level_obj,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger.setLevel(log_level_obj)

    @contextmanager
    def timer(
        self,
        operation_name: str,
        **extra_context: Any,
    ) -> Generator[dict[str, float], None, None]:
        """Context manager to time an operation.

        Measures the duration of an operation and logs it if telemetry is enabled.
        Additional context can be provided as keyword arguments.

        Parameters
        ----------
        operation_name : str
            Name of the operation being timed
        **extra_context : Any
            Additional context to include in the log

        Yields
        ------
        dict[str, float]
            Dictionary that will contain 'duration' key after operation completes

        Example
        -------
        >>> with telemetry.timer("embed_text", model="all-MiniLM-L6-v2") as ctx:
        ...     embeddings = embed_model.encode(text)
        >>> print(f"Took {ctx['duration']:.2f}s")
        """
        result_ctx: dict[str, float] = {}
        start_time = time.perf_counter()

        try:
            yield result_ctx
        finally:
            duration = time.perf_counter() - start_time
            result_ctx["duration"] = duration

            payload = self._build_payload(
                level="timer",
                event=operation_name,
                context={
                    **extra_context,
                    "duration_ms": int(duration * 1000),
                },
            )
            self._notify_observers(payload)

            if self.enabled:
                context_str = ", ".join(f"{k}={v}" for k, v in extra_context.items())
                log_msg = f"{operation_name} completed in {duration:.3f}s"
                if context_str:
                    log_msg += f" ({context_str})"
                if payload["run_id"]:
                    log_msg += f" [run_id={payload['run_id']}]"
                logger.info(log_msg)

    def start_trace(self, operation: str) -> str:
        """Start a new trace/run and return its identifier.

        This creates a unique run_id that will be propagated through all
        subsequent telemetry calls until a new trace is started.

        Parameters
        ----------
        operation : str
            Name of the high-level operation being traced

        Returns
        -------
        str
            Unique run identifier

        Example
        -------
        >>> run_id = telemetry.start_trace("analyze_bias_text")
        >>> # All subsequent operations inherit this run_id
        """
        run_id = str(uuid4())
        self._current_run_id = run_id

        if self.enabled:
            logger.info(f"Starting trace: {operation} [run_id={run_id}]")

        return run_id

    def end_trace(self, run_id: str, status: str = "success") -> None:
        """End a trace/run.

        Parameters
        ----------
        run_id : str
            The run identifier to end
        status : str, optional
            Final status of the run (success, failure, timeout), by default "success"
        """
        if self.enabled:
            logger.info(f"Ending trace: [run_id={run_id}] status={status}")

        if self._current_run_id == run_id:
            self._current_run_id = None

    def log_info(self, event: str, **context: Any) -> None:
        """Log an informational event with structured context.

        Parameters
        ----------
        event : str
            Name/description of the event
        **context : Any
            Additional structured context

        Example
        -------
        >>> telemetry.log_info("router_decision", workflow="bias_text", confidence=0.95)
        """
        payload = self._build_payload(
            level="info",
            event=event,
            context=context,
        )
        self._notify_observers(payload)

        if self.enabled:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            log_msg = f"[EVENT] {event}"
            if context_str:
                log_msg += f" | {context_str}"
            if payload["run_id"]:
                log_msg += f" [run_id={payload['run_id']}]"
            logger.info(log_msg)

    def log_warning(self, event: str, **context: Any) -> None:
        """Log a warning event with structured context.

        Parameters
        ----------
        event : str
            Name/description of the warning
        **context : Any
            Additional structured context
        """
        payload = self._build_payload(
            level="warning",
            event=event,
            context=context,
        )
        self._notify_observers(payload)

        if self.enabled:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            log_msg = f"[WARNING] {event}"
            if context_str:
                log_msg += f" | {context_str}"
            if payload["run_id"]:
                log_msg += f" [run_id={payload['run_id']}]"
            logger.warning(log_msg)

    def log_error(
        self,
        event: str,
        error: Exception | None = None,
        **context: Any,
    ) -> None:
        """Log an error event with structured context.

        Parameters
        ----------
        event : str
            Name/description of the error
        error : Exception | None, optional
            The exception that occurred, by default None
        **context : Any
            Additional structured context
        """
        payload = self._build_payload(
            level="error",
            event=event,
            context={**context, "exception": type(error).__name__ if error else None},
        )
        self._notify_observers(payload)

        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        log_msg = f"[ERROR] {event}"
        if context_str:
            log_msg += f" | {context_str}"
        if error:
            log_msg += f" | exception={type(error).__name__}: {error!s}"
        if payload["run_id"]:
            log_msg += f" [run_id={payload['run_id']}]"

        # Always log errors, even if telemetry is disabled
        logger.error(log_msg)

    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration: float,
        cost: float | None = None,
    ) -> None:
        """Record metrics for an LLM API call.

        Parameters
        ----------
        model : str
            Model identifier (e.g., "gpt-4", "claude-3-5-sonnet")
        prompt_tokens : int
            Number of tokens in the prompt
        completion_tokens : int
            Number of tokens in the completion
        duration : float
            Duration of the API call in seconds
        cost : float | None, optional
            Estimated cost in USD, by default None

        Example
        -------
        >>> telemetry.record_llm_call(
        ...     model="gpt-4",
        ...     prompt_tokens=500,
        ...     completion_tokens=150,
        ...     duration=2.3,
        ...     cost=0.045,
        ... )
        """
        payload = self._build_payload(
            level="llm_call",
            event="llm_call",
            context={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "duration": duration,
                "cost": cost,
            },
        )
        self._notify_observers(payload)

        if self.enabled:
            total_tokens = prompt_tokens + completion_tokens
            log_msg = (
                f"[LLM_CALL] model={model}, "
                f"tokens={total_tokens} (prompt={prompt_tokens}, "
                f"completion={completion_tokens}), "
                f"duration={duration:.2f}s"
            )
            if cost is not None:
                log_msg += f", cost=${cost:.4f}"
            if self._current_run_id:
                log_msg += f" [run_id={self._current_run_id}]"
            logger.info(log_msg)

    def register_observer(self, observer: Callable[[dict[str, Any]], None]) -> None:
        """Register an observer to receive telemetry payloads."""
        self._observers.append(observer)

    def unregister_observer(self, observer: Callable[[dict[str, Any]], None]) -> None:
        """Remove a previously registered observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def _build_payload(
        self,
        *,
        level: str,
        event: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Create normalized payload for observers."""
        run_id = context.get("run_id") or self._current_run_id
        return {
            "level": level,
            "event": event,
            "context": context,
            "run_id": run_id,
            "timestamp": time.time(),
        }

    def _notify_observers(self, payload: dict[str, Any]) -> None:
        """Send payload to registered observers."""
        if not self._observers:
            return
        for observer in list(self._observers):
            try:
                observer(payload)
            except Exception:  # pragma: no cover - observer failures are logged only
                logger.debug("Telemetry observer failed", exc_info=True)

    def record_cache_hit(self, operation: str, key: str) -> None:
        """Record a cache hit.

        Parameters
        ----------
        operation : str
            Operation that resulted in cache hit (e.g., "ocr", "embedding")
        key : str
            Cache key (truncated for logging)
        """
        if self.enabled:
            truncated_key = key[:16] + "..." if len(key) > 16 else key
            log_msg = f"[CACHE_HIT] operation={operation}, key={truncated_key}"
            if self._current_run_id:
                log_msg += f" [run_id={self._current_run_id}]"
            logger.debug(log_msg)

    def record_cache_miss(self, operation: str, key: str) -> None:
        """Record a cache miss.

        Parameters
        ----------
        operation : str
            Operation that resulted in cache miss
        key : str
            Cache key (truncated for logging)
        """
        if self.enabled:
            truncated_key = key[:16] + "..." if len(key) > 16 else key
            log_msg = f"[CACHE_MISS] operation={operation}, key={truncated_key}"
            if self._current_run_id:
                log_msg += f" [run_id={self._current_run_id}]"
            logger.debug(log_msg)


# Singleton instance configured from settings
telemetry = TelemetryService(
    enabled=settings.telemetry_enabled,
    log_level=settings.log_level,
)
