"""Fake text generation tool implementations for testing."""

import json
from typing import Any


class FakeLLMTool:
    """Fake LLM tool for testing.

    Returns deterministic bias analysis text. Simulates token counting and
    tracks calls for testing.

    Examples
    --------
    >>> llm = FakeLLMTool()
    >>> result = llm.predict("Analyze this text for bias")
    >>> print(result[:50])
    '**Bias Analysis Report**...'
    >>> print(llm.get_token_count("Sample text"))
    2
    """

    def __init__(self, *, return_text: str | None = None) -> None:
        """Initialize fake LLM with optional fixed return value."""
        self.return_text = return_text
        self.call_count = 0
        self.last_call_args: dict[str, Any] = {}

    def predict(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: int = 60,
    ) -> str:
        """Generate fake LLM prediction.

        Returns structured JSON matching BiasAnalysisOutput schema for
        deterministic behavior that mimics real LLM structured outputs.
        """
        self.call_count += 1
        self.last_call_args = {
            "prompt_len": len(prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }

        if self.return_text is not None:
            return self.return_text

        import hashlib  # noqa: PLC0415

        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        seed = prompt_hash % 100 / 100.0

        return json.dumps(
            {
                "bias_detected": True,
                "bias_instances": [
                    {
                        "type": "age",
                        "severity": "medium",
                        "text_span": "sample text",
                        "visual_element": "",
                        "explanation": (
                            f"Deterministic fake bias instance (seed={seed:.2f})"
                        ),
                        "start_char": 0,
                        "end_char": 11,
                        "evidence_source": "text",
                    },
                ],
                "overall_assessment": (
                    "Fake LLM deterministic output. The text demonstrates "
                    f"potential bias patterns (analysis seed={seed:.2f})."
                ),
                "risk_level": "medium",
            },
            indent=2,
        )

    def get_token_count(self, text: str) -> int:
        """Estimate token count (simple word-based approximation)."""
        return len(text.split())


class FakeSummarizerTool:
    """Fake summarization tool for testing.

    Returns deterministic summaries based on input text.

    Examples
    --------
    >>> summarizer = FakeSummarizerTool()
    >>> summary = summarizer.summarize("Long text here...", max_length=100)
    >>> print(summary)
    '**Summary**: Minimal bias detected...'
    """

    def __init__(self, *, return_summary: str | None = None) -> None:
        """Initialize fake summarizer with optional fixed return value."""
        self.return_summary = return_summary
        self.call_count = 0
        self.last_call_args: dict[str, Any] = {}

    def summarize(
        self,
        text: str,
        max_length: int = 200,
    ) -> str:
        """Generate fake summary."""
        self.call_count += 1
        self.last_call_args = {
            "text_len": len(text),
            "max_length": max_length,
        }

        if self.return_summary is not None:
            return self.return_summary[:max_length]

        summary = (
            "**Summary**: Minimal bias detected. Text is generally inclusive with\n"
            "neutral language across gender, age, and racial dimensions. "
            "Consider adding explicit\n"
            "diversity statements for improvement. Confidence: 0.85"
        )
        return summary[:max_length]
