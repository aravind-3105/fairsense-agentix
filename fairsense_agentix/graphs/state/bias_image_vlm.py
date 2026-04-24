"""State model for the VLM-based image bias analysis workflow."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class BiasImageVLMState(BaseModel):
    """State for VLM-based image bias analysis workflow.

    This workflow uses Vision-Language Models (GPT-4o Vision or Claude Sonnet Vision)
    with Chain-of-Thought reasoning for direct image analysis. Much faster than
    traditional OCR + Caption approach (2-5s vs 30-180s).

    The VLM analyzes images directly with embedded reasoning traces, eliminating
    the need for text extraction intermediate steps.

    Note: Quality assessment is handled by orchestrator's posthoc_eval node.

    Attributes
    ----------
    image_bytes : bytes
        Raw image data to analyze
    options : dict[str, Any]
        Configuration options
    vlm_analysis : BiasVisualAnalysisOutput | None
        VLM output with CoT traces and bias analysis
    summary : str | None
        Optional condensed summary
    highlighted_html : str | None
        HTML visualization of bias findings
    image_base64 : str | None
        Base64-encoded image with data URL for UI display
    run_id : str
        Unique identifier for tracing

    Examples
    --------
    >>> state = BiasImageVLMState(image_bytes=b"...", options={"temperature": 0.3})
    >>> # After visual_analyze node:
    >>> state.vlm_analysis.visual_description
    'Image shows 3 people in office setting...'
    >>> state.vlm_analysis.bias_analysis.bias_detected
    True
    >>> # After highlight node:
    >>> state.image_base64
    'data:image/jpeg;base64,/9j/4AAQSkZJRg...'
    """

    image_bytes: bytes = Field(description="Raw image data to analyze")

    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration options for workflow",
    )

    # VLM Analysis (single call with embedded CoT)
    vlm_analysis: Any = Field(
        default=None,
        description=(
            "VLM output with reasoning traces (BiasVisualAnalysisOutput object)"
        ),
    )

    # Post-processing (reuses existing formatter/summarizer)
    summary: str | None = Field(
        default=None,
        description="Optional condensed summary of findings",
    )

    highlighted_html: str | None = Field(
        default=None,
        description="HTML visualization of bias findings",
    )

    image_base64: str | None = Field(
        default=None,
        description=(
            "Base64-encoded image with data URL for UI display "
            "(data:image/...;base64,...)"
        ),
    )

    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for tracing",
    )
