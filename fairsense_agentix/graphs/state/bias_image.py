"""State model for the image bias analysis workflow."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class BiasImageState(BaseModel):
    """State for image bias analysis workflow.

    This workflow extracts text from images (OCR) and generates captions
    (captioning), then analyzes the combined text for bias. Parallel execution
    of OCR and captioning enables faster processing.

    Note: Quality assessment is handled by orchestrator's posthoc_eval node,
    not by the subgraph itself.

    Attributes
    ----------
    image_bytes : bytes
        Raw image data to analyze
    options : dict[str, Any]
        Configuration options
    ocr_text : str | None
        Text extracted via OCR
    caption_text : str | None
        Generated image caption
    merged_text : str | None
        Combined OCR + caption text (deduplicated)
    bias_analysis : str | None
        Detailed bias analysis from LLM
    summary : str | None
        Optional condensed summary
    highlighted_html : str | None
        HTML with bias spans highlighted
    run_id : str
        Unique identifier for tracing

    Examples
    --------
    >>> state = BiasImageState(image_bytes=b"...", options={"ocr_tool": "tesseract"})
    """

    image_bytes: bytes = Field(description="Raw image data to analyze")

    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration options for workflow",
    )

    # Parallel extraction
    ocr_text: str | None = Field(
        default=None,
        description="Text extracted via OCR",
    )

    caption_text: str | None = Field(
        default=None,
        description="Generated image caption",
    )

    merged_text: str | None = Field(
        default=None,
        description="Combined OCR + caption text",
    )

    # Analysis (same as BiasTextState after merge)
    bias_analysis: Any = Field(
        default=None,
        description="Structured bias analysis from LLM (BiasAnalysisOutput object)",
    )

    summary: str | None = Field(
        default=None,
        description="Optional condensed summary",
    )

    highlighted_html: str | None = Field(
        default=None,
        description="HTML with bias spans highlighted",
    )

    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for tracing",
    )
