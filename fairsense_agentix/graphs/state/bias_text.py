"""State model for the text bias analysis workflow."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class BiasTextState(BaseModel):
    """State for text bias analysis workflow.

    This workflow analyzes text for various forms of bias (gender, age, race,
    ability, etc.), provides a detailed analysis, optional summary, and
    highlighted HTML output.

    Note: Quality assessment is handled by orchestrator's posthoc_eval node,
    not by the subgraph itself.

    Attributes
    ----------
    text : str
        Input text to analyze for bias
    options : dict[str, Any]
        Configuration options (temperature, prompt_version, etc.)
    bias_analysis : str | None
        Detailed bias analysis from LLM
    summary : str | None
        Optional condensed summary of bias findings
    highlighted_html : str | None
        HTML with bias spans highlighted
    run_id : str
        Unique identifier inherited from orchestrator

    Examples
    --------
    >>> state = BiasTextState(
    ...     text="Sample job posting text", options={"temperature": 0.3}
    ... )
    """

    text: str = Field(description="Input text to analyze for bias")

    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration options for workflow",
    )

    bias_analysis: Any = Field(
        default=None,
        description="Structured bias analysis from LLM (BiasAnalysisOutput object)",
    )

    summary: str | None = Field(
        default=None,
        description="Optional condensed summary of findings",
    )

    highlighted_html: str | None = Field(
        default=None,
        description="HTML with bias spans highlighted",
    )

    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for tracing",
    )
