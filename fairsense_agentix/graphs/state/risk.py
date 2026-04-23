"""State model for the AI risk assessment workflow."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class RiskState(BaseModel):
    """State for AI risk assessment workflow.

    This workflow embeds scenario text, searches for relevant AI risks via
    FAISS, retrieves corresponding AI-RMF (Risk Management Framework)
    recommendations, and outputs results as an HTML table and CSV.

    Note: Quality assessment is handled by orchestrator's posthoc_eval node,
    not by the subgraph itself.

    Attributes
    ----------
    scenario_text : str
        Input scenario to assess for AI risks
    options : dict[str, Any]
        Configuration options (top_k, etc.)
    embedding : list[float] | None
        Vector embedding of scenario text
    risks : list[dict[str, Any]] | None
        Top-k AI risks from FAISS search
    rmf_recommendations : dict[str, list[dict[str, Any]]] | None
        RMF recommendations per risk (keyed by risk ID)
    joined_table : list[dict[str, Any]] | None
        Risks + RMF combined into table rows
    html_table : str | None
        Rendered HTML table
    csv_path : str | None
        Path to generated CSV file
    run_id : str
        Unique identifier for tracing

    Examples
    --------
    >>> state = RiskState(
    ...     scenario_text="Deploying facial recognition in public spaces",
    ...     options={"top_k": 5},
    ... )
    """

    scenario_text: str = Field(description="Input scenario to assess for AI risks")

    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration options for workflow",
    )

    # Embedding
    embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding of scenario text",
    )

    # FAISS search results
    risks: list[dict[str, Any]] | None = Field(
        default=None,
        description="Top-k AI risks from FAISS search",
    )

    rmf_recommendations: dict[str, list[dict[str, Any]]] | None = Field(
        default=None,
        description="RMF recommendations per risk (keyed by risk ID)",
    )

    # Join and formatting
    joined_table: list[dict[str, Any]] | None = Field(
        default=None,
        description="Risks + RMF combined into table rows",
    )

    html_table: str | None = Field(
        default=None,
        description="Rendered HTML table",
    )

    csv_path: str | None = Field(
        default=None,
        description="Path to generated CSV file",
    )

    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for tracing",
    )
