"""Pydantic schemas for FastAPI surface."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, validator

from fairsense_agentix.api import BiasResult, RiskResult


WorkflowID = Literal["bias_text", "bias_image", "bias_image_vlm", "risk"]


class AnalyzeRequest(BaseModel):
    """Request payload for /v1/analyze."""

    content: str = Field(
        description="Input text or base64 data (for uploads use /upload)"
    )
    input_type: WorkflowID | None = Field(
        default=None,
        description="Optional explicit workflow hint (auto-detected when omitted)",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-request overrides (temperature, faiss_top_k, etc.)",
    )


class AnalyzeStartResponse(BaseModel):
    """Response from /v1/analyze/start with run_id for WebSocket connection."""

    run_id: str
    status: Literal["started"] = "started"
    message: str = "Analysis started. Connect to WebSocket to receive events."


class AnalyzeResponse(BaseModel):
    """Wrapper for analysis outputs."""

    workflow_id: WorkflowID
    run_id: str
    bias_result: BiasResult | None = None
    risk_result: RiskResult | None = None
    metadata: dict[str, Any]

    @classmethod
    def from_bias_result(cls, result: BiasResult) -> "AnalyzeResponse":
        """Create response from a bias workflow result."""
        return cls(
            workflow_id=result.metadata.workflow_id,  # type: ignore[arg-type]
            run_id=result.metadata.run_id,
            bias_result=result,
            metadata=result.metadata.model_dump(),
        )

    @classmethod
    def from_risk_result(cls, result: RiskResult) -> "AnalyzeResponse":
        """Create response from a risk workflow result."""
        return cls(
            workflow_id="risk",
            run_id=result.metadata.run_id,
            risk_result=result,
            metadata=result.metadata.model_dump(),
        )


class BatchItem(BaseModel):
    """Single entry within a batch job."""

    content: str
    input_type: WorkflowID | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class BatchRequest(BaseModel):
    """Request body for /v1/batch."""

    items: list[BatchItem]

    @validator("items")
    @classmethod
    def validate_items(cls, value: list[BatchItem]) -> list[BatchItem]:
        """Ensure batch requests include at least one item."""
        if not value:
            raise ValueError("Batch must include at least one item")
        return value


class BatchStatus(BaseModel):
    """Response describing batch progress."""

    job_id: str
    status: Literal["pending", "running", "completed", "failed"]
    total: int
    completed: int
    errors: list[str]
    results: list[AnalyzeResponse] = Field(default_factory=list)
