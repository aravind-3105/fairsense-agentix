"""High-level API for FairSense-AgentiX.

This module provides a clean, user-friendly interface for bias detection and
risk assessment. It wraps the internal graph orchestration with Pydantic models
that are easy to use in scripts, FastAPI servers, and Streamlit apps.

Examples
--------
Simple usage (quick scripts):

    >>> from fairsense_agentix import analyze_text
    >>> result = analyze_text("Job posting text")
    >>> print(result.bias_detected)
    >>> print(result.highlighted_html)

Class-based usage (reusable, performant):

    >>> from fairsense_agentix import FairSense
    >>> fs = FairSense()  # Initialize once, reuse for many calls
    >>> result = fs.analyze_text("Text 1", temperature=0.3)
    >>> result2 = fs.analyze_text("Text 2", temperature=0.3)

FastAPI integration:

    >>> from fastapi import FastAPI
    >>> from fairsense_agentix import FairSense, BiasResult
    >>> app = FastAPI()
    >>> fs = FairSense()
    >>> @app.post("/analyze", response_model=BiasResult)
    >>> def analyze(text: str):
    ...     return fs.analyze_text(text)

Configuration is controlled via environment variables (see Settings docs):
    - FAIRSENSE_LLM_PROVIDER: "openai", "anthropic", "local", "fake"
    - FAIRSENSE_LLM_API_KEY: Your API key
    - FAIRSENSE_LLM_MODEL_NAME: Model to use (e.g., "gpt-4")
    - FAIRSENSE_OCR_TOOL: "tesseract", "paddleocr", "fake"
    - FAIRSENSE_CAPTION_MODEL: "blip", "blip2", "fake"

Per-request options can be passed to analysis methods (see options parameter docs).
"""

import time
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from fairsense_agentix.graphs.orchestrator_graph import create_orchestrator_graph
from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput


# ============================================================================
# Result Models (Pydantic for FastAPI compatibility)
# ============================================================================


class ResultMetadata(BaseModel):
    """Metadata about analysis execution.

    Attributes
    ----------
    run_id : str
        Unique identifier for tracing this execution
    workflow_id : str
        Which workflow was executed ("bias_text", "bias_image", "risk")
    execution_time_seconds : float
        Total execution time in seconds
    router_reasoning : str
        Human-readable explanation of why this workflow was selected
    router_confidence : float
        Router's confidence in workflow selection (0.0-1.0)
    refinement_count : int
        Number of refinement iterations performed (0 if none)
    preflight_score : float | None
        Pre-execution quality score (None if not evaluated, Phase 7+)
    posthoc_score : float | None
        Post-execution quality score (None if not evaluated, Phase 7+)
    """

    run_id: str = Field(description="Unique execution identifier for tracing")
    workflow_id: str = Field(description="Workflow executed")
    execution_time_seconds: float = Field(description="Total execution time")
    router_reasoning: str = Field(description="Why this workflow was selected")
    router_confidence: float = Field(
        ge=0.0, le=1.0, description="Router confidence (0.0-1.0)"
    )
    refinement_count: int = Field(ge=0, description="Number of refinement iterations")
    preflight_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Pre-execution quality score (Phase 7+)"
    )
    posthoc_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Post-execution quality score (Phase 7+)"
    )


class BiasResult(BaseModel):
    """Result from text or image bias analysis.

    This model is compatible with FastAPI (Pydantic-based) and provides
    attribute-style access to all results and metadata.

    Attributes
    ----------
    status : str
        Execution status: "success", "failed", or "partial"
    bias_detected : bool
        Whether bias was detected (None if using fake LLM)
    risk_level : str
        Risk level: "low", "medium", "high" (None if using fake LLM)
    bias_analysis : BiasAnalysisOutput | str | None
        Structured bias analysis (BiasAnalysisOutput) from real LLM,
        or string summary from fake LLM, or None if failed
    bias_instances : list[dict] | None
        List of bias instances with type, severity, text_span, explanation.
        None if using fake LLM or if no structured output
    summary : str | None
        Optional condensed summary of findings
    highlighted_html : str | None
        HTML with bias spans color-coded by type
    ocr_text : str | None
        Text extracted via OCR (image workflow only)
    caption_text : str | None
        Generated image caption (image workflow only)
    merged_text : str | None
        Combined OCR + caption text (image workflow only)
    metadata : ResultMetadata
        Execution metadata (timing, routing, quality scores)
    errors : list[str]
        Error messages (empty if successful)
    warnings : list[str]
        Warning messages (empty if none)

    Examples
    --------
    >>> result = analyze_text("Job posting")
    >>> if result.status == "success":
    ...     print(f"Bias detected: {result.bias_detected}")
    ...     print(f"Risk level: {result.risk_level}")
    ...     if result.bias_instances:
    ...         for inst in result.bias_instances:
    ...             print(f"  - {inst['type']}: {inst['text_span']}")
    >>> print(f"Execution time: {result.metadata.execution_time_seconds:.2f}s")
    >>> print(f"Router confidence: {result.metadata.router_confidence:.2%}")
    """

    status: Literal["success", "failed", "partial"] = Field(
        description="Execution status"
    )
    bias_detected: Optional[bool] = Field(None, description="Whether bias was detected")
    risk_level: Optional[Literal["low", "medium", "high"]] = Field(
        None, description="Overall risk level"
    )
    bias_analysis: Optional[Union[BiasAnalysisOutput, str]] = Field(
        None, description="Structured or string bias analysis"
    )
    bias_instances: Optional[list[dict[str, Any]]] = Field(
        None, description="Individual bias instances with details"
    )
    summary: Optional[str] = Field(None, description="Optional condensed summary")
    highlighted_html: Optional[str] = Field(
        None, description="HTML with color-coded bias highlighting"
    )

    # Image-specific fields (None for text workflow)
    ocr_text: Optional[str] = Field(None, description="OCR extracted text (image only)")
    caption_text: Optional[str] = Field(
        None, description="Generated caption (image only)"
    )
    merged_text: Optional[str] = Field(
        None, description="Combined OCR + caption (image only)"
    )

    # Metadata and diagnostics
    metadata: ResultMetadata = Field(description="Execution metadata")
    errors: list[str] = Field(default_factory=list, description="Error messages")
    warnings: list[str] = Field(default_factory=list, description="Warning messages")

    class Config:
        """Pydantic config."""

        # Allow BiasAnalysisOutput to be serialized
        arbitrary_types_allowed = True


class RiskResult(BaseModel):
    """Result from AI risk assessment.

    This model is compatible with FastAPI (Pydantic-based) and provides
    attribute-style access to all results and metadata.

    Attributes
    ----------
    status : str
        Execution status: "success", "failed", or "partial"
    embedding : list[float] | None
        Vector embedding of scenario (384 dimensions for default model)
    risks : list[dict[str, Any]]
        Top-k AI risks found via semantic search, each with:
        - name: str (risk name)
        - description: str (risk description)
        - score: float (similarity score 0.0-1.0)
        - category: str (risk category)
    rmf_recommendations : dict[str, list[dict[str, Any]]]
        AI-RMF recommendations per risk, keyed by risk ID
    html_table : str | None
        Formatted HTML table of risks + recommendations
    csv_path : str | None
        Path to exported CSV file
    metadata : ResultMetadata
        Execution metadata (timing, routing, quality scores)
    errors : list[str]
        Error messages (empty if successful)
    warnings : list[str]
        Warning messages (empty if none)

    Examples
    --------
    >>> result = assess_risk("AI deployment scenario")
    >>> if result.status == "success":
    ...     print(f"Found {len(result.risks)} risks")
    ...     for risk in result.risks[:3]:
    ...         print(f"  - {risk['name']}: {risk['score']:.3f}")
    >>> print(f"Execution time: {result.metadata.execution_time_seconds:.2f}s")
    """

    status: Literal["success", "failed", "partial"] = Field(
        description="Execution status"
    )
    embedding: Optional[list[float]] = Field(
        None, description="Vector embedding of scenario"
    )
    risks: list[dict[str, Any]] = Field(
        default_factory=list, description="Top-k AI risks from semantic search"
    )
    rmf_recommendations: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict, description="AI-RMF recommendations per risk"
    )
    html_table: Optional[str] = Field(
        None, description="Formatted HTML table of risks + recommendations"
    )
    csv_path: Optional[str] = Field(None, description="Path to exported CSV file")

    # Metadata and diagnostics
    metadata: ResultMetadata = Field(description="Execution metadata")
    errors: list[str] = Field(default_factory=list, description="Error messages")
    warnings: list[str] = Field(default_factory=list, description="Warning messages")


# ============================================================================
# Main API Class
# ============================================================================


class FairSense:
    """High-level API for bias detection and AI risk assessment.

    This class provides a clean interface for analyzing text, images, and
    AI deployment scenarios. It manages graph initialization internally and
    can be reused for multiple analyses (recommended for performance).

    Configuration
    -------------
    Configuration is controlled via environment variables or .env file.
    Set these before initializing FairSense:

        - FAIRSENSE_LLM_PROVIDER: "openai", "anthropic", "local", "fake"
        - FAIRSENSE_LLM_API_KEY: Your API key (required for real analysis)
        - FAIRSENSE_LLM_MODEL_NAME: Model to use (default per provider)
        - FAIRSENSE_OCR_TOOL: "tesseract", "paddleocr", "fake" (default "tesseract")
        - FAIRSENSE_CAPTION_MODEL: "blip", "blip2", "fake" (default "blip")
        - FAIRSENSE_TELEMETRY_ENABLED: "true" or "false" (default "true")

    See Settings documentation for all available configuration options.

    Per-Request Options
    -------------------
    All analysis methods accept optional **options kwargs:

    Text/Image Bias Detection:
        - temperature: LLM temperature (0.0-1.0, default 0.3)
        - max_tokens: Maximum LLM response tokens (default 2000)
        - ocr_language: OCR language code (default "eng")
        - caption_max_length: Max caption length (default 100)

    Risk Assessment:
        - top_k: Number of risks to retrieve (default 5, max 100)
        - rmf_per_risk: RMF recommendations per risk (default 3, max 20)

    Examples
    --------
    Basic usage:

        >>> fs = FairSense()
        >>> result = fs.analyze_text("Job posting text")
        >>> print(result.bias_detected, result.risk_level)

    With configuration:

        >>> result = fs.analyze_text(
        ...     "Job posting",
        ...     temperature=0.2,  # Lower for consistency
        ...     max_tokens=3000,  # More detailed analysis
        ... )

    Batch processing:

        >>> fs = FairSense()  # Initialize once
        >>> for text in texts:
        ...     result = fs.analyze_text(text)
        ...     print(f"{result.metadata.run_id}: {result.status}")

    FastAPI integration:

        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> fs = FairSense()
        >>> @app.post("/analyze", response_model=BiasResult)
        >>> def analyze(text: str):
        ...     return fs.analyze_text(text)
    """

    def __init__(self) -> None:
        """Initialize FairSense API.

        This creates and compiles the orchestrator graph. Graph compilation is
        expensive (~1-2s), so reuse this instance for multiple analyses.
        """
        self._graph = create_orchestrator_graph()

    def analyze_text(
        self,
        text: str,
        **options: Any,
    ) -> BiasResult:
        """Detect bias in text.

        Analyzes text for various forms of bias including gender, age, racial,
        disability, and socioeconomic bias. Uses LLM to identify bias instances,
        generate explanations, and produce color-coded HTML output.

        Parameters
        ----------
        text : str
            Text to analyze for bias
        **options
            Optional configuration overrides:
            - temperature: LLM temperature (0.0-1.0, default 0.3)
            - max_tokens: Maximum response tokens (default 2000)

        Returns
        -------
        BiasResult
            Complete analysis result with bias detection, instances, HTML output,
            execution metadata, and any errors/warnings

        Examples
        --------
        >>> fs = FairSense()
        >>> result = fs.analyze_text("We need young rockstar developers")
        >>> if result.bias_detected:
        ...     print(f"Risk: {result.risk_level}")
        ...     for inst in result.bias_instances or []:
        ...         print(f"  {inst['type']}: {inst['text_span']}")
        """
        start_time = time.time()

        result = self._graph.invoke(
            {
                "input_type": "text",
                "content": text,
                "options": options,
            }
        )

        execution_time = time.time() - start_time

        return self._build_bias_result(result, execution_time)

    def analyze_image(
        self,
        image: Union[bytes, Path],
        **options: Any,
    ) -> BiasResult:
        """Detect bias in images.

        Analyzes images by extracting text (OCR), generating captions, and
        performing bias analysis on the combined text. Uses parallel execution
        for OCR and captioning for optimal performance.

        Parameters
        ----------
        image : bytes | Path
            Image data as bytes or path to image file
        **options
            Optional configuration overrides:
            - ocr_language: OCR language code (default "eng")
            - caption_max_length: Max caption length (default 100)
            - temperature: LLM temperature (0.0-1.0, default 0.3)

        Returns
        -------
        BiasResult
            Complete analysis result including OCR text, caption, bias detection,
            HTML output, execution metadata, and any errors/warnings

        Examples
        --------
        >>> fs = FairSense()
        >>> result = fs.analyze_image(Path("hiring_ad.png"))
        >>> print(f"OCR: {result.ocr_text}")
        >>> print(f"Caption: {result.caption_text}")
        >>> if result.bias_detected:
        ...     print(f"Bias found: {result.risk_level}")
        """
        start_time = time.time()

        # Load image if path provided
        if isinstance(image, Path):
            image = image.read_bytes()

        result = self._graph.invoke(
            {
                "input_type": "image",
                "content": image,
                "options": options,
            }
        )

        execution_time = time.time() - start_time

        return self._build_bias_result(result, execution_time)

    def assess_risk(
        self,
        scenario: str,
        **options: Any,
    ) -> RiskResult:
        """Assess AI risks in deployment scenario.

        Performs semantic search against 280+ AI risks from the NIST AI Risk
        Management Framework. Retrieves relevant risks and corresponding
        mitigation recommendations.

        Parameters
        ----------
        scenario : str
            AI deployment scenario description
        **options
            Optional configuration overrides:
            - top_k: Number of risks to retrieve (default 5, max 100)
            - rmf_per_risk: Recommendations per risk (default 3, max 20)

        Returns
        -------
        RiskResult
            Complete risk assessment including identified risks, RMF recommendations,
            HTML table, CSV export path, execution metadata, and any errors/warnings

        Examples
        --------
        >>> fs = FairSense()
        >>> result = fs.assess_risk(
        ...     "Facial recognition for employee monitoring", top_k=10
        ... )
        >>> print(f"Found {len(result.risks)} risks")
        >>> for risk in result.risks:
        ...     print(f"  {risk['name']}: {risk['score']:.3f}")
        >>> print(f"CSV exported to: {result.csv_path}")
        """
        start_time = time.time()

        result = self._graph.invoke(
            {
                "input_type": "csv",  # CSV triggers risk workflow
                "content": scenario,
                "options": options,
            }
        )

        execution_time = time.time() - start_time

        return self._build_risk_result(result, execution_time)

    def _build_bias_result(
        self, raw_result: dict[str, Any], execution_time: float
    ) -> BiasResult:
        """Build BiasResult from orchestrator output."""
        final = raw_result["final_result"]
        output = final.get("output") or {}

        # Extract bias analysis (always BiasAnalysisOutput now)
        bias_analysis = output.get("bias_analysis")

        # Determine if structured or string
        bias_detected = None
        risk_level = None
        bias_instances = None

        if isinstance(bias_analysis, BiasAnalysisOutput):
            # Structured output (always the case now with new design)
            bias_detected = bias_analysis.bias_detected
            risk_level = bias_analysis.risk_level
            # Convert Pydantic models to dicts for JSON serialization
            bias_instances = [
                inst.model_dump() for inst in bias_analysis.bias_instances
            ]
        elif bias_analysis is None:
            # No analysis (shouldn't happen in normal flow)
            bias_detected = None
            risk_level = None
        else:
            # Unexpected type (shouldn't happen with new design)
            raise TypeError(
                f"Unexpected bias_analysis type: {type(bias_analysis)}. "
                "Expected BiasAnalysisOutput."
            )

        # Build metadata
        metadata = ResultMetadata(
            run_id=final["run_id"],
            workflow_id=final["workflow_id"],
            execution_time_seconds=execution_time,
            router_reasoning=final["metadata"]["plan_reasoning"] or "",
            router_confidence=final["metadata"]["plan_confidence"] or 0.0,
            refinement_count=final["refinement_count"],
            preflight_score=final["metadata"].get("preflight_score"),
            posthoc_score=final["metadata"].get("posthoc_score"),
        )

        return BiasResult(
            status=final["status"],
            bias_detected=bias_detected,
            risk_level=risk_level,
            bias_analysis=bias_analysis,
            bias_instances=bias_instances,
            summary=output.get("summary"),
            highlighted_html=output.get("highlighted_html"),
            ocr_text=output.get("ocr_text"),
            caption_text=output.get("caption_text"),
            merged_text=output.get("merged_text"),
            metadata=metadata,
            errors=final["errors"],
            warnings=final["warnings"],
        )

    def _build_risk_result(
        self, raw_result: dict[str, Any], execution_time: float
    ) -> RiskResult:
        """Build RiskResult from orchestrator output."""
        final = raw_result["final_result"]
        output = final.get("output") or {}

        # Build metadata
        metadata = ResultMetadata(
            run_id=final["run_id"],
            workflow_id=final["workflow_id"],
            execution_time_seconds=execution_time,
            router_reasoning=final["metadata"]["plan_reasoning"] or "",
            router_confidence=final["metadata"]["plan_confidence"] or 0.0,
            refinement_count=final["refinement_count"],
            preflight_score=final["metadata"].get("preflight_score"),
            posthoc_score=final["metadata"].get("posthoc_score"),
        )

        return RiskResult(
            status=final["status"],
            embedding=output.get("embedding"),
            risks=output.get("risks", []),
            rmf_recommendations=output.get("rmf_recommendations", {}),
            html_table=output.get("html_table"),
            csv_path=output.get("csv_path"),
            metadata=metadata,
            errors=final["errors"],
            warnings=final["warnings"],
        )


# ============================================================================
# Convenience Functions (for quick scripts)
# ============================================================================


def analyze_text(text: str, **options: Any) -> BiasResult:
    """Detect bias in text (convenience function).

    This function creates a new FairSense instance for each call. For better
    performance when analyzing multiple texts, create a FairSense instance
    and reuse it.

    Parameters
    ----------
    text : str
        Text to analyze for bias
    **options
        Optional configuration (see FairSense.analyze_text for details)

    Returns
    -------
    BiasResult
        Complete analysis result

    Examples
    --------
    >>> from fairsense_agentix import analyze_text
    >>> result = analyze_text("Job posting text")
    >>> print(result.bias_detected)
    """
    fs = FairSense()
    return fs.analyze_text(text, **options)


def analyze_image(image: Union[bytes, Path], **options: Any) -> BiasResult:
    """Detect bias in image (convenience function).

    This function creates a new FairSense instance for each call. For better
    performance when analyzing multiple images, create a FairSense instance
    and reuse it.

    Parameters
    ----------
    image : bytes | Path
        Image data or path to image file
    **options
        Optional configuration (see FairSense.analyze_image for details)

    Returns
    -------
    BiasResult
        Complete analysis result

    Examples
    --------
    >>> from fairsense_agentix import analyze_image
    >>> result = analyze_image(Path("hiring_ad.png"))
    >>> print(result.ocr_text, result.bias_detected)
    """
    fs = FairSense()
    return fs.analyze_image(image, **options)


def assess_risk(scenario: str, **options: Any) -> RiskResult:
    """Assess AI risks in scenario (convenience function).

    This function creates a new FairSense instance for each call. For better
    performance when analyzing multiple scenarios, create a FairSense instance
    and reuse it.

    Parameters
    ----------
    scenario : str
        AI deployment scenario description
    **options
        Optional configuration (see FairSense.assess_risk for details)

    Returns
    -------
    RiskResult
        Complete risk assessment result

    Examples
    --------
    >>> from fairsense_agentix import assess_risk
    >>> result = assess_risk("Facial recognition deployment")
    >>> print(f"Found {len(result.risks)} risks")
    """
    fs = FairSense()
    return fs.assess_risk(scenario, **options)
