"""BiasImageGraph subgraph for image bias analysis.

This workflow extracts information from images via parallel OCR and captioning,
merges the extracted text, then analyzes for bias. The parallel extraction
pattern enables faster processing.

Workflow:
    START → [extract_ocr ∥ generate_caption] → merge_text → analyze_bias
         → summarize → highlight → END

Phase 2 (Checkpoint 2.5): Stub implementations with fake OCR/caption/LLM
Phase 5+: Real OCR integration, captioning models, LLM analysis
Phase 6.2: Span extraction with evidence source tracking (caption vs OCR)

Example:
    >>> from fairsense_agentix.graphs.bias_image_graph import create_bias_image_graph
    >>> graph = create_bias_image_graph()
    >>> result = graph.invoke({"image_bytes": b"fake_image_data", "options": {}})
    >>> result["bias_analysis"]
    '...'
"""

from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fairsense_agentix.configs import settings
from fairsense_agentix.graphs.state import BiasImageState
from fairsense_agentix.tools import get_tool_registry
from fairsense_agentix.tools.llm.output_schemas import BiasAnalysisOutput


# ============================================================================
# Helper Functions
# ============================================================================


def _extract_spans_from_analysis(
    bias_analysis: str | BiasAnalysisOutput,
    merged_text: str,
    ocr_text: str,
    caption_text: str,
) -> list[tuple[int, int, str]]:
    """Extract bias spans with character positions from image analysis.

    Phase 6.2: Parse BiasAnalysisOutput to extract (start_char, end_char, type)
    tuples for HTML highlighting. Handles multiple text sources (OCR + caption)
    with evidence_source tracking.

    For image analysis, merged_text combines OCR and caption with headers.
    We map character positions from individual sources to merged_text.

    Parameters
    ----------
    bias_analysis : str | BiasAnalysisOutput
        Bias analysis output (JSON string or Pydantic object)
    merged_text : str
        Combined OCR + caption text that was analyzed
    ocr_text : str
        Original OCR extracted text
    caption_text : str
        Original caption generated text

    Returns
    -------
    list[tuple[int, int, str]]
        List of (start_char, end_char, bias_type) tuples for highlighting
        in the merged_text

    Examples
    --------
    >>> analysis = BiasAnalysisOutput(
    ...     bias_detected=True,
    ...     bias_instances=[
    ...         BiasInstance(
    ...             type="gender",
    ...             evidence_source="ocr_text",
    ...             start_char=0,
    ...             end_char=8,
    ...         )
    ...     ],
    ... )
    >>> _extract_spans_from_analysis(analysis, merged, "ocr", "caption")
    [(27, 35, 'gender')]  # Adjusted for merged_text offset
    """
    # Parse analysis if string
    if isinstance(bias_analysis, str):
        try:
            analysis = BiasAnalysisOutput.model_validate_json(bias_analysis)
        except Exception:
            return []
    else:
        analysis = bias_analysis

    # Calculate offsets for OCR and caption in merged_text
    # Format: "**OCR Extracted Text:**\n{ocr}\n\n**Image Caption:**\n{caption}"
    ocr_header = "**OCR Extracted Text:**\n"
    caption_header = "\n\n**Image Caption:**\n"

    ocr_offset = len(ocr_header)
    caption_offset = len(ocr_header) + len(ocr_text) + len(caption_header)

    # Extract spans with validation and offset adjustment
    spans: list[tuple[int, int, str]] = []
    merged_length = len(merged_text)

    for instance in analysis.bias_instances:
        # Determine which text source this span refers to
        evidence_source = instance.evidence_source

        if evidence_source == "ocr_text":
            source_text = ocr_text
            offset = ocr_offset
        elif evidence_source == "caption":
            source_text = caption_text
            offset = caption_offset
        else:
            # Unknown source or "text" (generic) - skip
            continue

        # Get positions relative to source text
        start = instance.start_char
        end = instance.end_char

        # Validate bounds in source text
        if not (0 <= start < len(source_text) and start < end <= len(source_text)):
            continue

        # Validate text_span matches (if provided)
        if instance.text_span and source_text[start:end] != instance.text_span:
            continue

        # Adjust to merged_text coordinates
        merged_start = offset + start
        merged_end = offset + end

        # Final validation in merged_text
        if not (
            0 <= merged_start < merged_length
            and merged_start < merged_end <= merged_length
        ):
            continue

        # Valid span - add to list
        spans.append((merged_start, merged_end, instance.type))

    return spans


# ============================================================================
# Node Functions
# ============================================================================


def extract_ocr(state: BiasImageState) -> dict[str, str]:
    """OCR node: Extract text from image using OCR tool.

    Uses OCR (Optical Character Recognition) to extract text content from
    the image. Supports multiple OCR backends (Tesseract, PaddleOCR, etc.).

    Executes in parallel with generate_caption for faster processing.

    Phase 4+: Uses tool registry to get OCR implementation (fake or real).
    Phase 5+: Real OCR integration with confidence thresholds.

    Parameters
    ----------
    state : BiasImageState
        Current state with image_bytes

    Returns
    -------
    dict
        State update with ocr_text field

    Examples
    --------
    >>> state = BiasImageState(image_bytes=b"image_data")
    >>> update = extract_ocr(state)
    >>> "ocr_text" in update
    True
    """
    # Get tools from registry
    registry = get_tool_registry()

    # Extract options
    language = state.options.get("ocr_language", "eng")
    confidence_threshold = state.options.get("ocr_confidence", 0.5)

    # Validate parameters
    if not isinstance(confidence_threshold, (int, float)) or not (
        0.0 <= confidence_threshold <= 1.0
    ):
        confidence_threshold = 0.5  # Use default for invalid values
    if not isinstance(language, str) or not language:
        language = "eng"  # Use default for invalid values

    # Use OCR tool from registry with error handling
    try:
        ocr_text = registry.ocr.extract(
            image_bytes=state.image_bytes,
            language=language,
            confidence_threshold=confidence_threshold,
        )
    except Exception:
        # TODO Phase 8: Log OCR failure with telemetry
        # Fallback: Empty string allows workflow to continue with caption-only
        # OCR can fail for many reasons (bad image format, corrupted data, etc.)
        # Since caption runs in parallel, we can continue with single data source
        ocr_text = ""

    return {"ocr_text": ocr_text}


def generate_caption(state: BiasImageState) -> dict[str, str]:
    """Captioning node: Generate descriptive caption for image.

    Uses image captioning models (BLIP, BLIP-2, LLaVA, etc.) to generate
    a natural language description of the image content. This captures
    visual elements that OCR might miss.

    Executes in parallel with extract_ocr for faster processing.

    Phase 4+: Uses tool registry to get caption implementation (fake or real).
    Phase 5+: Real captioning using BLIP-2 or similar models.

    Parameters
    ----------
    state : BiasImageState
        Current state with image_bytes

    Returns
    -------
    dict
        State update with caption_text field

    Examples
    --------
    >>> state = BiasImageState(image_bytes=b"image_data")
    >>> update = generate_caption(state)
    >>> "caption_text" in update
    True
    """
    # Get tools from registry
    registry = get_tool_registry()

    # Extract options
    max_length = state.options.get("caption_max_length", 100)

    # Validate max_length parameter
    if not isinstance(max_length, int) or max_length <= 0:
        max_length = 100  # Use default for invalid values

    # Use caption tool from registry with error handling
    try:
        caption_text = registry.caption.caption(
            image_bytes=state.image_bytes, max_length=max_length
        )
    except Exception:
        # TODO Phase 8: Log caption failure with telemetry
        # Fallback: Empty string allows workflow to continue with OCR-only
        # Caption models can fail (model loading error, OOM, invalid image, etc.)
        # Since OCR runs in parallel, we can continue with single data source
        caption_text = ""

    return {"caption_text": caption_text}


def merge_text(state: BiasImageState) -> dict[str, str]:
    """Merge node: Combine OCR and caption text into unified input.

    Merges the extracted OCR text and generated caption into a single
    text input for bias analysis. Performs deduplication to avoid
    redundancy.

    Parameters
    ----------
    state : BiasImageState
        Current state with ocr_text and caption_text

    Returns
    -------
    dict
        State update with merged_text field

    Examples
    --------
    >>> state = BiasImageState(
    ...     image_bytes=b"...",
    ...     ocr_text="Text from OCR",
    ...     caption_text="Description from caption",
    ... )
    >>> update = merge_text(state)
    >>> "merged_text" in update
    True
    """
    # Phase 2: Simple concatenation
    # Phase 5+: Smart merging with deduplication and formatting

    ocr_text = state.ocr_text or ""
    caption_text = state.caption_text or ""

    # If BOTH extraction methods failed, we can't continue
    if not ocr_text and not caption_text:
        # TODO Phase 8: Log dual-extraction failure with telemetry
        # Raise error that orchestrator will catch and handle
        raise ValueError(
            "Both OCR and caption extraction failed - no text available for analysis"
        )

    # Simple merge with section headers
    merged = f"""**OCR Extracted Text:**
{ocr_text}

**Image Caption:**
{caption_text}"""

    return {"merged_text": merged}


def analyze_bias(state: BiasImageState) -> dict[str, str]:
    """LLM node: Analyze merged text for various forms of bias.

    Uses LLM to identify and explain bias in the merged OCR + caption text
    across multiple dimensions (gender, age, race, disability, socioeconomic).

    This is conceptually similar to BiasTextGraph's analyze_bias, but operates
    on the merged text from image analysis.

    Phase 4+: Uses tool registry to get LLM implementation (fake or real).
    Phase 5+: Real LLM call with bias detection prompt.
    Phase 6.2: Uses bias_analysis_v1.txt prompt template with structured output
               and evidence_source tracking (caption vs ocr_text).

    Parameters
    ----------
    state : BiasImageState
        Current state with merged_text

    Returns
    -------
    dict
        State update with bias_analysis field

    Examples
    --------
    >>> state = BiasImageState(image_bytes=b"...", merged_text="...")
    >>> update = analyze_bias(state)
    >>> "bias_analysis" in update
    True
    """
    # Get tools from registry
    registry = get_tool_registry()

    # Extract options
    merged_text = state.merged_text or ""
    temperature = state.options.get("temperature", 0.3)
    max_tokens = state.options.get("llm_max_tokens", 2000)

    # Validate parameters
    if not isinstance(temperature, (int, float)) or not (0.0 <= temperature <= 1.0):
        temperature = 0.3  # Use default for invalid values
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = 2000  # Use default for invalid values

    # Phase 6.2: Load prompt template for bias analysis
    # Template instructs LLM to return precise character positions
    prompt_template_path = (
        Path(__file__).parent.parent / "prompts" / "templates" / "bias_analysis_v1.txt"
    )

    try:
        with open(prompt_template_path) as f:
            prompt_template = f.read()
    except FileNotFoundError:
        # Fallback to hardcoded prompt if template not found
        prompt_template = """You are a bias detection specialist analyzing text for various forms of bias.

Your task is to carefully analyze the provided text and identify ANY instances of bias across these categories:
- **Gender bias**: Gendered language, stereotypes, exclusionary terms
- **Age bias**: Age discrimination, ageist assumptions, generational stereotypes
- **Racial/ethnic bias**: Racial stereotypes, cultural assumptions, exclusionary language
- **Disability bias**: Ableist language, accessibility assumptions, capability stereotypes
- **Socioeconomic bias**: Class-based assumptions, privilege bias, economic stereotypes

**TEXT TO ANALYZE:**
{text}

**IMPORTANT FOR IMAGE ANALYSIS:** Specify evidence_source as "caption" or "ocr_text" for each bias instance.

Return valid JSON with structure: {{"bias_detected": bool, "bias_instances": [...], "overall_assessment": str, "risk_level": str}}"""

    # Build prompt with merged text
    prompt = prompt_template.format(text=merged_text)

    # Call LLM via registry with structured output
    bias_analysis = registry.llm.predict(
        prompt=prompt, temperature=temperature, max_tokens=max_tokens
    )

    # Convert structured output to JSON string if needed
    # With .with_structured_output(), LLM returns Pydantic objects
    from pydantic import BaseModel  # noqa: PLC0415

    if isinstance(bias_analysis, BaseModel):
        # Structured output: convert to formatted JSON string
        bias_analysis_str = bias_analysis.model_dump_json(indent=2)
    else:
        # Plain string output (fake tools or legacy)
        bias_analysis_str = bias_analysis

    return {"bias_analysis": bias_analysis_str}


def summarize(state: BiasImageState) -> dict[str, str | None]:
    """Summarization node: Generate concise summary of bias findings.

    Creates a short executive summary of the bias analysis. Useful for
    quick insights or when merged text is long.

    Parameters
    ----------
    state : BiasImageState
        Current state with bias_analysis

    Returns
    -------
    dict
        State update with summary field

    Examples
    --------
    >>> state = BiasImageState(image_bytes=b"...", bias_analysis="...")
    >>> update = summarize(state)
    >>> "summary" in update
    True
    """
    # Get tools from registry
    registry = get_tool_registry()

    if state.bias_analysis is None:
        return {"summary": None}

    # Use summarizer tool to generate summary with error handling
    max_length = state.options.get("summary_max_length", 200)

    # Validate max_length parameter
    if not isinstance(max_length, int) or max_length <= 0:
        max_length = 200  # Use default for invalid values

    try:
        summary = registry.summarizer.summarize(
            text=state.bias_analysis, max_length=max_length
        )
    except Exception:
        # TODO Phase 8: Log summarization failure with telemetry
        # Fallback: None (summary is already optional in the workflow)
        # If summarization fails, workflow continues without summary
        summary = None

    return {"summary": summary}


def highlight(state: BiasImageState) -> dict[str, str]:
    """Highlighting node: Generate HTML with bias spans highlighted.

    Creates HTML representation of the merged text with problematic spans
    highlighted by bias type. Since this is image-derived text, the HTML
    includes both OCR and caption sections.

    Phase 6.2: Extracts precise character positions from bias_analysis
               with evidence source tracking (caption vs OCR).

    Parameters
    ----------
    state : BiasImageState
        Current state with merged_text and bias_analysis

    Returns
    -------
    dict
        State update with highlighted_html field

    Examples
    --------
    >>> state = BiasImageState(
    ...     image_bytes=b"...", merged_text="...", bias_analysis="..."
    ... )
    >>> update = highlight(state)
    >>> "<html>" in update["highlighted_html"]
    True
    """
    # Get tools from registry
    registry = get_tool_registry()

    merged_text = state.merged_text or ""

    # Phase 6.2: Extract spans from bias analysis with evidence source tracking
    spans: list[tuple[int, int, str]] = []

    if state.bias_analysis:
        # Extract (start_char, end_char, bias_type) tuples from analysis
        # This handles mapping from OCR/caption positions to merged_text
        spans = _extract_spans_from_analysis(
            bias_analysis=state.bias_analysis,
            merged_text=merged_text,
            ocr_text=state.ocr_text or "",
            caption_text=state.caption_text or "",
        )

    # Get bias type colors from configuration
    bias_types = settings.get_bias_type_colors()

    # Use formatter tool to generate highlighted HTML with error handling
    try:
        highlighted_html = registry.formatter.highlight(
            text=merged_text, spans=spans, bias_types=bias_types
        )
    except Exception:
        # TODO Phase 8: Log formatting failure with telemetry
        # Fallback: Plain HTML with unformatted text
        # Formatting is presentation layer - if it fails, provide plain text
        # rather than killing the entire workflow
        highlighted_html = f"<pre>{merged_text}</pre>"

    return {"highlighted_html": highlighted_html}


# Note: Quality assessment removed from subgraph
# Orchestrator's posthoc_eval node owns all evaluation (Phase 7+)


# ============================================================================
# Graph Construction
# ============================================================================


def create_bias_image_graph() -> CompiledStateGraph:
    """Create and compile the BiasImageGraph subgraph.

    Builds the image bias analysis workflow with parallel OCR and captioning.

    Graph Structure:
        START → extract_ocr (parallel with generate_caption)
        START → generate_caption (parallel with extract_ocr)
        Both → merge_text → analyze_bias → summarize → highlight → END

    Note: Quality assessment is handled by orchestrator's posthoc_eval node.

    Returns
    -------
    StateGraph
        Compiled bias image graph ready for invocation

    Examples
    --------
    >>> graph = create_bias_image_graph()
    >>> result = graph.invoke({"image_bytes": b"fake_image_data", "options": {}})
    >>> result["bias_analysis"]
    '...'
    """
    # Create graph with BiasImageState
    workflow = StateGraph(BiasImageState)

    # Add nodes
    workflow.add_node("extract_ocr", extract_ocr)
    workflow.add_node("generate_caption", generate_caption)
    workflow.add_node("merge_text", merge_text)
    workflow.add_node("analyze_bias", analyze_bias)
    workflow.add_node("summarize", summarize)
    workflow.add_node("highlight", highlight)

    # Add edges for parallel execution
    # START → both OCR and caption (parallel)
    workflow.add_edge(START, "extract_ocr")
    workflow.add_edge(START, "generate_caption")

    # Both parallel nodes → merge (fan-in)
    workflow.add_edge("extract_ocr", "merge_text")
    workflow.add_edge("generate_caption", "merge_text")

    # Sequential after merge
    workflow.add_edge("merge_text", "analyze_bias")
    workflow.add_edge("analyze_bias", "summarize")
    workflow.add_edge("summarize", "highlight")
    workflow.add_edge("highlight", END)

    # Compile graph
    return workflow.compile()
